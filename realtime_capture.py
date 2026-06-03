#!/usr/bin/env python3
"""
Real-time Network Traffic Feature Extraction and Anomaly Detection
Captures network packets, extracts CIC-IDS style features, and runs through
the new ConvAttention+LSTM VAE (aligned with cic-ids-vae.ipynb / cic_neural.ipynb).

OOD scoring uses the same reconstruction-loss fusion approach as the neural notebook:
  attack_score = (1 - w) * (1 - proba_benign) + w * recon_score
No CatBoost dependency -- purely neural.
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from threading import Thread, Lock
from typing import Tuple
from sklearn.preprocessing import QuantileTransformer

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import sys as _sys
    from scapy.all import sniff, IP, TCP, UDP, conf as scapy_conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠ scapy not installed – packet capture disabled. Run: pip install scapy")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
WINDOW_SIZE = 30          # Sliding window – same as cic_ids_vae_benchmark.py
DF_DEG = 3.0
RECON_STUDENT_WEIGHT = 0.4
RECON_MSE_WEIGHT = 0.2
RECON_OOD_WEIGHT = 0.45   # Match neural notebook fusion weight
EPS = 1e-8

# OOD binary threshold (can be overridden at runtime)
DEFAULT_OOD_THRESHOLD = 0.55

ATTACK_CLASS_NAMES = {
    0: "Benign",
    1: "DoS",
    2: "DDoS",
    3: "Brute-Force",
    4: "Web Attack",
}

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [PROJECT_ROOT, PROJECT_ROOT / "datasets"]

TOP_FEATURES = [
    "Fwd Pkts/s", "Init Bwd Win Byts", "Flow Pkts/s", "Fwd Seg Size Min",
    "Init Fwd Win Byts", "Flow IAT Std", "Pkt Len Max", "ACK Flag Cnt",
    "Fwd Header Len", "Fwd Pkt Len Std", "Bwd Pkts/s", "Flow Byts/s",
    "Fwd Pkt Len Max", "Bwd Header Len", "Fwd IAT Tot", "Bwd Pkt Len Max",
    "Fwd Pkt Len Mean", "URG Flag Cnt", "Fwd IAT Std", "Pkt Len Std",
    "Flow IAT Min", "Flow IAT Mean", "Down/Up Ratio",
]


# ---------------------------------------------------------------------------
# NEW VAE Architecture (ConvAttention + LSTM) – must match cic-ids-vae.ipynb
# ---------------------------------------------------------------------------
class ConvAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1, conv_kernel_size: int = 3):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        conv_padding = conv_kernel_size // 2
        self.local_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=conv_kernel_size, padding=conv_padding, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        ).double()
        self.norm_local = nn.LayerNorm(embed_dim).double()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        ).double()
        self.norm_attention = nn.LayerNorm(embed_dim).double()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim),
        ).double()
        self.norm_ffn = nn.LayerNorm(embed_dim).double()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm_local(x + local_features)
        attn_out, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.norm_attention(x + self.dropout(attn_out))
        return self.norm_ffn(x + self.dropout(self.ffn(x)))


class Encoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, conv_channels: int = 128, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(n_features, conv_channels, kernel_size=kernel_size, padding=0).double()
        self.relu = nn.ReLU()
        self.attention = ConvAttentionBlock(embed_dim=conv_channels)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=conv_channels, batch_first=True).double()
        self.fc_mu = nn.Linear(conv_channels, latent_dim).double()
        self.fc_logvar = nn.Linear(conv_channels, latent_dim).double()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = self.attention(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, conv_channels: int = 128, kernel_size: int = 3, output_window_size: int = WINDOW_SIZE):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_window_size = output_window_size
        self.fc = nn.Linear(latent_dim, output_window_size * conv_channels).double()
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=conv_channels, batch_first=True).double()
        self.deconv1 = nn.Conv1d(conv_channels, n_features, kernel_size=kernel_size, padding=0).double()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z)
        z = z.view(z.size(0), self.output_window_size, -1)
        x, _ = self.lstm(z)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.deconv1(x)
        return x.permute(0, 2, 1)


class VAE(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, conv_channels: int = 128, kernel_size: int = 3, window_size: int = WINDOW_SIZE):
        super().__init__()
        self.encoder = Encoder(n_features, latent_dim, conv_channels, kernel_size)
        self.decoder = Decoder(n_features, latent_dim, conv_channels, kernel_size, output_window_size=window_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, deterministic)
        return self.decoder(z), mu, logvar


# ---------------------------------------------------------------------------
# Neural Classifier Architecture (BinaryFirstTabularNet / OpenWorldTabularNet)
# ---------------------------------------------------------------------------
MODEL_WIDTH = 224
MODEL_DEPTH = 5
EMBED_DIM = 112
PROTOTYPE_TEMPERATURE = 0.14
MODEL_DROPOUT = 0.14
N_ROW_ORDER_DOMAINS = 6

class GatedResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, width * 2)
        self.out = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = self.gate(self.norm(x)).chunk(2, dim=-1)
        update = values * F.silu(gates)
        return x + self.dropout(self.out(update))

class BinaryFirstTabularNet(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 5, n_attack_classes: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, MODEL_WIDTH),
            nn.LayerNorm(MODEL_WIDTH),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
        )
        self.blocks = nn.Sequential(*[GatedResidualBlock(MODEL_WIDTH, MODEL_DROPOUT) for _ in range(MODEL_DEPTH)])
        self.embed = nn.Sequential(
            nn.LayerNorm(MODEL_WIDTH),
            nn.Linear(MODEL_WIDTH, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
            nn.Linear(EMBED_DIM // 2, 2),
        )
        self.attack_subtype_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
            nn.Linear(EMBED_DIM, n_attack_classes),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.SiLU(),
            nn.Dropout(MODEL_DROPOUT),
            nn.Linear(EMBED_DIM // 2, N_ROW_ORDER_DOMAINS),
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, EMBED_DIM) * 0.05)

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        hidden = self.blocks(self.stem(x))
        embedding = self.embed(hidden)
        return {
            "embedding": embedding,
            "binary_logits": self.binary_head(embedding),
            "attack_subtype_logits": self.attack_subtype_head(embedding),
            "prototype_logits": self.prototype_logits(embedding),
        }

    def prototype_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embedding, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return z @ proto.T / PROTOTYPE_TEMPERATURE

OpenWorldTabularNet = BinaryFirstTabularNet


# ---------------------------------------------------------------------------
# Reconstruction scorer (same as neural notebook)
# ---------------------------------------------------------------------------
class ReconScorer:
    """Normalises reconstruction loss to [0, 1] using train-split percentiles."""

    def __init__(self, p5: np.ndarray, p95: np.ndarray):
        self.p5 = p5.astype(np.float32)
        self.p95 = p95.astype(np.float32)

    @classmethod
    def load(cls, path: Path) -> "ReconScorer":
        obj = joblib.load(path)
        return cls(obj["p5"], obj["p95"])

    def score(self, mse: float, kld: float) -> float:
        vals = np.array([[mse, kld]], dtype=np.float32)
        span = np.maximum(self.p95 - self.p5, EPS)
        scaled = (vals - self.p5) / span
        raw = float(np.clip(scaled.mean(), 0.0, None))
        return float(np.clip(1.0 - np.exp(-raw), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Flow Tracker (unchanged logic)
# ---------------------------------------------------------------------------
class FlowTracker:
    """Tracks network flows and exports them when finished (TCP FIN/RST or idle timeout)."""

    def __init__(self, idle_timeout: float = 5.0):
        self.flows = defaultdict(lambda: {
            "fwd_packets": [], "bwd_packets": [],
            "fwd_timestamps": [], "bwd_timestamps": [],
            "fwd_header_lens": [], "bwd_header_lens": [],
            "init_fwd_win": None, "init_bwd_win": None,
            "ack_count": 0, "urg_count": 0, "start_time": None,
            "last_active": None, "finished": False,
        })
        self.idle_timeout = idle_timeout
        self.lock = Lock()

    def get_flow_key(self, pkt):
        if IP in pkt:
            src_ip, dst_ip, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
            sp = pkt.sport if (TCP in pkt or UDP in pkt) else 0
            dp = pkt.dport if (TCP in pkt or UDP in pkt) else 0
            return (src_ip, sp, dst_ip, dp, proto) if (src_ip, sp) < (dst_ip, dp) else (dst_ip, dp, src_ip, sp, proto)
        return None

    def is_forward(self, pkt, key):
        if IP in pkt:
            sp = pkt.sport if (TCP in pkt or UDP in pkt) else 0
            return (pkt[IP].src, sp) == (key[0], key[1])
        return True

    def process_packet(self, pkt):
        key = self.get_flow_key(pkt)
        if key is None:
            return
        with self.lock:
            f = self.flows[key]
            ts = time.time()
            if f["start_time"] is None:
                f["start_time"] = ts
            f["last_active"] = ts
            pkt_len = len(pkt)
            fwd = self.is_forward(pkt, key)
            hdr = 20
            if TCP in pkt:
                hdr = pkt[TCP].dataofs * 4
                flags = str(pkt[TCP].flags)
                if "A" in flags:
                    f["ack_count"] += 1
                if "U" in flags:
                    f["urg_count"] += 1
                win = pkt[TCP].window
                if fwd and f["init_fwd_win"] is None:
                    f["init_fwd_win"] = win
                elif not fwd and f["init_bwd_win"] is None:
                    f["init_bwd_win"] = win
                # End flow on FIN or RST
                if "F" in flags or "R" in flags:
                    f["finished"] = True
            if fwd:
                f["fwd_packets"].append(pkt_len)
                f["fwd_timestamps"].append(ts)
                f["fwd_header_lens"].append(hdr)
            else:
                f["bwd_packets"].append(pkt_len)
                f["bwd_timestamps"].append(ts)
                f["bwd_header_lens"].append(hdr)

    def pop_finished_flows(self) -> list:
        now = time.time()
        finished_keys = []
        with self.lock:
            for key, f in self.flows.items():
                if f["finished"] or (f["last_active"] is not None and (now - f["last_active"]) >= self.idle_timeout):
                    finished_keys.append(key)
            
            features = []
            for key in finished_keys:
                f = self.flows.pop(key)
                fp, bp = f["fwd_packets"], f["bwd_packets"]
                ft, bt = f["fwd_timestamps"], f["bwd_timestamps"]
                total = len(fp) + len(bp)
                if total == 0:
                    continue
                dur = max(max(ft) if ft else 0, max(bt) if bt else 0) - f["start_time"]
                dur = max(dur, 1e-3)
                all_ts = sorted(ft + bt)
                iats = np.diff(all_ts) if len(all_ts) > 1 else [0]
                fwd_iats = np.diff(sorted(ft)) if len(ft) > 1 else [0]
                feat = {
                    "_flow_key": f"{key[0]}:{key[1]} -> {key[2]}:{key[3]} (proto={key[4]})",
                    "Fwd Pkts/s": len(fp) / dur,
                    "Bwd Pkts/s": len(bp) / dur,
                    "Flow Pkts/s": total / dur,
                    "Flow Byts/s": (sum(fp) + sum(bp)) / dur,
                    "Init Fwd Win Byts": f["init_fwd_win"] if f["init_fwd_win"] is not None else -1,
                    "Init Bwd Win Byts": f["init_bwd_win"] if f["init_bwd_win"] is not None else -1,
                    "Fwd Pkt Len Max": max(fp) if fp else 0,
                    "Fwd Pkt Len Mean": np.mean(fp) if fp else 0,
                    "Fwd Pkt Len Std": np.std(fp) if len(fp) > 1 else 0,
                    "Fwd Seg Size Min": min(f["fwd_header_lens"]) if f["fwd_header_lens"] else 0,
                    "Bwd Pkt Len Max": max(bp) if bp else 0,
                    "Pkt Len Max": max(fp + bp) if (fp + bp) else 0,
                    "Pkt Len Std": np.std(fp + bp) if len(fp + bp) > 1 else 0,
                    "Fwd Header Len": sum(f["fwd_header_lens"]),
                    "Bwd Header Len": sum(f["bwd_header_lens"]),
                    "Flow IAT Std": float(np.std(iats)) * 1e6,
                    "Flow IAT Min": float(np.min(iats)) * 1e6,
                    "Flow IAT Mean": float(np.mean(iats)) * 1e6,
                    "Fwd IAT Tot": float(np.sum(fwd_iats)) * 1e6,
                    "Fwd IAT Std": float(np.std(fwd_iats)) * 1e6,
                    "ACK Flag Cnt": f["ack_count"],
                    "URG Flag Cnt": f["urg_count"],
                    "Down/Up Ratio": len(bp) / len(fp) if len(fp) > 0 else 0,
                }
                features.append(feat)
            return features


# ---------------------------------------------------------------------------
# Real-Time Detector (neural only)
# ---------------------------------------------------------------------------
class RealTimeDetector:
    """Real-time anomaly detection using the new ConvAttention+LSTM VAE."""

    def __init__(
        self,
        vae_path: str = "vae_full_model.pth",
        classifier_path: str = "nn_classifier_model.pth",
        nn_scaler_path: str = "nn_scaler.joblib",
        column_mapping_path: str = "column_min_max_mapping.csv",
        top_features_path: str = "top_features.joblib",
        recon_scorer_path: str = "anomaly_thresholds.joblib",
        ood_threshold: float = DEFAULT_OOD_THRESHOLD,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.ood_threshold = ood_threshold
        self.flow_tracker = FlowTracker()
        self.capturing = False

        # Sliding window buffer (WINDOW_SIZE × n_features)
        self.window_buffer: np.ndarray | None = None
        self.last_valid_features: np.ndarray | None = None
        self.inference_times: list[float] = []

        self.vae: VAE | None = None
        self.classifier: BinaryFirstTabularNet | None = None
        self.nn_scaler: QuantileTransformer | None = None
        self.column_mapping: pd.DataFrame | None = None
        self.top_features: list | None = None
        self.recon_scorer: ReconScorer | None = None

        self._load_artifacts(
            vae_path, classifier_path, nn_scaler_path, column_mapping_path, top_features_path, recon_scorer_path
        )

    # ------------------------------------------------------------------
    def _find(self, filename: str) -> Path:
        for base in DATA_DIR_CANDIDATES:
            p = base / filename
            if p.exists():
                return p
        return Path(filename)  # fall back to CWD

    def _load_artifacts(self, vae_path, classifier_path, nn_scaler_path, col_map_path, top_feat_path, recon_path):
        # Column min/max mapping
        p = self._find(col_map_path)
        if p.exists():
            self.column_mapping = pd.read_csv(p).set_index("column")
            print(f"✓ Column mapping loaded  ({len(self.column_mapping)} cols)")
        else:
            print(f"⚠ Column mapping not found: {p}")

        # Top features list
        p = self._find(top_feat_path)
        if p.exists():
            self.top_features = joblib.load(p)
            print(f"✓ Top features loaded   ({len(self.top_features)} features)")
        else:
            print(f"⚠ Top features not found: {p}")
            self.top_features = TOP_FEATURES

        # VAE checkpoint
        p = self._find(vae_path)
        if p.exists():
            self.vae = torch.load(p, map_location=self.device, weights_only=False)
            self.vae.to(self.device).eval()
            print(f"✓ VAE loaded            ({type(self.vae).__name__}) → {self.device}")
        else:
            print(f"⚠ VAE checkpoint not found: {p}")

        # Classifier Scaler
        p = self._find(nn_scaler_path)
        if p.exists():
            self.nn_scaler = joblib.load(p)
            print(f"✓ Classifier scaler loaded ({type(self.nn_scaler).__name__})")
        else:
            print(f"⚠ Classifier scaler not found: {p}")

        # Classifier Model
        p = self._find(classifier_path)
        if p.exists():
            self.classifier = BinaryFirstTabularNet(input_dim=8, n_classes=5, n_attack_classes=4)
            self.classifier.load_state_dict(torch.load(p, map_location=self.device))
            self.classifier.to(self.device).eval()
            print(f"✓ Classifier loaded     ({type(self.classifier).__name__}) → {self.device}")
        else:
            print(f"⚠ Classifier model not found: {p}")

        # Reconstruction scorer (saved as dict {"p5":…, "p95":…} inside anomaly_thresholds.joblib)
        p = self._find(recon_path)
        if p.exists():
            obj = joblib.load(p)
            rs = obj.get("recon_scorer") if isinstance(obj, dict) else obj
            if rs and "p5" in rs and "p95" in rs:
                self.recon_scorer = ReconScorer(np.array(rs["p5"]), np.array(rs["p95"]))
                print(f"✓ Recon scorer loaded   (p5={rs['p5']}, p95={rs['p95']})")
            else:
                print("⚠ recon_scorer key not found in thresholds file – scoring disabled")
            if isinstance(obj, dict) and "threshold" in obj:
                self.ood_threshold = float(obj["threshold"])
                print(f"✓ OOD threshold loaded  (threshold={self.ood_threshold:.3f})")
        else:
            print(f"⚠ Thresholds file not found: {p}")

        # Initialise window buffer
        n_feat = len(self.top_features)
        self.window_buffer = np.zeros((WINDOW_SIZE, n_feat), dtype=np.float64)
        self.window_fill_count = 0  # tracks cold-start; scoring starts after WINDOW_SIZE fills

    # ------------------------------------------------------------------
    def _scale(self, raw_vec: np.ndarray) -> np.ndarray:
        """Min-max scale to 0-50 using column_mapping."""
        if self.column_mapping is None:
            return raw_vec
        scaled = np.zeros_like(raw_vec, dtype=np.float64)
        for i, col in enumerate(self.top_features):
            if col in self.column_mapping.index:
                mn = float(self.column_mapping.loc[col, "min"])
                mx = float(self.column_mapping.loc[col, "max"])
                scaled[i] = ((raw_vec[i] - mn) / (mx - mn)) * 50 if mx != mn else 0.0
        return scaled

    # ------------------------------------------------------------------
    def process_features(self, flow_feat: dict) -> dict | None:
        if not flow_feat:
            return None

        # Build feature vector from flow features dictionary
        raw_vec = np.zeros(len(self.top_features), dtype=np.float64)
        for i, feat in enumerate(self.top_features):
            raw_vec[i] = flow_feat.get(feat, 0.0)

        self.last_valid_features = raw_vec.copy()

        # Scale and push into sliding window
        current_scaled = self._scale(raw_vec)
        self.window_buffer[:-1] = self.window_buffer[1:]
        self.window_buffer[-1] = current_scaled
        self.window_fill_count += 1

        result = {
            "raw_features": raw_vec,
            "flow_key": flow_feat.get("_flow_key", "Unknown Flow")
        }

        if self.vae is None:
            return result

        # Skip scoring until the window is fully filled (cold-start warmup)
        if self.window_fill_count < WINDOW_SIZE:
            remaining = WINDOW_SIZE - self.window_fill_count
            result["warming_up"] = True
            result["warmup_remaining"] = remaining
            return result

        start_t = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(self.window_buffer).unsqueeze(0).to(self.device)  # (1, W, F)
            recon, mu, logvar = self.vae(x, deterministic=True)

            last_recon = recon[:, -1, :]
            last_x = x[:, -1, :]
            diff = last_recon - last_x
            mse_loss = float((diff * diff).mean())
            student_loss = float(torch.log1p((diff * diff) / DF_DEG).mean())
            kld_loss = float(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())

            latent = mu.squeeze(0).cpu().numpy()  # shape (latent_dim,)

        # Recon score (normalised anomaly signal)
        recon_score = self.recon_scorer.score(mse_loss, kld_loss) if self.recon_scorer else mse_loss

        # Construct raw features vector for classifier: [latent_0..latent_4, student_loss, kld_loss, mse_loss]
        clf_raw = np.concatenate([latent, [student_loss, kld_loss, mse_loss]]).astype(np.float32)

        # Scaler and neural classifier prediction
        if self.nn_scaler is not None and self.classifier is not None:
            clf_raw_2d = clf_raw.reshape(1, -1)
            clf_scaled = self.nn_scaler.transform(clf_raw_2d).astype(np.float32)
            np.clip(clf_scaled, -5.0, 5.0, out=clf_scaled)
            
            with torch.no_grad():
                x_clf = torch.from_numpy(clf_scaled).to(self.device)
                outputs = self.classifier(x_clf)
                binary_prob = float(torch.softmax(outputs["binary_logits"], dim=1)[:, 1].cpu().item())
                subtype_prob = torch.softmax(outputs["attack_subtype_logits"], dim=1).squeeze(0).cpu().numpy()
        else:
            binary_prob = 0.0
            subtype_prob = np.zeros(4, dtype=np.float32)

        # OOD attack score: fusion of classifier prediction and reconstruction score
        if self.classifier is not None:
            w = float(RECON_OOD_WEIGHT)
            attack_score = float(np.clip((1.0 - w) * binary_prob + w * recon_score, 0.0, 1.0))
        else:
            attack_score = float(np.clip(recon_score, 0.0, 1.0))

        is_attack = attack_score >= self.ood_threshold

        # Determine classification
        pred_class_name = "Benign"
        if is_attack:
            if self.classifier is not None:
                # Class mapping: index 0..3 of subtype_prob maps to class 1..4 in ATTACK_CLASS_NAMES
                pred_idx = int(np.argmax(subtype_prob)) + 1
                pred_class_name = ATTACK_CLASS_NAMES.get(pred_idx, f"Attack Class {pred_idx}")
            else:
                pred_class_name = "Attack"

        result.update({
            "latent": latent,
            "mse_loss": mse_loss,
            "student_loss": student_loss,
            "kld_loss": kld_loss,
            "recon_score": recon_score,
            "binary_prob": binary_prob,
            "attack_score": attack_score,
            "is_attack": is_attack,
            "pred_class": pred_class_name,
            "subtype_probs": subtype_prob.tolist(),
        })
        self.inference_times.append(time.perf_counter() - start_t)
        return result

    # ------------------------------------------------------------------
    def packet_callback(self, pkt):
        self.flow_tracker.process_packet(pkt)

    def start_capture(self, interface=None):
        if not SCAPY_AVAILABLE:
            print("⚠ scapy unavailable – cannot capture packets. Run: pip install scapy")
            return None
        self.capturing = True

        def _run():
            import sys as _sys
            print(f"Capturing on: {interface or 'default interface'}")
            try:
                if _sys.platform == "win32":
                    # On Windows, raw packet capture needs EITHER:
                    #   (a) Npcap installed  → works for any user
                    #   (b) Run as Administrator  → uses Windows native L3 raw socket
                    # We try Npcap first (scapy auto-detects it), then fall back to L3 raw.
                    sock = scapy_conf.L3socket(iface=interface)
                    sniff(opened_socket=sock, prn=self.packet_callback, store=False,
                          stop_filter=lambda _: not self.capturing)
                else:
                    sniff(iface=interface, prn=self.packet_callback, store=False,
                          stop_filter=lambda _: not self.capturing)
            except PermissionError:
                print("⚠ Permission denied – run as Administrator (Windows) or with sudo (Linux).")
                self.capturing = False
            except OSError as e:
                msg = str(e)
                if "administrator" in msg.lower() or "10013" in msg:
                    print("⚠ Raw socket access denied.")
                    print("   Fix option 1 (recommended): Install Npcap → https://npcap.com")
                    print("   Fix option 2: Re-run this script as Administrator")
                elif "winpcap" in msg.lower() or "npcap" in msg.lower():
                    print("⚠ Npcap not found. Install from https://npcap.com (free, lightweight).")
                else:
                    print(f"Capture error: {e}")
                self.capturing = False
            except Exception as e:
                print(f"Capture error: {e}")
                self.capturing = False

        t = Thread(target=_run, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    def run(self, interface=None, poll_interval: float = 0.1, duration: int = 120):
        print("=" * 65)
        print("Real-Time Network Traffic Detection (VAE Neural OOD)")
        print("=" * 65)
        print(f"  Poll interval    : {poll_interval}s  |  Duration: {duration}s")
        print(f"  Window size      : {WINDOW_SIZE} steps")
        print(f"  OOD threshold    : {self.ood_threshold:.3f}")
        print(f"  Device           : {self.device}")
        print("-" * 65)

        self.start_capture(interface)

        total_records = 0
        total_benign = 0
        total_attacks = 0
        attack_classes = defaultdict(int)
        start = time.time()

        try:
            while (time.time() - start) < duration:
                time.sleep(poll_interval)
                features_list = self.flow_tracker.pop_finished_flows()
                for flow_feat in features_list:
                    result = self.process_features(flow_feat)
                    if result is None:
                        continue

                    if result.get("warming_up"):
                        print(f"[{time.strftime('%H:%M:%S')}] Warming up window buffer... {result['warmup_remaining']} steps remaining. (Flow: {result.get('flow_key')})")
                        continue

                    total_records += 1
                    elapsed = time.time() - start
                    remaining = duration - elapsed
                    print(f"\n[Record {total_records}] {time.strftime('%H:%M:%S')} | ⏱ {remaining:.0f}s remaining")
                    print(f"  Flow: {result.get('flow_key')}")
                    print(f"  Metrics: recon={result.get('recon_score', 0.0):.4f} | clf_prob={result.get('binary_prob', 0.0):.4f} | attack_score={result.get('attack_score', 0.0):.4f}")

                    if result.get("is_attack"):
                        total_attacks += 1
                        sc = result.get("attack_score", 0.0)
                        cls = result.get("pred_class", "Attack")
                        print(f"  ⚠️  ATTACK DETECTED: {cls}  (score={sc:.3f} ≥ threshold={self.ood_threshold:.3f})")
                        attack_classes[cls] += 1
                    else:
                        total_benign += 1
                        print(f"  ✓  Benign  (score={result.get('attack_score', 0.0):.3f})")

        except KeyboardInterrupt:
            print("\n\nStopped early.")

        self.capturing = False
        self._print_report(total_records, total_benign, total_attacks, attack_classes)

    # ------------------------------------------------------------------
    def _print_report(self, total, benign, attacks, attack_classes):
        print("\n" + "!" * 65)
        print(f"DEPLOYMENT REPORT  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("!" * 65)
        print(f"\nTotal flows     : {total}")
        b_pct = 100 * benign / max(total, 1)
        a_pct = 100 * attacks / max(total, 1)
        print(f"  ✓ Benign       : {benign} ({b_pct:.1f}%)")
        print(f"  ⚠ Attacks      : {attacks} ({a_pct:.1f}%)")
        if attack_classes:
            print("\nAttack breakdown:")
            for cls, cnt in sorted(attack_classes.items()):
                print(f"  {cls}: {cnt}")
        if self.inference_times:
            avg_ms = np.mean(self.inference_times) * 1000
            print(f"\nPipeline Performance:")
            print(f"  Avg inference latency: {avg_ms:.2f} ms")
        print("!" * 65)


# ---------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Real-time network anomaly detection (VAE neural)")
    p.add_argument("-i", "--interface", default=None, help="Network interface")
    p.add_argument("-t", "--interval", type=float, default=0.1, help="Polling interval for finished flows (s)")
    p.add_argument("-d", "--duration", type=int, default=120, help="Total duration (s)")
    p.add_argument("--vae", default="vae_full_model.pth", help="VAE checkpoint path")
    p.add_argument("--classifier", default="nn_classifier_model.pth", help="Classifier checkpoint path")
    p.add_argument("--scaler", default="nn_scaler.joblib", help="Scaler path")
    p.add_argument("--threshold", type=float, default=DEFAULT_OOD_THRESHOLD, help="OOD threshold")
    p.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    args = p.parse_args()

    detector = RealTimeDetector(
        vae_path=args.vae,
        classifier_path=args.classifier,
        nn_scaler_path=args.scaler,
        ood_threshold=args.threshold,
        device=args.device,
    )
    detector.run(interface=args.interface, poll_interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()
