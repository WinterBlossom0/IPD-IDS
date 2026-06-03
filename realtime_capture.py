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

try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠ scapy not installed – packet capture disabled. Feature inference still works.")

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
    """Tracks network flows and computes CIC-IDS features."""

    def __init__(self):
        self.flows = defaultdict(lambda: {
            "fwd_packets": [], "bwd_packets": [],
            "fwd_timestamps": [], "bwd_timestamps": [],
            "fwd_header_lens": [], "bwd_header_lens": [],
            "init_fwd_win": None, "init_bwd_win": None,
            "ack_count": 0, "urg_count": 0, "start_time": None,
        })
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
            if fwd:
                f["fwd_packets"].append(pkt_len)
                f["fwd_timestamps"].append(ts)
                f["fwd_header_lens"].append(hdr)
            else:
                f["bwd_packets"].append(pkt_len)
                f["bwd_timestamps"].append(ts)
                f["bwd_header_lens"].append(hdr)

    def compute_features(self):
        with self.lock:
            if not self.flows:
                return None
            all_features = []
            for _, f in self.flows.items():
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
                    "Fwd Pkts/s": len(fp) / dur,
                    "Bwd Pkts/s": len(bp) / dur,
                    "Flow Pkts/s": total / dur,
                    "Flow Byts/s": (sum(fp) + sum(bp)) / dur,
                    "Init Fwd Win Byts": f["init_fwd_win"] or 0,
                    "Init Bwd Win Byts": f["init_bwd_win"] or 0,
                    "Fwd Pkt Len Max": max(fp) if fp else 0,
                    "Fwd Pkt Len Mean": np.mean(fp) if fp else 0,
                    "Fwd Pkt Len Std": np.std(fp) if len(fp) > 1 else 0,
                    "Fwd Seg Size Min": min(fp) if fp else 0,
                    "Bwd Pkt Len Max": max(bp) if bp else 0,
                    "Pkt Len Max": max(fp + bp) if (fp + bp) else 0,
                    "Pkt Len Std": np.std(fp + bp) if len(fp + bp) > 1 else 0,
                    "Fwd Header Len": sum(f["fwd_header_lens"]),
                    "Bwd Header Len": sum(f["bwd_header_lens"]),
                    "Flow IAT Std": float(np.std(iats)),
                    "Flow IAT Min": float(np.min(iats)),
                    "Flow IAT Mean": float(np.mean(iats)),
                    "Fwd IAT Tot": float(np.sum(fwd_iats)),
                    "Fwd IAT Std": float(np.std(fwd_iats)),
                    "ACK Flag Cnt": f["ack_count"],
                    "URG Flag Cnt": f["urg_count"],
                    "Down/Up Ratio": len(bp) / len(fp) if len(fp) > 0 else 0,
                }
                all_features.append(feat)
            self.flows.clear()
            return all_features


# ---------------------------------------------------------------------------
# Real-Time Detector (neural only)
# ---------------------------------------------------------------------------
class RealTimeDetector:
    """Real-time anomaly detection using the new ConvAttention+LSTM VAE."""

    def __init__(
        self,
        vae_path: str = "vae_full_model.pth",
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

        self.vae: VAE | None = None
        self.column_mapping: pd.DataFrame | None = None
        self.top_features: list | None = None
        self.recon_scorer: ReconScorer | None = None

        self._load_artifacts(vae_path, column_mapping_path, top_features_path, recon_scorer_path)

    # ------------------------------------------------------------------
    def _find(self, filename: str) -> Path:
        for base in DATA_DIR_CANDIDATES:
            p = base / filename
            if p.exists():
                return p
        return Path(filename)  # fall back to CWD

    def _load_artifacts(self, vae_path, col_map_path, top_feat_path, recon_path):
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
        else:
            print(f"⚠ Thresholds file not found: {p}")

        # Initialise window buffer
        n_feat = len(self.top_features)
        self.window_buffer = np.zeros((WINDOW_SIZE, n_feat), dtype=np.float64)

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
    def process_features(self, features_list: list) -> dict | None:
        if not features_list:
            return None

        df = pd.DataFrame(features_list)
        for feat in self.top_features:
            if feat not in df.columns:
                df[feat] = 0.0
        df = df[self.top_features].fillna(0.0)

        # Aggregate flows → single feature vector
        current_raw = df.mean().values
        self.last_valid_features = current_raw.copy()

        # Scale and push into sliding window
        current_scaled = self._scale(current_raw)
        self.window_buffer[:-1] = self.window_buffer[1:]
        self.window_buffer[-1] = current_scaled

        result = {"num_flows": len(features_list), "raw_features": current_raw}

        if self.vae is None:
            return result

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

        # OOD attack score: pure reconstruction signal (no CatBoost needed)
        attack_score = float(np.clip(recon_score, 0.0, 1.0))
        is_attack = attack_score >= self.ood_threshold

        result.update({
            "latent": latent,
            "mse_loss": mse_loss,
            "student_loss": student_loss,
            "kld_loss": kld_loss,
            "recon_score": recon_score,
            "attack_score": attack_score,
            "is_attack": is_attack,
        })
        return result

    # ------------------------------------------------------------------
    def packet_callback(self, pkt):
        self.flow_tracker.process_packet(pkt)

    def start_capture(self, interface=None):
        if not SCAPY_AVAILABLE:
            print("⚠ scapy unavailable – cannot capture packets.")
            return None
        self.capturing = True

        def _run():
            print(f"Capturing on: {interface or 'default interface'}")
            try:
                sniff(iface=interface, prn=self.packet_callback, store=False,
                      stop_filter=lambda _: not self.capturing)
            except PermissionError:
                print("⚠ Permission denied – run with admin/sudo.")
                self.capturing = False
            except Exception as e:
                print(f"Capture error: {e}")
                self.capturing = False

        t = Thread(target=_run, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    def run(self, interface=None, interval: int = 3, duration: int = 300):
        print("=" * 65)
        print("Real-Time Network Traffic Detection (VAE Neural OOD)")
        print("=" * 65)
        print(f"  Capture interval : {interval}s  |  Duration: {duration}s")
        print(f"  Window size      : {WINDOW_SIZE} steps")
        print(f"  OOD threshold    : {self.ood_threshold:.3f}")
        print(f"  Device           : {self.device}")
        print("-" * 65)

        self.start_capture(interface)

        total_records = 0
        total_benign = 0
        total_attacks = 0
        attack_classes: dict[float, int] = defaultdict(int)
        start = time.time()

        try:
            while (time.time() - start) < duration:
                time.sleep(interval)
                features_list = self.flow_tracker.compute_features()
                if not features_list:
                    print(f"[{time.strftime('%H:%M:%S')}] No flows captured")
                    continue

                result = self.process_features(features_list)
                if result is None:
                    continue

                total_records += 1
                elapsed = time.time() - start
                remaining = duration - elapsed
                print(f"\n[Record {total_records}] {time.strftime('%H:%M:%S')} | ⏱ {remaining:.0f}s remaining")
                print(f"  Flows: {result['num_flows']}  |  recon_score={result.get('recon_score', 'n/a'):.4f}  attack_score={result.get('attack_score', 'n/a'):.4f}")

                if result.get("is_attack"):
                    total_attacks += 1
                    sc = result.get("attack_score", 0.0)
                    print(f"  ⚠️  ATTACK DETECTED  (score={sc:.3f} ≥ threshold={self.ood_threshold:.3f})")
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
        print(f"\nTotal intervals : {total}")
        b_pct = 100 * benign / max(total, 1)
        a_pct = 100 * attacks / max(total, 1)
        print(f"  ✓ Benign       : {benign} ({b_pct:.1f}%)")
        print(f"  ⚠ Attacks      : {attacks} ({a_pct:.1f}%)")
        if attack_classes:
            print("\nAttack breakdown:")
            for cls, cnt in sorted(attack_classes.items()):
                name = ATTACK_CLASS_NAMES.get(int(cls), f"Class {cls}")
                print(f"  {name}: {cnt}")
        print("!" * 65)


# ---------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Real-time network anomaly detection (VAE neural)")
    p.add_argument("-i", "--interface", default=None, help="Network interface")
    p.add_argument("-t", "--interval", type=int, default=3, help="Capture interval (s)")
    p.add_argument("-d", "--duration", type=int, default=300, help="Total duration (s)")
    p.add_argument("--vae", default="vae_full_model.pth", help="VAE checkpoint path")
    p.add_argument("--threshold", type=float, default=DEFAULT_OOD_THRESHOLD, help="OOD threshold")
    p.add_argument("--device", default="cpu", help="Torch device (cpu/cuda)")
    args = p.parse_args()

    detector = RealTimeDetector(
        vae_path=args.vae,
        ood_threshold=args.threshold,
        device=args.device,
    )
    detector.run(interface=args.interface, interval=args.interval, duration=args.duration)


if __name__ == "__main__":
    main()
