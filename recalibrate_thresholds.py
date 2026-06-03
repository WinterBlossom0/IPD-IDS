"""
Recalibrate anomaly_thresholds.joblib p5/p95 using actual benign training data.
Run from the project root after the VAE has been trained.
"""
import sys, pathlib, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, joblib
from typing import Tuple

# ── replicate the model classes exactly as in realtime_capture.py ──────────
WINDOW_SIZE = 30
EPS = 1e-8

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

# ── load artifacts ──────────────────────────────────────────────────────────
print("Loading VAE …")
vae = torch.load("vae_full_model.pth", map_location="cpu", weights_only=False)
vae.eval()

top_features = joblib.load("top_features.joblib")
col_map = pd.read_csv("column_min_max_mapping.csv").set_index("column")
n_feat = len(top_features)
print(f"  features={n_feat}, window={WINDOW_SIZE}")

# ── load benign training samples ────────────────────────────────────────────
print("Loading benign training data …")
df = pd.read_csv("combined_train.csv", nrows=200_000)
benign = df[df["label"] == 0] if "label" in df.columns else df
for feat in top_features:
    if feat not in benign.columns:
        benign[feat] = 0.0
benign = benign[top_features].fillna(0.0)

# Scale (same as _scale in realtime_capture.py)
scaled = np.zeros_like(benign.values, dtype=np.float64)
for i, col in enumerate(top_features):
    if col in col_map.index:
        mn = float(col_map.loc[col, "min"])
        mx = float(col_map.loc[col, "max"])
        scaled[:, i] = ((benign.values[:, i] - mn) / (mx - mn + EPS)) * 50

print(f"  {len(scaled):,} benign rows available")

# ── build sliding windows and compute losses ─────────────────────────────────
print("Computing reconstruction losses …")
losses = []
stride = 5   # step every 5 rows to avoid extreme correlation
with torch.no_grad():
    for start in range(0, len(scaled) - WINDOW_SIZE, stride):
        window = scaled[start : start + WINDOW_SIZE]
        x = torch.from_numpy(window).unsqueeze(0)  # (1, W, F)
        recon, mu, logvar = vae(x, deterministic=True)
        diff = recon[:, -1, :] - x[:, -1, :]
        mse = float((diff * diff).mean())
        kld = float(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
        losses.append((mse, kld))
        if len(losses) % 1000 == 0:
            print(f"  … {len(losses)} windows", end="\r")

arr = np.array(losses, dtype=np.float32)
p5  = np.percentile(arr, 5,  axis=0)
p95 = np.percentile(arr, 95, axis=0)
print(f"\n  p5  = {p5}")
print(f"  p95 = {p95}")
print(f"  mean mse={arr[:,0].mean():.4f}  mean kld={arr[:,1].mean():.4f}")

# ── patch and save thresholds ────────────────────────────────────────────────
existing = joblib.load("anomaly_thresholds.joblib")
existing["recon_scorer"] = {"p5": p5, "p95": p95}
joblib.dump(existing, "anomaly_thresholds.joblib")
print("\n✓ anomaly_thresholds.joblib updated with real benign p5/p95")
