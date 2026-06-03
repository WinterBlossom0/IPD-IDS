#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor 


WINDOW_SIZE = 30
DF_DEG = 3.0
RECON_STUDENT_WEIGHT = 0.4
RECON_MSE_WEIGHT = 0.2


PROJECT_ROOT = Path.cwd()
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else PROJECT_ROOT

DATA_DIR_CANDIDATES = [
    PROJECT_ROOT,
    PROJECT_ROOT / "datasets",
    KAGGLE_INPUT,
    KAGGLE_INPUT / "compressed-cic-2018",
    KAGGLE_WORKING,
]


def find_data_file(filename: str) -> Path:
    for base in DATA_DIR_CANDIDATES:
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}")


def artifact_path(filename: str) -> Path:
    return OUTPUT_DIR / filename


def make_windows(X_array: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    X_array = np.asarray(X_array)
    n_rows = len(X_array)
    if n_rows < window_size:
        raise ValueError(f"Need at least {window_size} rows to build one window, got {n_rows}")
    return np.stack([X_array[i : i + window_size] for i in range(n_rows - window_size + 1)])


class ConvAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        conv_padding = conv_kernel_size // 2
        self.local_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=conv_kernel_size,
                padding=conv_padding,
                groups=embed_dim,
            ),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        ).double()
        self.norm_local = nn.LayerNorm(embed_dim).double()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        ).double()
        self.norm_attention = nn.LayerNorm(embed_dim).double()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        ).double()
        self.norm_ffn = nn.LayerNorm(embed_dim).double()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        local_features = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm_local(x + local_features)
        attention_features, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.norm_attention(x + self.dropout(attention_features))
        return self.norm_ffn(x + self.dropout(self.ffn(x)))


class Encoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        conv_channels: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=0,
        ).double()
        self.relu = nn.ReLU()
        self.attention = ConvAttentionBlock(embed_dim=conv_channels)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=conv_channels,
            batch_first=True,
        ).double()
        self.fc_mu = nn.Linear(conv_channels, latent_dim).double()
        self.fc_logvar = nn.Linear(conv_channels, latent_dim).double()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = self.attention(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        conv_channels: int = 128,
        kernel_size: int = 3,
        output_window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_window_size = output_window_size
        self.fc = nn.Linear(latent_dim, output_window_size * conv_channels).double()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=conv_channels,
            batch_first=True,
        ).double()
        self.deconv1 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=n_features,
            kernel_size=kernel_size,
            padding=0,
        ).double()

    def forward(self, z: Tensor) -> Tensor:
        z = self.fc(z)
        z = z.view(z.size(0), self.output_window_size, -1)
        x, _ = self.lstm(z)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.deconv1(x)
        x = x.permute(0, 2, 1)
        return x


class VAE(nn.Module):
    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.encoder = Encoder(n_features, latent_dim, conv_channels, kernel_size)
        self.decoder = Decoder(n_features, latent_dim, conv_channels, kernel_size, output_window_size=window_size)

    def reparameterize(self, mu: Tensor, logvar: Tensor, deterministic: bool = False) -> Tensor:
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def prepare_features(df_raw: pd.DataFrame) -> np.ndarray:
    min_max_df = pd.read_csv(artifact_path("column_min_max_mapping.csv"))
    min_max_dict = {row["column"]: (row["min"], row["max"]) for _, row in min_max_df.iterrows()}
    top_features = joblib.load(artifact_path("top_features.joblib"))

    X_new = pd.get_dummies(df_raw.drop(columns=["Label"]), columns=["Protocol"], prefix="Protocol", drop_first=False)
    X_new.drop(columns=["Protocol_0"], inplace=True, errors="ignore")
    X_new = X_new.reindex(columns=top_features, fill_value=0)

    X_normalized = X_new.copy()
    for col in X_normalized.columns:
        col_min, col_max = min_max_dict[col]
        if col_max - col_min != 0:
            X_normalized[col] = ((X_new[col] - col_min) / (col_max - col_min)) * 50
        else:
            X_normalized[col] = 0
    return X_normalized.values.astype(np.float64)


def benchmark_vae_inference(variant_name: str, batch_size: int, device_name: str, input_file: str) -> pd.DataFrame:
    input_path = find_data_file(input_file)
    df_raw = pd.read_csv(input_path)
    X_array = prepare_features(df_raw)
    X_windowed = make_windows(X_array, WINDOW_SIZE)

    device = torch.device(device_name)
    model_path = artifact_path(f"vae_{variant_name}_model.pth")
    vae = torch.load(model_path, weights_only=False, map_location=device)
    vae.to(device)
    vae.eval()

    start_time = time.perf_counter()
    with torch.no_grad():
        for start_idx in range(0, len(X_windowed), batch_size):
            batch = torch.from_numpy(X_windowed[start_idx : start_idx + batch_size]).to(
                device=device,
                dtype=torch.double,
            )
            mu, logvar = vae.encoder(batch)
            recon = vae.decoder(mu)
            recon_last = recon[:, -1, :]
            batch_last = batch[:, -1, :]
            diff = recon_last - batch_last
            _student_loss = torch.log1p((diff * diff) / DF_DEG).mean(dim=1)
            _mse_loss = (diff * diff).mean(dim=1)
            _recon_loss = (RECON_STUDENT_WEIGHT * _student_loss) + (RECON_MSE_WEIGHT * _mse_loss)
            _kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1)
            del batch, mu, logvar, recon
    elapsed = time.perf_counter() - start_time

    sample_rows = len(df_raw)
    windows = len(X_windowed)
    speed_df = pd.DataFrame([
        {
            "variant": variant_name,
            "input_file": str(input_path),
            "device": str(device),
            "batch_size": batch_size,
            "sample_rows": sample_rows,
            "compressed_windows": windows,
            "seconds": elapsed,
            "rows_per_second": sample_rows / elapsed if elapsed > 0 else np.nan,
            "windows_per_second": windows / elapsed if elapsed > 0 else np.nan,
            "milliseconds_per_row": (elapsed / sample_rows) * 1000 if sample_rows > 0 else np.nan,
            "milliseconds_per_window": (elapsed / windows) * 1000 if windows > 0 else np.nan,
        }
    ])

    output_path = artifact_path(f"inference_speed_benchmark_{variant_name}.csv")
    speed_df.to_csv(output_path, index=False)
    if variant_name == "full":
        speed_df.to_csv(artifact_path("inference_speed_benchmark.csv"), index=False)

    del X_array, X_windowed, vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return speed_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CIC-IDS VAE compression inference speed.")
    parser.add_argument("--variant", default="full", help="VAE variant name, e.g. full, no_time_adv, no_student.")
    parser.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size. Use 1 for per-flow latency.")
    parser.add_argument("--device", default="cpu", help="Torch device for benchmarking, usually cpu.")
    parser.add_argument("--input-file", default="combined_test.csv", help="Raw CIC test CSV with Label column.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = benchmark_vae_inference(
        variant_name=args.variant,
        batch_size=args.batch_size,
        device_name=args.device,
        input_file=args.input_file,
    )
    print(result)
