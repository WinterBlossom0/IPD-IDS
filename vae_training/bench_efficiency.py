"""Deployment cost of every trained encoder. This is the x-axis of an edge paper.

Only the ENCODER is measured. In deployment the VAE is a feature extractor: a window goes
in, a latent comes out, a small head classifies it. The decoder exists solely to shape the
representation during training and never ships. Reporting whole-VAE parameter counts would
overstate on-device cost by ~2x.

Reports per model:
  encoder params, fp32 / int8 footprint, MACs+FLOPs per window,
  CPU single-window latency (the edge-relevant number - no GPU on an ESP32),
  CPU batched throughput (windows/s).

Run:  torch-python bench_efficiency.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_vae import ColumnVAE

RUNS = Path(__file__).resolve().parent / "runs"
OUT = RUNS / "frontier"
WARMUP, ITERS = 5, 30
LATENCY_BATCH = 256


def count_flops(encoder, window, n_feat):
    """MACs via forward hooks on Linear/Conv1d, plus the attention matmuls."""
    macs = {"v": 0}
    hooks = []

    def lin_hook(m, i, o):
        macs["v"] += int(np.prod(o.shape[:-1])) * m.in_features * m.out_features

    def conv_hook(m, i, o):
        macs["v"] += int(o.shape[0] * o.shape[2]) * m.in_channels * m.out_channels * m.kernel_size[0] // m.groups

    for m in encoder.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))

    with torch.no_grad():
        encoder(torch.randn(1, window, n_feat))
    for h in hooks:
        h.remove()

    # Attention score+context matmuls are not Linear layers: 2 * L^2 * E per block.
    n_blocks = len(encoder.blocks)
    embed = encoder.in_proj.out_features
    macs["v"] += n_blocks * 2 * window * window * embed
    return macs["v"]


def main():
    ckpts = sorted(RUNS.glob("*.pt"))
    if not ckpts:
        raise SystemExit("no checkpoints in runs/")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)          # single-core: the honest edge assumption
    rows = []

    for ck in ckpts:
        blob = torch.load(ck, map_location="cpu")
        hid = blob.get("hidden_dim", 48)
        nb = blob.get("n_blocks", 2)
        model = ColumnVAE(blob["n_feat"], hidden_dim=hid,
                          latent_dim=blob["latent_dim"], n_blocks=nb)
        model.load_state_dict(blob["model"])
        model.eval()
        enc = model.encoder
        W, F_ = blob["window"], blob["n_feat"]

        p = sum(q.numel() for q in enc.parameters())
        macs = count_flops(enc, W, F_)

        x1 = torch.randn(1, W, F_)
        with torch.no_grad():
            for _ in range(WARMUP):
                enc(x1)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                enc(x1)
            lat_ms = (time.perf_counter() - t0) / ITERS * 1000

            xb = torch.randn(LATENCY_BATCH, W, F_)
            for _ in range(2):
                enc(xb)
            t0 = time.perf_counter()
            for _ in range(5):
                enc(xb)
            thr = LATENCY_BATCH * 5 / (time.perf_counter() - t0)

        rows.append({
            "tag": ck.stem,
            "size_name": blob.get("size_name", "?"),
            "lam": blob.get("lam"),
            "drop_port": blob.get("drop_port"),
            "hidden_dim": hid, "latent_dim": blob["latent_dim"], "n_blocks": nb,
            "encoder_params": p,
            "fp32_KB": p * 4 / 1024,
            "int8_KB": p / 1024,
            "MACs_per_window": macs,
            "FLOPs_per_window": macs * 2,
            "cpu_latency_ms_1window": lat_ms,
            "cpu_throughput_windows_per_s": thr,
        })
        print(f"{ck.stem:22s} params={p:>7,}  fp32={p * 4 / 1024:>7.1f}KB  "
              f"int8={p / 1024:>6.1f}KB  MACs={macs / 1e6:>6.2f}M  "
              f"lat={lat_ms:>6.2f}ms  thr={thr:>8.0f}/s")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "efficiency.csv", index=False)
    print(f"\nefficiency -> {OUT / 'efficiency.csv'}")


if __name__ == "__main__":
    main()
