"""Freeze each swept model and dump its latents for fit/val/test.

Everything downstream - MDL port-probes, linear-erasure baselines, detection heads,
transfer evaluation - runs on these frozen arrays and needs no GPU. That is what keeps
the evaluation matrix cheap: the encoder is trained once per lambda, then every
evaluation axis is a forward pass.

A window's latent is mu at its LAST row (the flow the window is "about"), matching the
causal design and the adversary's target during training. Labels and port buckets are
aligned to the same row.

Run:  torch-python extract_latents.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from train_vae import ColumnVAE, amp_ctx, DEVICE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SHORTCUT_DIR = DATA_DIR / "shortcut"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
LAT_DIR = RUNS_DIR / "latents"
BATCH = 1024


def windows_for(split, drop_port, window):
    X = np.load(DATA_DIR / f"features_{split}.npy")
    if drop_port:
        X = np.delete(X, 0, axis=1)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    return Xt.unfold(0, window, 1).permute(0, 2, 1)


@torch.no_grad()
def encode_all(model, w, batch=BATCH):
    out = torch.empty(w.shape[0], model.encoder.to_mu.out_features,
                      dtype=torch.float32, device="cpu")
    for s in range(0, w.shape[0], batch):
        blk = w[s:s + batch].contiguous()
        with amp_ctx():
            mu, _ = model.encoder(blk)
        out[s:s + batch] = mu[:, -1, :].float().cpu()
    return out.numpy()


def main():
    LAT_DIR.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(RUNS_DIR.glob("*.pt"))
    if not ckpts:
        raise SystemExit("no checkpoints in runs/ - has the sweep finished any runs?")
    print(f"found {len(ckpts)} checkpoints")

    # Row-aligned side information, emitted once (identical across models).
    first = torch.load(ckpts[0], map_location="cpu")
    W = first["window"]
    for split in ("fit", "val", "test"):
        y = np.load(DATA_DIR / f"labels_{split}.npy")[W - 1:]
        pb = np.load(SHORTCUT_DIR / f"portbucket_{split}.npy")[W - 1:]
        np.save(LAT_DIR / f"y_{split}.npy", y)
        np.save(LAT_DIR / f"portbucket_{split}.npy", pb)
        for mname in ("mask_dedup", "mask_purged"):
            src = SHORTCUT_DIR / f"{mname}_{split}.npy"
            if src.exists():
                np.save(LAT_DIR / f"{mname}_{split}.npy", np.load(src)[W - 1:])
    print(f"wrote aligned labels/buckets/masks (window={W})")

    cache = {}
    for ck in ckpts:
        tag = ck.stem
        blob = torch.load(ck, map_location=DEVICE)
        model = ColumnVAE(blob["n_feat"], hidden_dim=blob.get("hidden_dim", 48),
                          latent_dim=blob["latent_dim"],
                          n_blocks=blob.get("n_blocks", 2)).to(DEVICE)
        model.load_state_dict(blob["model"])
        model.eval()
        dp = bool(blob["drop_port"])

        for split in ("fit", "val", "test"):
            dst = LAT_DIR / f"{tag}_{split}.npy"
            if dst.exists():
                continue
            key = (split, dp)
            if key not in cache:
                cache.clear()                       # only ever hold one split on GPU
                cache[key] = windows_for(split, dp, blob["window"])
            z = encode_all(model, cache[key])
            np.save(dst, z.astype(np.float32))
            print(f"  {tag:12s} {split:5s} -> {z.shape}")
        del model
        torch.cuda.empty_cache()

    (LAT_DIR / "index.json").write_text(json.dumps(
        {"tags": [c.stem for c in ckpts], "window": W}, indent=2))
    print(f"\nlatents -> {LAT_DIR}")


if __name__ == "__main__":
    main()
