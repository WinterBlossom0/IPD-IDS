"""Stage 0 for the shortcut-frontier experiments.

Three jobs, all CPU:

1. Recover the raw `Dst Port` value for every row. The analysis notebook applied
   signed_log then RobustScaler and did not persist the scaler, but the transform is
   monotonic and invertible: scaled = (log1p(port) - center) / scale. Solving the two
   unknowns off the observed min/max (port 0 and port 65535 both occur) recovers
   center=4.394449, scale=1.705870 - which puts the median destination port at exactly
   80.00 and reproduces 80/53/443/3389/445/8080/21 as the head of the distribution,
   with 100% of values landing within 0.5 of an integer. Verified, not assumed.

2. Bucket those ports into PORT_BUCKETS classes (top-K by fit frequency + "other"),
   which become the adversary's target in train_invariant.py.

3. Build evaluation-time deduplication masks. These are masks, never destructive edits:
   the VAE consumes contiguous windows, so physically dropping rows would corrupt the
   temporal adjacency the sequence model depends on. Duplicate flows are real traffic.
   What is *not* legitimate is scoring on test rows the model already memorised from
   fit, so detection metrics get reported on both the raw and the purged test set.

Writes everything to data/shortcut/.
"""
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "shortcut"

# Recovered RobustScaler parameters for feature 0 (Dst Port); see module docstring.
PORT_SCALE = 1.7058696
PORT_CENTER = 4.3944492
PORT_FEATURE_IDX = 0

N_TOP_PORTS = 15          # + 1 "other" bucket -> PORT_BUCKETS classes
PORT_BUCKETS = N_TOP_PORTS + 1


def recover_ports(x_col: np.ndarray) -> np.ndarray:
    """Invert signed_log + RobustScaler on the Dst Port column."""
    return np.expm1(x_col.astype(np.float64) * PORT_SCALE + PORT_CENTER)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in ("fit", "val", "test"):
        X = np.load(DATA_DIR / f"features_{name}.npy", mmap_mode="r")
        y = np.load(DATA_DIR / f"labels_{name}.npy")
        splits[name] = (X, y)
        print(f"{name}: {X.shape[0]:,} rows x {X.shape[1]} features")

    # ---------------------------------------------------------------- ports
    print("\n--- port recovery ---")
    ports = {}
    for name, (X, _) in splits.items():
        raw = recover_ports(np.asarray(X[:, PORT_FEATURE_IDX]))
        err = np.abs(raw - np.round(raw))
        p = np.round(raw).astype(np.int32)
        p = np.clip(p, 0, 65535)
        ports[name] = p
        print(f"  {name}: max_integrality_err={err.max():.4f}  "
              f"n_unique={len(np.unique(p)):,}  median={int(np.median(p))}")

    # Bucket vocabulary is defined on fit only - val/test ports that never appear in
    # fit fall into "other", exactly as an unseen port would at deployment time.
    vals, cnts = np.unique(ports["fit"], return_counts=True)
    top_ports = vals[np.argsort(-cnts)[:N_TOP_PORTS]]
    top_ports = np.sort(top_ports)
    port_to_bucket = {int(p): i for i, p in enumerate(top_ports)}
    print(f"\n  top-{N_TOP_PORTS} ports (bucket 0..{N_TOP_PORTS - 1}): {top_ports.tolist()}")
    print(f"  bucket {N_TOP_PORTS} = other")

    buckets = {}
    for name in splits:
        b = np.full(len(ports[name]), N_TOP_PORTS, dtype=np.int8)
        for p, i in port_to_bucket.items():
            b[ports[name] == p] = i
        buckets[name] = b
        frac_other = (b == N_TOP_PORTS).mean()
        print(f"  {name}: {frac_other:.1%} land in 'other'")

    # ---------------------------------------------------------------- dedup masks
    print("\n--- duplicate analysis (masks only, non-destructive) ---")

    def row_hashes(X, chunk=200_000):
        out = np.empty(X.shape[0], dtype=np.int64)
        for s in range(0, X.shape[0], chunk):
            blk = np.ascontiguousarray(X[s:s + chunk])
            out[s:s + chunk] = [hash(r.tobytes()) for r in blk]
        return out

    h = {name: row_hashes(X) for name, (X, _) in splits.items()}

    masks = {}
    for name in splits:
        _, first_idx = np.unique(h[name], return_index=True)
        m = np.zeros(len(h[name]), dtype=bool)
        m[first_idx] = True          # keep first occurrence, preserves temporal order
        masks[name] = m
        print(f"  {name}: {len(m):,} rows -> {m.sum():,} unique ({1 - m.mean():.1%} duplicate)")

    fit_set = set(h["fit"].tolist())
    for name in ("val", "test"):
        seen = np.fromiter((x in fit_set for x in h[name]), dtype=bool, count=len(h[name]))
        purged = masks[name] & ~seen
        n_leaked = (masks[name] & seen).sum()
        print(f"  {name}: {n_leaked:,} unique rows also present in fit "
              f"({n_leaked / masks[name].sum():.2%} of unique) -> purged mask keeps {purged.sum():,}")
        np.save(OUT_DIR / f"mask_purged_{name}.npy", purged)

    for name in splits:
        np.save(OUT_DIR / f"ports_{name}.npy", ports[name])
        np.save(OUT_DIR / f"portbucket_{name}.npy", buckets[name])
        np.save(OUT_DIR / f"mask_dedup_{name}.npy", masks[name])

    meta = {
        "port_scale": PORT_SCALE,
        "port_center": PORT_CENTER,
        "n_top_ports": N_TOP_PORTS,
        "port_buckets": PORT_BUCKETS,
        "top_ports": top_ports.tolist(),
        "rows": {k: int(v[0].shape[0]) for k, v in splits.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
