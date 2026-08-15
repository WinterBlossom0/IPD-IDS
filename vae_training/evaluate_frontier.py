"""Stage 3+4: turn frozen latents into the shortcut frontier. CPU only.

Produces, for every swept model:

  X-AXIS  how much destination-port information survives in the latent, measured by
          ONLINE (PREQUENTIAL) MDL following Voita & Titov (EMNLP 2020) rather than probe
          accuracy. This matters: Ravfogel et al. (NeurIPS 2022) showed an encoder can beat
          its own adversary while a fresh probe still recovers the attribute, so adversary
          loss is not evidence of erasure. Codelength is reported as a compression ratio
          against the uniform code - 1.0 means the representation says nothing about port.
          Two probe families (linear, MLP) because erasure that only defeats linear probes
          is not erasure.

  Y-AXIS  detection performance from a LightGBM head on the frozen latent, scored on
          val (near) and test (temporally shifted = environment transfer), each on the
          raw split and on the leakage-purged split.

Also reports:
  * SRR, the Shortcut Reliance Ratio, on raw features - the diagnostic that motivated all this.
  * LEACE (Belrose et al. 2023) closed-form linear erasure as a reference point, with a
    self-check that a linear probe really does collapse to majority-class after erasure.

Run:  ds-python evaluate_frontier.py
"""
import json
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_curve
import lightgbm as lgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNS = Path(__file__).resolve().parent / "runs"
LAT = RUNS / "latents"
OUT = RUNS / "frontier"

SEED = 42
PROBE_POOL = 150_000       # probe sample budget (MDL is sequential; keep it bounded)
HEAD_POOL = 400_000        # head-training subsample; keeps 22 models x 2 heads tractable
MDL_FRACTIONS = [0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1.0]
TARGET_FPR = 0.01
WORKERS = max(1, min(8, (os.cpu_count() or 4) // 3))


# ---------------------------------------------------------------- MDL probing

def online_mdl(Z, y, probe, n_classes, rng):
    """Prequential codelength in bits (Voita & Titov 2020).

    Cost of transmitting y given Z: the first block is sent under the uniform code, and
    each later block is sent under a probe fitted only on everything before it. A
    representation that carries no information about y cannot beat the uniform code.
    """
    n = len(y)
    idx = rng.permutation(n)
    Z, y = Z[idx], y[idx]
    blocks = [int(f * n) for f in MDL_FRACTIONS]

    codelength = blocks[0] * np.log2(n_classes)      # first block: uniform code
    for i in range(len(blocks) - 1):
        a, b = blocks[i], blocks[i + 1]
        ytr = y[:a]
        if len(np.unique(ytr)) < 2:                  # degenerate prefix -> uniform
            codelength += (b - a) * np.log2(n_classes)
            continue
        p = probe()
        p.fit(Z[:a], ytr)
        proba = p.predict_proba(Z[a:b])
        # Probe may not have seen every class yet; map to the full label space.
        full = np.full((b - a, n_classes), 1e-12)
        for j, c in enumerate(p.classes_):
            full[:, c] = proba[:, j]
        full /= full.sum(axis=1, keepdims=True)
        picked = full[np.arange(b - a), y[a:b]]
        codelength += float(-np.log2(np.clip(picked, 1e-12, None)).sum())

    uniform = n * np.log2(n_classes)
    return {"codelength_bits": codelength,
            "uniform_bits": uniform,
            "compression": uniform / codelength}


def probe_suite(Z, y, n_classes, rng):
    out = {}
    for name, ctor in [
        ("linear", lambda: LogisticRegression(max_iter=300, n_jobs=2)),
        ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=60,
                                       random_state=SEED, early_stopping=False)),
    ]:
        out[name] = online_mdl(Z, y, ctor, n_classes, np.random.default_rng(SEED))
    return out


# ---------------------------------------------------------------- LEACE

def leace_eraser(X, Z_onehot, ridge=1e-4):
    """Closed-form linear concept erasure (Belrose et al., NeurIPS 2023).

    Returns (mean, P) so that erased = (X - mean) @ P.T + mean removes every linear
    direction that carries concept information, while making the smallest possible
    change in the whitened norm.
    """
    Xc = X - X.mean(0, keepdims=True)
    Zc = Z_onehot - Z_onehot.mean(0, keepdims=True)
    n, d = Xc.shape

    S_xx = (Xc.T @ Xc) / n + ridge * np.eye(d)
    S_xz = (Xc.T @ Zc) / n

    evals, evecs = np.linalg.eigh(S_xx)
    evals = np.clip(evals, 1e-10, None)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T          # whitening
    W_inv = evecs @ np.diag(evals ** 0.5) @ evecs.T

    A = W @ S_xz
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    rank = int((s > 1e-8 * max(1.0, s[0])).sum())
    U = U[:, :rank]

    P = np.eye(d) - W_inv @ U @ U.T @ W
    return X.mean(0, keepdims=True), P


# ---------------------------------------------------------------- detection

def head_specs(d):
    """Two detection heads, deliberately at opposite ends of the deployment budget.

    LightGBM measures how much class information the latent carries - it is a
    representation probe. But it serialises to ~1.4 MB, which is 333x the XS encoder it
    sits on, so quoting encoder size alone as "deployment cost" while scoring with this
    head would misstate the deployed system by three orders of magnitude.

    MLP-8 is the honest edge configuration: a 49-parameter head that actually fits
    alongside the encoder on a microcontroller. Reporting both makes the head a measured
    axis instead of a hidden cost.
    """
    return [
        ("lgbm", lambda: lgb.LGBMClassifier(n_estimators=200, num_leaves=63,
                                             learning_rate=0.1, n_jobs=3,
                                             random_state=SEED, verbose=-1),
         12600 * 2, 1379.3),
        ("mlp8", lambda: MLPClassifier((8,), max_iter=40, random_state=SEED),
         d * 8 + 8 + 8 + 1, (d * 8 + 17) * 4 / 1024),
    ]


def detection_scores(Ztr, ytr, evals_, head_factory, head_pool=HEAD_POOL):
    """Fit one head on frozen latents and score it on each evaluation split."""
    btr = (ytr != 0).astype(int)
    if len(btr) > head_pool:
        idx = np.random.default_rng(SEED).choice(len(btr), head_pool, replace=False)
        Ztr, btr = Ztr[idx], btr[idx]
    clf = head_factory()
    clf.fit(Ztr, btr)
    res = {}
    for name, (Ze, ye, mask) in evals_.items():
        be = (ye != 0).astype(int)
        if mask is not None:
            Ze, be = Ze[mask], be[mask]
        proba = clf.predict_proba(Ze)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fpr, tpr, thr = roc_curve(be, proba)
        i = int(np.searchsorted(fpr, TARGET_FPR, side="right") - 1)
        i = max(0, min(i, len(thr) - 1))
        tp = ((proba >= thr[i]) & (be == 1)).sum()
        fp = ((proba >= thr[i]) & (be == 0)).sum()
        res[name] = {
            "f1": float(f1_score(be, pred)),
            "mcc": float(matthews_corrcoef(be, pred)),
            f"recall_at_fpr{TARGET_FPR}": float(tpr[i]),
            f"precision_at_fpr{TARGET_FPR}": float(tp / max(tp + fp, 1)),
            "n": int(len(be)), "attack_rate": float(be.mean()),
        }
    return res


def compute_srr(window):
    """Shortcut Reliance Ratio on RAW features: F1(port only) / F1(all features)."""
    Xf = np.load(DATA_DIR / "features_fit.npy")
    yf = (np.load(DATA_DIR / "labels_fit.npy") != 0).astype(int)
    Xv = np.load(DATA_DIR / "features_val.npy")
    yv = (np.load(DATA_DIR / "labels_val.npy") != 0).astype(int)
    out = {}
    for name, cols in [("port_only", [0]), ("all_features", list(range(Xf.shape[1])))]:
        t = DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(Xf[:, cols], yf)
        out[name] = float(f1_score(yv, t.predict(Xv[:, cols])))
    out["SRR"] = out["port_only"] / out["all_features"]
    return out


# ---------------------------------------------------------------- main

def score_one(args):
    """All per-model work. Top-level and picklable so a process pool can own it.

    Scoring is embarrassingly parallel across models - each reads its own frozen latents
    and shares nothing - and single-model scoring costs ~6 min, so serialising 30 models
    wastes hours on a 24-core box.
    """
    tag, n_buckets, probe_idx = args
    Zf = np.load(LAT / f"{tag}_fit.npy")
    Zv = np.load(LAT / f"{tag}_val.npy")
    Zt = np.load(LAT / f"{tag}_test.npy")
    y = {k: np.load(LAT / f"y_{k}.npy") for k in ("fit", "val", "test")}
    pb_fit = np.load(LAT / "portbucket_fit.npy")
    purged = {}
    for k in ("val", "test"):
        f_ = LAT / f"mask_purged_{k}.npy"
        purged[k] = np.load(f_) if f_.exists() else None

    rng = np.random.default_rng(SEED)
    mdl = probe_suite(Zf[probe_idx], pb_fit[probe_idx], n_buckets, rng)
    evals_ = {
        "val": (Zv, y["val"], None),
        "val_purged": (Zv, y["val"], purged["val"]),
        "test": (Zt, y["test"], None),
        "test_purged": (Zt, y["test"], purged["test"]),
    }
    det_all, head_meta = {}, {}
    for hname, factory, hparams, hkb in head_specs(Zf.shape[1]):
        det_all[hname] = detection_scores(Zf, y["fit"], evals_, factory)
        head_meta[hname] = (hparams, hkb)

    sp = RUNS / f"{tag}_summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    row = {
        "tag": tag, "lam": summ.get("lam"), "drop_port": summ.get("drop_port"),
        "d_eff": summ.get("d_eff"), "best_epoch": summ.get("best_epoch"),
        "seed": summ.get("seed", SEED), "latent_dim": int(Zf.shape[1]),
        "mdl_linear_compression": mdl["linear"]["compression"],
        "mdl_mlp_compression": mdl["mlp"]["compression"],
        **{f"{h}_{k}_{m}": det_all[h][k][m] for h in det_all for k in det_all[h]
           for m in ("f1", "mcc", f"precision_at_fpr{TARGET_FPR}")},
        **{f"{h}_head_params": head_meta[h][0] for h in head_meta},
        **{f"{h}_head_KB": head_meta[h][1] for h in head_meta},
    }
    return tag, row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((LAT / "index.json").read_text())
    tags = index["tags"]
    rng = np.random.default_rng(SEED)

    y = {s: np.load(LAT / f"y_{s}.npy") for s in ("fit", "val", "test")}
    pb = {s: np.load(LAT / f"portbucket_{s}.npy") for s in ("fit", "val", "test")}
    purged = {}
    for s in ("val", "test"):
        p = LAT / f"mask_purged_{s}.npy"
        purged[s] = np.load(p) if p.exists() else None
    n_buckets = int(max(pb["fit"].max(), pb["val"].max())) + 1

    print("=== SRR on raw features ===")
    srr = compute_srr(index["window"])
    print(f"  port_only F1={srr['port_only']:.4f}  all_features F1={srr['all_features']:.4f}"
          f"  -> SRR={srr['SRR']:.4f}")
    (OUT / "srr.json").write_text(json.dumps(srr, indent=2))

    probe_idx = rng.choice(len(y["fit"]), size=min(PROBE_POOL, len(y["fit"])), replace=False)

    cache_p = OUT / "_scored.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = [t for t in tags if t not in cache]
    print(f"\n=== scoring {len(todo)} model(s); {len(cache)} cached ===", flush=True)

    if todo:
        work = [(t, n_buckets, probe_idx) for t in todo]
        n_proc = max(1, min(WORKERS, len(todo)))
        print(f"    {n_proc} workers on {os.cpu_count()} cores", flush=True)
        with mp.get_context("spawn").Pool(n_proc) as pool:
            for i, (tag, row) in enumerate(pool.imap_unordered(score_one, work), 1):
                cache[tag] = row
                cache_p.write_text(json.dumps(cache, indent=1))
                print(f"  [{i}/{len(todo)}] {tag:16s} "
                      f"MDL lin/mlp={row['mdl_linear_compression']:.2f}/"
                      f"{row['mdl_mlp_compression']:.2f}  test F1 "
                      f"lgbm={row.get('lgbm_test_purged_f1', float('nan')):.3f} "
                      f"mlp8={row.get('mlp8_test_purged_f1', float('nan')):.3f}", flush=True)

    rows = [cache[t] for t in tags if t in cache]
    pd.DataFrame(rows).to_csv(OUT / "frontier.csv", index=False)

    # LEACE reference point on the unconstrained (lambda=0) latent, if present.
    base = "lam0"
    if (LAT / f"{base}_fit.npy").exists():
        print(f"\n=== LEACE reference on {base} ===")
        Zf = np.load(LAT / f"{base}_fit.npy")
        sub = probe_idx
        onehot = np.eye(n_buckets)[pb["fit"][sub]]
        mean, P = leace_eraser(Zf[sub], onehot)
        Ze = (Zf[sub] - mean) @ P.T + mean
        before = LogisticRegression(max_iter=300, n_jobs=-1).fit(Zf[sub], pb["fit"][sub])
        after = LogisticRegression(max_iter=300, n_jobs=-1).fit(Ze, pb["fit"][sub])
        maj = float(np.bincount(pb["fit"][sub]).max() / len(sub))
        acc_b = float(before.score(Zf[sub], pb["fit"][sub]))
        acc_a = float(after.score(Ze, pb["fit"][sub]))
        print(f"  linear port-probe acc: {acc_b:.4f} -> {acc_a:.4f} (majority={maj:.4f})")
        print(f"  self-check {'PASS' if abs(acc_a - maj) < 0.02 else 'FAIL'}: "
              f"erased probe should sit at majority class")
        (OUT / "leace_check.json").write_text(json.dumps(
            {"probe_acc_before": acc_b, "probe_acc_after": acc_a,
             "majority_baseline": maj, "passes": bool(abs(acc_a - maj) < 0.02)}, indent=2))

    print(f"\nfrontier -> {OUT / 'frontier.csv'}")


if __name__ == "__main__":
    main()
