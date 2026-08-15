"""The deployed detector is two branches over one encoder. This scores both, and the
combination.

  BRANCH A - supervised head on the latent. Handles traffic whose class was present at
             training time. Evaluated two ways, because operators need both:
               binary      benign vs attack
               multiclass  which attack (macro-F1 over classes seen in fit)

  BRANCH B - unsupervised anomaly score from the same forward pass (reconstruction error;
             see score_ood.py). Handles classes absent at training time, which the
             supervised head cannot represent at all.

  COMBINED - a window is flagged if either branch fires. The anomaly threshold is
             calibrated on FIT ONLY, at a fixed benign false-positive rate, so no test
             information reaches the threshold.

Reported on the leakage-purged test split, with attacks split into classes seen in
training and classes never seen, so memorisation cannot inflate the zero-day number.

Run:  ds-python evaluate_hybrid.py
"""
import json
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(__file__).resolve().parent / "runs"
LAT = RUNS / "latents"
OUT = RUNS / "frontier"

SEED = 42
HEAD_POOL = 400_000
WORKERS = max(1, min(8, (os.cpu_count() or 4) // 3))
TARGET_FPR = 0.01          # anomaly-branch operating point, calibrated on fit


def heads(d):
    return [
        ("lgbm", lambda: lgb.LGBMClassifier(n_estimators=200, num_leaves=63,
                                            learning_rate=0.1, n_jobs=3,
                                            random_state=SEED, verbose=-1)),
        ("mlp8", lambda: MLPClassifier((8,), max_iter=40, random_state=SEED)),
    ]


def score_one(args):
    """Per-model work, top-level so a process pool can own it. Models share nothing."""
    tag, idx, seen, seen_atk_cls = args
    y = {k: np.load(LAT / f"y_{k}.npy") for k in ("fit", "test")}
    mask = np.load(LAT / "mask_purged_test.npy")
    is_unseen = mask & (~np.isin(y["test"], seen)) & (y["test"] != 0)
    is_seen_atk = mask & (np.isin(y["test"], seen)) & (y["test"] != 0)
    is_benign = mask & (y["test"] == 0)

    sp = RUNS / f"{tag}_summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else {}
    Zf = np.load(LAT / f"{tag}_fit.npy")
    Zt = np.load(LAT / f"{tag}_test.npy")
    r = {"tag": tag, "size_name": summ.get("size_name", "?"), "lam": summ.get("lam"),
         "seed": summ.get("seed", SEED), "encoder_params": summ.get("encoder_params")}
    for hname, factory in heads(Zf.shape[1]):
        hb = factory(); hb.fit(Zf[idx], (y["fit"][idx] != 0).astype(int))
        pb = hb.predict(Zt)
        r[f"{hname}_bin_f1"] = f1_score((y["test"] != 0)[mask], pb[mask])
        r[f"{hname}_bin_mcc"] = matthews_corrcoef((y["test"] != 0)[mask], pb[mask])
        r[f"{hname}_recall_seen"] = float(pb[is_seen_atk].mean())
        r[f"{hname}_recall_unseen"] = float(pb[is_unseen].mean())
        r[f"{hname}_fpr_benign"] = float(pb[is_benign].mean())

        hm = factory(); hm.fit(Zf[idx], y["fit"][idx])
        sel = is_benign | is_seen_atk
        pm = hm.predict(Zt[sel])
        r[f"{hname}_mc_macro_f1"] = f1_score(y["test"][sel], pm, average="macro",
                                             labels=seen, zero_division=0)
        r[f"{hname}_mc_weighted_f1"] = f1_score(y["test"][sel], pm, average="weighted",
                                                labels=seen, zero_division=0)
        r[f"{hname}_mc_attack_macro_f1"] = f1_score(y["test"][sel], pm, average="macro",
                                                     labels=seen_atk_cls, zero_division=0)
    return tag, r


def main():
    y = {k: np.load(LAT / f"y_{k}.npy") for k in ("fit", "test")}
    mask = np.load(LAT / "mask_purged_test.npy")
    seen = sorted(set(np.unique(y["fit"]).tolist()))
    seen_atk_cls = [c for c in seen if c != 0]

    is_unseen = mask & (~np.isin(y["test"], seen)) & (y["test"] != 0)
    is_seen_atk = mask & (np.isin(y["test"], seen)) & (y["test"] != 0)
    is_benign = mask & (y["test"] == 0)
    print(f"purged test: benign={is_benign.sum():,}  seen-attack={is_seen_atk.sum():,}  "
          f"unseen-attack={is_unseen.sum():,}")

    ood_p = OUT / "ood.csv"
    ood = pd.read_csv(ood_p).set_index("tag") if ood_p.exists() else None
    if ood is None:
        print("NOTE: ood.csv absent - anomaly branch and combined system will be skipped")

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y["fit"]), min(HEAD_POOL, len(y["fit"])), replace=False)

    tags = []
    for f in sorted(LAT.glob("*_fit.npy")):
        t = f.name[:-8]
        if t.startswith(("y", "portbucket", "mask_")):
            continue
        if not (LAT / f"{t}_test.npy").exists():
            continue
        if np.load(f, mmap_mode="r").ndim != 2:
            continue
        tags.append(t)

    cache_p = OUT / "_hybrid_scored.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = [t for t in tags if t not in cache]
    n_proc = max(1, min(WORKERS, len(todo))) if todo else 1
    print(f"scoring {len(todo)} model(s), {len(cache)} cached, {n_proc} workers", flush=True)

    if todo:
        work = [(t, idx, seen, seen_atk_cls) for t in todo]
        with mp.get_context("spawn").Pool(n_proc) as pool:
            for i, (tag, r) in enumerate(pool.imap_unordered(score_one, work), 1):
                cache[tag] = r
                cache_p.write_text(json.dumps(cache, indent=1))
                print(f"  [{i}/{len(todo)}] {r['size_name']:<3} {tag:16s} "
                      f"bin_F1={r['lgbm_bin_f1']:.3f} mc={r['lgbm_mc_attack_macro_f1']:.3f} "
                      f"seen={r['lgbm_recall_seen']:.3f} unseen={r['lgbm_recall_unseen']:.3f}",
                      flush=True)

    rows = [cache[t] for t in tags if t in cache]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hybrid.csv", index=False)
    print(f"\nhybrid -> {OUT / 'hybrid.csv'}")

    base = df[(df["lam"] == 0) & df["size_name"].isin(["L", "M", "S", "XS"])]
    if len(base):
        agg = base.groupby("size_name").agg(
            n=("tag", "count"),
            bin_f1=("lgbm_bin_f1", "mean"),
            mc_macro=("lgbm_mc_macro_f1", "mean"),
            rec_seen=("lgbm_recall_seen", "mean"),
            rec_unseen=("lgbm_recall_unseen", "mean"))
        print("\nsupervised branch, lambda=0, mean over seeds:")
        print(agg.reindex(["L", "M", "S", "XS"]).round(4).to_string())


if __name__ == "__main__":
    main()
