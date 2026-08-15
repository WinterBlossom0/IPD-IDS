"""Unsupervised out-of-distribution scoring - the deployment answer for zero-day attacks.

A supervised head can only flag attack classes it was trained on. Measured on this split,
that limitation is severe: the largest encoder detects 0.2% of attack classes absent from
training, i.e. it is effectively blind to novel attacks.

A VAE gives a label-free alternative from the same forward pass. Reconstruction error
answers "does this window look like the traffic I was fitted on?", which does not require
having seen the attack. We score three quantities per window and evaluate each as a
detector of the four attack classes that appear only in test:

  recon  - blended Student-t / MSE reconstruction error (the anomaly signal)
  kl     - KL to the ARD prior (how far the posterior is pushed)
  score  - recon + beta * kl, the training objective read as an anomaly score

Evaluation is AUROC of benign vs unseen-attack windows only: seen attack classes are
excluded so the number cannot be inflated by memorisation.

Run:  torch-python score_ood.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_vae import ColumnVAE, amp_ctx, DEVICE, RECON_STUDENT_WEIGHT, RECON_MSE_WEIGHT, DF_DEG

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = Path(__file__).resolve().parent / "runs"
LAT = RUNS / "latents"
OUT = RUNS / "frontier"
BATCH = 1024
BETA = 0.4
TARGET_FPR = 0.01


@torch.no_grad()
def per_window_scores(model, windows, sigma_hat_sq):
    n = windows.shape[0]
    rec = torch.empty(n, dtype=torch.float32)
    kl = torch.empty(n, dtype=torch.float32)
    for s in range(0, n, BATCH):
        x = windows[s:s + BATCH].contiguous()
        with amp_ctx():
            recon, mu, logvar = model(x, deterministic=True)
        recon, mu, logvar = recon.float(), mu.float(), logvar.float()
        diff = recon - x
        student = torch.log1p((diff * diff) / DF_DEG).mean(dim=(1, 2))
        mse = (diff * diff).mean(dim=(1, 2))
        rec[s:s + BATCH] = (RECON_STUDENT_WEIGHT * student + RECON_MSE_WEIGHT * mse).cpu()
        kld = 0.5 * (torch.log(sigma_hat_sq) - logvar
                     + (logvar.exp() + mu.pow(2)) / sigma_hat_sq - 1)
        kl[s:s + BATCH] = kld.mean(dim=(1, 2)).cpu()
    return rec.numpy(), kl.numpy()


def auroc(scores, labels):
    """Rank-based AUROC; labels 1 = anomaly. Avoids a sklearn dependency here."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


CAL_WINDOWS = 200_000      # fit subsample for threshold calibration


def main():
    y_fit = np.load(LAT / "y_fit.npy")
    y_test = np.load(LAT / "y_test.npy")
    mask = np.load(LAT / "mask_purged_test.npy")
    seen = set(np.unique(y_fit).tolist())

    unseen = mask & (~np.isin(y_test, list(seen))) & (y_test != 0)
    benign = mask & (y_test == 0)
    seen_atk = mask & (np.isin(y_test, list(seen))) & (y_test != 0)
    print(f"benign={benign.sum():,}  seen-attack={seen_atk.sum():,}  "
          f"UNSEEN-attack={unseen.sum():,}")

    rows = []
    cache = {}
    fitcache = {}
    for ck in sorted(RUNS.glob("*.pt")):
        tag = ck.stem
        blob = torch.load(ck, map_location=DEVICE)
        dp = bool(blob["drop_port"])
        if dp not in cache:
            cache.clear()
            X = np.load(DATA / "features_test.npy")
            if dp:
                X = np.delete(X, 0, axis=1)
            Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
            cache[dp] = Xt.unfold(0, blob["window"], 1).permute(0, 2, 1)
        w = cache[dp]

        model = ColumnVAE(blob["n_feat"], hidden_dim=blob.get("hidden_dim", 48),
                          latent_dim=blob["latent_dim"],
                          n_blocks=blob.get("n_blocks", 2)).to(DEVICE)
        model.load_state_dict(blob["model"])
        model.eval()
        sig = blob["sigma_hat_sq"].to(DEVICE)

        rec, kl = per_window_scores(model, w, sig)
        combo = rec + BETA * kl

        # Operating point: threshold set on BENIGN FIT windows only, so no test
        # information reaches it. Reported recall is then honest at a fixed alarm budget.
        if dp not in fitcache:
            Xf = np.load(DATA / "features_fit.npy")
            if dp:
                Xf = np.delete(Xf, 0, axis=1)
            Xf = torch.as_tensor(Xf, dtype=torch.float32, device=DEVICE)
            fitcache.clear()
            fitcache[dp] = Xf.unfold(0, blob["window"], 1).permute(0, 2, 1)
        wf = fitcache[dp]
        step = max(1, wf.shape[0] // CAL_WINDOWS)
        rec_f, kl_f = per_window_scores(model, wf[::step].contiguous(), sig)
        yb = y_fit[::step][: len(rec_f)] == 0
        thr = float(np.quantile(rec_f[yb], 1.0 - TARGET_FPR))

        r = {"tag": tag, "size_name": blob.get("size_name", "?"),
             "lam": blob.get("lam"), "seed": blob.get("seed", 42),
             "encoder_params": blob.get("encoder_params")}
        for nm, sc in [("recon", rec), ("kl", kl), ("elbo", combo)]:
            # unseen-vs-benign is the zero-day question; seen-vs-benign is the sanity check
            sel_u = benign | unseen
            sel_s = benign | seen_atk
            r[f"auroc_{nm}_unseen"] = auroc(sc[sel_u], unseen[sel_u].astype(int))
            r[f"auroc_{nm}_seen"] = auroc(sc[sel_s], seen_atk[sel_s].astype(int))
        flag = rec >= thr
        r["recon_thr_fit"] = thr
        r["ood_recall_unseen_at_fpr1"] = float(flag[unseen].mean())
        r["ood_recall_seen_at_fpr1"] = float(flag[seen_atk].mean())
        r["ood_fpr_benign_test"] = float(flag[benign].mean())
        rows.append(r)
        print(f"  {tag:16s} AUROC unseen: recon={r['auroc_recon_unseen']:.3f} "
              f"kl={r['auroc_kl_unseen']:.3f} elbo={r['auroc_elbo_unseen']:.3f} "
              f"| recall@1%FPR unseen={r['ood_recall_unseen_at_fpr1']:.3f} "
              f"benignFPR(test)={r['ood_fpr_benign_test']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(OUT / "ood.csv", index=False)

    print(f"\nood -> {OUT / 'ood.csv'}")


if __name__ == "__main__":
    main()
