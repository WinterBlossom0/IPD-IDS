"""Port-invariant column VAE - the sweep that produces the shortcut frontier.

One run = one erasure strength lambda. The encoder is trained to reconstruct as usual
while an adversary tries to read the destination-port bucket off the latent; the encoder
is simultaneously pushed to make that latent *uninformative* about port.

Two departures from the earlier chronological-discriminator experiment, both deliberate,
both fixing observed failures:

  1. CONFUSION, NOT INVERSION. The encoder minimises KL(adversary_softmax || uniform)
     rather than maximising the adversary's cross-entropy. Maximising an adversary's loss
     is unbounded above - the encoder can always make the adversary worse - which is what
     drove sigma_hat_sq to ~5e3 and stopped validation loss ever recovering. KL-to-uniform
     is bounded below at zero and attained exactly when the latent carries no port
     information, so an equilibrium exists.

  2. SCALE-FREE ADVERSARIAL INPUT. mu is L2-normalised before the adversary sees it, so
     adversarial pressure cannot be relieved by inflating latent magnitude. That closes the
     feedback loop into the ARD prior-variance estimate, which is computed from mu.

WINDOW_SIZE drops 256 -> 64: attention is O(L^2) and this is what makes a 6-point sweep
fit in one session (benchmarked 4.3x faster per batch on an RTX 5080).

Usage:  python train_invariant.py --lam 0.25
        python train_invariant.py --lam 0.0 --drop-port    # port-deleted baseline
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_vae import ColumnVAE, vae_loss, estimate_sigma_hat_sq, estimate_relevance, amp_ctx, DEVICE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SHORTCUT_DIR = DATA_DIR / "shortcut"
OUT_DIR = Path(__file__).resolve().parent / "runs"

SEED = 42
WINDOW_SIZE = 64
LATENT_DIM = 32
BATCH_SIZE = 512
EPOCHS = 8
BETA_MAX = 0.4
BETA_WARMUP_EPOCHS = 2
LR = 1e-3
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0

DISC_HIDDEN = 96
DISC_LR = 2e-4
DISC_STEPS = 1
ALPHA_POOL_ROWS = 20_000
VAL_SUBSAMPLE = 0.25      # sweep-time only; final numbers come from evaluate_frontier.py


class PortDiscriminator(nn.Module):
    """Reads the destination-port bucket off a (already L2-normalised) latent vector."""

    def __init__(self, latent_dim, n_buckets, hidden=DISC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, n_buckets),
        )

    def forward(self, z):
        return self.net(z)


def confusion_loss(logits):
    """KL(softmax(logits) || uniform), averaged. Zero iff the adversary is at chance."""
    logp = F.log_softmax(logits, dim=-1)
    uniform = torch.full_like(logp, 1.0 / logits.shape[-1])
    return F.kl_div(logp, uniform, reduction="batchmean")


def load_data(drop_port):
    X_fit = np.load(DATA_DIR / "features_fit.npy")
    X_val = np.load(DATA_DIR / "features_val.npy")
    b_fit = np.load(SHORTCUT_DIR / "portbucket_fit.npy")
    b_val = np.load(SHORTCUT_DIR / "portbucket_val.npy")
    meta = json.loads((SHORTCUT_DIR / "meta.json").read_text())

    if drop_port:
        # Ablation: delete the column outright rather than erase it from the latent.
        X_fit = np.delete(X_fit, 0, axis=1)
        X_val = np.delete(X_val, 0, axis=1)

    Xf = torch.as_tensor(X_fit, dtype=torch.float32, device=DEVICE)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=DEVICE)
    bf = torch.as_tensor(b_fit.astype(np.int64), device=DEVICE)
    bv = torch.as_tensor(b_val.astype(np.int64), device=DEVICE)

    sgd_end = Xf.shape[0] - ALPHA_POOL_ROWS
    w_fit = Xf[:sgd_end].unfold(0, WINDOW_SIZE, 1).permute(0, 2, 1)
    w_alpha = Xf[sgd_end:].unfold(0, WINDOW_SIZE, 1).permute(0, 2, 1)
    w_val = Xv.unfold(0, WINDOW_SIZE, 1).permute(0, 2, 1)

    # A window's port target is its *last* row - the flow the window is "about",
    # consistent with the causal design (row t sees rows <= t only).
    t_fit = bf[WINDOW_SIZE - 1:sgd_end]
    t_val = bv[WINDOW_SIZE - 1:]

    n_feat = Xf.shape[1]
    print(f"windows: fit={w_fit.shape[0]:,} val={w_val.shape[0]:,} alpha={w_alpha.shape[0]:,} "
          f"(W={WINDOW_SIZE}, n_feat={n_feat}, port_buckets={meta['port_buckets']})")
    return n_feat, meta["port_buckets"], w_fit, w_val, w_alpha, t_fit, t_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True, help="erasure strength")
    ap.add_argument("--drop-port", action="store_true", help="delete Dst Port column instead")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag", type=str, default=None)
    # Capacity axis - the primary axis for edge deployment. Only the ENCODER ships at
    # inference time (the decoder exists to train the representation), so encoder
    # parameter count is the number that matters on-device.
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--latent", type=int, default=LATENT_DIM)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--size-name", type=str, default=None, help="short label, e.g. XS")
    ap.add_argument("--seed", type=int, default=SEED,
                     help="repeat runs with different seeds to estimate run-to-run variance")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    size_lbl = args.size_name or f"h{args.hidden}l{args.latent}b{args.blocks}"
    tag = args.tag or (f"{size_lbl}_dropport" if args.drop_port else f"{size_lbl}_lam{args.lam:g}")
    if args.seed != SEED:
        tag = f"{tag}_s{args.seed}"
    if (OUT_DIR / f"{tag}.pt").exists():
        print(f"[{tag}] checkpoint exists, skipping")
        return

    n_feat, n_buckets, w_fit, w_val, w_alpha, t_fit, t_val = load_data(args.drop_port)
    n_fit, n_val = w_fit.shape[0], w_val.shape[0]

    model = ColumnVAE(n_feat, hidden_dim=args.hidden, latent_dim=args.latent,
                      n_blocks=args.blocks).to(DEVICE)
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"[{tag}] encoder params={enc_params:,}  ({enc_params * 4 / 1024:.1f} KB fp32)")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    disc = PortDiscriminator(args.latent, n_buckets).to(DEVICE)
    disc_opt = torch.optim.AdamW(disc.parameters(), lr=DISC_LR, weight_decay=1e-4)

    sigma_hat_sq = torch.ones(args.latent, device=DEVICE)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(args.seed)
    best = {"val": float("inf"), "state": None, "sigma": sigma_hat_sq.clone(), "epoch": -1}
    history = []

    n_val_use = max(1, int(n_val * VAL_SUBSAMPLE))
    val_idx = torch.randperm(n_val, device=DEVICE, generator=gen)[:n_val_use].sort().values

    for epoch in range(1, args.epochs + 1):
        sigma_hat_sq = estimate_sigma_hat_sq(model, w_alpha, BATCH_SIZE)
        model.train(); disc.train()
        beta = BETA_MAX * min(1.0, epoch / BETA_WARMUP_EPOCHS)
        perm = torch.randperm(n_fit, device=DEVICE, generator=gen)
        t0 = time.time()
        tot = {"loss": 0., "recon": 0., "kld": 0., "adv": 0., "dacc": 0., "n": 0}

        for s in range(0, n_fit, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            batch = w_fit[idx].contiguous()
            tgt = t_fit[idx]

            # --- adversary step: learn to read port off the (detached) latent
            for _ in range(DISC_STEPS):
                disc_opt.zero_grad(set_to_none=True)
                with torch.no_grad(), amp_ctx():
                    mu_d, _ = model.encoder(batch)
                z_d = F.normalize(mu_d[:, -1, :].float().detach(), dim=-1)
                d_loss = F.cross_entropy(disc(z_d), tgt)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), GRAD_CLIP)
                disc_opt.step()

            # --- encoder/decoder step: reconstruct, and confuse the adversary
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                recon_x, mu_b, logvar_b = model(batch, deterministic=False)
            loss, recon, kld, _ = vae_loss(recon_x.float(), batch, mu_b.float(), logvar_b.float(),
                                            beta=beta, prior_var=sigma_hat_sq)
            z_e = F.normalize(mu_b[:, -1, :].float(), dim=-1)
            logits_e = disc(z_e)
            adv = confusion_loss(logits_e)
            (loss + args.lam * adv).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            n = batch.size(0)
            with torch.no_grad():
                dacc = (logits_e.argmax(-1) == tgt).float().mean().item()
            tot["loss"] += loss.item() * n; tot["recon"] += recon.item() * n
            tot["kld"] += kld.item() * n;   tot["adv"] += adv.item() * n
            tot["dacc"] += dacc * n;        tot["n"] += n

        sched.step()

        model.eval(); disc.eval()
        v = {"loss": 0., "recon": 0., "dacc": 0., "n": 0}
        with torch.no_grad():
            for s in range(0, n_val_use, BATCH_SIZE):
                vi = val_idx[s:s + BATCH_SIZE]
                batch = w_val[vi].contiguous()
                with amp_ctx():
                    recon_x, mu_b, logvar_b = model(batch, deterministic=False)
                loss, recon, _, _ = vae_loss(recon_x.float(), batch, mu_b.float(), logvar_b.float(),
                                              beta=beta, prior_var=sigma_hat_sq)
                z = F.normalize(mu_b[:, -1, :].float(), dim=-1)
                n = batch.size(0)
                v["loss"] += loss.item() * n; v["recon"] += recon.item() * n
                v["dacc"] += (disc(z).argmax(-1) == t_val[vi]).float().mean().item() * n
                v["n"] += n

        row = {
            "epoch": epoch, "lam": args.lam, "beta": beta,
            "train_loss": tot["loss"] / tot["n"], "train_recon": tot["recon"] / tot["n"],
            "train_kld": tot["kld"] / tot["n"], "train_adv": tot["adv"] / tot["n"],
            "train_disc_acc": tot["dacc"] / tot["n"],
            "val_loss": v["loss"] / v["n"], "val_recon": v["recon"] / v["n"],
            "val_disc_acc": v["dacc"] / v["n"],
            "sigma_max": sigma_hat_sq.max().item(), "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(OUT_DIR / f"{tag}_history.csv", index=False)

        # Warmup epochs are graded on a lower KLD weight -> not comparable, excluded.
        mark = ""
        if epoch > BETA_WARMUP_EPOCHS and row["val_recon"] < best["val"]:
            best.update(val=row["val_recon"], epoch=epoch,
                        state={k: t.detach().clone() for k, t in model.state_dict().items()},
                        sigma=sigma_hat_sq.clone())
            mark = "  <-- best"
        elif epoch <= BETA_WARMUP_EPOCHS:
            mark = "  (warmup)"

        print(f"[{tag}] ep{epoch}/{args.epochs} recon={row['train_recon']:.5f} "
              f"val_recon={row['val_recon']:.5f} adv={row['train_adv']:.4f} "
              f"disc_acc={row['train_disc_acc']:.3f}/val {row['val_disc_acc']:.3f} "
              f"sig_max={row['sigma_max']:.3g} ({row['seconds']:.0f}s){mark}", flush=True)

    if best["state"] is None:                       # every epoch was warmup (short runs)
        best["state"] = {k: t.detach().clone() for k, t in model.state_dict().items()}
        best["sigma"] = sigma_hat_sq.clone()
        best["epoch"] = args.epochs
        best["val"] = history[-1]["val_recon"]
    model.load_state_dict(best["state"])
    torch.save({"model": best["state"], "sigma_hat_sq": best["sigma"].cpu(),
                "lam": args.lam, "drop_port": args.drop_port, "n_feat": n_feat,
                "window": WINDOW_SIZE, "latent_dim": args.latent,
                "hidden_dim": args.hidden, "n_blocks": args.blocks,
                "size_name": size_lbl, "encoder_params": enc_params, "seed": args.seed,
                "best_epoch": best["epoch"]},
               OUT_DIR / f"{tag}.pt")

    _, _, order, cum, d_eff = estimate_relevance(model, w_alpha, best["sigma"], BATCH_SIZE)
    summary = {"tag": tag, "lam": args.lam, "drop_port": args.drop_port,
               "size_name": size_lbl, "hidden_dim": args.hidden, "latent_dim": args.latent,
               "n_blocks": args.blocks, "encoder_params": enc_params, "seed": args.seed,
               "best_epoch": best["epoch"], "best_val_recon": best["val"],
               "final_val_disc_acc": history[-1]["val_disc_acc"],
               "d_eff": d_eff, "relevant_axes": order[:d_eff].cpu().tolist(),
               "total_seconds": sum(h["seconds"] for h in history)}
    (OUT_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] done. best_epoch={best['epoch']} val_recon={best['val']:.5f} "
          f"d_eff={d_eff}/{args.latent} -> {OUT_DIR / f'{tag}.pt'}")


if __name__ == "__main__":
    main()
