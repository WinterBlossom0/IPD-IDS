"""Causal column-wise VAE for CICIoT2023. Latent shape (B, L, latent_dim) - the
row axis L is never reduced, only the feature axis is compressed. Attention
and conv are causal (row t sees rows <= t only). Latent size is selected by
ARD-VAE (Saha et al., WACV 2025 - sources.md [2]) instead of fixed by hand.

Two-stage run: stage 1 trains the full LATENT_DIM model for STAGE1_EPOCHS to
get an ARD relevance ranking and effective dimension d_eff; stage 2 retrains
a fresh, smaller model at latent_dim=d_eff for STAGE2_EPOCHS. Run stages
independently with `--stage 1` / `--stage 2` (stage 2 reads d_eff back out of
ard_relevance.csv), or both back-to-back with `--stage both` (default).
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def amp_ctx():
    # bf16 autocast for the encoder/decoder matmuls (attention/conv/ffn); the
    # loss (log/exp-heavy KLD + student-t recon) is cast back to fp32 before
    # vae_loss, since that math already showed instability at fp32 precision.
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEVICE.type == "cuda"))

# Data
WINDOW_SIZE = 256

# Model
HIDDEN_DIM = 48
LATENT_DIM = 32
NUM_HEADS = 4
KERNEL_SIZE = 3
N_BLOCKS = 2
DROPOUT = 0.1

# ARD-VAE (sources.md [2])
ALPHA_POOL_ROWS = 20_000
ALPHA_BATCH_WINDOWS = 8
ESTIMATE_SIGMA_EVERY = 1
SIGMA_HAT_FLOOR = 1e-6
RELEVANCE_PROBES = 8
RELEVANCE_THRESHOLD = 0.99

# Loss
RECON_STUDENT_WEIGHT = 0.4
RECON_MSE_WEIGHT = 0.2
BETA_MAX = 0.4
BETA_WARMUP_EPOCHS = 4
FREE_BITS = 0.02
DF_DEG = 3.0

# Chronological discriminator (ported from old_ipd/cic-ids-vae.ipynb - adversarially
# strips cross-window temporal-order signal out of the latent; the time-of-day
# discriminator from that notebook is intentionally not included).
ADV_SEQ_LEN = 8
DISC_HIDDEN_DIM = 64
DISC_N_LAYERS = 2
DISC_LR = 2e-4
DISC_STEPS_PER_BATCH = 2
DISC_CHUNK_SIZE = 1024
ADV_LOSS_WEIGHT = 0.2

# Training
BATCH_SIZE = 512
STAGE1_EPOCHS = 16  # initial training, full LATENT_DIM
STAGE2_EPOCHS = 8   # retrain at the ARD-reduced latent dimension
LR = 1e-3
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0


# --------------------------------------------------------------------------- data

def load_windows():
    feature_cols = json.load(open(DATA_DIR / "feature_columns.json"))
    n_feat = len(feature_cols)

    # Already RobustScaler-scaled (fit-only) upstream in cicids2018-analysis.ipynb - no
    # re-standardization here, that would just redundantly rescale already-scaled values.
    X_fit_raw = np.load(DATA_DIR / "features_fit.npy")
    X_val_raw = np.load(DATA_DIR / "features_val.npy")
    print(f"rows: fit={X_fit_raw.shape[0]:,} val={X_val_raw.shape[0]:,} (pre-split, contiguous, no shuffling)")

    X_fit_all = torch.as_tensor(X_fit_raw, dtype=torch.float32, device=DEVICE)
    X_val = torch.as_tensor(X_val_raw, dtype=torch.float32, device=DEVICE)
    del X_fit_raw, X_val_raw

    sgd_row_end = X_fit_all.shape[0] - ALPHA_POOL_ROWS
    X_sgd = X_fit_all[:sgd_row_end]
    X_alpha = X_fit_all[sgd_row_end:]
    del X_fit_all

    windows_fit = X_sgd.unfold(0, WINDOW_SIZE, 1).permute(0, 2, 1)
    windows_val = X_val.unfold(0, WINDOW_SIZE, 1).permute(0, 2, 1)
    windows_alpha = X_alpha.unfold(0, WINDOW_SIZE, 1).permute(0, 2, 1)
    print(f"windows: fit={windows_fit.shape[0]:,} val={windows_val.shape[0]:,} "
          f"alpha_pool={windows_alpha.shape[0]:,} (window_size={WINDOW_SIZE})")
    return feature_cols, n_feat, windows_fit, windows_val, windows_alpha


# --------------------------------------------------------------------------- model

def causal_mask(L: int, device) -> torch.Tensor:
    return torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2-style MLA: Q/K/V from a shared low-rank latent."""

    def __init__(self, embed_dim, num_heads, rank, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_down = nn.Linear(embed_dim, rank)
        self.kv_down = nn.Linear(embed_dim, rank)
        self.q_up = nn.Linear(rank, embed_dim)
        self.k_up = nn.Linear(rank, embed_dim)
        self.v_up = nn.Linear(rank, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, E = x.shape
        H, D = self.num_heads, self.head_dim

        c_q = self.q_down(x)
        c_kv = self.kv_down(x)

        q = self.q_up(c_q).view(B, L, H, D).transpose(1, 2)
        k = self.k_up(c_kv).view(B, L, H, D).transpose(1, 2)
        v = self.v_up(c_kv).view(B, L, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale + attn_mask
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(B, L, E)
        return self.out_proj(out)


class CausalBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=NUM_HEADS, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        self.kernel_size = kernel_size
        self.attn = MultiHeadLatentAttention(embed_dim, num_heads, rank=embed_dim // 4, dropout=dropout)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.depthwise = nn.Conv1d(embed_dim, embed_dim, kernel_size, groups=embed_dim)
        self.pointwise = nn.Conv1d(embed_dim, embed_dim, 1)
        self.norm_conv = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        attn_out = self.attn(x, attn_mask=causal_mask(L, x.device))
        x = self.norm_attn(x + self.dropout(attn_out))

        c = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        c = self.pointwise(self.depthwise(c)).transpose(1, 2)
        x = self.norm_conv(x + self.dropout(c))

        return self.norm_ffn(x + self.ffn(x))


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, n_blocks):
        super().__init__()
        self.in_proj = nn.Linear(n_features, hidden_dim)
        self.blocks = nn.ModuleList([CausalBlock(hidden_dim) for _ in range(n_blocks)])
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.to_mu(h), self.to_logvar(h)


class Decoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, n_blocks):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([CausalBlock(hidden_dim) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden_dim, n_features)

    def forward(self, z):
        h = self.from_latent(z)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


class ColumnVAE(nn.Module):
    def __init__(self, n_features, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, n_blocks=N_BLOCKS):
        super().__init__()
        self.encoder = Encoder(n_features, hidden_dim, latent_dim, n_blocks)
        self.decoder = Decoder(n_features, hidden_dim, latent_dim, n_blocks)

    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, deterministic=False):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, deterministic)
        return self.decoder(z), mu, logvar


# --------------------------------------------------------------------------- chronological discriminator

def _all_subsequences(z: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Contiguous overlapping subsequences along dim 0: (N, D) -> (N-seq_len+1, seq_len, D)."""
    return z.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()


class TemporalDiscriminator(nn.Module):
    """LSTM discriminator for chronological-vs-shuffled window order (ported from
    old_ipd/cic-ids-vae.ipynb). Operates on each window's last-row mu as that
    window's summary latent, since (unlike the old conv VAE) this encoder emits
    one mu per row rather than one per window.
    """

    def __init__(self, latent_dim, hidden_dim=DISC_HIDDEN_DIM, num_layers=DISC_N_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                             dropout=0.2 if num_layers > 1 else 0.0)
        mlp_in = hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seqs):
        out, (h_n, _) = self.lstm(seqs)
        last_hidden = h_n[-1]
        mean_pooled = out.mean(dim=1)
        return self.mlp(torch.cat([last_hidden, mean_pooled], dim=-1)).squeeze(-1)

    def forward_chunked(self, seqs, chunk_size=DISC_CHUNK_SIZE):
        return torch.cat([self.forward(c) for c in seqs.split(chunk_size, dim=0)], dim=0)


# --------------------------------------------------------------------------- loss

def vae_loss(recon_x, x, mu, logvar, beta, prior_var, df=DF_DEG, free_bits=FREE_BITS):
    diff = recon_x - x
    student_loss = torch.log1p((diff * diff) / df).mean()
    mse_loss = (diff * diff).mean()
    recon_loss = RECON_STUDENT_WEIGHT * student_loss + RECON_MSE_WEIGHT * mse_loss

    kld_per_dim = 0.5 * (torch.log(prior_var) - logvar + (logvar.exp() + mu.pow(2)) / prior_var - 1)
    raw_kld = kld_per_dim.mean()
    kld_loss = torch.clamp(kld_per_dim, min=free_bits).mean() if free_bits > 0 else raw_kld

    total = recon_loss + beta * kld_loss
    return total, recon_loss, raw_kld, mse_loss


# --------------------------------------------------------------------------- ARD-VAE (sources.md [2])

def estimate_sigma_hat_sq(model, windows_alpha, batch_size):
    model.eval()
    n_pool = windows_alpha.shape[0]
    idx = torch.randperm(n_pool, device=windows_alpha.device)[:ALPHA_BATCH_WINDOWS]
    sq_sum = None
    n_samples = 0
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            batch = windows_alpha[idx[s:s + batch_size]].contiguous()
            mu, _ = model.encoder(batch)
            batch_sq_sum = mu.pow(2).sum(dim=(0, 1))
            sq_sum = batch_sq_sum if sq_sum is None else sq_sum + batch_sq_sum
            n_samples += mu.shape[0] * mu.shape[1]
    sigma_hat_sq = sq_sum / n_samples
    return torch.clamp(sigma_hat_sq, min=SIGMA_HAT_FLOOR)


def estimate_relevance(model, windows_alpha, sigma_hat_sq, batch_size):
    model.eval()
    n_pool = windows_alpha.shape[0]
    idx = torch.randperm(n_pool, device=windows_alpha.device)[:ALPHA_BATCH_WINDOWS]
    w_sum = torch.zeros(sigma_hat_sq.shape[0], device=windows_alpha.device)
    n_samples = 0
    for s in range(0, len(idx), batch_size):
        batch = windows_alpha[idx[s:s + batch_size]].contiguous()
        with torch.no_grad():
            mu, _ = model.encoder(batch)
        z = mu.detach().clone().requires_grad_(True)
        x_hat = model.decoder(z)
        for _ in range(RELEVANCE_PROBES):
            v = torch.randn_like(x_hat)
            grad_z, = torch.autograd.grad((v * x_hat).sum(), z, retain_graph=True)
            w_sum += grad_z.pow(2).sum(dim=(0, 1))
        n_samples += z.shape[0] * z.shape[1] * RELEVANCE_PROBES

    w_hat = w_sum / n_samples
    relevance = w_hat * sigma_hat_sq
    order = torch.argsort(relevance, descending=True)
    cum_frac = torch.cumsum(relevance[order], dim=0) / relevance.sum()
    d_eff = int(torch.searchsorted(cum_frac, RELEVANCE_THRESHOLD).item()) + 1
    return w_hat, relevance, order, cum_frac, d_eff


# --------------------------------------------------------------------------- train

def train_model(n_feat, windows_fit, windows_val, windows_alpha, latent_dim, epochs, out_prefix):
    """Train one ColumnVAE end to end and checkpoint the best-val-loss state.

    Returns (model, best_sigma_hat_sq, best_val) so a caller can chain a second
    stage (e.g. ARD relevance -> retrain at the reduced dimension) without
    reloading data.
    """
    n_win_fit, n_win_val = windows_fit.shape[0], windows_val.shape[0]

    model = ColumnVAE(n_feat, latent_dim=latent_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    discriminator = TemporalDiscriminator(latent_dim=latent_dim).to(DEVICE)
    disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=DISC_LR, weight_decay=1e-4)

    sigma_hat_sq = torch.ones(latent_dim, device=DEVICE)

    best_val = float("inf")
    best_state = None
    best_sigma_hat_sq = sigma_hat_sq
    history = []
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(SEED)

    for epoch in range(1, epochs + 1):
        if (epoch - 1) % ESTIMATE_SIGMA_EVERY == 0:
            sigma_hat_sq = estimate_sigma_hat_sq(model, windows_alpha, BATCH_SIZE)

        model.train()
        discriminator.train()
        epoch_beta = BETA_MAX * min(1.0, epoch / BETA_WARMUP_EPOCHS)
        perm = torch.randperm(n_win_fit, device=DEVICE, generator=gen)
        t0 = time.time()
        tot = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "mse": 0.0, "adv": 0.0, "n": 0}

        for s in range(0, n_win_fit, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            batch = windows_fit[idx].contiguous()

            # Chronologically-ordered batch for the discriminator: a contiguous
            # (unshuffled) window-index span, so consecutive rows are genuinely
            # consecutive in time - unlike `batch` above, which is drawn via `perm`.
            adv_start = int(torch.randint(0, n_win_fit - BATCH_SIZE, (1,), device=DEVICE).item())
            x_adv = windows_fit[adv_start:adv_start + BATCH_SIZE].contiguous()

            for _ in range(DISC_STEPS_PER_BATCH):
                disc_opt.zero_grad(set_to_none=True)
                with torch.no_grad(), amp_ctx():
                    _, mu_adv_d, _ = model(x_adv, deterministic=False)
                mu_adv_d = mu_adv_d[:, -1, :].float().detach()
                perm_d = torch.randperm(mu_adv_d.size(0), device=DEVICE)
                real_seqs = _all_subsequences(mu_adv_d, ADV_SEQ_LEN)
                fake_seqs = _all_subsequences(mu_adv_d[perm_d], ADV_SEQ_LEN)
                logit_real = discriminator.forward_chunked(real_seqs)
                logit_fake = discriminator.forward_chunked(fake_seqs)
                chron_disc_loss = (
                    F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
                    + F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))
                )
                chron_disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
                disc_opt.step()

            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                recon_x, mu_b, logvar_b = model(batch, deterministic=False)
                _, mu_adv_enc, _ = model(x_adv, deterministic=False)
            loss, recon, kld, mse = vae_loss(recon_x.float(), batch, mu_b.float(), logvar_b.float(),
                                              beta=epoch_beta, prior_var=sigma_hat_sq)

            perm_enc = torch.randperm(mu_adv_enc.size(0), device=DEVICE)
            shuffled_seqs = _all_subsequences(mu_adv_enc[:, -1, :].float()[perm_enc], ADV_SEQ_LEN)
            logit_shuffled = discriminator.forward_chunked(shuffled_seqs)
            chron_adv_loss = F.binary_cross_entropy_with_logits(logit_shuffled, torch.ones_like(logit_shuffled))

            (loss + ADV_LOSS_WEIGHT * chron_adv_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            n = batch.size(0)
            tot["loss"] += loss.item() * n
            tot["recon"] += recon.item() * n
            tot["kld"] += kld.item() * n
            tot["mse"] += mse.item() * n
            tot["adv"] += chron_adv_loss.item() * n
            tot["n"] += n

        sched.step()

        model.eval()
        vtot = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "mse": 0.0, "n": 0}
        with torch.no_grad():
            for s in range(0, n_win_val, BATCH_SIZE):
                batch = windows_val[s:s + BATCH_SIZE].contiguous()
                with amp_ctx():
                    recon_x, mu_b, logvar_b = model(batch, deterministic=False)
                loss, recon, kld, mse = vae_loss(recon_x.float(), batch, mu_b.float(), logvar_b.float(),
                                                  beta=epoch_beta, prior_var=sigma_hat_sq)
                n = batch.size(0)
                vtot["loss"] += loss.item() * n
                vtot["recon"] += recon.item() * n
                vtot["kld"] += kld.item() * n
                vtot["mse"] += mse.item() * n
                vtot["n"] += n

        train_loss = tot["loss"] / tot["n"]
        val_loss = vtot["loss"] / vtot["n"]
        # Epochs still inside the beta warmup are graded on a lower KLD weight, so
        # their val_loss isn't comparable to later epochs - excluded from "best".
        if epoch > BETA_WARMUP_EPOCHS and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_sigma_hat_sq = sigma_hat_sq.clone()
            marker = "  <-- best"
        elif epoch <= BETA_WARMUP_EPOCHS:
            marker = "  (warmup, not eligible)"
        else:
            marker = ""

        history.append({
            "epoch": epoch, "beta": epoch_beta,
            "train_loss": train_loss, "train_recon": tot["recon"] / tot["n"],
            "train_kld": tot["kld"] / tot["n"], "train_mse": tot["mse"] / tot["n"],
            "train_adv": tot["adv"] / tot["n"],
            "val_loss": val_loss, "val_recon": vtot["recon"] / vtot["n"],
            "val_kld": vtot["kld"] / vtot["n"], "val_mse": vtot["mse"] / vtot["n"],
            "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0,
            "sigma_hat_sq_min": sigma_hat_sq.min().item(), "sigma_hat_sq_max": sigma_hat_sq.max().item(),
        })
        pd.DataFrame(history).to_csv(OUT_DIR / f"{out_prefix}_history.csv", index=False)

        print(f"[{out_prefix}] epoch {epoch}/{epochs}  train_loss={train_loss:.5f} "
              f"(recon={tot['recon']/tot['n']:.5f} kld={tot['kld']/tot['n']:.6f} adv={tot['adv']/tot['n']:.5f})  "
              f"val_loss={val_loss:.5f} (recon={vtot['recon']/vtot['n']:.5f} kld={vtot['kld']/vtot['n']:.6f})  "
              f"beta={epoch_beta:.3f}  sigma_hat_sq=[{sigma_hat_sq.min().item():.4f}, {sigma_hat_sq.max().item():.4f}]  "
              f"({time.time()-t0:.1f}s){marker}", flush=True)

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT_DIR / f"{out_prefix}.pth")
    print(f"\n[{out_prefix}] done training. best_val_loss={best_val:.5f}  -> {OUT_DIR/f'{out_prefix}.pth'}")

    return model, best_sigma_hat_sq, best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                         help="run only stage 1, only stage 2 (reads d_eff from ard_relevance.csv), or both")
    args = parser.parse_args()

    feature_cols, n_feat, windows_fit, windows_val, windows_alpha = load_windows()
    d_eff = None

    if args.stage in ("1", "both"):
        print(f"=== stage 1: initial training (latent_dim={LATENT_DIM}, epochs={STAGE1_EPOCHS}) ===", flush=True)
        model, best_sigma_hat_sq, _ = train_model(
            n_feat, windows_fit, windows_val, windows_alpha,
            latent_dim=LATENT_DIM, epochs=STAGE1_EPOCHS, out_prefix="column_vae",
        )

        w_hat, relevance, order, cum_frac, d_eff = estimate_relevance(model, windows_alpha, best_sigma_hat_sq, BATCH_SIZE)
        ranked = pd.DataFrame({
            "latent_axis": order.cpu().numpy(),
            "sigma_hat_sq": best_sigma_hat_sq[order].cpu().numpy(),
            "decoder_sensitivity": w_hat[order].cpu().numpy(),
            "relevance_score": relevance[order].cpu().numpy(),
            "cumulative_fraction": cum_frac.cpu().numpy(),
            "kept": np.arange(1, LATENT_DIM + 1) <= d_eff,
        })
        ranked.to_csv(OUT_DIR / "ard_relevance.csv", index=False)
        print(f"ARD: d_eff={d_eff}/{LATENT_DIM} latent axes reach {RELEVANCE_THRESHOLD:.0%} cumulative relevance")
        print(f"  relevant axes (original indices): {order[:d_eff].cpu().tolist()}")
        print(f"  -> {OUT_DIR/'ard_relevance.csv'}")

    if args.stage in ("2", "both"):
        if d_eff is None:
            ranked = pd.read_csv(OUT_DIR / "ard_relevance.csv")
            d_eff = int(ranked["kept"].sum())
            print(f"loaded d_eff={d_eff} from {OUT_DIR/'ard_relevance.csv'}", flush=True)

        print(f"\n=== stage 2: retrain at reduced dimension (latent_dim={d_eff}, epochs={STAGE2_EPOCHS}) ===", flush=True)
        train_model(
            n_feat, windows_fit, windows_val, windows_alpha,
            latent_dim=d_eff, epochs=STAGE2_EPOCHS, out_prefix="column_vae_reduced",
        )


if __name__ == "__main__":
    main()
