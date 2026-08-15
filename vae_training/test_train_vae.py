"""Shape/gradient smoke test for the column VAE - no real data, no GPU wait.
Run: python test_train_vae.py
"""
import torch

from train_vae import (ColumnVAE, vae_loss, causal_mask, MultiHeadLatentAttention,
                        estimate_sigma_hat_sq, estimate_relevance)

B, L, F, LATENT = 8, 30, 46, 16

model = ColumnVAE(n_features=F, hidden_dim=64, latent_dim=LATENT, n_blocks=2)
x = torch.randn(B, L, F)
prior_var = torch.ones(LATENT)

recon_x, mu, logvar = model(x, deterministic=False)
assert recon_x.shape == (B, L, F), recon_x.shape
assert mu.shape == (B, L, LATENT), mu.shape          # row axis (L) preserved -> column-wise compression only
assert logvar.shape == (B, L, LATENT), logvar.shape

loss, recon, kld, mse = vae_loss(recon_x, x, mu, logvar, beta=0.1, prior_var=prior_var)
loss.backward()
assert all(p.grad is not None for p in model.parameters() if p.requires_grad), "dead parameter in backward graph"

# prior_var=1 must reduce exactly to the plain N(0,I) KLD formula used before ARD was added
plain_kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
assert torch.allclose(kld, plain_kld, atol=1e-5), (kld.item(), plain_kld.item())

# ARD: sigma_hat estimation and relevance ranking, on a tiny synthetic alpha pool
windows_alpha = torch.randn(20, L, F)
sigma_hat_sq = estimate_sigma_hat_sq(model, windows_alpha, batch_size=4)
assert sigma_hat_sq.shape == (LATENT,) and (sigma_hat_sq > 0).all()

w_hat, relevance, order, cum_frac, d_eff = estimate_relevance(model, windows_alpha, sigma_hat_sq, batch_size=4)
assert w_hat.shape == (LATENT,) and relevance.shape == (LATENT,)
assert order.shape == (LATENT,) and sorted(order.tolist()) == list(range(LATENT))  # a permutation
assert torch.all(cum_frac[:-1] <= cum_frac[1:] + 1e-6)  # non-decreasing cumulative sum
assert 1 <= d_eff <= LATENT

m = causal_mask(5, torch.device("cpu"))
assert torch.isinf(m[0, 1]) and m[0, 1] < 0          # row 0 cannot see column 1 (future)
assert m[4, 0] == 0                                  # row 4 can see column 0 (past)

mla = MultiHeadLatentAttention(embed_dim=64, num_heads=4, rank=16)
assert mla.q_down.out_features == 16 and mla.kv_down.out_features == 16  # rank = embed_dim // 4
out = mla(torch.randn(2, 10, 64), attn_mask=causal_mask(10, torch.device("cpu")))
assert out.shape == (2, 10, 64)

print(f"OK: recon={recon_x.shape} mu={mu.shape} loss={loss.item():.4f} "
      f"(recon={recon.item():.4f} kld={kld.item():.4f} mse={mse.item():.4f})")
