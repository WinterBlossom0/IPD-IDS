# Sources and provenance

Every external asset this project depends on, and the licence or attribution condition
attached to it. The bibliography for the manuscript is `paper/refs.bib`, which is generated
by `paper/verify_refs.py` from DBLP and the arXiv API — **this file records provenance and
obligations, not citation formatting.** Where a work appears in both, the BibTeX in
`refs.bib` is authoritative.

---

## 1. Dataset — CSE-CIC-IDS2018

**Provenance.** Canadian Institute for Cybersecurity, University of New Brunswick.
Dataset page: <https://www.unb.ca/cic/datasets/ids-2018.html>

**Raw files used.** Official public S3 bucket, ten daily CSVs of CICFlowMeter flow
features:

```
s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/
```

Retrieved into `data/cicids2018_raw/` (git-ignored; rebuild per `data/README.md`).

**Attribution obligation.** CIC's terms require citing the dataset paper *and*
acknowledging CIC/UNB as the source. Both are satisfied in the manuscript: the paper is
cited as `sharafaldin2018toward`, and the dataset page is credited in Section IV.

> I. Sharafaldin, A. Habibi Lashkari, and A. A. Ghorbani, "Toward generating a new
> intrusion detection dataset and intrusion traffic characterization," in *Proc. 4th Int.
> Conf. Information Systems Security and Privacy (ICISSP)*, 2018, pp. 108–116.
> doi: 10.5220/0006639801080116

**Known defects — material to how this repo uses the data.** Two audits document errors in
this capture, and the manuscript reports results in light of them:

- Engelen, Rimmer & Joosen (IEEE SPW/WTMC 2021) — CICFlowMeter faults including premature
  TCP termination and flag-counting errors. `refs.bib: engelen2021troubleshooting`
- Liu, Engelen, Lynar, Essam & Joosen (IEEE CNS 2022) — extends the audit to the 2018
  capture. Records that **`DoS-SlowHTTPTest` was never successfully executed** (the tool was
  launched against port 21) and that **`FTP-BruteForce` and `Infiltration` carry substantial
  label corruption**. `refs.bib: liu2022error`

Corrected labels are published by that group at
<https://github.com/GintsEngelen/CNS2022_Code>. **This repo's current results use the
original labels**; migrating to the corrected release is listed as future work in the
manuscript. The affected classes are named explicitly so the limitation is visible.

---

## 2. Method — ARD-VAE (latent dimensionality selection)

> S. Saha, S. Joshi, and R. Whitaker, "ARD-VAE: A statistical formulation to find the
> relevant latent dimensions of variational autoencoders," in *Proc. IEEE/CVF Winter Conf.
> Applications of Computer Vision (WACV)*, 2025, pp. 889–898. arXiv:2501.10901

**Used in:** `vae_training/train_vae.py` — `estimate_sigma_hat_sq` (online prior-variance
estimation) and `estimate_relevance` (post-training relevance ranking), which together give
the effective latent dimension rather than fixing it by hand. `refs.bib: saha2025ardvae`

---

## 3. Method — adversarial attribute removal

The port-erasure branch descends from domain-adversarial training. We claim no novelty in
the mechanism; the contribution is the measurement apparatus around it.

> Y. Ganin *et al.*, "Domain-adversarial training of neural networks."
> `refs.bib: ganin2016dann`

**Departure from the standard formulation**, documented in
`vae_training/train_invariant.py`: the encoder minimises KL-to-uniform (a *confusion* loss,
bounded below) rather than maximising the adversary's cross-entropy (unbounded above, no
equilibrium). The unbounded form was tried first and drove the ARD prior variance to
~5×10³ without recovery.

---

## 4. Method — verification of concept removal

> A. Kumar, C. Tan, and A. Sharma, "Probing classifiers are unreliable for concept removal
> and detection," NeurIPS 35, 2022, pp. 17994–18008. `refs.bib: kumar2022probing`

Establishes that an encoder defeating its adversary is **not** evidence the concept is gone.
Directly motivates our use of description-length probing instead of adversary accuracy — a
choice vindicated empirically here, since adversary accuracy pins at exactly 0.443, the
majority-class rate of the port-bucket distribution.

> E. Voita and I. Titov, "Information-theoretic probing with minimum description length,"
> EMNLP 2020. `refs.bib: voita2020mdl`
> N. Belrose *et al.*, "LEACE: Perfect linear concept erasure in closed form," NeurIPS 36,
> 2023. `refs.bib: belrose2023leace`

**Used in:** `vae_training/evaluate_frontier.py` — `online_mdl` (prequential codelength) and
`leace_eraser` (closed-form linear erasure, validated on synthetic data where a linear probe
falls from 0.9891 to the 0.2527 majority baseline).

---

## 5. Architecture components

> A. Vaswani *et al.*, "Attention is all you need," NeurIPS 2017. `refs.bib: vaswani2017attention`
> DeepSeek-AI, "DeepSeek-V2," arXiv:2405.04434 — low-rank latent attention.
> `refs.bib: deepseekv2`
> D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," 2014. `refs.bib: kingma2014vae`
> I. Higgins *et al.*, "beta-VAE," ICLR 2017 — the $\beta$ warm-up. `refs.bib: higgins2017betavae`
> D. P. Kingma *et al.*, "Improving variational inference with inverse autoregressive flow,"
> 2016 — the free-bits floor. `refs.bib: kingma2016iaf`

**Used in:** `vae_training/train_vae.py` — `MultiHeadLatentAttention`, `CausalBlock`,
`ColumnVAE`, `vae_loss`.

---

## 6. Software

| Tool | Version | Role | Licence |
|---|---|---|---|
| PyTorch | 2.13.0+cu130 | training, latent extraction, efficiency benchmark | BSD-3-Clause |
| scikit-learn | ≥1.9 | probes, MLP-8 head, decision-tree SRR baseline | BSD-3-Clause |
| LightGBM | ≥4.7 | representation-quality detection head | MIT |
| matplotlib | ≥3.11 | figures | PSF-based (matplotlib licence) |
| pandas / NumPy / SciPy | — | data handling | BSD-3-Clause |
| tectonic | 0.15.0 | LaTeX engine for `paper/` | MIT |
| IEEEtran | — | manuscript class | LPPL |

Bibliographic records were resolved programmatically against **DBLP**
(<https://dblp.org>, records offered under CC0) and the **arXiv API**
(<https://arxiv.org/help/api>). No entry in `refs.bib` derives from a secondary summary;
see the header of that file and `paper/verify_refs.py`.

---

## 7. Superseded first-generation pipeline

An earlier CIC-IDS2018 pipeline (conv VAE with chronological and time-of-day adversaries,
plus its ablations) previously lived in `old_ipd/`. That directory has been removed. Its
code and notebooks remain recoverable from this repository's git history at their original
root paths; its bulk derived arrays were regenerable outputs and were not retained. Small
run metadata is preserved under `archive/old_ipd_metadata/`.
