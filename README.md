# Shortcut Reliance and Capacity in Parameter-Efficient Network Intrusion Detection

Research code and manuscript for a study of **how much of flow-based intrusion-detection
accuracy is real detection**, and what it costs to remove the part that is not.

**Authors** — Vihaan Ovalekar, Ojas Gore, Yash Pradhan, Ishan Tanawade
**Mentor** — Prof. Deepali Patil
Department of Artificial Intelligence and Data Science,
SVKM's Dwarkadas J. Sanghvi College of Engineering, Mumbai, India

---

## Summary

On CSE-CIC-IDS2018, a depth-8 decision tree given **only the destination port** attains
binary `F1 = 0.9257`, which is **96.7%** of the `F1 = 0.9572` obtained from all 46 features.
Attacks in that capture were orchestrated against fixed ports, so port identity is very
nearly the label. We formalise this as the **Shortcut Reliance Ratio (SRR)**, then train a
causal column-wise VAE across capacities from 40,576 down to 1,060 encoder parameters,
crossed with an adversarial objective that erases port information from the latent.

Erasure is verified by **prequential minimum description length** on frozen latents, not by
adversary loss — on this data the adversary is degenerate, pinning at exactly `0.443`, the
majority-class rate of the port-bucket distribution.

### Headline results

| Result | Status |
|---|---|
| SRR = **0.9672** — the benchmark is nearly answerable from one field | measured |
| Erasure works: residual port information falls **7.16 → 2.38** (MDL, size L) | measured |
| Zero-day AUROC **0.97–0.99** at every capacity, incl. 1,060 params | measured, 5 seeds |
| Supervised and anomaly branches are complementary (0.675/0.041 vs 0.366/0.989) | measured |
| Smaller models lean harder on the shortcut | **refuted** — larger retain more |
| Erasure improves cross-environment transfer | **refuted** — best transfer at λ=0 |
| A 3,024-param encoder matches a 40,576-param one on transfer | **inside noise** |

The last row is *not* statistically supported: between-size range at λ=0 is 0.144 while
within-size spread across λ reaches 0.199. Do not cite it without error bars. The deployed
system is encoder **+ classifier**: a LightGBM head adds ~1,379 KB regardless of encoder
size, so only the 49-parameter MLP-8 configuration is microcontroller-deployable.

---

## Repository layout

```
paper/              LaTeX manuscript (IEEE conference, 6-page limit)
  main.tex            paper source; results tables are \input, never hand-edited
  refs.bib            32 references, every one resolved against DBLP or the arXiv API
  verify_refs.py      rebuilds refs.bib from authoritative records only
  make_tables.py      generates tables + prose macros from the experiment CSVs
  check_pdf.py        post-build gate: page limit, sections, unresolved citations
  build.sh            regenerate tables, then compile with tectonic

vae_training/       experiment code
  train_vae.py          the causal column-wise VAE (ColumnVAE) and ARD machinery
  prep_shortcut.py      port recovery, port bucketing, duplicate/leakage masks
  train_invariant.py    capacity x erasure training; one run per grid cell
  extract_latents.py    freeze each model, dump latents (all evaluation runs on these)
  evaluate_frontier.py  SRR, MDL probes, LEACE, LightGBM detection heads
  bench_efficiency.py   params, FLOPs, int8 size, single-core CPU latency
  make_figures.py       publication figures
  score_ood.py          unsupervised zero-day scoring from reconstruction error
  evaluate_hybrid.py    binary + multiclass supervised branch, seen vs unseen split
  run_grid.sh           capacity x erasure grid
  run_seeds.sh          seed repeats (serial)
  run_seeds_parallel.sh seed repeats, 4-way concurrent (~1.8 GB VRAM each)
  run_eval_pipeline.sh  latents -> efficiency -> frontier -> tables -> figures

data/               inputs and derived arrays (git-ignored; see data/README.md)
archive/            small metadata preserved from the superseded first-generation pipeline
sources.md          dataset provenance and citation requirements
```

---

## Reproducing

Requires an NVIDIA GPU (developed on an RTX 5080, 16 GB) and Python 3.14 environments as
described in `requirements.txt`.

```bash
# 0. obtain CSE-CIC-IDS2018 and build the splits   (see data/README.md)
jupyter nbconvert --execute cicids2018-analysis.ipynb

# 1. port recovery, bucketing, leakage masks       (CPU, ~5 min)
ds-python vae_training/prep_shortcut.py

# 2. capacity x erasure grid                       (GPU, ~1h45m)
bash vae_training/run_grid.sh

# 3. latents, efficiency, MDL/LEACE/detection, figures   (~45 min)
bash vae_training/run_eval_pipeline.sh

# 4. zero-day branch + supervised branch
torch-python vae_training/score_ood.py
ds-python   vae_training/evaluate_hybrid.py

# 5. build and verify the paper
./paper/build.sh
python3 paper/check_pdf.py 6      # page limit, sections, citations, overflow
ds-python paper/audit_claims.py   # numbers vs data, abstract/body consistency
```

### Methodological commitments

These are deliberate and load-bearing; changing them changes what the results mean.

- **Chronological splits.** Each capture file is split 25/25/50 *before* cross-file
  concatenation, nothing is shuffled, and the scaler is fitted on the fit partition only.
- **Leakage purged.** 22.2% of fit rows and 27.8% of test rows are exact duplicates, and
  90,022 unique test vectors (5.76%) also occur in fit. Detection metrics are reported on
  the purged split. Duplicates are masked at evaluation, not deleted, because the sequence
  model needs contiguous windows.
- **Deployment cost counts the classifier too.** The decoder is training-only and never
  ships, so parameter counts are the encoder's — but the *system* is encoder + classifier,
  and a LightGBM head adds ~1,379 KB irrespective of encoder size. Only the 49-parameter
  MLP-8 configuration fits a microcontroller budget; both are reported.
- **Two branches, one encoder.** A supervised head covers attack classes seen in training
  (binary and multiclass); reconstruction error covers classes absent from it. Neither alone
  is a detector — the anomaly branch scores *below chance* on seen attacks, and the
  supervised head near-zero on unseen ones.
- **Erasure is verified, not asserted.** Probing classifiers are unreliable evidence of
  concept removal, so we report MDL codelength across two probe families.

---

## Citation

See `CITATION.cff`. Dataset provenance and the citations required by the Canadian Institute
for Cybersecurity's redistribution terms are recorded in `sources.md`.
