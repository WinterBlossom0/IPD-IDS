# Research Figures

Generated from:

- `neural_ood_binary_metrics.csv`
- `neural_multiclass_metrics.csv`

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_research_figures.ps1
```

Figures:

The SVGs are sized for single-column placement in a two-column paper
(`3.45in x 2.35in`). Abbreviations:

- `CL`: primary compressed + losses
- `C`: primary compressed only
- `ACF`: primary all compressed features
- `-T`: no time adversary
- `-S`: no student loss

- `ood_ablation_metrics.svg`: final OOD attack recall and macro-F1 across feature and VAE ablations.
- `multiclass_ablation_metrics.svg`: in-distribution hierarchical-F1 and macro-F1 across the same ablations.
- `ood_precision_recall.svg`: attack recall vs. false-positive-rate operating point comparison.
- `ablation_delta_vs_primary.svg`: attack-recall and macro-F1 retained relative to `primary__compressed_plus_losses`.
- `ablation_summary.csv`: compact table of paper-ready ablation metrics and deltas.
