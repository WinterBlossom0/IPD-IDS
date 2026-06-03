# Research Figures

Generated from:

- `neural_ood_binary_metrics.csv`
- `neural_multiclass_metrics.csv`

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_research_figures.ps1
```

Figures:

- `ood_ablation_metrics.svg`: final OOD attack detection metrics across feature and VAE ablations.
- `multiclass_ablation_metrics.svg`: in-distribution multiclass performance across the same ablations.
- `ood_precision_recall.svg`: attack and benign precision/recall operating point comparison.
- `ablation_delta_vs_primary.svg`: metric deltas relative to `primary__compressed_plus_losses`.
- `ablation_summary.csv`: compact table of paper-ready ablation metrics and deltas.
