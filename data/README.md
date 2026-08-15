# Data

Everything in this directory is git-ignored: the raw capture is ~6.7 GB and the derived
arrays are larger. This file records how to rebuild it all from scratch.

## 1. Raw capture

CSE-CIC-IDS2018, ten daily CSVs of CICFlowMeter flow features, from the official public
bucket:

    aws s3 sync --no-sign-request \
      "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
      data/cicids2018_raw/

Dataset page: https://www.unb.ca/cic/datasets/ids-2018.html
Citation requirements: see `sources.md` in the repository root.

**Known defects.** Liu et al. (IEEE CNS 2022) document that `DoS-SlowHTTPTest` was never
successfully executed (the tool was launched against port 21), and that `FTP-BruteForce`
and `Infiltration` carry substantial label corruption. Corrected labels are available at
https://github.com/GintsEngelen/CNS2022_Code and are the recommended input.

## 2. Derived arrays

Produced by `cicids2018-analysis.ipynb`:

| File | Contents |
|---|---|
| `features_{fit,val,test}.npy` | 46 signed-log + RobustScaler features, float32 |
| `labels_{fit,val,test}.npy` | integer class ids, see `label_mapping.json` |
| `feature_columns.json` | ordered feature names |
| `label_mapping.json` | class name to id |

Splits are per-file chronological (fit 0-25%, val 25-50%, test 50-100%) applied *before*
cross-file concatenation. Nothing is shuffled. The scaler is fitted on fit only.

## 3. Shortcut artifacts

Produced by `vae_training/prep_shortcut.py` into `data/shortcut/`:

| File | Contents |
|---|---|
| `ports_{split}.npy` | destination port recovered by inverting the scaler |
| `portbucket_{split}.npy` | top-15 ports + "other", the adversary's target |
| `mask_dedup_{split}.npy` | first-occurrence mask for exact duplicate rows |
| `mask_purged_{split}.npy` | dedup mask minus any row also present in fit |
| `meta.json` | recovered scaler parameters, bucket vocabulary |

The scaler was not persisted by the notebook, but the transform is monotonic and
invertible; solving for its two parameters recovers ports with 100% of values within 0.5 of
an integer and a median of port 80.
