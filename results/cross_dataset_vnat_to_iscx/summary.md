# Cross-dataset generalization

Trained on `data\vnat_features.csv`, tested on `data\vpn_dataset_cleaned.csv`. Mean ± std over seeds [0, 1, 2], in %.

|                                                 | accuracy     | f1_macro     | roc_auc      |
|:------------------------------------------------|:-------------|:-------------|:-------------|
| ('same dataset (test split)', 'Random Forest')  | 99.15 ± 0.06 | 99.15 ± 0.06 | 99.94 ± 0.02 |
| ('other dataset (all flows)', 'Random Forest')  | 55.91 ± 0.92 | 52.52 ± 0.83 | 54.56 ± 0.66 |
| ('same dataset (test split)', 'XGBoost')        | 99.45 ± 0.10 | 99.45 ± 0.10 | 99.98 ± 0.00 |
| ('other dataset (all flows)', 'XGBoost')        | 54.84 ± 0.81 | 52.36 ± 1.05 | 54.28 ± 0.79 |
| ('same dataset (test split)', 'FT-Transformer') | 98.95 ± 0.30 | 98.95 ± 0.30 | 99.93 ± 0.05 |
| ('other dataset (all flows)', 'FT-Transformer') | 50.53 ± 2.26 | 47.48 ± 0.62 | 51.70 ± 5.46 |

The drop from the first block to the second is the generalization gap.
