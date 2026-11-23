# Cross-dataset generalization

Trained on `data\vpn_dataset_cleaned.csv`, tested on `data\vnat_features.csv`. Mean ± std over seeds [0, 1, 2], in %.

|                                                 | accuracy     | f1_macro      | roc_auc      |
|:------------------------------------------------|:-------------|:--------------|:-------------|
| ('same dataset (test split)', 'Random Forest')  | 92.14 ± 0.58 | 92.12 ± 0.58  | 97.59 ± 0.36 |
| ('other dataset (all flows)', 'Random Forest')  | 53.15 ± 2.87 | 45.97 ± 4.29  | 57.96 ± 0.27 |
| ('same dataset (test split)', 'XGBoost')        | 92.94 ± 0.59 | 92.93 ± 0.59  | 98.32 ± 0.24 |
| ('other dataset (all flows)', 'XGBoost')        | 51.78 ± 1.29 | 43.31 ± 1.30  | 49.85 ± 2.58 |
| ('same dataset (test split)', 'FT-Transformer') | 90.47 ± 1.09 | 90.45 ± 1.09  | 96.83 ± 0.63 |
| ('other dataset (all flows)', 'FT-Transformer') | 56.16 ± 7.55 | 50.86 ± 11.42 | 65.19 ± 4.29 |

The drop from the first block to the second is the generalization gap.
