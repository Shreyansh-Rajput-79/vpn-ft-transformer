# Model comparison (vnat_features.csv)

Mean ± std over 5 seeds (70/15/15 split per seed). Values in %.

## Test-set performance

| model               | accuracy     | precision_macro   | recall_macro   | f1_macro     | vpn_precision   | vpn_recall   | roc_auc      |
|:--------------------|:-------------|:------------------|:---------------|:-------------|:----------------|:-------------|:-------------|
| Logistic Regression | 93.53 ± 1.34 | 93.45 ± 1.31      | 93.41 ± 1.28   | 93.41 ± 1.29 | 92.74 ± 2.09    | 94.85 ± 1.92 | 95.28 ± 1.16 |
| Random Forest       | 98.89 ± 0.54 | 98.88 ± 0.52      | 98.86 ± 0.56   | 98.87 ± 0.54 | 98.83 ± 0.92    | 99.05 ± 0.35 | 99.92 ± 0.06 |
| XGBoost             | 99.25 ± 0.37 | 99.25 ± 0.37      | 99.22 ± 0.38   | 99.24 ± 0.37 | 99.30 ± 0.57    | 99.28 ± 0.33 | 99.96 ± 0.03 |
| FT-Transformer      | 98.49 ± 0.80 | 98.51 ± 0.76      | 98.42 ± 0.85   | 98.45 ± 0.81 | 98.59 ± 1.11    | 98.49 ± 1.42 | 99.89 ± 0.06 |

## Is the difference real? (FT-Transformer vs each baseline)

Paired over 84 test capture files: each unit's accuracy is averaged over the seeds in which it was in the test split, and every model is scored on the same units. Positive = FT-Transformer more accurate. Wilcoxon signed-rank test, Holm-corrected for 3 comparisons, significance level 0.05. The CI is a bootstrap over units. Per-unit accuracy weights every capture file equally, so it can differ from the flow-level accuracy in the first table.

| comparison                           |   mean acc. diff per unit (pp) | 95% CI (pp)    | units better / worse / tied   |   p (Wilcoxon) |   p (Holm) | per-seed acc. diff (pp)   | verdict                   |
|:-------------------------------------|-------------------------------:|:---------------|:------------------------------|---------------:|-----------:|:--------------------------|:--------------------------|
| FT-Transformer - Logistic Regression |                           5.2  | [+3.99, +6.49] | 78 / 4 / 2                    |       1.39e-12 |   4.18e-12 | +4.96 ± 1.15              | FT-Transformer better     |
| FT-Transformer - Random Forest       |                          -0.75 | [-1.38, -0.18] | 33 / 25 / 26                  |       0.679    |   0.679    | -0.40 ± 0.41              | no significant difference |
| FT-Transformer - XGBoost             |                          -1.36 | [-2.18, -0.67] | 14 / 32 / 38                  |       0.000523 |   0.00105  | -0.77 ± 0.49              | FT-Transformer worse      |

## Robustness (test accuracy under perturbed features)

`noise_σ`: Gaussian noise added to standardized features. `missing_r`: fraction of feature values replaced by the training mean.

Clean accuracy for reference is in the table above.

| model               | acc_noise_0.1   | acc_noise_0.25   | acc_noise_0.5   | acc_missing_0.1   | acc_missing_0.3   |
|:--------------------|:----------------|:-----------------|:----------------|:------------------|:------------------|
| Logistic Regression | 72.01 ± 2.14    | 62.00 ± 1.38     | 56.89 ± 1.16    | 72.93 ± 1.23      | 58.96 ± 0.79      |
| Random Forest       | 75.47 ± 1.15    | 58.74 ± 4.25     | 51.72 ± 5.64    | 93.78 ± 0.86      | 75.40 ± 3.01      |
| XGBoost             | 77.32 ± 1.25    | 63.98 ± 1.83     | 56.42 ± 2.51    | 93.08 ± 0.68      | 76.31 ± 1.14      |
| FT-Transformer      | 76.50 ± 3.25    | 63.24 ± 2.81     | 56.18 ± 2.28    | 86.67 ± 3.19      | 67.91 ± 6.17      |

## Cost

`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF, XGBoost).

| model               | train_time_s   | inference_us_per_flow   | size                 |
|:--------------------|:---------------|:------------------------|:---------------------|
| Logistic Regression | 0.3 ± 0.0      | 0.2 ± 0.0               | 24.0 ± 0.0           |
| Random Forest       | 1.7 ± 0.1      | 17.9 ± 1.7              | 463,415.6 ± 16,930.8 |
| XGBoost             | 1.5 ± 0.1      | 1.1 ± 0.3               | 29,992.0 ± 694.2     |
| FT-Transformer      | 300.6 ± 98.0   | 10.8 ± 0.6              | 52,450.0 ± 0.0       |

## Confusion matrices (seed 42, rows = true, cols = predicted: ['Non-VPN', 'VPN'])

**Logistic Regression**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      2879 |   257 |
| VPN     |       197 |  2319 |

**Random Forest**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      3114 |    22 |
| VPN     |        38 |  2478 |

**XGBoost**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      3126 |    10 |
| VPN     |        31 |  2485 |

**FT-Transformer**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      3118 |    18 |
| VPN     |        99 |  2417 |
