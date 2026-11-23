# Model comparison (vnat_nfstream.csv)

Mean ± std over 5 seeds (70/15/15 split per seed). Values in %.

## Test-set performance

| model               | accuracy      | precision_macro   | recall_macro   | f1_macro      | vpn_precision   | vpn_recall    | roc_auc       |
|:--------------------|:--------------|:------------------|:---------------|:--------------|:----------------|:--------------|:--------------|
| Logistic Regression | 99.99 ± 0.02  | 99.99 ± 0.02      | 99.99 ± 0.02   | 99.99 ± 0.02  | 99.99 ± 0.01    | 99.98 ± 0.04  | 100.00 ± 0.00 |
| Random Forest       | 100.00 ± 0.00 | 100.00 ± 0.00     | 100.00 ± 0.00  | 100.00 ± 0.00 | 100.00 ± 0.00   | 100.00 ± 0.00 | 100.00 ± 0.00 |
| XGBoost             | 100.00 ± 0.00 | 100.00 ± 0.00     | 100.00 ± 0.00  | 100.00 ± 0.00 | 100.00 ± 0.00   | 100.00 ± 0.00 | 100.00 ± 0.00 |
| FT-Transformer      | 100.00 ± 0.00 | 100.00 ± 0.00     | 100.00 ± 0.00  | 100.00 ± 0.00 | 100.00 ± 0.00   | 100.00 ± 0.00 | 100.00 ± 0.00 |

## Is the difference real? (FT-Transformer vs each baseline)

Paired over 84 test capture files: each unit's accuracy is averaged over the seeds in which it was in the test split, and every model is scored on the same units. Positive = FT-Transformer more accurate. Wilcoxon signed-rank test, Holm-corrected for 3 comparisons, significance level 0.05. The CI is a bootstrap over units. Per-unit accuracy weights every capture file equally, so it can differ from the flow-level accuracy in the first table.

| comparison                           |   mean acc. diff per unit (pp) | 95% CI (pp)    | units better / worse / tied   |   p (Wilcoxon) |   p (Holm) | per-seed acc. diff (pp)   | verdict                   |
|:-------------------------------------|-------------------------------:|:---------------|:------------------------------|---------------:|-----------:|:--------------------------|:--------------------------|
| FT-Transformer - Logistic Regression |                           0.05 | [+0.00, +0.13] | 2 / 0 / 82                    |           0.18 |      0.539 | +0.01 ± 0.02              | no significant difference |
| FT-Transformer - Random Forest       |                           0    | [+0.00, +0.00] | 0 / 0 / 84                    |           1    |      1     | +0.00 ± 0.00              | no significant difference |
| FT-Transformer - XGBoost             |                           0    | [+0.00, +0.00] | 0 / 0 / 84                    |           1    |      1     | +0.00 ± 0.00              | no significant difference |

## Robustness (test accuracy under perturbed features)

`noise_σ`: Gaussian noise added to standardized features. `missing_r`: fraction of feature values replaced by the training mean.

Clean accuracy for reference is in the table above.

| model               | acc_noise_0.1   | acc_noise_0.25   | acc_noise_0.5   | acc_missing_0.1   | acc_missing_0.3   |
|:--------------------|:----------------|:-----------------|:----------------|:------------------|:------------------|
| Logistic Regression | 99.99 ± 0.02    | 99.96 ± 0.06     | 99.54 ± 0.28    | 99.75 ± 0.16      | 96.85 ± 1.38      |
| Random Forest       | 99.99 ± 0.02    | 98.23 ± 1.52     | 89.26 ± 4.02    | 98.99 ± 0.49      | 84.82 ± 2.92      |
| XGBoost             | 94.30 ± 4.97    | 91.82 ± 4.54     | 84.86 ± 5.02    | 94.86 ± 1.05      | 84.29 ± 2.15      |
| FT-Transformer      | 100.00 ± 0.00   | 99.40 ± 0.77     | 94.19 ± 3.80    | 96.18 ± 2.06      | 84.40 ± 4.46      |

## Cost

`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF, XGBoost).

| model               | train_time_s   | inference_us_per_flow   | size            |
|:--------------------|:---------------|:------------------------|:----------------|
| Logistic Regression | 0.1 ± 0.0      | 0.4 ± 0.1               | 58.0 ± 0.0      |
| Random Forest       | 1.0 ± 0.0      | 14.4 ± 1.4              | 5,722.4 ± 534.5 |
| XGBoost             | 0.5 ± 0.0      | 0.6 ± 0.1               | 686.8 ± 197.4   |
| FT-Transformer      | 200.0 ± 105.3  | 23.9 ± 1.8              | 54,626.0 ± 0.0  |

## Confusion matrices (seed 42, rows = true, cols = predicted: ['Non-VPN', 'VPN'])

**Logistic Regression**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      3138 |     0 |
| VPN     |         0 |  2516 |

**Random Forest**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      3138 |     0 |
| VPN     |         0 |  2516 |

**XGBoost**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      3138 |     0 |
| VPN     |         0 |  2516 |

**FT-Transformer**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      3138 |     0 |
| VPN     |         0 |  2516 |
