# Model comparison (vnat_nfstream.csv, features: nfstream_noflags)

Mean ± std over 5 seeds (70/15/15 split per seed). Values in %.

## Test-set performance

| model               | accuracy      | precision_macro   | recall_macro   | f1_macro      | vpn_precision   | vpn_recall    | roc_auc       |
|:--------------------|:--------------|:------------------|:---------------|:--------------|:----------------|:--------------|:--------------|
| Logistic Regression | 99.99 ± 0.03  | 99.99 ± 0.03      | 99.99 ± 0.03   | 99.99 ± 0.03  | 99.99 ± 0.01    | 99.98 ± 0.04  | 100.00 ± 0.00 |
| Random Forest       | 100.00 ± 0.00 | 100.00 ± 0.00     | 100.00 ± 0.00  | 100.00 ± 0.00 | 100.00 ± 0.00   | 100.00 ± 0.00 | 100.00 ± 0.00 |
| XGBoost             | 100.00 ± 0.00 | 100.00 ± 0.00     | 100.00 ± 0.00  | 100.00 ± 0.00 | 100.00 ± 0.00   | 100.00 ± 0.00 | 100.00 ± 0.00 |
| FT-Transformer      | 100.00 ± 0.00 | 100.00 ± 0.00     | 100.00 ± 0.00  | 100.00 ± 0.00 | 100.00 ± 0.00   | 100.00 ± 0.00 | 100.00 ± 0.00 |

## Is the difference real? (FT-Transformer vs each baseline)

Paired over 84 test capture files: each unit's accuracy is averaged over the seeds in which it was in the test split, and every model is scored on the same units. Positive = FT-Transformer more accurate. Wilcoxon signed-rank test, Holm-corrected for 3 comparisons, significance level 0.05. The CI is a bootstrap over units. Per-unit accuracy weights every capture file equally, so it can differ from the flow-level accuracy in the first table.

| comparison                           | mean acc. diff per unit (pp)   | 95% CI (pp)    | units better / worse / tied   | p (Wilcoxon)   | p (Holm)   | per-seed acc. diff (pp)   | verdict                   |
|:-------------------------------------|:-------------------------------|:---------------|:------------------------------|:---------------|:-----------|:--------------------------|:--------------------------|
| FT-Transformer - Logistic Regression | +0.04                          | [+0.00, +0.13] | 2 / 0 / 82                    | 0.18           | 0.539      | +0.01 ± 0.03              | no significant difference |
| FT-Transformer - Random Forest       | +0.00                          | [+0.00, +0.00] | 0 / 0 / 84                    | 1              | 1          | +0.00 ± 0.00              | no significant difference |
| FT-Transformer - XGBoost             | +0.00                          | [+0.00, +0.00] | 0 / 0 / 84                    | 1              | 1          | +0.00 ± 0.00              | no significant difference |

## Robustness (test accuracy under perturbed features)

`noise_σ`: Gaussian noise added to standardized features. `missing_r`: fraction of feature values replaced by the training mean.

Clean accuracy for reference is in the table above.

| model               | acc_noise_0.1   | acc_noise_0.25   | acc_noise_0.5   | acc_missing_0.1   | acc_missing_0.3   |
|:--------------------|:----------------|:-----------------|:----------------|:------------------|:------------------|
| Logistic Regression | 99.99 ± 0.01    | 99.95 ± 0.03     | 99.00 ± 0.21    | 99.40 ± 0.15      | 95.02 ± 0.79      |
| Random Forest       | 99.93 ± 0.11    | 98.13 ± 1.85     | 94.17 ± 3.33    | 98.36 ± 1.04      | 88.92 ± 3.77      |
| XGBoost             | 94.18 ± 4.97    | 91.55 ± 4.88     | 85.04 ± 4.44    | 94.71 ± 0.76      | 84.20 ± 2.08      |
| FT-Transformer      | 99.98 ± 0.04    | 99.33 ± 0.86     | 95.62 ± 2.87    | 95.56 ± 2.24      | 85.21 ± 5.85      |

## Cost

`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF, XGBoost).

| model               | train_time_s   | inference_us_per_flow   | size            |
|:--------------------|:---------------|:------------------------|:----------------|
| Logistic Regression | 0.1 ± 0.0      | 0.3 ± 0.1               | 34.0 ± 0.0      |
| Random Forest       | 1.1 ± 0.1      | 15.5 ± 2.5              | 8,848.8 ± 659.9 |
| XGBoost             | 0.5 ± 0.1      | 0.6 ± 0.1               | 686.8 ± 197.4   |
| FT-Transformer      | 190.7 ± 68.7   | 14.5 ± 0.8              | 53,090.0 ± 0.0  |

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
