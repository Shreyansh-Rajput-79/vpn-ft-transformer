# Model comparison (ISCX time-based features)

Mean ± std over 5 seeds (70/15/15 split per seed). Values in %.

## Test-set performance

| model               | accuracy     | precision_macro   | recall_macro   | f1_macro     | vpn_precision   | vpn_recall   | roc_auc      |
|:--------------------|:-------------|:------------------|:---------------|:-------------|:----------------|:-------------|:-------------|
| Logistic Regression | 67.29 ± 0.76 | 67.26 ± 0.76      | 67.17 ± 0.76   | 67.18 ± 0.76 | 67.70 ± 0.76    | 70.62 ± 0.65 | 72.73 ± 0.49 |
| Random Forest       | 92.40 ± 0.57 | 92.44 ± 0.56      | 92.35 ± 0.58   | 92.38 ± 0.57 | 91.82 ± 0.77    | 93.70 ± 0.61 | 97.62 ± 0.30 |
| XGBoost             | 93.32 ± 0.72 | 93.32 ± 0.72      | 93.30 ± 0.72   | 93.31 ± 0.72 | 93.34 ± 0.84    | 93.81 ± 0.80 | 98.40 ± 0.20 |
| FT-Transformer      | 90.64 ± 0.83 | 90.66 ± 0.86      | 90.59 ± 0.82   | 90.62 ± 0.83 | 90.32 ± 0.56    | 91.77 ± 1.23 | 96.90 ± 0.47 |

## Is the difference real? (FT-Transformer vs each baseline)

Paired over 10063 test flows: each unit's accuracy is averaged over the seeds in which it was in the test split, and every model is scored on the same units. Positive = FT-Transformer more accurate. Wilcoxon signed-rank test, Holm-corrected for 3 comparisons, significance level 0.05. The CI is a bootstrap over units. Per-unit accuracy weights every flow equally, so it can differ from the flow-level accuracy in the first table.

| comparison                           | mean acc. diff per unit (pp)   | 95% CI (pp)      | units better / worse / tied   | p (Wilcoxon)   | p (Holm)   | per-seed acc. diff (pp)   | verdict               |
|:-------------------------------------|:-------------------------------|:-----------------|:------------------------------|:---------------|:-----------|:--------------------------|:----------------------|
| FT-Transformer - Logistic Regression | +23.33                         | [+22.35, +24.32] | 2812 / 441 / 6810             | 0              | 0          | +23.34 ± 1.58             | FT-Transformer better |
| FT-Transformer - Random Forest       | -1.78                          | [-2.29, -1.25]   | 353 / 560 / 9150              | 2.05e-10       | 2.05e-10   | -1.77 ± 0.54              | FT-Transformer worse  |
| FT-Transformer - XGBoost             | -2.72                          | [-3.21, -2.25]   | 254 / 574 / 9235              | 3.71e-25       | 7.43e-25   | -2.68 ± 0.50              | FT-Transformer worse  |

## Robustness (test accuracy under perturbed features)

`noise_σ`: Gaussian noise added to standardized features. `missing_r`: fraction of feature values replaced by the training mean.

Clean accuracy for reference is in the table above.

| model               | acc_noise_0.1   | acc_noise_0.25   | acc_noise_0.5   | acc_missing_0.1   | acc_missing_0.3   |
|:--------------------|:----------------|:-----------------|:----------------|:------------------|:------------------|
| Logistic Regression | 61.51 ± 0.84    | 55.99 ± 0.83     | 52.14 ± 1.74    | 57.01 ± 0.98      | 52.67 ± 0.75      |
| Random Forest       | 79.02 ± 0.54    | 68.17 ± 0.92     | 60.65 ± 0.86    | 89.87 ± 0.95      | 81.30 ± 0.78      |
| XGBoost             | 78.54 ± 0.75    | 64.95 ± 0.94     | 57.70 ± 1.69    | 89.28 ± 1.14      | 78.85 ± 0.70      |
| FT-Transformer      | 81.48 ± 0.79    | 67.95 ± 0.89     | 58.37 ± 2.01    | 81.98 ± 0.38      | 67.91 ± 1.41      |

## Cost

`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF, XGBoost).

| model               | train_time_s   | inference_us_per_flow   | size                |
|:--------------------|:---------------|:------------------------|:--------------------|
| Logistic Regression | 0.1 ± 0.0      | 0.2 ± 0.0               | 24.0 ± 0.0          |
| Random Forest       | 0.8 ± 0.1      | 37.7 ± 2.3              | 585,482.8 ± 2,159.8 |
| XGBoost             | 1.5 ± 0.0      | 1.8 ± 0.2               | 46,284.8 ± 707.9    |
| FT-Transformer      | 140.3 ± 18.4   | 10.9 ± 1.4              | 103,682.0 ± 0.0     |

## Confusion matrices (seed 42, rows = true, cols = predicted: ['Non-VPN', 'VPN'])

**Logistic Regression**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |       833 |   470 |
| VPN     |       423 |   980 |

**Random Forest**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      1205 |    98 |
| VPN     |        91 |  1312 |

**XGBoost**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      1230 |    73 |
| VPN     |        82 |  1321 |

**FT-Transformer**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      1166 |   137 |
| VPN     |       101 |  1302 |
