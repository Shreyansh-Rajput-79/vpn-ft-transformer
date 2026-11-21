# Model comparison: VPN vs Non-VPN (ISCX time-based features)

Mean ± std over 5 seeds (stratified 70/15/15 split per seed). Values in %.

## Test-set performance

| model               | accuracy     | precision_macro   | recall_macro   | f1_macro     | vpn_precision   | vpn_recall   | roc_auc      |
|:--------------------|:-------------|:------------------|:---------------|:-------------|:----------------|:-------------|:-------------|
| Logistic Regression | 67.32 ± 0.76 | 67.28 ± 0.76      | 67.19 ± 0.77   | 67.20 ± 0.77 | 67.72 ± 0.76    | 70.65 ± 0.64 | 72.73 ± 0.49 |
| Random Forest       | 92.40 ± 0.57 | 92.44 ± 0.56      | 92.35 ± 0.58   | 92.38 ± 0.57 | 91.82 ± 0.77    | 93.70 ± 0.61 | 97.62 ± 0.30 |
| FT-Transformer      | 90.83 ± 0.31 | 90.94 ± 0.35      | 90.75 ± 0.28   | 90.80 ± 0.30 | 89.88 ± 1.16    | 92.79 ± 1.79 | 97.02 ± 0.19 |

## Robustness (test accuracy under perturbed features)

`noise_σ`: Gaussian noise added to standardized features. `missing_r`: fraction of feature values replaced by the training mean.

Clean accuracy for reference is in the table above.

| model               | acc_noise_0.1   | acc_noise_0.25   | acc_noise_0.5   | acc_missing_0.1   | acc_missing_0.3   |
|:--------------------|:----------------|:-----------------|:----------------|:------------------|:------------------|
| Logistic Regression | 61.50 ± 0.84    | 56.00 ± 0.83     | 52.15 ± 1.74    | 57.01 ± 0.97      | 52.68 ± 0.76      |
| Random Forest       | 79.02 ± 0.54    | 68.17 ± 0.92     | 60.65 ± 0.86    | 89.87 ± 0.95      | 81.30 ± 0.78      |
| FT-Transformer      | 82.19 ± 0.49    | 67.71 ± 0.86     | 58.00 ± 1.08    | 82.72 ± 0.36      | 68.10 ± 0.98      |

## Cost

`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF).

| model               | train_time_s   | inference_us_per_flow   | size                |
|:--------------------|:---------------|:------------------------|:--------------------|
| Logistic Regression | 0.1 ± 0.0      | 0.2 ± 0.0               | 24.0 ± 0.0          |
| Random Forest       | 0.8 ± 0.0      | 21.7 ± 2.9              | 585,482.8 ± 2,159.8 |
| FT-Transformer      | 352.3 ± 7.4    | 52.4 ± 2.7              | 103,682.0 ± 0.0     |

## Confusion matrices (seed 42, rows = true, cols = predicted: ['Non-VPN', 'VPN'])

**Logistic Regression**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |       833 |   470 |
| VPN     |       422 |   981 |

**Random Forest**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      1205 |    98 |
| VPN     |        91 |  1312 |

**FT-Transformer**

|         |   Non-VPN |   VPN |
|:--------|----------:|------:|
| Non-VPN |      1143 |   160 |
| VPN     |        97 |  1306 |
