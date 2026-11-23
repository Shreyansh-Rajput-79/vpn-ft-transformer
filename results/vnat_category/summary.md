# Model comparison (vnat_nfstream.csv, label: category)

Mean ± std over 5 seeds (70/15/15 split per seed). Values in %.

## Test-set performance

| model               | accuracy      | precision_macro   | recall_macro   | f1_macro      | roc_auc      |
|:--------------------|:--------------|:------------------|:---------------|:--------------|:-------------|
| Logistic Regression | 84.15 ± 10.31 | 73.13 ± 15.73     | 76.04 ± 6.64   | 71.10 ± 13.25 | 96.14 ± 2.67 |
| Random Forest       | 93.30 ± 5.94  | 90.24 ± 5.22      | 88.55 ± 4.39   | 87.90 ± 4.75  | 98.94 ± 0.55 |
| XGBoost             | 91.88 ± 8.35  | 87.76 ± 7.51      | 87.04 ± 6.40   | 85.05 ± 9.18  | 99.58 ± 0.24 |
| FT-Transformer      | 92.79 ± 5.83  | 85.47 ± 9.79      | 88.16 ± 4.76   | 85.18 ± 8.89  | 99.19 ± 0.44 |

## Is the difference real? (FT-Transformer vs each baseline)

Paired over 87 test capture files: each unit's accuracy is averaged over the seeds in which it was in the test split, and every model is scored on the same units. Positive = FT-Transformer more accurate. Wilcoxon signed-rank test, Holm-corrected for 3 comparisons, significance level 0.05. The CI is a bootstrap over units. Per-unit accuracy weights every capture file equally, so it can differ from the flow-level accuracy in the first table.

| comparison                           | mean acc. diff per unit (pp)   | 95% CI (pp)     | units better / worse / tied   | p (Wilcoxon)   | p (Holm)   | per-seed acc. diff (pp)   | verdict                   |
|:-------------------------------------|:-------------------------------|:----------------|:------------------------------|:---------------|:-----------|:--------------------------|:--------------------------|
| FT-Transformer - Logistic Regression | +6.92                          | [+4.30, +10.05] | 62 / 5 / 20                   | 2.09e-09       | 6.26e-09   | +8.64 ± 8.09              | FT-Transformer better     |
| FT-Transformer - Random Forest       | -0.25                          | [-2.09, +1.79]  | 13 / 25 / 49                  | 0.117          | 0.235      | -0.52 ± 8.79              | no significant difference |
| FT-Transformer - XGBoost             | +0.00                          | [-2.13, +2.37]  | 10 / 23 / 54                  | 0.183          | 0.235      | +0.91 ± 10.95             | no significant difference |

## Robustness (test accuracy under perturbed features)

`noise_σ`: Gaussian noise added to standardized features. `missing_r`: fraction of feature values replaced by the training mean.

Clean accuracy for reference is in the table above.

| model               | acc_noise_0.1   | acc_noise_0.25   | acc_noise_0.5   | acc_missing_0.1   | acc_missing_0.3   |
|:--------------------|:----------------|:-----------------|:----------------|:------------------|:------------------|
| Logistic Regression | 81.33 ± 7.98    | 61.81 ± 3.59     | 43.19 ± 1.51    | 68.72 ± 5.30      | 52.37 ± 2.74      |
| Random Forest       | 91.14 ± 6.29    | 88.28 ± 5.43     | 77.91 ± 7.37    | 92.34 ± 6.31      | 88.19 ± 5.61      |
| XGBoost             | 88.49 ± 5.16    | 81.45 ± 4.83     | 64.30 ± 5.79    | 91.22 ± 8.25      | 86.75 ± 7.41      |
| FT-Transformer      | 91.38 ± 5.25    | 85.63 ± 4.93     | 68.60 ± 6.83    | 87.55 ± 4.01      | 77.42 ± 6.95      |

## Cost

`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF, XGBoost).

| model               | train_time_s   | inference_us_per_flow   | size                 |
|:--------------------|:---------------|:------------------------|:---------------------|
| Logistic Regression | 4.0 ± 0.3      | 0.5 ± 0.0               | 290.0 ± 0.0          |
| Random Forest       | 1.5 ± 0.1      | 18.4 ± 2.8              | 497,725.2 ± 41,925.5 |
| XGBoost             | 8.9 ± 0.4      | 3.4 ± 1.3               | 104,461.6 ± 5,753.1  |
| FT-Transformer      | 254.3 ± 84.2   | 24.0 ± 2.7              | 54,725.0 ± 0.0       |

## Confusion matrices (seed 42, rows = true, cols = predicted: ['C2', 'Chat', 'File Transfer', 'Streaming', 'VoIP'])

**Logistic Regression**

|               |   C2 |   Chat |   File Transfer |   Streaming |   VoIP |
|:--------------|-----:|-------:|----------------:|------------:|-------:|
| C2            |  338 |    608 |             401 |           1 |     11 |
| Chat          |   40 |   3333 |               0 |          43 |      1 |
| File Transfer |    9 |      0 |              72 |           9 |      0 |
| Streaming     |  152 |     15 |              13 |         390 |      1 |
| VoIP          |    5 |      0 |               1 |           0 |    116 |

**Random Forest**

|               |   C2 |   Chat |   File Transfer |   Streaming |   VoIP |
|:--------------|-----:|-------:|----------------:|------------:|-------:|
| C2            |  480 |    872 |               3 |           4 |      0 |
| Chat          |   26 |   3389 |               1 |           1 |      0 |
| File Transfer |    3 |      0 |              76 |          11 |      0 |
| Streaming     |    1 |      4 |               4 |         562 |      0 |
| VoIP          |    1 |      1 |               0 |           0 |    120 |

**XGBoost**

|               |   C2 |   Chat |   File Transfer |   Streaming |   VoIP |
|:--------------|-----:|-------:|----------------:|------------:|-------:|
| C2            |  166 |    864 |               8 |         281 |     40 |
| Chat          |   25 |   3391 |               0 |           1 |      0 |
| File Transfer |    3 |      0 |              71 |          16 |      0 |
| Streaming     |    1 |      4 |               5 |         561 |      0 |
| VoIP          |    1 |      1 |               0 |           0 |    120 |

**FT-Transformer**

|               |   C2 |   Chat |   File Transfer |   Streaming |   VoIP |
|:--------------|-----:|-------:|----------------:|------------:|-------:|
| C2            | 1258 |     85 |               1 |           4 |     11 |
| Chat          |   28 |   3387 |               0 |           2 |      0 |
| File Transfer |    9 |      0 |              62 |          19 |      0 |
| Streaming     |   16 |      4 |               1 |         550 |      0 |
| VoIP          |    0 |      1 |               0 |           0 |    121 |
