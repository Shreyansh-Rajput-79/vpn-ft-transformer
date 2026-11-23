# Per-class metrics: VPN detection, NFStream 57 features

Test split of seed 42 (5654 flows), the split behind the confusion matrices in summary.md. Precision, recall, F1 and FPR in %. For each class, that class is the positive one (one-vs-rest).

| model               | class   | support   | TP   | FP   | FN   | TN   | precision   | recall   | f1     | FPR   |
|:--------------------|:--------|:----------|:-----|:-----|:-----|:-----|:------------|:---------|:-------|:------|
| Logistic Regression | Non-VPN | 3138      | 3138 | 0    | 0    | 2516 | 100.00      | 100.00   | 100.00 | 0.00  |
| Logistic Regression | VPN     | 2516      | 2516 | 0    | 0    | 3138 | 100.00      | 100.00   | 100.00 | 0.00  |
| Random Forest       | Non-VPN | 3138      | 3138 | 0    | 0    | 2516 | 100.00      | 100.00   | 100.00 | 0.00  |
| Random Forest       | VPN     | 2516      | 2516 | 0    | 0    | 3138 | 100.00      | 100.00   | 100.00 | 0.00  |
| XGBoost             | Non-VPN | 3138      | 3138 | 0    | 0    | 2516 | 100.00      | 100.00   | 100.00 | 0.00  |
| XGBoost             | VPN     | 2516      | 2516 | 0    | 0    | 3138 | 100.00      | 100.00   | 100.00 | 0.00  |
| FT-Transformer      | Non-VPN | 3138      | 3138 | 0    | 0    | 2516 | 100.00      | 100.00   | 100.00 | 0.00  |
| FT-Transformer      | VPN     | 2516      | 2516 | 0    | 0    | 3138 | 100.00      | 100.00   | 100.00 | 0.00  |

Figures: roc_curves.png, pr_curves.png, confusion_matrices.png
