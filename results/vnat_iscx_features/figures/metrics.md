# Per-class metrics: VPN detection, timing features only

Test split of seed 42 (5652 flows), the split behind the confusion matrices in summary.md. Precision, recall, F1 and FPR in %. For each class, that class is the positive one (one-vs-rest).

| model               | class   | support   | TP   | FP   | FN   | TN   | precision   | recall   | f1    | FPR   |
|:--------------------|:--------|:----------|:-----|:-----|:-----|:-----|:------------|:---------|:------|:------|
| Logistic Regression | Non-VPN | 3136      | 2879 | 197  | 257  | 2319 | 93.60       | 91.80    | 92.69 | 7.83  |
| Logistic Regression | VPN     | 2516      | 2319 | 257  | 197  | 2879 | 90.02       | 92.17    | 91.08 | 8.20  |
| Random Forest       | Non-VPN | 3136      | 3114 | 38   | 22   | 2478 | 98.79       | 99.30    | 99.05 | 1.51  |
| Random Forest       | VPN     | 2516      | 2478 | 22   | 38   | 3114 | 99.12       | 98.49    | 98.80 | 0.70  |
| XGBoost             | Non-VPN | 3136      | 3126 | 31   | 10   | 2485 | 99.02       | 99.68    | 99.35 | 1.23  |
| XGBoost             | VPN     | 2516      | 2485 | 10   | 31   | 3126 | 99.60       | 98.77    | 99.18 | 0.32  |
| FT-Transformer      | Non-VPN | 3136      | 3118 | 99   | 18   | 2417 | 96.92       | 99.43    | 98.16 | 3.93  |
| FT-Transformer      | VPN     | 2516      | 2417 | 18   | 99   | 3118 | 99.26       | 96.07    | 97.64 | 0.57  |

Figures: roc_curves.png, pr_curves.png, confusion_matrices.png
