# Per-class metrics: 5-category classification, NFStream

Test split of seed 42 (5559 flows), the split behind the confusion matrices in summary.md. Precision, recall, F1 and FPR in %. For each class, that class is the positive one (one-vs-rest).

| model               | class         | support   | TP   | FP   | FN   | TN   | precision   | recall   | f1    | FPR   |
|:--------------------|:--------------|:----------|:-----|:-----|:-----|:-----|:------------|:---------|:------|:------|
| Logistic Regression | C2            | 1359      | 338  | 206  | 1021 | 3994 | 62.13       | 24.87    | 35.52 | 4.90  |
| Logistic Regression | Chat          | 3417      | 3333 | 623  | 84   | 1519 | 84.25       | 97.54    | 90.41 | 29.08 |
| Logistic Regression | File Transfer | 90        | 72   | 415  | 18   | 5054 | 14.78       | 80.00    | 24.96 | 7.59  |
| Logistic Regression | Streaming     | 571       | 390  | 53   | 181  | 4935 | 88.04       | 68.30    | 76.92 | 1.06  |
| Logistic Regression | VoIP          | 122       | 116  | 13   | 6    | 5424 | 89.92       | 95.08    | 92.43 | 0.24  |
| Random Forest       | C2            | 1359      | 480  | 31   | 879  | 4169 | 93.93       | 35.32    | 51.34 | 0.74  |
| Random Forest       | Chat          | 3417      | 3389 | 877  | 28   | 1265 | 79.44       | 99.18    | 88.22 | 40.94 |
| Random Forest       | File Transfer | 90        | 76   | 8    | 14   | 5461 | 90.48       | 84.44    | 87.36 | 0.15  |
| Random Forest       | Streaming     | 571       | 562  | 16   | 9    | 4972 | 97.23       | 98.42    | 97.82 | 0.32  |
| Random Forest       | VoIP          | 122       | 120  | 0    | 2    | 5437 | 100.00      | 98.36    | 99.17 | 0.00  |
| XGBoost             | C2            | 1359      | 166  | 30   | 1193 | 4170 | 84.69       | 12.21    | 21.35 | 0.71  |
| XGBoost             | Chat          | 3417      | 3391 | 869  | 26   | 1273 | 79.60       | 99.24    | 88.34 | 40.57 |
| XGBoost             | File Transfer | 90        | 71   | 13   | 19   | 5456 | 84.52       | 78.89    | 81.61 | 0.24  |
| XGBoost             | Streaming     | 571       | 561  | 298  | 10   | 4690 | 65.31       | 98.25    | 78.46 | 5.97  |
| XGBoost             | VoIP          | 122       | 120  | 40   | 2    | 5397 | 75.00       | 98.36    | 85.11 | 0.74  |
| FT-Transformer      | C2            | 1359      | 1258 | 53   | 101  | 4147 | 95.96       | 92.57    | 94.23 | 1.26  |
| FT-Transformer      | Chat          | 3417      | 3387 | 90   | 30   | 2052 | 97.41       | 99.12    | 98.26 | 4.20  |
| FT-Transformer      | File Transfer | 90        | 62   | 2    | 28   | 5467 | 96.88       | 68.89    | 80.52 | 0.04  |
| FT-Transformer      | Streaming     | 571       | 550  | 25   | 21   | 4963 | 95.65       | 96.32    | 95.99 | 0.50  |
| FT-Transformer      | VoIP          | 122       | 121  | 11   | 1    | 5426 | 91.67       | 99.18    | 95.28 | 0.20  |

Figures: roc_curves.png, pr_curves.png, confusion_matrices.png
