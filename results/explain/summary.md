# Explainability report

Seed-42 test split (same as train.py), 2706 flows, 23 features. Per-flow explanations: `test_flow_explanations.csv`.

## Top-5 features by method

- FT attention (mean): min_biat, duration, min_fiat, std_flowiat, flowBytesPerSecond
- FT permutation: min_biat, std_flowiat, flowBytesPerSecond, min_flowiat, min_fiat
- RF TreeSHAP (mean |SHAP|): min_biat, std_flowiat, min_flowiat, max_flowiat, flowBytesPerSecond
- RF permutation: min_biat, flowBytesPerSecond, min_flowiat, min_fiat, std_flowiat
- RF impurity: min_biat, flowBytesPerSecond, min_flowiat, flowPktsPerSecond, max_flowiat

## Agreement between methods (Spearman rank correlation over 23 features)

- FT attention vs FT permutation: 0.879
- RF TreeSHAP vs RF permutation: 0.726
- FT permutation vs RF permutation: 0.750

## Faithfulness (FT-Transformer, all test flows)

Mean confidence (%) in the original prediction after removing k features per flow. Lower = removing those features hurts more.

|   features removed |   Most-attended first |   Random order |   Least-attended first |
|-------------------:|----------------------:|---------------:|-----------------------:|
|                  0 |                 92.33 |          92.33 |                  92.33 |
|                  1 |                 63.25 |          88.14 |                  91.87 |
|                  2 |                 56.83 |          84.05 |                  91.46 |
|                  3 |                 53.35 |          80.17 |                  90.98 |
|                  4 |                 50.23 |          76.69 |                  90.24 |
|                  5 |                 49.26 |          73.51 |                  89.52 |
|                  6 |                 47.47 |          70.64 |                  88.58 |
|                  7 |                 46.57 |          67.66 |                  87.96 |
|                  8 |                 46.39 |          65.31 |                  86.93 |
|                  9 |                 47.18 |          63.22 |                  84.38 |
|                 10 |                 45.48 |          61.51 |                  83.13 |

## FT attention vs RF TreeSHAP (same 500 sampled flows)

Each model is tested with its own explanation. The gap between the random curve and the top-first curve measures how well the explanation identifies what the model relies on. Attention is produced in the same forward pass as the prediction; TreeSHAP is computed afterwards.

| explanation                |   confidence after removing top-1 (%) |   after removing random-1 (%) |   AUC top-first |   AUC random |   faithfulness gap (AUC random - top) |   explanation cost (ms / flow) |
|:---------------------------|--------------------------------------:|------------------------------:|----------------:|-------------:|--------------------------------------:|-------------------------------:|
| FT-Transformer (attention) |                                 62.68 |                         88.37 |           53.07 |        75.67 |                                 22.6  |                          0.058 |
| Random Forest (TreeSHAP)   |                                 70.78 |                         82.66 |           55.89 |        69    |                                 13.12 |                         92.382 |

## Example flows

| true    | prediction   |   confidence | top_features                                             |
|:--------|:-------------|-------------:|:---------------------------------------------------------|
| Non-VPN | Non-VPN      |       0.9992 | min_biat (10%), flowPktsPerSecond (9%), duration (8%)    |
| Non-VPN | Non-VPN      |       0.9805 | duration (10%), min_biat (9%), max_flowiat (9%)          |
| Non-VPN | Non-VPN      |       0.9991 | std_biat (20%), min_fiat (7%), duration (6%)             |
| VPN     | VPN          |       0.9953 | flowBytesPerSecond (9%), min_flowiat (8%), duration (7%) |
| VPN     | VPN          |       0.9936 | std_flowiat (16%), max_fiat (11%), max_flowiat (7%)      |
| VPN     | VPN          |       0.9956 | std_flowiat (15%), max_fiat (11%), max_flowiat (7%)      |
| VPN     | Non-VPN      |       0.998  | mean_biat (23%), max_idle (6%), min_fiat (6%)            |
| Non-VPN | VPN          |       0.9975 | min_biat (10%), std_flowiat (8%), duration (8%)          |

## Figures

- `global_importance.png`
- `attention_by_class.png`
- `deletion_test.png`
- `faithfulness_ft_vs_rf.png`
- `example_flows.png`
