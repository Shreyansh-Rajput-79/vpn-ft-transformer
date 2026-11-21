# FT-Transformer explainability report

Seed-42 test split (same as train.py), 2706 flows. Per-flow explanations: `test_flow_explanations.csv`.

## Top-5 features by method

- FT attention (mean): min_biat, duration, min_fiat, std_flowiat, flowBytesPerSecond
- FT permutation: min_biat, std_flowiat, flowBytesPerSecond, min_flowiat, min_fiat
- RF permutation: min_biat, flowBytesPerSecond, min_flowiat, min_fiat, std_flowiat
- RF impurity: min_biat, flowBytesPerSecond, min_flowiat, flowPktsPerSecond, max_flowiat

## Agreement between methods (Spearman rank correlation over 23 features)

- FT attention vs FT permutation: 0.879
- FT permutation vs RF permutation: 0.750

## Faithfulness (deletion test)

Mean confidence (%) in the original prediction after removing k features per flow. Lower area = removing those features hurts more.

|   features removed |   Most-attended first |   Random order |   Least-attended first |
|-------------------:|----------------------:|---------------:|-----------------------:|
|                  0 |                 92.33 |          92.33 |                  92.33 |
|                  1 |                 63.25 |          88.18 |                  91.87 |
|                  2 |                 56.83 |          84.21 |                  91.46 |
|                  3 |                 53.35 |          80.2  |                  90.98 |
|                  4 |                 50.23 |          76.55 |                  90.24 |
|                  5 |                 49.26 |          73.13 |                  89.52 |
|                  6 |                 47.47 |          70.13 |                  88.58 |
|                  7 |                 46.57 |          67.43 |                  87.96 |
|                  8 |                 46.39 |          65.21 |                  86.93 |
|                  9 |                 47.18 |          63.09 |                  84.38 |
|                 10 |                 45.48 |          61.28 |                  83.13 |

Area under curve (mean confidence, %): Most-attended first: 52.94, Random order: 74.49, Least-attended first: 88.96

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
- `example_flows.png`
