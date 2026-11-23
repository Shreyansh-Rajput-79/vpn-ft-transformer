# Explainability report

Seed-42 test split (same as train.py), 5652 flows, 23 features. Per-flow explanations: `test_flow_explanations.csv`.

## Top-5 features by method

- FT attention (mean): flowBytesPerSecond, mean_flowiat, flowPktsPerSecond, min_active, mean_fiat
- FT permutation: flowBytesPerSecond, mean_flowiat, mean_fiat, min_flowiat, mean_biat
- RF TreeSHAP (mean |SHAP|): flowBytesPerSecond, min_flowiat, mean_flowiat, duration, min_active
- RF permutation: flowBytesPerSecond, min_active, min_biat, mean_flowiat, flowPktsPerSecond
- RF impurity: flowBytesPerSecond, min_flowiat, duration, mean_flowiat, mean_fiat

## Agreement between methods (Spearman rank correlation over 23 features)

- FT attention vs FT permutation: 0.699
- RF TreeSHAP vs RF permutation: 0.704
- FT permutation vs RF permutation: 0.700

## Faithfulness (FT-Transformer, all test flows)

Mean confidence (%) in the original prediction after removing k features per flow. Lower = removing those features hurts more.

|   features removed |   Most-attended first |   Random order |   Least-attended first |
|-------------------:|----------------------:|---------------:|-----------------------:|
|                  0 |                 99    |          99    |                  99    |
|                  1 |                 53.85 |          92.96 |                  98.96 |
|                  2 |                 53.38 |          86.83 |                  98.79 |
|                  3 |                 54.62 |          81.41 |                  98.37 |
|                  4 |                 63.99 |          76.25 |                  98.07 |
|                  5 |                 71.47 |          71.76 |                  97.73 |
|                  6 |                 70.05 |          68.11 |                  97.19 |
|                  7 |                 72.54 |          64.84 |                  96.43 |
|                  8 |                 71.19 |          62.31 |                  94.38 |
|                  9 |                 68.42 |          60.28 |                  90.03 |
|                 10 |                 66.36 |          58.56 |                  86.02 |

## FT attention vs RF TreeSHAP (same 500 sampled flows)

Each model is tested with its own explanation. The gap between the random curve and the top-first curve measures how well the explanation identifies what the model relies on. Attention is produced in the same forward pass as the prediction; TreeSHAP is computed afterwards.

| explanation                |   confidence after removing top-1 (%) |   after removing random-1 (%) |   AUC top-first |   AUC random |   faithfulness gap (AUC random - top) |   explanation cost (ms / flow) |
|:---------------------------|--------------------------------------:|------------------------------:|----------------:|-------------:|--------------------------------------:|-------------------------------:|
| FT-Transformer (attention) |                                 54.04 |                         93.14 |           65.74 |        73.69 |                                  7.95 |                          0.09  |
| Random Forest (TreeSHAP)   |                                 62.6  |                         87.05 |           50.45 |        70.07 |                                 19.61 |                        127.111 |

## Example flows

| true    | prediction   |   confidence | top_features                                                       |
|:--------|:-------------|-------------:|:-------------------------------------------------------------------|
| Non-VPN | Non-VPN      |       0.9985 | flowBytesPerSecond (7%), mean_flowiat (6%), flowPktsPerSecond (5%) |
| Non-VPN | Non-VPN      |       0.9982 | flowBytesPerSecond (7%), mean_flowiat (6%), flowPktsPerSecond (6%) |
| Non-VPN | Non-VPN      |       0.9984 | flowBytesPerSecond (6%), std_idle (6%), std_fiat (5%)              |
| VPN     | VPN          |       0.999  | flowBytesPerSecond (6%), mean_flowiat (6%), flowPktsPerSecond (6%) |
| VPN     | VPN          |       0.9987 | flowBytesPerSecond (7%), mean_flowiat (6%), max_idle (5%)          |
| VPN     | VPN          |       0.9986 | flowBytesPerSecond (6%), mean_flowiat (6%), flowPktsPerSecond (5%) |
| VPN     | Non-VPN      |       0.9978 | flowBytesPerSecond (6%), std_idle (6%), std_fiat (5%)              |
| Non-VPN | VPN          |       0.9976 | flowBytesPerSecond (7%), mean_flowiat (6%), std_biat (5%)          |

### The same flows in plain language

Features named = the model's own top-3 attention (see deletion test above); the comparison with typical values per class comes from the training split.

- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the data rate was 277 bytes/s, typical of Non-VPN traffic (Non-VPN median 420 bytes/s, VPN median 458 bytes/s); (2) the average gap between packets was 679 ms, typical of VPN traffic (Non-VPN median 522 ms, VPN median 652 ms); (3) the packet rate was 1.56 packets/s, typical of VPN traffic (Non-VPN median 1.98 packets/s, VPN median 1.61 packets/s).
- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the data rate was 321 bytes/s, typical of Non-VPN traffic (Non-VPN median 420 bytes/s, VPN median 458 bytes/s); (2) the average gap between packets was 714 ms, typical of VPN traffic (Non-VPN median 522 ms, VPN median 652 ms); (3) the packet rate was 1.47 packets/s, typical of VPN traffic (Non-VPN median 1.98 packets/s, VPN median 1.61 packets/s).
- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the data rate was 189 kB/s, higher than usual for either class, on the VPN side (Non-VPN median 420 bytes/s, VPN median 458 bytes/s); (2) the variation in the idle period was 0 us, lower than usual for either class, on the Non-VPN side (Non-VPN median 133 ms, VPN median 699 ms); (3) the variation in the gap between client->server packets was 0 us, lower than usual for either class, on the Non-VPN side (Non-VPN median 1.25 s, VPN median 1.29 s).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the data rate was 451 bytes/s, typical of VPN traffic (VPN median 458 bytes/s, Non-VPN median 420 bytes/s); (2) the average gap between packets was 633 ms, typical of VPN traffic (VPN median 652 ms, Non-VPN median 522 ms); (3) the packet rate was 1.67 packets/s, typical of VPN traffic (VPN median 1.61 packets/s, Non-VPN median 1.98 packets/s).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the data rate was 374 bytes/s, typical of Non-VPN traffic (VPN median 458 bytes/s, Non-VPN median 420 bytes/s); (2) the average gap between packets was 789 ms, typical of VPN traffic (VPN median 652 ms, Non-VPN median 522 ms); (3) the longest idle period was 3.69 s, typical of Non-VPN traffic (VPN median 4.34 s, Non-VPN median 3.24 s).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the data rate was 511 bytes/s, typical of VPN traffic (VPN median 458 bytes/s, Non-VPN median 420 bytes/s); (2) the average gap between packets was 625 ms, typical of VPN traffic (VPN median 652 ms, Non-VPN median 522 ms); (3) the packet rate was 1.67 packets/s, typical of VPN traffic (VPN median 1.61 packets/s, Non-VPN median 1.98 packets/s).
- *True VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the data rate was 25.2 MB/s, higher than usual for either class, on the VPN side (Non-VPN median 420 bytes/s, VPN median 458 bytes/s); (2) the variation in the idle period was 0 us, lower than usual for either class, on the Non-VPN side (Non-VPN median 133 ms, VPN median 699 ms); (3) the variation in the gap between client->server packets was 175 us, lower than usual for either class, on the Non-VPN side (Non-VPN median 1.25 s, VPN median 1.29 s).
- *True Non-VPN:* Classified as VPN (100% confident). Main reasons: (1) the data rate was 248 bytes/s, lower than usual for either class, on the Non-VPN side (VPN median 458 bytes/s, Non-VPN median 420 bytes/s); (2) the average gap between packets was 1.26 s, higher than usual for either class, on the VPN side (VPN median 652 ms, Non-VPN median 522 ms); (3) the variation in the gap between server->client packets was 5.27 ms, lower than usual for either class, on the Non-VPN side (VPN median 1.28 s, Non-VPN median 1.25 s).

## Figures

- `global_importance.png`
- `attention_by_class.png`
- `deletion_test.png`
- `faithfulness_ft_vs_rf.png`
- `example_flows.png`
