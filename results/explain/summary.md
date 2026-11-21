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
| FT-Transformer (attention) |                                 62.68 |                         88.37 |           53.07 |        75.67 |                                 22.6  |                          0.139 |
| Random Forest (TreeSHAP)   |                                 70.78 |                         82.66 |           55.89 |        69    |                                 13.12 |                        146.176 |

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

### The same flows in plain language

Features named = the model's own top-3 attention (see deletion test above); the comparison with typical values per class comes from the training split.

- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the shortest gap between server->client packets was 1 us, lower than usual for either class, on the Non-VPN side (Non-VPN median 10 us, VPN median 30 us); (2) the packet rate was 102 packets/s, higher than usual for either class, on the Non-VPN side (Non-VPN median 40.3 packets/s, VPN median 8.81 packets/s); (3) the flow duration was 15 s, typical of Non-VPN traffic (Non-VPN median 13.4 s, VPN median 10.1 s).
- *True Non-VPN:* Classified as Non-VPN (98% confident). Main reasons: (1) the flow duration was 255 us, lower than usual for either class, on the VPN side (Non-VPN median 13.4 s, VPN median 10.1 s); (2) the shortest gap between server->client packets could not be measured (too few packets); (3) the longest gap between packets was 255 us, lower than usual for either class, on the Non-VPN side (Non-VPN median 173 ms, VPN median 922 ms).
- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the variation in the gap between server->client packets was 3.14 ms, lower than usual for either class, on the Non-VPN side (Non-VPN median 6.75 ms, VPN median 38.5 ms); (2) the shortest gap between client->server packets was 15 ms, higher than usual for either class, on the Non-VPN side (Non-VPN median 84 us, VPN median 55 us); (3) the flow duration was 15 s, typical of Non-VPN traffic (Non-VPN median 13.4 s, VPN median 10.1 s).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the data rate was 21 bytes/s, lower than usual for either class, on the VPN side (VPN median 1.58 kB/s, Non-VPN median 10.3 kB/s); (2) the shortest gap between packets was 38.3 ms, higher than usual for either class, on the VPN side (VPN median 20 us, Non-VPN median 13 us); (3) the flow duration was 10.1 s, typical of VPN traffic (VPN median 10.1 s, Non-VPN median 13.4 s).
- *True VPN:* Classified as VPN (99% confident). Main reasons: (1) the variation in the gap between packets was 6.58 ms, lower than usual for either class, on the Non-VPN side (VPN median 200 ms, Non-VPN median 14.1 ms); (2) the longest gap between client->server packets was 30.7 ms, lower than usual for either class, on the Non-VPN side (VPN median 961 ms, Non-VPN median 92.1 ms); (3) the longest gap between packets was 27.2 ms, lower than usual for either class, on the Non-VPN side (VPN median 922 ms, Non-VPN median 173 ms). None of these values on its own is typical of VPN; the decision comes from how the features combine.
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the variation in the gap between packets was 6.09 ms, lower than usual for either class, on the Non-VPN side (VPN median 200 ms, Non-VPN median 14.1 ms); (2) the longest gap between client->server packets was 33.3 ms, lower than usual for either class, on the Non-VPN side (VPN median 961 ms, Non-VPN median 92.1 ms); (3) the longest gap between packets was 23 ms, lower than usual for either class, on the Non-VPN side (VPN median 922 ms, Non-VPN median 173 ms). None of these values on its own is typical of VPN; the decision comes from how the features combine.
- *True VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the average gap between server->client packets was 33.3 ms, closer to Non-VPN than VPN traffic (Non-VPN median 20 ms, VPN median 168 ms); (2) the longest idle period could not be measured (too few packets); (3) the shortest gap between client->server packets was 11 us, lower than usual for either class, on the VPN side (Non-VPN median 84 us, VPN median 55 us).
- *True Non-VPN:* Classified as VPN (100% confident). Main reasons: (1) the shortest gap between server->client packets was 0 us, lower than usual for either class, on the Non-VPN side (VPN median 30 us, Non-VPN median 10 us); (2) the variation in the gap between packets was 90.3 ms, closer to VPN than Non-VPN traffic (VPN median 200 ms, Non-VPN median 14.1 ms); (3) the flow duration was 15 s, typical of Non-VPN traffic (VPN median 10.1 s, Non-VPN median 13.4 s).

## Figures

- `global_importance.png`
- `attention_by_class.png`
- `deletion_test.png`
- `faithfulness_ft_vs_rf.png`
- `example_flows.png`
