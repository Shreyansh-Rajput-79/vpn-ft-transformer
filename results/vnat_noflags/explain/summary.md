# Explainability report

Seed-42 test split (same as train.py), 5654 flows, 33 features. Per-flow explanations: `test_flow_explanations.csv`.

## Top-5 features by method

- FT attention (mean): src2dst_min_ps, bidirectional_min_ps, bidirectional_max_piat_ms, bidirectional_stddev_ps, dst2src_min_ps
- FT permutation: bidirectional_min_ps, src2dst_min_ps, bidirectional_duration_ms, bidirectional_packets, bidirectional_bytes
- RF TreeSHAP (mean |SHAP|): bidirectional_min_ps, src2dst_min_ps, dst2src_min_ps, dst2src_mean_ps, src2dst_mean_ps
- RF permutation: bidirectional_min_ps, src2dst_min_ps, bidirectional_duration_ms, bidirectional_packets, bidirectional_bytes
- RF impurity: bidirectional_min_ps, src2dst_min_ps, dst2src_min_ps, src2dst_mean_ps, dst2src_mean_ps

## Agreement between methods (Spearman rank correlation over 33 features)

- FT attention vs FT permutation: 0.413
- RF TreeSHAP vs RF permutation: 0.414
- FT permutation vs RF permutation: 1.000

## Faithfulness (FT-Transformer, all test flows)

Mean confidence (%) in the original prediction after removing k features per flow. Lower = removing those features hurts more.

|   features removed |   Most-attended first |   Random order |   Least-attended first |
|-------------------:|----------------------:|---------------:|-----------------------:|
|                  0 |                 99.99 |          99.99 |                  99.99 |
|                  1 |                 99.98 |          99.98 |                  99.99 |
|                  2 |                 99.93 |          99.89 |                  99.99 |
|                  3 |                 99.89 |          99.66 |                  99.99 |
|                  4 |                 99.88 |          99.34 |                  99.99 |
|                  5 |                 99.81 |          98.95 |                  99.99 |
|                  6 |                 99.76 |          98.4  |                  99.99 |
|                  7 |                 97.99 |          97.74 |                  99.99 |
|                  8 |                 99.3  |          96.96 |                  99.98 |
|                  9 |                 99.24 |          96.1  |                  99.98 |
|                 10 |                 99.18 |          95.18 |                  99.98 |

## FT attention vs RF TreeSHAP (same 500 sampled flows)

Each model is tested with its own explanation. The gap between the random curve and the top-first curve measures how well the explanation identifies what the model relies on. Attention is produced in the same forward pass as the prediction; TreeSHAP is computed afterwards.

| explanation                |   confidence after removing top-1 (%) |   after removing random-1 (%) |   AUC top-first |   AUC random |   faithfulness gap (AUC random - top) |   explanation cost (ms / flow) |
|:---------------------------|--------------------------------------:|------------------------------:|----------------:|-------------:|--------------------------------------:|-------------------------------:|
| FT-Transformer (attention) |                                 99.98 |                         99.98 |           99.46 |        98.42 |                                 -1.03 |                          0.147 |
| Random Forest (TreeSHAP)   |                                 84.94 |                         98.58 |           64.59 |        93.79 |                                 29.19 |                          0.776 |

## Example flows

| true    | prediction   |   confidence | top_features                                                                            |
|:--------|:-------------|-------------:|:----------------------------------------------------------------------------------------|
| Non-VPN | Non-VPN      |       0.9998 | bidirectional_min_ps (3%), src2dst_mean_ps (3%), bidirectional_mean_ps (3%)             |
| Non-VPN | Non-VPN      |       0.9998 | src2dst_min_ps (4%), bidirectional_min_ps (3%), dst2src_min_ps (3%)                     |
| Non-VPN | Non-VPN      |       0.9998 | src2dst_mean_ps (4%), src2dst_min_ps (3%), bidirectional_stddev_ps (3%)                 |
| VPN     | VPN          |       0.9999 | bidirectional_min_piat_ms (4%), dst2src_mean_piat_ms (3%), bidirectional_stddev_ps (3%) |
| VPN     | VPN          |       0.9999 | src2dst_bytes (4%), dst2src_packets (3%), bidirectional_mean_ps (3%)                    |
| VPN     | VPN          |       0.9999 | bidirectional_stddev_ps (4%), dst2src_duration_ms (3%), bidirectional_max_ps (3%)       |

### The same flows in plain language

Features named = the model's own top-3 attention (see deletion test above); the comparison with typical values per class comes from the training split.

- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the smallest packet size was 52 bytes, typical of Non-VPN traffic (Non-VPN median 40 bytes, VPN median 154 bytes); (2) the average client->server packet size was 65 bytes, lower than usual for either class, on the Non-VPN side (Non-VPN median 178 bytes, VPN median 278 bytes); (3) the average packet size was 1.06 kB, higher than usual for either class, on the VPN side (Non-VPN median 196 bytes, VPN median 287 bytes).
- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the smallest client->server packet size was 40 bytes, typical of Non-VPN traffic (Non-VPN median 40 bytes, VPN median 154 bytes); (2) the smallest packet size was 40 bytes, typical of Non-VPN traffic (Non-VPN median 40 bytes, VPN median 154 bytes); (3) the smallest server->client packet size was 40 bytes, typical of Non-VPN traffic (Non-VPN median 40 bytes, VPN median 154 bytes).
- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the average client->server packet size was 52 bytes, lower than usual for either class, on the Non-VPN side (Non-VPN median 178 bytes, VPN median 278 bytes); (2) the smallest client->server packet size was 52 bytes, typical of Non-VPN traffic (Non-VPN median 40 bytes, VPN median 154 bytes); (3) the variation in the packet size was 5 bytes, lower than usual for either class, on the VPN side (Non-VPN median 145 bytes, VPN median 116 bytes).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the shortest gap between packets was 4.9 s (both VPN and Non-VPN typically 1 ms); (2) the average gap between server->client packets was 0 us, lower than usual for either class, on the Non-VPN side (VPN median 1.37 s, Non-VPN median 1.07 s); (3) the variation in the packet size was 0 bytes, lower than usual for either class, on the VPN side (VPN median 116 bytes, Non-VPN median 145 bytes).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the client->server data volume was 12.6 MB, higher than usual for either class, on the VPN side (VPN median 2.9 kB, Non-VPN median 2.07 kB); (2) the number of server->client packets was 156,780 (both VPN and Non-VPN typically 10); (3) the average packet size was 1.03 kB, higher than usual for either class, on the VPN side (VPN median 287 bytes, Non-VPN median 196 bytes).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the variation in the packet size was 131 bytes, typical of Non-VPN traffic (VPN median 116 bytes, Non-VPN median 145 bytes); (2) the server->client duration was 15 s, typical of VPN traffic (VPN median 12.1 s, Non-VPN median 11.6 s); (3) the largest packet size was 666 bytes, typical of Non-VPN traffic (VPN median 570 bytes, Non-VPN median 617 bytes).

## Figures

- `global_importance.png`
- `attention_by_class.png`
- `deletion_test.png`
- `faithfulness_ft_vs_rf.png`
- `example_flows.png`
