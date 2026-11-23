# Explainability report

Seed-42 test split (same as train.py), 5654 flows, 57 features. Per-flow explanations: `test_flow_explanations.csv`.

## Top-5 features by method

- FT attention (mean): bidirectional_min_ps, bidirectional_psh_packets, bidirectional_urg_packets, bidirectional_packets, bidirectional_duration_ms
- FT permutation: bidirectional_min_ps, bidirectional_packets, bidirectional_bytes, src2dst_duration_ms, src2dst_packets
- RF TreeSHAP (mean |SHAP|): bidirectional_min_ps, src2dst_min_ps, dst2src_min_ps, bidirectional_psh_packets, bidirectional_ack_packets
- RF permutation: bidirectional_min_ps, bidirectional_packets, bidirectional_bytes, src2dst_duration_ms, src2dst_packets
- RF impurity: bidirectional_min_ps, src2dst_min_ps, dst2src_min_ps, bidirectional_ack_packets, bidirectional_psh_packets

## Agreement between methods (Spearman rank correlation over 57 features)

- FT attention vs FT permutation: 0.227
- RF TreeSHAP vs RF permutation: 0.228
- FT permutation vs RF permutation: 1.000

## Faithfulness (FT-Transformer, all test flows)

Mean confidence (%) in the original prediction after removing k features per flow. Lower = removing those features hurts more.

|   features removed |   Most-attended first |   Random order |   Least-attended first |
|-------------------:|----------------------:|---------------:|-----------------------:|
|                  0 |                 99.98 |          99.98 |                  99.98 |
|                  1 |                 97.32 |          99.73 |                  99.98 |
|                  2 |                 59.05 |          99.38 |                  99.98 |
|                  3 |                 59.2  |          98.97 |                  99.98 |
|                  4 |                 60.36 |          98.53 |                  99.98 |
|                  5 |                 60.5  |          98.07 |                  99.98 |
|                  6 |                 60.93 |          97.58 |                  99.98 |
|                  7 |                 60.96 |          97.05 |                  99.98 |
|                  8 |                 61.08 |          96.55 |                  99.98 |
|                  9 |                 60.62 |          96.01 |                  99.98 |
|                 10 |                 60.95 |          95.43 |                  99.98 |

## FT attention vs RF TreeSHAP (same 500 sampled flows)

Each model is tested with its own explanation. The gap between the random curve and the top-first curve measures how well the explanation identifies what the model relies on. Attention is produced in the same forward pass as the prediction; TreeSHAP is computed afterwards.

| explanation                |   confidence after removing top-1 (%) |   after removing random-1 (%) |   AUC top-first |   AUC random |   faithfulness gap (AUC random - top) |   explanation cost (ms / flow) |
|:---------------------------|--------------------------------------:|------------------------------:|----------------:|-------------:|--------------------------------------:|-------------------------------:|
| FT-Transformer (attention) |                                 97.87 |                         99.62 |           67.87 |        98.37 |                                 30.5  |                          0.323 |
| Random Forest (TreeSHAP)   |                                 88.56 |                         98.97 |           68.41 |        94.95 |                                 26.55 |                          0.548 |

## Example flows

| true    | prediction   |   confidence | top_features                                                                              |
|:--------|:-------------|-------------:|:------------------------------------------------------------------------------------------|
| Non-VPN | Non-VPN      |       0.9998 | bidirectional_packets (2%), bidirectional_min_ps (2%), bidirectional_urg_packets (2%)     |
| VPN     | VPN          |       0.9998 | bidirectional_min_ps (3%), bidirectional_psh_packets (2%), bidirectional_urg_packets (2%) |

### The same flows in plain language

Features named = the model's own top-3 attention (see deletion test above); the comparison with typical values per class comes from the training split.

- *True Non-VPN:* Classified as Non-VPN (100% confident). Main reasons: (1) the number of packets was 4,127, higher than usual for either class, on the Non-VPN side (Non-VPN median 22, VPN median 20); (2) the smallest packet size was 52 bytes, typical of Non-VPN traffic (Non-VPN median 40 bytes, VPN median 154 bytes); (3) the number of URG-flag packets was 0 (both Non-VPN and VPN typically 0).
- *True VPN:* Classified as VPN (100% confident). Main reasons: (1) the smallest packet size was 122 bytes, typical of VPN traffic (VPN median 154 bytes, Non-VPN median 40 bytes); (2) the number of PSH-flag packets was 0, typical of VPN traffic (VPN median 0, Non-VPN median 12); (3) the number of URG-flag packets was 0 (both VPN and Non-VPN typically 0).

## Figures

- `global_importance.png`
- `attention_by_class.png`
- `deletion_test.png`
- `faithfulness_ft_vs_rf.png`
- `example_flows.png`
