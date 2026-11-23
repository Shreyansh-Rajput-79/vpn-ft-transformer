# Explainability report

Seed-42 test split (same as train.py), 5559 flows, 57 features. Per-flow explanations: `test_flow_explanations.csv`.

## Top-5 features by method

- FT attention (mean): bidirectional_packets, bidirectional_max_piat_ms, src2dst_psh_packets, bidirectional_mean_ps, dst2src_stddev_piat_ms
- FT permutation: src2dst_stddev_ps, src2dst_ack_packets, bidirectional_packets, src2dst_min_piat_ms, src2dst_packets
- RF TreeSHAP (mean |SHAP|): src2dst_stddev_ps, dst2src_stddev_piat_ms, dst2src_mean_piat_ms, src2dst_stddev_piat_ms, src2dst_mean_ps
- RF permutation: bidirectional_stddev_ps, src2dst_stddev_ps, src2dst_max_ps, dst2src_stddev_ps, dst2src_mean_ps
- RF impurity: dst2src_stddev_piat_ms, dst2src_mean_piat_ms, src2dst_mean_ps, src2dst_stddev_piat_ms, bidirectional_mean_piat_ms

## Agreement between methods (Spearman rank correlation over 57 features)

- FT attention vs FT permutation: 0.405
- RF TreeSHAP vs RF permutation: 0.331
- FT permutation vs RF permutation: 0.333

## Faithfulness (FT-Transformer, all test flows)

Mean confidence (%) in the original prediction after removing k features per flow. Lower = removing those features hurts more.

|   features removed |   Most-attended first |   Random order |   Least-attended first |
|-------------------:|----------------------:|---------------:|-----------------------:|
|                  0 |                 95.09 |          95.09 |                  95.09 |
|                  1 |                 86.65 |          93.42 |                  94.86 |
|                  2 |                 81.76 |          91.88 |                  93.93 |
|                  3 |                 80.79 |          90.51 |                  93.84 |
|                  4 |                 79.47 |          89.11 |                  94.16 |
|                  5 |                 79.81 |          87.86 |                  94.02 |
|                  6 |                 78.16 |          86.61 |                  93.18 |
|                  7 |                 74.38 |          85.45 |                  92.86 |
|                  8 |                 67.35 |          84.29 |                  92.18 |
|                  9 |                 59.34 |          83.3  |                  91.72 |
|                 10 |                 54.74 |          82.38 |                  91.22 |

## FT attention vs RF TreeSHAP (same 500 sampled flows)

Each model is tested with its own explanation. The gap between the random curve and the top-first curve measures how well the explanation identifies what the model relies on. Attention is produced in the same forward pass as the prediction; TreeSHAP is computed afterwards.

| explanation                |   confidence after removing top-1 (%) |   after removing random-1 (%) |   AUC top-first |   AUC random |   faithfulness gap (AUC random - top) |   explanation cost (ms / flow) |
|:---------------------------|--------------------------------------:|------------------------------:|----------------:|-------------:|--------------------------------------:|-------------------------------:|
| FT-Transformer (attention) |                                 87.59 |                         93.06 |           77.62 |        88.12 |                                 10.49 |                          0.381 |
| Random Forest (TreeSHAP)   |                                 83.42 |                         85.81 |           63.33 |        78.92 |                                 15.59 |                        160.882 |

## Example flows

| true          | prediction    |   confidence | top_features                                                                         |
|:--------------|:--------------|-------------:|:-------------------------------------------------------------------------------------|
| C2            | C2            |       0.9577 | src2dst_stddev_ps (5%), bidirectional_packets (3%), bidirectional_syn_packets (3%)   |
| Chat          | Chat          |       0.9999 | bidirectional_packets (3%), dst2src_ack_packets (3%), bidirectional_max_piat_ms (2%) |
| File Transfer | File Transfer |       0.9821 | src2dst_max_ps (4%), bidirectional_packets (3%), dst2src_syn_packets (3%)            |
| Streaming     | Streaming     |       0.9991 | bidirectional_syn_packets (3%), src2dst_syn_packets (3%), dst2src_min_ps (3%)        |
| VoIP          | VoIP          |       0.9991 | bidirectional_min_ps (3%), bidirectional_psh_packets (3%), src2dst_stddev_ps (3%)    |
| File Transfer | Streaming     |       0.9983 | bidirectional_packets (4%), bidirectional_mean_ps (3%), src2dst_psh_packets (3%)     |
| File Transfer | Streaming     |       0.9979 | bidirectional_packets (5%), dst2src_ack_packets (4%), src2dst_psh_packets (3%)       |

### The same flows in plain language

Features named = the model's own top-3 attention (see deletion test above); the comparison with typical values per class comes from the training split.

- *True C2:* Classified as C2 (96% confident). Main reasons: (1) the variation in the client->server packet size was 28 bytes, closer to File Transfer than C2 traffic (C2 median 0 bytes, File Transfer median 64 bytes); (2) the number of packets was 22, typical of Chat traffic (C2 median 4, Chat median 23); (3) the number of SYN-flag packets was 0 (both C2 and Chat typically 0). None of these values on its own is typical of C2; the decision comes from how the features combine.
- *True Chat:* Classified as Chat (100% confident). Main reasons: (1) the number of packets was 18, typical of Streaming traffic (Chat median 23, Streaming median 21); (2) the number of server->client ACK-flag packets was 9, typical of Streaming traffic (Chat median 0, Streaming median 8); (3) the longest gap between packets was 4.88 s, typical of Chat traffic (Chat median 3.89 s, File Transfer median 9.01 s).
- *True File Transfer:* Classified as File Transfer (98% confident). Main reasons: (1) the largest client->server packet size was 1.38 kB, typical of File Transfer traffic (File Transfer median 1.38 kB, Streaming median 651 bytes); (2) the number of packets was 47, closer to Chat than File Transfer traffic (File Transfer median 174, Chat median 23); (3) the number of server->client SYN-flag packets was 1 (both File Transfer and C2 typically 0).
- *True Streaming:* Classified as Streaming (100% confident). Main reasons: (1) the number of SYN-flag packets was 2 (both Streaming and C2 typically 0); (2) the number of client->server SYN-flag packets was 1 (both Streaming and C2 typically 0); (3) the smallest server->client packet size was 52 bytes, typical of Streaming traffic (Streaming median 52 bytes, VoIP median 73 bytes).
- *True VoIP:* Classified as VoIP (100% confident). Main reasons: (1) the smallest packet size was 138 bytes, typical of Chat traffic (VoIP median 73 bytes, Chat median 154 bytes); (2) the number of PSH-flag packets was 0 (both VoIP and C2 typically 0); (3) the variation in the client->server packet size was 2 bytes (both VoIP and C2 typically 0 bytes). None of these values on its own is typical of VoIP; the decision comes from how the features combine.
- *True File Transfer:* Classified as Streaming (100% confident). Main reasons: (1) the number of packets was 22,691, higher than usual for either class, on the File Transfer side (Streaming median 21, File Transfer median 174); (2) the average packet size was 912 bytes, typical of File Transfer traffic (Streaming median 296 bytes, File Transfer median 727 bytes); (3) the number of client->server PSH-flag packets was 1, typical of Streaming traffic (Streaming median 2, C2 median 0).
- *True File Transfer:* Classified as Streaming (100% confident). Main reasons: (1) the number of packets was 65,533, higher than usual for either class, on the File Transfer side (Streaming median 21, File Transfer median 174); (2) the number of server->client ACK-flag packets was 23,225, higher than usual for either class, on the Streaming side (Streaming median 8, C2 median 0); (3) the number of client->server PSH-flag packets was 2, typical of Streaming traffic (Streaming median 2, C2 median 0).

## Figures

- `global_importance.png`
- `attention_by_class.png`
- `deletion_test.png`
- `faithfulness_ft_vs_rf.png`
- `example_flows.png`
