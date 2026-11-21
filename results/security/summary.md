# Adversarial robustness: VPN evasion

Attacker goal: make detected VPN flows look Non-VPN. Seed-42 test split. Evasion rate = % of VPN flows the model detected that are classified Non-VPN after the attack. FT-Transformer: 1306 detected flows attacked, Random Forest: 1312 detected flows attacked.

## 1. Traffic shaping (realistic, black-box)

Attacker slows the flow down (every gap stretched) and optionally pads packets (byte overhead up to 100%). Evasion rate (%) within each slowdown budget:

| slowdown   |   FT-Transformer / Delay only |   FT-Transformer / Delay + up to 100% padding |   Random Forest / Delay only |   Random Forest / Delay + up to 100% padding |
|:-----------|------------------------------:|----------------------------------------------:|-----------------------------:|---------------------------------------------:|
| <= 1.1x    |                           0.9 |                                          13.2 |                         25.9 |                                         29.6 |
| <= 1.25x   |                           3.7 |                                          17   |                         30.6 |                                         33.9 |
| <= 1.5x    |                          12.5 |                                          26.4 |                         35   |                                         38.9 |
| <= 2.0x    |                          18.5 |                                          32.8 |                         39.7 |                                         43.6 |
| <= 3.0x    |                          30.3 |                                          45.9 |                         47.3 |                                         52.3 |
| <= 5.0x    |                          46.2 |                                          60.3 |                         63   |                                         68.8 |

Median slowdown needed among flows that can evade within 5x: FT-Transformer / Delay only: 2.35x; FT-Transformer / Delay + up to 100% padding: 1.80x; Random Forest / Delay only: 1.30x; Random Forest / Delay + up to 100% padding: 1.30x

## 2. Feature-space attacks (worst case)

L-inf budget eps on standardized features (1.0 = one standard deviation per feature). Evasion rate (%):

|                                                         |   0.05 |   0.1 |   0.25 |   0.5 |   1.0 |
|:--------------------------------------------------------|-------:|------:|-------:|------:|------:|
| ('Black-box random search', 'FT-Transformer')           |   25.4 |  62.6 |   97.6 | 100   | 100   |
| ('Black-box random search', 'Random Forest')            |   54.1 |  80   |   98.6 |  99.9 | 100   |
| ('Transfer from FT-Transformer', 'Random Forest')       |   33.4 |  38.6 |   71.6 |  78.6 |  84.3 |
| ('Transfer from FT-Transformer (PGD)', 'Random Forest') |   32.8 |  39.6 |   60.6 |  71.2 |  80.8 |
| ('Transfer from Random Forest', 'FT-Transformer')       |   13.7 |  37.7 |   79   |  83.7 |  94.5 |
| ('White-box PGD', 'FT-Transformer')                     |   19.1 |  53.9 |   86.8 |  99.4 | 100   |

## Figures

- `shaping_attack.png`
- `feature_space_attacks.png`

## Caveats

- Traffic shaping is simulated on flow features, not re-captured packets; stretched gaps crossing the 1 s activity threshold would also create idle periods in a real capture.
- Feature-space attacks ignore whether the perturbed feature vector is physically achievable: they are an upper bound on attacker power, not a realistic attack.
