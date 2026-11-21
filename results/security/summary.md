# Adversarial robustness: VPN evasion

Data: `data/vpn_dataset_cleaned.csv`, seeds [0, 1, 2, 3, 42]. Values: mean ± std over seeds, in %. Evasion rate = % of VPN flows the model detected that are classified Non-VPN after the attack.

## Clean accuracy

|                                         | clean_accuracy   |
|:----------------------------------------|:-----------------|
| ('standard', 'FT-Transformer')          | 90.8 ± 0.3       |
| ('standard', 'Random Forest')           | 92.4 ± 0.6       |
| ('shaping-augmented', 'FT-Transformer') | 90.8 ± 0.8       |
| ('shaping-augmented', 'Random Forest')  | 92.3 ± 0.3       |

## Traffic shaping: delay only

Evasion rate within each slowdown budget.

|                                         | Delay only <= 1.1x   | Delay only <= 1.25x   | Delay only <= 1.5x   | Delay only <= 2.0x   | Delay only <= 3.0x   | Delay only <= 5.0x   |
|:----------------------------------------|:---------------------|:----------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| ('standard', 'FT-Transformer')          | 1.0 ± 0.5            | 4.4 ± 1.6             | 12.8 ± 2.2           | 21.8 ± 2.0           | 34.4 ± 2.6           | 51.1 ± 3.0           |
| ('standard', 'Random Forest')           | 23.7 ± 5.8           | 29.0 ± 3.8            | 34.6 ± 0.8           | 39.7 ± 0.4           | 50.0 ± 1.7           | 67.7 ± 3.1           |
| ('shaping-augmented', 'FT-Transformer') | 0.4 ± 0.1            | 0.8 ± 0.1             | 1.6 ± 0.3            | 2.7 ± 0.4            | 5.5 ± 1.2            | 14.9 ± 2.8           |
| ('shaping-augmented', 'Random Forest')  | 3.8 ± 0.6            | 6.9 ± 0.6             | 10.2 ± 0.7           | 13.4 ± 0.8           | 17.8 ± 1.4           | 43.3 ± 2.2           |

## Traffic shaping: delay + up to 100% padding

|                                         | Delay + up to 100% padding <= 1.1x   | Delay + up to 100% padding <= 1.25x   | Delay + up to 100% padding <= 1.5x   | Delay + up to 100% padding <= 2.0x   | Delay + up to 100% padding <= 3.0x   | Delay + up to 100% padding <= 5.0x   |
|:----------------------------------------|:-------------------------------------|:--------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|
| ('standard', 'FT-Transformer')          | 12.2 ± 2.7                           | 16.5 ± 3.3                            | 25.5 ± 2.9                           | 34.8 ± 3.7                           | 47.8 ± 4.1                           | 62.9 ± 3.7                           |
| ('standard', 'Random Forest')           | 28.7 ± 2.7                           | 34.6 ± 0.8                            | 39.1 ± 1.4                           | 44.6 ± 1.6                           | 55.5 ± 1.9                           | 73.0 ± 2.5                           |
| ('shaping-augmented', 'FT-Transformer') | 8.1 ± 3.9                            | 8.7 ± 4.2                             | 9.5 ± 4.1                            | 10.8 ± 4.2                           | 14.0 ± 4.1                           | 24.2 ± 4.5                           |
| ('shaping-augmented', 'Random Forest')  | 8.1 ± 0.8                            | 11.5 ± 0.6                            | 14.4 ± 0.8                           | 16.8 ± 0.8                           | 21.3 ± 2.1                           | 47.4 ± 2.5                           |

`shaping-augmented` = models retrained with one randomly shaped copy of every training flow (slowdown 1-3x, padding 1-2x, both classes, same label).

## Feature-space attacks (standard models)

L-inf budget eps on standardized features.

|                                               | 0.05       | 0.1        | 0.25       | 0.5        | 1.0         |
|:----------------------------------------------|:-----------|:-----------|:-----------|:-----------|:------------|
| ('Black-box random search', 'FT-Transformer') | 26.9 ± 2.1 | 59.8 ± 2.4 | 93.5 ± 6.6 | 99.9 ± 0.1 | 100.0 ± 0.0 |
| ('Black-box random search', 'Random Forest')  | 55.0 ± 0.7 | 79.3 ± 0.9 | 97.7 ± 1.0 | 99.9 ± 0.1 | 100.0 ± 0.0 |
| ('White-box PGD', 'FT-Transformer')           | 22.7 ± 3.3 | 51.9 ± 3.1 | 85.0 ± 3.6 | 97.9 ± 1.7 | 99.9 ± 0.1  |

## Caveats

- Traffic shaping is simulated on flow features, not re-captured packets; stretched gaps crossing the 1 s activity threshold would also create idle periods in a real capture.
- Feature-space attacks ignore whether the perturbed feature vector is physically achievable: they are an upper bound on attacker power.
- The defense is evaluated against the same shaping family it was trained on, up to 3x slowdown; budgets beyond 3x are outside its training range.
