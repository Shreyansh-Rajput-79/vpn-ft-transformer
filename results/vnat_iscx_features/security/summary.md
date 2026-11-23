# Adversarial robustness: VPN evasion

Data: `data\vnat_features.csv`, seeds [0, 1, 2, 3, 42]. Values: mean ± std over seeds, in %. Evasion rate = % of VPN flows the model detected that are classified Non-VPN after the attack.

## Clean accuracy

|                                         | clean_accuracy   |
|:----------------------------------------|:-----------------|
| ('standard', 'FT-Transformer')          | 98.4 ± 0.7       |
| ('standard', 'Random Forest')           | 98.9 ± 0.5       |
| ('shaping-augmented', 'FT-Transformer') | 98.5 ± 0.4       |
| ('shaping-augmented', 'Random Forest')  | 98.7 ± 0.5       |

## Traffic shaping: delay only

Evasion rate within each slowdown budget.

|                                         | Delay only <= 1.1x   | Delay only <= 1.25x   | Delay only <= 1.5x   | Delay only <= 2.0x   | Delay only <= 3.0x   | Delay only <= 5.0x   |
|:----------------------------------------|:---------------------|:----------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| ('standard', 'FT-Transformer')          | 0.1 ± 0.1            | 0.2 ± 0.1             | 0.6 ± 0.3            | 7.7 ± 7.8            | 20.7 ± 13.3          | 27.1 ± 17.6          |
| ('standard', 'Random Forest')           | 15.1 ± 7.2           | 32.7 ± 6.6            | 52.0 ± 5.2           | 79.3 ± 2.5           | 96.5 ± 1.1           | 99.4 ± 0.8           |
| ('shaping-augmented', 'FT-Transformer') | 20.2 ± 16.5          | 58.6 ± 12.6           | 71.5 ± 16.3          | 73.7 ± 16.7          | 77.9 ± 14.2          | 81.6 ± 14.6          |
| ('shaping-augmented', 'Random Forest')  | 35.5 ± 10.8          | 63.2 ± 14.4           | 72.5 ± 16.2          | 73.9 ± 16.5          | 75.3 ± 15.8          | 90.2 ± 3.5           |

## Traffic shaping: delay + up to 100% padding

|                                         | Delay + up to 100% padding <= 1.1x   | Delay + up to 100% padding <= 1.25x   | Delay + up to 100% padding <= 1.5x   | Delay + up to 100% padding <= 2.0x   | Delay + up to 100% padding <= 3.0x   | Delay + up to 100% padding <= 5.0x   |
|:----------------------------------------|:-------------------------------------|:--------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|
| ('standard', 'FT-Transformer')          | 5.1 ± 4.8                            | 5.3 ± 5.0                             | 6.0 ± 5.1                            | 13.5 ± 9.7                           | 26.7 ± 11.3                          | 32.6 ± 15.7                          |
| ('standard', 'Random Forest')           | 15.3 ± 7.5                           | 32.8 ± 6.6                            | 52.1 ± 5.2                           | 79.8 ± 2.6                           | 96.7 ± 1.2                           | 99.4 ± 0.7                           |
| ('shaping-augmented', 'FT-Transformer') | 35.0 ± 19.7                          | 63.8 ± 13.9                           | 71.9 ± 16.4                          | 73.9 ± 16.7                          | 78.3 ± 14.3                          | 81.8 ± 14.7                          |
| ('shaping-augmented', 'Random Forest')  | 35.7 ± 10.6                          | 63.3 ± 14.4                           | 72.5 ± 16.2                          | 73.9 ± 16.5                          | 75.3 ± 15.8                          | 90.2 ± 3.5                           |

`shaping-augmented` = models retrained with one randomly shaped copy of every training flow (slowdown 1-3x, padding 1-2x, both classes, same label).

## Feature-space attacks (standard models)

L-inf budget eps on standardized features.

|                                               | 0.05        | 0.1         | 0.25       | 0.5         | 1.0         |
|:----------------------------------------------|:------------|:------------|:-----------|:------------|:------------|
| ('Black-box random search', 'FT-Transformer') | 73.7 ± 16.2 | 90.5 ± 8.9  | 98.8 ± 0.9 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| ('Black-box random search', 'Random Forest')  | 90.9 ± 5.4  | 97.0 ± 2.3  | 99.9 ± 0.1 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| ('White-box PGD', 'FT-Transformer')           | 51.6 ± 16.5 | 61.9 ± 10.6 | 91.4 ± 8.4 | 98.7 ± 1.8  | 99.4 ± 1.0  |

## Caveats

- Traffic shaping is simulated on flow features, not re-captured packets; stretched gaps crossing the 1 s activity threshold would also create idle periods in a real capture.
- Feature-space attacks ignore whether the perturbed feature vector is physically achievable: they are an upper bound on attacker power.
- The defense is evaluated against the same shaping family it was trained on, up to 3x slowdown; budgets beyond 3x are outside its training range.
