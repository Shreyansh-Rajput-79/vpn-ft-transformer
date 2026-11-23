# Adversarial robustness: VPN evasion

Data: `data\vnat_nfstream.csv`, seeds [0, 1, 2, 3, 42]. Values: mean ± std over seeds, in %. Evasion rate = % of VPN flows the model detected that are classified Non-VPN after the attack.

## Clean accuracy

|                                         | clean_accuracy   |
|:----------------------------------------|:-----------------|
| ('standard', 'FT-Transformer')          | 100.0 ± 0.0      |
| ('standard', 'Random Forest')           | 100.0 ± 0.0      |
| ('shaping-augmented', 'FT-Transformer') | 100.0 ± 0.0      |
| ('shaping-augmented', 'Random Forest')  | 100.0 ± 0.0      |

## Traffic shaping: delay only

Evasion rate within each slowdown budget.

|                                         | Delay only <= 1.1x   | Delay only <= 1.25x   | Delay only <= 1.5x   | Delay only <= 2.0x   | Delay only <= 3.0x   | Delay only <= 5.0x   |
|:----------------------------------------|:---------------------|:----------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| ('standard', 'FT-Transformer')          | 0.0 ± 0.0            | 0.0 ± 0.0             | 0.0 ± 0.0            | 0.0 ± 0.0            | 0.0 ± 0.0            | 0.0 ± 0.0            |
| ('standard', 'Random Forest')           | 0.0 ± 0.0            | 0.0 ± 0.0             | 0.0 ± 0.0            | 0.0 ± 0.0            | 0.0 ± 0.0            | 0.0 ± 0.0            |
| ('shaping-augmented', 'FT-Transformer') | 0.0 ± 0.0            | 0.0 ± 0.0             | 0.0 ± 0.0            | 0.0 ± 0.1            | 0.0 ± 0.1            | 0.0 ± 0.1            |
| ('shaping-augmented', 'Random Forest')  | 0.0 ± 0.0            | 0.0 ± 0.0             | 0.0 ± 0.0            | 0.0 ± 0.0            | 0.0 ± 0.0            | 0.0 ± 0.0            |

## Traffic shaping: delay + up to 100% padding

|                                         | Delay + up to 100% padding <= 1.1x   | Delay + up to 100% padding <= 1.25x   | Delay + up to 100% padding <= 1.5x   | Delay + up to 100% padding <= 2.0x   | Delay + up to 100% padding <= 3.0x   | Delay + up to 100% padding <= 5.0x   |
|:----------------------------------------|:-------------------------------------|:--------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|:-------------------------------------|
| ('standard', 'FT-Transformer')          | 0.0 ± 0.0                            | 0.0 ± 0.0                             | 0.0 ± 0.0                            | 0.0 ± 0.0                            | 0.0 ± 0.0                            | 0.0 ± 0.0                            |
| ('standard', 'Random Forest')           | 0.0 ± 0.0                            | 0.0 ± 0.0                             | 0.0 ± 0.0                            | 0.0 ± 0.0                            | 0.0 ± 0.0                            | 0.0 ± 0.0                            |
| ('shaping-augmented', 'FT-Transformer') | 0.0 ± 0.0                            | 0.0 ± 0.0                             | 0.0 ± 0.0                            | 0.0 ± 0.1                            | 0.0 ± 0.1                            | 0.0 ± 0.1                            |
| ('shaping-augmented', 'Random Forest')  | 0.0 ± 0.0                            | 0.0 ± 0.0                             | 0.0 ± 0.0                            | 0.0 ± 0.0                            | 0.0 ± 0.0                            | 0.0 ± 0.0                            |

`shaping-augmented` = models retrained with one randomly shaped copy of every training flow (slowdown 1-3x, padding 1-2x, both classes, same label).

## Feature-space attacks (standard models)

L-inf budget eps on standardized features.

|                                               | 0.05      | 0.1       | 0.25        | 0.5         | 1.0         |
|:----------------------------------------------|:----------|:----------|:------------|:------------|:------------|
| ('Black-box random search', 'FT-Transformer') | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.8 ± 6.1   | 78.7 ± 26.5 | 100.0 ± 0.0 |
| ('Black-box random search', 'Random Forest')  | 0.0 ± 0.0 | 0.0 ± 0.0 | 29.5 ± 20.8 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| ('White-box PGD', 'FT-Transformer')           | 0.0 ± 0.0 | 0.0 ± 0.0 | 3.5 ± 5.0   | 47.9 ± 34.7 | 83.7 ± 11.8 |

## Caveats

- Traffic shaping is simulated on flow features, not re-captured packets; stretched gaps crossing the 1 s activity threshold would also create idle periods in a real capture.
- Feature-space attacks ignore whether the perturbed feature vector is physically achievable: they are an upper bound on attacker power.
- The defense is evaluated against the same shaping family it was trained on, up to 3x slowdown; budgets beyond 3x are outside its training range.
