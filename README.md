# Explainable VPN Traffic Classification (FT-Transformer)

VPN vs Non-VPN (and VNAT category / app) flow classification with an
FT-Transformer, compared against Logistic Regression, Random Forest and XGBoost.
Includes explainability (attention rollout, faithfulness test, TreeSHAP on the
Random Forest) and an adversarial robustness evaluation with a defense.

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# GPU machine: install the CUDA build of torch matching the driver, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"
```

## VNAT on the GPU machine: one command
```
.\run_vnat.ps1 -VnatDir D:\path\to\VNAT
```
Builds both feature sets, tunes the FT-Transformer, trains the final models and
writes every comparison, explainability, security and cross-dataset report to
`results\`. Re-running skips finished steps. If PowerShell blocks the script:
`powershell -ExecutionPolicy Bypass -File .\run_vnat.ps1 -VnatDir D:\path\to\VNAT`.

## Scripts
| Script | What it does |
|---|---|
| `build_vnat_dataset.py` | VNAT pcaps -> feature CSV (`--extractor iscx` or `nfstream`), labels `class1` / `app` / `category` from the file names |
| `train.py` | Train + test report. `--data`, `--label-col`, `--group-col source_file`, `--config` |
| `compare_models.py` | LR / RF / XGBoost / FT-Transformer over 5 seeds: metrics, robustness, cost, confusion matrices |
| `explain.py` | Attention explanations, permutation importance, TreeSHAP on RF, faithfulness of both |
| `security_eval.py` | Traffic-shaping and feature-space evasion attacks, shaping-augmentation defense, 5 seeds |
| `tune.py` | Random hyperparameter search for the FT-Transformer (validation split only) |
| `cross_dataset.py` | Train on one dataset, test on another (ISCX <-> VNAT) |
| `predict.py` | Classify a pcap or CSV. `--explain`, `--by-device` (hotspot demo) |

`--group-col source_file` splits by capture file so flows from one recording
never appear in both train and test. Always use it for VNAT.

## ISCX VPN-nonVPN 2016 (data/vpn_dataset_cleaned.csv)
```
python train.py
python compare_models.py
python explain.py
python security_eval.py
python predict.py capture.pcap --explain --by-device -o out.csv
```

## Feature sets
- **iscx**: the 23 ISCX time-based features, re-implemented from pcaps at microsecond precision.
- **nfstream**: 57 NFStream behavioural features. IPs, ports, MACs, timestamps and all nDPI
  application fields are excluded on purpose, since they identify endpoints or leak the label.
