# Explainable VPN Traffic Classification (FT-Transformer)

VPN vs Non-VPN flow classification with an FT-Transformer, a Random Forest
baseline, explainability (attention rollout + faithfulness test) and an
adversarial robustness evaluation.

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# GPU machine: install the CUDA build of torch matching the driver, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## ISCX VPN-nonVPN 2016 (data/vpn_dataset_cleaned.csv)
```
python train.py                 # train + test report -> models/model.pth
python compare_models.py        # LR / RF / FT-Transformer, 5 seeds -> results/
python explain.py               # explainability report -> results/explain/
python security_eval.py         # evasion attacks -> results/security/
python predict.py capture.pcap --explain -o out.csv
```

## VNAT (raw pcaps, run on the lab machine)
```
python build_vnat_dataset.py path\to\VNAT -o data\vnat_features.csv
python train.py --data data\vnat_features.csv --group-col source_file --model-out models\vnat_model.pth
python compare_models.py --data data\vnat_features.csv --group-col source_file --out results\vnat
```
`--group-col source_file` splits by capture file so flows from one recording
never appear in both train and test.
