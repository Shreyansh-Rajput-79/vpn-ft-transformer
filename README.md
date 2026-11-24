# Explainable VPN Traffic Classification (FT-Transformer)

Detects whether network traffic goes through a VPN, and which kind of application
it is (Chat, C2, Streaming, File Transfer, VoIP), from **behaviour only**: packet
sizes, timing and volume. IP addresses, ports and timestamps are never used.
Every decision comes with a plain-language explanation, and it works on live
traffic from your own network card.

The model is an **FT-Transformer**, compared on equal terms with Logistic
Regression, Random Forest and XGBoost (5 seeds, splits by capture file,
paired significance tests). Datasets: **VNAT** (MIT Lincoln Laboratory) and
**ISCX VPN-nonVPN 2016**. The full write-up is in
[`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md).

| Main results (mean of 5 seeds) | LR | RF | XGBoost | FT-Transformer |
|---|---|---|---|---|
| VPN vs Non-VPN, timing features only | 93.5% | 98.9% | **99.3%** | 98.5% |
| 5 application categories, accuracy | 84.2% | **93.3%** | 91.9% | 92.8% |
| Evading the detector by slowing traffic 2× (lower = better) | | 79.3% (RF) | | **7.7%** |

The FT-Transformer ties the tree models on accuracy. Its advantages are explanations
fast enough for live traffic (0.09-0.38 ms per flow vs 127-161 ms for TreeSHAP) and far
stronger resistance to traffic shaping. No model generalises across the two datasets
(50-56%); see the limitations in the write-up.

---

## 1. Install (5 minutes)

You need **Python 3.10, 3.11 or 3.12** ([python.org](https://www.python.org/downloads/);
on Windows tick *"Add python.exe to PATH"*). A GPU is optional.

1. Download the project: green **Code** button → **Download ZIP**, then extract it
   (or `git clone https://github.com/Shreyansh-Rajput-79/vpn-ft-transformer.git`).
2. Open a terminal **inside the extracted folder** and run:

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
If PowerShell refuses to run `activate`, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional, NVIDIA GPU: replace the CPU build of PyTorch with the CUDA one, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu124`.
Only training gets faster; the demos below run fine on a CPU.

## 2. Try it: classify and explain a capture

Two small captures of the same Skype chat are included in [`samples/`](samples/),
one through OpenVPN and one without. The model never saw them during training.

```bash
python predict.py samples/vpn_skype-chat_capture11.pcap    -m models/vnat_noflags_model.pth --explain --trace 1
python predict.py samples/nonvpn_skype-chat_capture13.pcap -m models/vnat_noflags_model.pth --explain --trace 1
```

`--trace 1` prints every stage for the first flow: packets → flow → 33 features →
preprocessing → FT-Transformer (logits, probabilities) → explanation. For example:

```
VPN (100% confident). What the model looked at most:
  ...
  Clearest sign of VPN in this flow (from the training data, not the model's attention):
     Smallest packet size: 154 bytes  -> points to VPN  (typical: VPN 154 bytes, Non-VPN 40 bytes)
     Even its smallest packet was large: every packet seems to carry extra bytes. That is what
     a VPN does: it wraps every packet, even a bare acknowledgement, in extra headers and encryption data.
```

Any `.pcap` / `.pcapng` you record yourself (e.g. with Wireshark) works the same way.
Add `-o out.csv` to save one row per flow with the prediction and explanation.

## 3. Live traffic from your own network card

**Windows:** install [Npcap](https://npcap.com/#download) first (default options).
**Linux / macOS:** run the commands with `sudo` (use the venv's python, e.g. `sudo venv/bin/python ...`).

```bash
python predict_live.py --list                                    # find your interface name (Windows)
python predict_live.py -i Wi-Fi -m models/vnat_noflags_model.pth --trace 1 --seconds 60
```
On Linux/macOS use the interface name from `ip link` / `ifconfig`, e.g. `-i wlan0` or `-i en0`.

Each flow is judged when it reaches 15 s or goes quiet, with a reason in plain language.
DNS, DHCP and other housekeeping traffic is skipped. Turn a VPN on and off to compare.
Note: the model was trained on **OpenVPN over UDP** only; other VPNs (WireGuard,
commercial apps over TCP) may not be recognised. That is a known limitation.

To measure accuracy on your own recordings: `python score_captures.py --help`.

**No VPN at hand?** `tools/demo_tunnel.py` is a small encrypted UDP tunnel for demos
(not a real VPN). On Windows, with Npcap installed, run in three terminals:
```powershell
python tools/demo_tunnel.py --remote proof.ovh.net:443
python predict_live.py -i \Device\NPF_Loopback --bpf "udp port 1195" -m models/vnat_noflags_model.pth --trace 1
curl.exe --resolve proof.ovh.net:443:127.0.0.1 https://proof.ovh.net/files/10Mb.dat -o NUL --max-time 25
```
Result in our test: classified **Non-VPN** (missed). The explanation shows why: the tunnel's
small control packets (76 bytes) never occur in VNAT's OpenVPN traffic. See
`results/FINAL_RESULTS.md`, section 8.

**Traffic-shaping demo (no capture needed, ~20 s):** `python tools/demo_shaping.py` slows
VPN flows from unseen test recordings to half speed and shows how many fool the Random Forest
vs the FT-Transformer, with one example flow before and after. `--slowdown 3` for other budgets.
One split only, so numbers differ slightly from the 5-seed averages in section 6 of the write-up.

## 4. Which model to use

| File | Task | Features |
|---|---|---|
| `models/vnat_noflags_model.pth` | VPN vs Non-VPN, **recommended for live use** | 33 NFStream features, no TCP flags |
| `models/vnat_model.pth` | VPN vs Non-VPN | 57 NFStream features (TCP flags act as a protocol shortcut) |
| `models/vnat_timing_model.pth` | VPN vs Non-VPN, the hardest setting | 23 timing features only |
| `models/vnat_category_model.pth` | Chat / C2 / Streaming / File Transfer / VoIP | 57 NFStream features |
| `models/model.pth` | VPN vs Non-VPN on ISCX 2016 | 23 ISCX time-based features (CSV input) |

## 5. Reproduce the results

All results and trained models are already included (`results/`, `models/`). The cleaned
VNAT feature tables are in `data/`, so the raw VNAT captures are **not** needed.

Single steps (minutes on a GPU, longer on a CPU):
```bash
# train the live model again (writes a new file, does not touch the shipped one)
python train.py --data data/vnat_nfstream.csv --group-col source_file --feature-set nfstream_noflags --config results/tuning/best_config.json --model-out models/my_noflags_model.pth

# 5-seed comparison with significance tests, and ROC / confusion-matrix figures
python compare_models.py --data data/vnat_features.csv --group-col source_file --config results/tuning/best_config.json --out results/my_vnat_timing
python metrics_figures.py --help

# ISCX 2016
python train.py --model-out models/my_iscx_model.pth
python compare_models.py --out results/my_iscx
```

Everything (Windows): `.\run_vnat.ps1`. Each step is skipped when its output already
exists, so delete an output (e.g. `results\vnat\summary.md`) to recompute it. The full run
takes several hours on a GPU. The exact command behind every result is listed in
`run_vnat.ps1` and in section 10 of `results/FINAL_RESULTS.md`; on Linux/macOS run those
commands one by one.

To rebuild `data/vnat_*.csv` from the raw captures, download VNAT and run
`python build_vnat_dataset.py <VNAT folder> -o data/vnat_nfstream.csv --extractor nfstream`
(and without `--extractor` for `data/vnat_features.csv`), or `.\run_vnat.ps1 -VnatDir <VNAT folder>`
after deleting the two CSVs.

`--group-col source_file` splits by capture file, so flows of one recording never appear in
both train and test. Always use it for VNAT.

## 6. Scripts

| Script | What it does |
|---|---|
| `predict.py` | Classify a pcap or CSV. `--explain`, `--trace N` (every stage for N flows), `-o out.csv`, `--by-device` |
| `predict_live.py` | Classify and explain live traffic. `--list`, `-i <interface>`, `--trace N`, `--seconds`, `-o live.csv` |
| `score_captures.py` | Accuracy on your own labelled captures (VPN on / off) |
| `train.py` | Train one FT-Transformer + test report |
| `compare_models.py` | LR / RF / XGBoost / FT-Transformer over 5 seeds, significance tests, robustness, cost |
| `metrics_figures.py` | ROC / PR curves, confusion matrices, per-class TP / FP / FN / TN |
| `explain.py` | Attention explanations, TreeSHAP on Random Forest, faithfulness (deletion test) |
| `security_eval.py` | Traffic-shaping and feature-space evasion attacks, augmentation defence |
| `tools/demo_shaping.py` | Quick demo: slow VPN flows down, compare Random Forest vs FT-Transformer evasion |
| `cross_dataset.py` | Train on one dataset, test on the other (ISCX ↔ VNAT) |
| `tune.py` | Hyperparameter search (validation split only) |
| `build_vnat_dataset.py` | VNAT pcaps → cleaned feature CSV |
| `run_vnat.ps1` | The whole VNAT pipeline in one command (Windows) |

## 7. Troubleshooting

- **`pip install` fails on nfstream or torch**: use Python 3.10-3.12 (not 3.13) and a 64-bit Python.
- **`predict_live.py`: no interfaces / capture error**: install Npcap (Windows) or use `sudo` (Linux/macOS).
- **Everything is classified Non-VPN live**: expected without a VPN; with a non-OpenVPN VPN see the limitation above.
- **`python` not found on Windows**: reinstall Python with *Add to PATH*, or use `py` instead of `python`.

## Data and citation

- **VNAT**: S. Jorgensen et al., "Extensible Machine Learning for Encrypted Network Traffic
  Application Labeling via Uncertainty Quantification," IEEE Transactions on Artificial
  Intelligence, 2023. The feature tables in `data/` and the two captures in `samples/` are
  derived from / taken from VNAT.
- **ISCX VPN-nonVPN 2016**: G. Draper-Gil, A. H. Lashkari, M. S. I. Mamun and A. A. Ghorbani,
  "Characterization of Encrypted and VPN Traffic Using Time-Related Features," ICISSP 2016.

Author: Shreyansh Rajput, IIIT Allahabad.
