# Explainable VPN traffic classification with an FT-Transformer: final results

ISCX VPN-nonVPN 2016 and VNAT (MIT Lincoln Laboratory). Run of 22 Nov 2026 on 2× GTX 1080 Ti.
Every number below comes from a `summary.md` in `results/`; the source folder is given with each section.

## 1. Headline findings

1. **Accuracy: the FT-Transformer matches tree ensembles and never beats them.** In paired tests over 84–87 test capture files it is on par with Random Forest on every VNAT task, on par with XGBoost on the 5-category task, and 0.8 pp behind XGBoost (significant) with timing-only features. It is significantly better than Logistic Regression wherever the task is not trivial.
2. **VPN vs Non-VPN on VNAT is trivially separable with NFStream features.** Every model scores ~100%, including Logistic Regression (99.99%). The reason is physical: every VNAT VPN flow is OpenVPN over UDP, and its encapsulation makes even the smallest packet ≥ ~120 bytes (VPN median 154 B vs 40 B). This is a property of the dataset, not evidence of model strength.
3. **Explanation cost is the FT-Transformer's clearest advantage.** Attention explanations cost 0.09–0.38 ms per flow in every setting. TreeSHAP on the Random Forest costs 127–161 ms per flow on non-trivial tasks (≈400–1,400× more, ~6–8 flows/s), so only attention can explain live traffic.
4. **Explanation faithfulness is mixed.** Attention passes the deletion test in 4 of 5 settings and its top-1 feature is more decisive than TreeSHAP's on the hard VPN task, but TreeSHAP ranks features more faithfully overall in 3 of 5 settings. "Attention is more faithful" is **not** supported in general.
5. **The FT-Transformer is much harder to evade by traffic shaping.** On VNAT timing features, a 2× packet slowdown lets 7.7% of detected VPN flows evade it vs 79.3% for Random Forest (ISCX: 21.8% vs 39.7%). It also needs larger feature-space perturbations. But the shaping-augmentation defense that worked on ISCX **backfires** on VNAT (§6).
6. **No model generalizes across datasets.** Trained on ISCX and tested on VNAT, or the reverse, every model falls to 50–56% accuracy (chance ≈ 50%).

## 2. Data and protocol

| | ISCX 2016 | VNAT |
|---|---|---|
| Source | 23 published time-based features | 165 pcaps, re-extracted |
| Feature sets | 23 timing features | 23 timing features (same definitions, µs precision); 57 NFStream features; 33 NFStream features without TCP-flag counts |
| Flows used | 18,758 rows, ~18,000 after removing exact duplicates | 42,728 (timing) / 42,737 (NFStream) after cleaning |
| Split | stratified by flow, 70/15/15 | **by capture file**, 70/15/15 per class: no recording appears in both train and test |
| Seeds | 5 | 5 (tuning on seed 42's validation split only) |

**Flow construction.** Flows are cut 15 s after their first packet in both extractors. NFStream's default (30 min) turned each VPN capture into ~2 flows (1.4% VPN).

**Label cleaning (VNAT).** Labels come from file names, but captures contain traffic that is not the labelled app. In the non-VPN ssh/scp/sftp/rsync captures 86–95% of flows were DNS lookups, and VPN captures contained flows that bypassed the tunnel. Kept: VPN captures → OpenVPN tunnel flows only (port 1195); non-VPN captures → everything except DNS, DHCP, NTP, NetBIOS, SMB, LDAP, SSDP, mDNS, LLMNR and non-TCP/UDP. Ports decide only which flows are trustworthy; they are never features. Effect: 76,051 → 42,737 flows, VPN share 28% → 49%, flows of ≤ 2 packets 56% → 4%, no identical feature vector carries both labels. Raw files kept as `data/vnat_*_raw.csv`.

**Model.** FT-Transformer tuned by 20-trial random search on the 5-category task (VPN vs Non-VPN is saturated and cannot rank configurations). Best: d_model 32, 4 layers, 8 heads, FFN 128, dropout 0.1, lr 3e-4, batch 128 (validation accuracy 98.5%). Early stopping on validation loss (patience 10, max 60 epochs), AdamW, weight decay 1e-4. Baselines: Logistic Regression, Random Forest (300 trees), XGBoost (400 trees, depth 8), identical splits and preprocessing.

**Significance.** Unit = test capture file, its accuracy averaged over the seeds in which it was in the test split (flows from one recording are not independent). Paired Wilcoxon signed-rank test, Holm-corrected for 3 comparisons, α = 0.05; bootstrap 95% CI of the mean difference.

## 3. Accuracy (mean ± std over 5 seeds, %)

### VNAT, VPN vs Non-VPN (`results/vnat`, `results/vnat_noflags`, `results/vnat_iscx_features`)

| Model | NFStream (57) | NFStream, no TCP flags (33) | Timing only (23) |
|---|---|---|---|
| Logistic Regression | 99.99 ± 0.02 | 99.99 ± 0.03 | 93.53 ± 1.34 |
| Random Forest | 100.00 ± 0.00 | 100.00 ± 0.00 | 98.89 ± 0.54 |
| XGBoost | 100.00 ± 0.00 | 100.00 ± 0.00 | **99.25 ± 0.37** |
| FT-Transformer | 100.00 ± 0.00 | 100.00 ± 0.00 | 98.49 ± 0.80 |

### VNAT, 5 categories: Streaming, VoIP, Chat, C2, File Transfer (`results/vnat_category`)

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 84.15 ± 10.31 | 71.10 ± 13.25 | 96.14 ± 2.67 |
| Random Forest | **93.30 ± 5.94** | **87.90 ± 4.75** | 98.94 ± 0.55 |
| XGBoost | 91.88 ± 8.35 | 85.05 ± 9.18 | 99.58 ± 0.24 |
| FT-Transformer | 92.79 ± 5.83 | 85.18 ± 8.89 | 99.19 ± 0.44 |

The large ± comes from the file-level split: VoIP has 6 capture files, so one test recording swings every model together. The paired test below removes this.

### ISCX 2016, VPN vs Non-VPN (`results/summary.md`)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 67.32 ± 0.76 | 72.73 ± 0.49 |
| Random Forest | **92.40 ± 0.57** | 97.62 ± 0.30 |
| FT-Transformer | 90.83 ± 0.31 | 97.02 ± 0.19 |

### Is the difference real? FT-Transformer minus baseline, per test capture file

| Task | vs Logistic Regression | vs Random Forest | vs XGBoost |
|---|---|---|---|
| VPN, NFStream (84 files) | +0.05 pp, n.s. | 0.00, n.s. | 0.00, n.s. |
| VPN, no TCP flags (84 files) | +0.04 pp, n.s. | 0.00, n.s. | 0.00, n.s. |
| VPN, timing only (84 files) | **+5.20 pp, better** (p < 0.001) | −0.75 pp [−1.38, −0.18], n.s. (p = 0.68) | **−1.36 pp, worse** (p = 0.001) |
| 5 categories (87 files) | **+6.92 pp, better** (p < 0.001) | −0.25 pp [−2.09, +1.79], n.s. | +0.00 pp [−2.13, +2.37], n.s. |

n.s. = not significant after Holm correction. Against Random Forest on timing features the mean is negative, but the FT-Transformer is better on more files (33 vs 25): it loses large on a few files and wins small on many, which the rank test does not call.

### Cost (VNAT, mean over 5 seeds)

| Task | Model | Train time | Inference | Size |
|---|---|---|---|---|
| Timing only | Random Forest | 1.7 s | 17.9 µs/flow | 463k tree nodes |
| | XGBoost | 1.5 s | 1.1 µs/flow | 30k tree nodes |
| | FT-Transformer | 301 s | 10.8 µs/flow | 52k parameters |
| 5 categories | Random Forest | 1.5 s | 18.4 µs/flow | 498k tree nodes |
| | XGBoost | 8.9 s | 3.4 µs/flow | 104k tree nodes |
| | FT-Transformer | 254 s | 24.0 µs/flow | 55k parameters |

### ROC curves, confusion matrices and per-class TP / FP / FN / TN (`results/*/figures`)

Each task folder has `figures/roc_curves.png`, `pr_curves.png` (precision-recall), `confusion_matrices.png` and `metrics.md` / `per_class_metrics.csv` (TP, FP, FN, TN, precision, recall, F1, false-positive rate per class and model). Created by `metrics_figures.py`.

**These come from one split (seed 42), the one behind the confusion matrices in each `summary.md`. The models, preprocessing and split are those of the comparison, and the accuracies match the seed-42 row of `per_seed_metrics.csv` exactly.** They illustrate the error types. The mean ± std tables and the paired tests above are the results to cite.

ROC-AUC and average precision on this split (multi-class: one-vs-rest, macro):

| Task | Logistic Regression | Random Forest | XGBoost | FT-Transformer |
|---|---|---|---|---|
| VPN, NFStream / no TCP flags | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| VPN, timing only | 0.9470 / 0.9038 | 0.9994 / 0.9993 | 0.9998 / 0.9997 | 0.9990 / 0.9987 |
| 5 categories | 0.9690 / 0.8496 | 0.9924 / 0.9574 | 0.9926 / 0.9551 | 0.9889 / 0.9371 |

**VPN, NFStream (with or without TCP flags):** 0 false positives and 0 false negatives for every model (2,516 VPN, 3,138 Non-VPN test flows).

**VPN, timing only, VPN = positive class:**

| Model | TP | FN (VPN missed) | FP (false alarm) | TN | VPN recall | VPN precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 2319 | 197 | 257 | 2879 | 92.17 | 90.02 |
| Random Forest | 2478 | 38 | 22 | 3114 | 98.49 | 99.12 |
| XGBoost | 2485 | 31 | 10 | 3126 | 98.77 | 99.60 |
| FT-Transformer | 2417 | 99 | 18 | 3118 | 96.07 | 99.26 |

On this split the FT-Transformer's errors are mostly missed VPN flows rather than false alarms.

**5 categories, recall per class (%), seed-42 split:**

| Class (test flows) | Logistic Regression | Random Forest | XGBoost | FT-Transformer |
|---|---|---|---|---|
| C2 (1359) | 24.87 | 35.32 | 12.21 | 92.57 |
| Chat (3417) | 97.54 | 99.18 | 99.24 | 99.12 |
| File Transfer (90) | 80.00 | 84.44 | 78.89 | 68.89 |
| Streaming (571) | 68.30 | 98.42 | 98.25 | 96.32 |
| VoIP (122) | 95.08 | 98.36 | 98.36 | 99.18 |

On this split the tree models label most C2 flows as Chat, while the FT-Transformer does not; the FT-Transformer is weakest on File Transfer (the smallest class, 90 flows), mostly confused with Streaming. This split favours the FT-Transformer (96.7% accuracy vs 83.2% for Random Forest); over all 5 seeds the two are not significantly different (table above). The ROC-AUCs also show that the tree models rank C2 well and lose accuracy at the decision step, so their C2 errors are a threshold problem more than a separability problem.

## 4. Explainability (`results/*/explain`)

Deletion test: each model is tested with its own explanation. Features are removed (set to the training mean) in the order the explanation ranks them; a faithful explanation makes confidence fall faster than removing features at random. Gap = AUC(random) − AUC(explanation order) over the first 10 removals; higher is better. 500 sampled test flows.

| Setting | Attention gap | TreeSHAP gap | More faithful | Top-1 removal: attention | Top-1 removal: TreeSHAP | Cost: attention | Cost: TreeSHAP |
|---|---|---|---|---|---|---|---|
| ISCX 2016, VPN | **22.6** | 13.1 | attention | 88.4 → 62.7% | 82.7 → 70.8% | 0.14 ms | 146 ms |
| VNAT, VPN, 57 features | **30.5** | 26.6 | attention | 99.6 → 97.9% | 99.0 → 88.6% | 0.32 ms | 0.55 ms |
| VNAT, VPN, no TCP flags | −1.0 | **29.2** | TreeSHAP | 99.98 → 99.98% | 98.6 → 84.9% | 0.15 ms | 0.78 ms |
| VNAT, 5 categories | 10.5 | **15.6** | TreeSHAP | 93.1 → 87.6% | 85.8 → 83.4% | 0.38 ms | 161 ms |
| VNAT, VPN, timing only | 8.0 | **19.6** | TreeSHAP | **93.1 → 54.0%** | 87.1 → 62.6% | 0.09 ms | 127 ms |

(Top-1 columns: confidence after removing one *random* feature → after removing the explanation's top feature.)

Reading:
- **Cost.** Attention costs 0.09–0.38 ms per flow everywhere. TreeSHAP's cost grows with the forest: 0.55–0.78 ms on the saturated VPN task (small forests), 127–161 ms on the non-trivial tasks (≈ 460k–500k tree nodes). On non-trivial tasks TreeSHAP explains ~6–8 flows/s; attention explains thousands.
- **Faithfulness.** Attention passes in 4 of 5 settings (its curve falls faster than random). It fails only on the no-flag model, which is saturated (99.99% confidence) with several redundant size features, so removing one or two changes nothing. TreeSHAP's full ranking is more faithful in 3 of 5 settings, while attention's single top feature is the most decisive of all on the timing-only task (−39 pp).
- **Caveat.** Each explanation is evaluated on its own model, so the gap also reflects how each model spreads its reliance across features, not only explanation quality.
- **What the models rely on.** VPN with NFStream features: smallest packet size, in every method. VPN with timing features: data rate (`flowBytesPerSecond`), then inter-arrival times; all methods agree (Spearman 0.70). Categories: packet-size variation and inter-arrival statistics.

**Supported claim:** attention yields per-flow explanations that pass a faithfulness test in most settings, as a by-product of inference, at a cost that makes explaining live traffic feasible, which TreeSHAP cannot do on non-trivial tasks. TreeSHAP is the more precise ranking when time allows.

## 5. Robustness to feature perturbation (test accuracy, %)

| Task | Model | Noise σ = 0.25 | Noise σ = 0.5 | 30% of values missing |
|---|---|---|---|---|
| VPN, timing only | Random Forest | 58.7 ± 4.3 | 51.7 ± 5.6 | **75.4 ± 3.0** |
| | XGBoost | 64.0 ± 1.8 | 56.4 ± 2.5 | 76.3 ± 1.1 |
| | FT-Transformer | 63.2 ± 2.8 | 56.2 ± 2.3 | 67.9 ± 6.2 |
| 5 categories | Random Forest | 88.3 ± 5.4 | **77.9 ± 7.4** | **88.2 ± 5.6** |
| | XGBoost | 81.5 ± 4.8 | 64.3 ± 5.8 | 86.8 ± 7.4 |
| | FT-Transformer | 85.6 ± 4.9 | 68.6 ± 6.8 | 77.4 ± 7.0 |

The FT-Transformer is comparable under Gaussian noise but consistently weaker than the tree models when feature values are missing (as on ISCX).

## 6. Adversarial evasion (`results/*/security`)

Evasion rate = share of VPN flows the model detected that are classified Non-VPN after the attack. Traffic shaping (delay, padding) is simulated on flow features. Feature-space attacks edit standardized features directly within an L∞ budget ε: an upper bound on attacker power, not physically constrained.

### VNAT, NFStream features (`results/vnat/security`)

- Traffic shaping: **0% evasion** for every model at every budget (up to 5× slowdown + 100% padding). Delay and padding cannot make packets smaller, and the smallest packet size identifies the tunnel.
- Feature-space, black-box random search:

| ε | 0.25 | 0.5 | 1.0 |
|---|---|---|---|
| Random Forest | 29.5 ± 20.8 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| FT-Transformer | **4.8 ± 6.1** | **78.7 ± 26.5** | 100.0 ± 0.0 |
| FT-Transformer, white-box PGD | 3.5 ± 5.0 | 47.9 ± 34.7 | 83.7 ± 11.8 |

### VNAT, timing features only (`results/vnat_iscx_features/security`)

Clean accuracy: FT-Transformer 98.4 ± 0.7, Random Forest 98.9 ± 0.5; shaping-augmented 98.5 ± 0.4 and 98.7 ± 0.5.

Delay only (evasion, %):

| Slowdown budget | 1.1× | 1.5× | 2× | 3× | 5× |
|---|---|---|---|---|---|
| Random Forest | 15.1 ± 7.2 | 52.0 ± 5.2 | 79.3 ± 2.5 | 96.5 ± 1.1 | 99.4 ± 0.8 |
| **FT-Transformer** | **0.1 ± 0.1** | **0.6 ± 0.3** | **7.7 ± 7.8** | **20.7 ± 13.3** | **27.1 ± 17.6** |
| Random Forest, shaping-augmented | 35.5 ± 10.8 | 72.5 ± 16.2 | 73.9 ± 16.5 | 75.3 ± 15.8 | 90.2 ± 3.5 |
| FT-Transformer, shaping-augmented | 20.2 ± 16.5 | 71.5 ± 16.3 | 73.7 ± 16.7 | 77.9 ± 14.2 | 81.6 ± 14.6 |

Delay + up to 100% padding: FT-Transformer 5.1 ± 4.8 / 6.0 ± 5.1 / 13.5 ± 9.7 / 26.7 ± 11.3 at 1.1× / 1.5× / 2× / 3×; Random Forest 15.3 ± 7.5 / 52.1 ± 5.2 / 79.8 ± 2.6 / 96.7 ± 1.2.

Feature-space attacks:

| ε | 0.05 | 0.1 | 0.25 |
|---|---|---|---|
| Random Forest, black-box | 90.9 ± 5.4 | 97.0 ± 2.3 | 99.9 ± 0.1 |
| FT-Transformer, black-box | **73.7 ± 16.2** | **90.5 ± 8.9** | 98.8 ± 0.9 |
| FT-Transformer, white-box PGD | 51.6 ± 16.5 | 61.9 ± 10.6 | 91.4 ± 8.4 |

Reading:
- **The standard FT-Transformer resists traffic shaping far better than Random Forest**: at a 2× slowdown 7.7% of detected VPN flows evade it vs 79.3% for Random Forest, at 1.5× 0.6% vs 52.0%. This repeats the ISCX result with a larger margin and is the strongest evidence for the FT-Transformer in this study.
- **The shaping-augmentation defense backfires on VNAT.** It helped on ISCX (2× evasion 21.8% → 2.7%), but on VNAT it makes both models much easier to evade (FT-Transformer at 1.1×: 0.1% → 20.2%) without changing clean accuracy. Untested hypothesis: data rate is the main VPN cue on VNAT, and shaped copies of Non-VPN flows teach the model that slowed traffic is Non-VPN, so a slowed VPN flow falls among them. The defense is therefore dataset-dependent and should not be presented as a general fix.
- **Feature-space attacks defeat both models at small budgets** on timing features; the FT-Transformer is somewhat harder (73.7% vs 90.9% at ε = 0.05). Timing-feature classifiers are not robust to an attacker who can shape features freely.

### ISCX 2016 (`results/security`)

| Slowdown budget | 1.5× | 2× | 3× |
|---|---|---|---|
| Random Forest | 34.6 ± 0.8 | 39.7 ± 0.4 | 50.0 ± 1.7 |
| FT-Transformer | **12.8 ± 2.2** | **21.8 ± 2.0** | **34.4 ± 2.6** |
| Random Forest, shaping-augmented | 10.2 ± 0.7 | 13.4 ± 0.8 | 17.8 ± 1.4 |
| FT-Transformer, shaping-augmented | **1.6 ± 0.3** | **2.7 ± 0.4** | **5.5 ± 1.2** |

(Delay-only evasion, %.)

## 7. Cross-dataset generalization (timing features, 3 seeds; `results/cross_dataset_*`)

| | Same dataset | ISCX → VNAT | VNAT → ISCX |
|---|---|---|---|
| Random Forest | 92.1 / 99.2 | 53.2 ± 2.9 (AUC 58.0) | 55.9 ± 0.9 (AUC 54.6) |
| XGBoost | 92.9 / 99.5 | 51.8 ± 1.3 (AUC 49.9) | 54.8 ± 0.8 (AUC 54.3) |
| FT-Transformer | 90.5 / 99.0 | 56.2 ± 7.6 (AUC 65.2) | 50.5 ± 2.3 (AUC 51.7) |

Every model collapses to near chance in both directions. The FT-Transformer is best in one direction and worst in the other, so no model generalizes better. Timing statistics learned on one network and VPN setup do not transfer.

## 8. Live classification

`predict_live.py` captures from an interface (NFStream + Npcap, no admin rights needed), skips background flows, and prints a verdict and plain-language reason per flow 0–15 s after it ends. Tested on the lab Wi-Fi.

- The full NFStream VPN model explained live decisions mostly with TCP-flag counts ("the number of SYN-flag packets was 2"). In VNAT every VPN flow is UDP, so flags encode "not OpenVPN-over-UDP": a protocol shortcut that would miss any VPN over TCP.
- The **no-flag model** (`models/vnat_noflags_model.pth`) is equally accurate on VNAT (100%) and explains live decisions with packet sizes ("the smallest packet size was 54 bytes, typical of Non-VPN traffic; VPN median 154 bytes"). **Use it for live capture.**
- Live accuracy on other networks and VPN types is **not measured**. Given §7, expect it to be weak for VPNs other than OpenVPN over UDP. `score_captures.py` measures it on your own labelled recordings (VPN on / off); pass `--vpn-server` so flows that bypass the tunnel are not scored as VPN.

## 9. Limitations

1. VNAT's VPN traffic is one VPN (OpenVPN over UDP 1195) on one lab network; 100% VPN accuracy reflects that, not general VPN detection.
2. No model generalizes across datasets (§7).
3. Category results rest on few capture files per class (VoIP 6, Streaming 12, File Transfer 16); the ± of 6–10 pp is real uncertainty.
4. Hyperparameters were chosen on seed 42's validation split, and seed 42 is one of the 5 reported seeds (small optimistic bias for the FT-Transformer).
5. Traffic shaping is simulated on features, not re-captured packets. Feature-space attacks ignore physical feasibility.
6. The deletion test is uninformative on saturated models (≈100% confidence, redundant features).
7. Background flows (DNS etc.) are excluded from training and evaluation; a deployment must filter them the same way (the live tool does).
8. The shaping-augmentation defense helps on ISCX and hurts on VNAT; its effect is dataset-dependent and its mechanism on VNAT is a hypothesis, not tested.
9. Security results on timing features have large seed-to-seed spread (± up to 17 pp); cite means with their ±.
10. The NFStream features include per-direction byte and packet counts; on other datasets these may correlate with the application or capture setup rather than with VPN use.

## 10. Reproduce

```
powershell -ExecutionPolicy Bypass -File .\run_vnat.ps1 -VnatDir D:\Vnat -Python python
```
Re-running skips finished steps; delete an output to redo a step. Every table above is in the `summary.md` of the folder named with it.

ROC / precision-recall curves, confusion-matrix heatmaps and per-class TP/FP/FN/TN (about 5 min per task; add `--replot` to redraw from the saved probabilities without training):
```
python metrics_figures.py --data data\vnat_nfstream.csv --group-col source_file --label-col category --config results\tuning\best_config.json --out results\vnat_category
```
Other tasks: drop `--label-col category`; add `--feature-set nfstream_noflags` (`--out results\vnat_noflags`), or use `--data data\vnat_features.csv` (`--out results\vnat_iscx_features`).
