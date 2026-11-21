"""Baseline comparison: Logistic Regression, Random Forest and XGBoost vs. FT-Transformer.

Every model sees the same splits, the same preprocessing and the same test set.
The experiment is repeated over several seeds and reported as mean +/- std.

Usage:
    python compare_models.py                                   # ISCX 2016
    python compare_models.py --data data/vnat_features.csv --group-col source_file --out results/vnat
    python compare_models.py --data data/vnat_features.csv --group-col source_file \
        --label-col category --out results/vnat_category       # multi-class
    add --config results/tuning/best_config.json to use tuned FT-Transformer settings

Outputs (in --out, default results/):
    per_seed_metrics.csv     raw metrics for every model and seed
    summary.md               result tables + confusion matrices
"""
# ==========================================
# 1. Imports
# ==========================================
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from utils.preprocessing import (LABEL_COL, fit_standardizer, prepare_dataset,
                                 split_indices, standardize)
from utils.training import device, predict_proba, train_ft_transformer

SEEDS = [0, 1, 2, 3, 42]
OUT_DIR = "results"
DATASET_NAME = "ISCX time-based features"

# Robustness: perturbations applied to the (standardized) test features only,
# imitating feature drift when flows come from a different capture / extractor.
NOISE_LEVELS = [0.1, 0.25, 0.5]
MISSING_RATES = [0.1, 0.3]

# ==========================================
# 2. Models
# ==========================================
class SklearnModel:
    def __init__(self, make):
        self.make = make

    def fit(self, X_train, y_train, X_val, y_val, num_classes, seed):
        # Classical baselines do not need a validation set; train on train only
        # so every model sees exactly the same training data.
        self.model = self.make(seed).fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def num_params(self):
        m = self.model
        if hasattr(m, "coef_"):
            return m.coef_.size + m.intercept_.size
        if hasattr(m, "get_booster"):  # XGBoost: total tree nodes
            return sum(len(t.strip().splitlines()) for t in m.get_booster().get_dump())
        return sum(est.tree_.node_count for est in m.estimators_)  # tree nodes


class FTTransformerModel:
    tuned = {}  # filled from --config

    def fit(self, X_train, y_train, X_val, y_val, num_classes, seed):
        t = self.tuned
        self.model = train_ft_transformer(
            X_train, y_train, X_val, y_val, num_classes, seed=seed, verbose=False,
            config=t.get("model_config"), **{k: t[k] for k in ("lr", "batch_size") if k in t})
        return self

    def predict_proba(self, X):
        return predict_proba(self.model, X)

    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())


MODELS = {
    "Logistic Regression": lambda: SklearnModel(
        lambda s: LogisticRegression(max_iter=2000, random_state=s)),
    "Random Forest": lambda: SklearnModel(
        lambda s: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s)),
    "XGBoost": lambda: SklearnModel(
        lambda s: XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                                tree_method="hist", device=device.type, random_state=s)),
    "FT-Transformer": FTTransformerModel,
}

# ==========================================
# 3. Experiment
# ==========================================
def metrics(y_true, proba, classes):
    pred = proba.argmax(axis=1)
    labels = list(range(len(classes)))
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, labels=labels, average="macro", zero_division=0),
    }
    if classes == ["Non-VPN", "VPN"]:
        vpn = classes.index("VPN")
        out["vpn_precision"] = precision_score(y_true, pred, pos_label=vpn, zero_division=0)
        out["vpn_recall"] = recall_score(y_true, pred, pos_label=vpn, zero_division=0)
        out["roc_auc"] = roc_auc_score(y_true, proba[:, vpn])
    else:
        out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    return out


def robustness(model, X_test, y_test, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for sigma in NOISE_LEVELS:
        X_noisy = X_test + rng.normal(0, sigma, X_test.shape).astype(np.float32)
        out[f"acc_noise_{sigma}"] = accuracy_score(y_test, model.predict_proba(X_noisy).argmax(1))
    for rate in MISSING_RATES:
        # A dropped feature is replaced with the training mean (0 after standardizing).
        X_missing = np.where(rng.random(X_test.shape) < rate, 0, X_test).astype(np.float32)
        out[f"acc_missing_{rate}"] = accuracy_score(y_test, model.predict_proba(X_missing).argmax(1))
    return out


def main():
    global OUT_DIR, DATASET_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/vpn_dataset_cleaned.csv")
    parser.add_argument("--group-col", default=None,
                        help="split by this column (e.g. source_file) instead of by flow")
    parser.add_argument("--label-col", default=LABEL_COL,
                        help="class1 (VPN / Non-VPN, default), or category / app for VNAT")
    parser.add_argument("--config", default=None, help="tuned FT-Transformer settings (tune.py)")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out
    if args.data != parser.get_default("data"):
        DATASET_NAME = os.path.basename(args.data)
    if args.label_col != LABEL_COL:
        DATASET_NAME += f", label: {args.label_col}"
    if args.config:
        FTTransformerModel.tuned = json.load(open(args.config))
    os.makedirs(OUT_DIR, exist_ok=True)

    data = prepare_dataset(args.data, args.label_col)
    classes, y, Z = data["classes"], data["y"], data["Z"]
    groups = data["df"][args.group_col].to_numpy() if args.group_col else None

    print(f"[INFO] {len(Z)} flows, {Z.shape[1]} {data['feature_set']} features, "
          f"{len(classes)} classes, seeds: {SEEDS}")

    rows, confusions = [], {}

    for seed in SEEDS:
        tr, va, te = split_indices(y, seed=seed, groups=groups)
        mean, std = fit_standardizer(Z[tr])
        X_train, X_val, X_test = (standardize(Z[i], mean, std) for i in (tr, va, te))
        y_train, y_val, y_test = y[tr], y[va], y[te]

        for name, make in MODELS.items():
            model = make()

            start = time.perf_counter()
            model.fit(X_train, y_train, X_val, y_val, len(classes), seed)
            train_time = time.perf_counter() - start

            start = time.perf_counter()
            proba = model.predict_proba(X_test)
            infer_us = (time.perf_counter() - start) / len(X_test) * 1e6

            row = {"model": name, "seed": seed,
                   **metrics(y_test, proba, classes),
                   **robustness(model, X_test, y_test, seed),
                   "train_time_s": train_time,
                   "inference_us_per_flow": infer_us,
                   "size": model.num_params()}
            rows.append(row)

            if seed == SEEDS[-1]:
                confusions[name] = confusion_matrix(y_test, proba.argmax(1), labels=range(len(classes)))

            print(f"[seed {seed}] {name:20s} acc {row['accuracy']:.4f}  f1 {row['f1_macro']:.4f}  "
                  f"auc {row['roc_auc']:.4f}  ({train_time:.1f}s)")

    results = pd.DataFrame(rows)
    results.to_csv(f"{OUT_DIR}/per_seed_metrics.csv", index=False)
    write_summary(results, confusions, classes)

# ==========================================
# 4. Reporting
# ==========================================
def table(results, columns, fmt):
    order = list(MODELS)
    g = results.groupby("model")[columns]
    mean, std = g.mean().loc[order], g.std().loc[order]
    cells = mean.apply(lambda c: [fmt(m, s) for m, s in zip(c, std[c.name])])
    return cells.to_markdown()


def write_summary(results, confusions, classes):
    pct = lambda m, s: f"{m*100:.2f} ± {s*100:.2f}"

    main_cols = [c for c in ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                             "vpn_precision", "vpn_recall", "roc_auc"] if c in results.columns]
    robust_cols = [c for c in results.columns if c.startswith("acc_")]
    cost_cols = ["train_time_s", "inference_us_per_flow", "size"]

    parts = [
        f"# Model comparison ({DATASET_NAME})",
        f"Mean ± std over {len(SEEDS)} seeds (70/15/15 split per seed). Values in %.",
        "## Test-set performance",
        table(results, main_cols, pct),
        "## Robustness (test accuracy under perturbed features)",
        "`noise_σ`: Gaussian noise added to standardized features. "
        "`missing_r`: fraction of feature values replaced by the training mean.",
        "Clean accuracy for reference is in the table above.",
        table(results, robust_cols, pct),
        "## Cost",
        "`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF, XGBoost).",
        table(results, cost_cols, lambda m, s: f"{m:,.1f} ± {s:,.1f}"),
        f"## Confusion matrices (seed {SEEDS[-1]}, rows = true, cols = predicted: {classes})",
    ]
    for name, cm in confusions.items():
        parts.append(f"**{name}**\n\n" + pd.DataFrame(cm, index=classes, columns=classes).to_markdown())

    text = "\n\n".join(parts) + "\n"
    with open(f"{OUT_DIR}/summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text.encode("ascii", "replace").decode())  # Windows consoles choke on ± / σ
    print(f"[INFO] Full summary (with symbols) saved to {OUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
