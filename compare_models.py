"""Baseline comparison: Logistic Regression and Random Forest vs. FT-Transformer.

Every model sees the same splits, the same preprocessing and the same test set.
The experiment is repeated over several seeds and reported as mean +/- std.

Usage:
    python compare_models.py                                   # ISCX 2016
    python compare_models.py --data data/vnat_features.csv         --group-col source_file --out results/vnat              # VNAT, split by capture file

Outputs (in --out, default results/):
    per_seed_metrics.csv     raw metrics for every model and seed
    summary.md               result tables + confusion matrices
"""
# ==========================================
# 1. Imports
# ==========================================
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from utils.preprocessing import (fit_standardizer, load_dataset, split_by_group,
                                 standardize, to_matrix)
from utils.training import predict_proba, train_ft_transformer

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
        return sum(est.tree_.node_count for est in m.estimators_)  # tree nodes


class FTTransformerModel:
    def fit(self, X_train, y_train, X_val, y_val, num_classes, seed):
        self.model = train_ft_transformer(X_train, y_train, X_val, y_val,
                                          num_classes, seed=seed, verbose=False)
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
    "FT-Transformer": FTTransformerModel,
}

# ==========================================
# 3. Experiment
# ==========================================
def metrics(y_true, proba, positive):
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro"),
        "recall_macro": recall_score(y_true, pred, average="macro"),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "vpn_precision": precision_score(y_true, pred, pos_label=positive),
        "vpn_recall": recall_score(y_true, pred, pos_label=positive),
        "roc_auc": roc_auc_score(y_true, proba[:, positive]),
    }


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
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out
    if args.data != parser.get_default("data"):
        DATASET_NAME = os.path.basename(args.data)
    os.makedirs(OUT_DIR, exist_ok=True)

    df, labels = load_dataset(args.data)
    classes = sorted(labels.unique())
    y = labels.map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = to_matrix(df)
    vpn = classes.index("VPN")

    print(f"[INFO] {len(X)} flows, {X.shape[1]} features, seeds: {SEEDS}")

    rows, confusions = [], {}

    for seed in SEEDS:
        if args.group_col:
            tr, va, te = split_by_group(df[args.group_col].to_numpy(), y, seed=seed)
            X_train, X_val, X_test = X[tr], X[va], X[te]
            y_train, y_val, y_test = y[tr], y[va], y[te]
        else:
            X_train, X_tmp, y_train, y_tmp = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(
                X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
        mean, std = fit_standardizer(X_train)
        X_train, X_val, X_test = (standardize(a, mean, std) for a in (X_train, X_val, X_test))

        for name, make in MODELS.items():
            model = make()

            start = time.perf_counter()
            model.fit(X_train, y_train, X_val, y_val, len(classes), seed)
            train_time = time.perf_counter() - start

            start = time.perf_counter()
            proba = model.predict_proba(X_test)
            infer_us = (time.perf_counter() - start) / len(X_test) * 1e6

            row = {"model": name, "seed": seed,
                   **metrics(y_test, proba, vpn),
                   **robustness(model, X_test, y_test, seed),
                   "train_time_s": train_time,
                   "inference_us_per_flow": infer_us,
                   "size": model.num_params()}
            rows.append(row)

            if seed == SEEDS[-1]:
                confusions[name] = confusion_matrix(y_test, proba.argmax(1))

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

    main_cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
                 "vpn_precision", "vpn_recall", "roc_auc"]
    robust_cols = [c for c in results.columns if c.startswith("acc_")]
    cost_cols = ["train_time_s", "inference_us_per_flow", "size"]

    parts = [
        f"# Model comparison: VPN vs Non-VPN ({DATASET_NAME})",
        f"Mean ± std over {len(SEEDS)} seeds (70/15/15 split per seed). Values in %.",
        "## Test-set performance",
        table(results, main_cols, pct),
        "## Robustness (test accuracy under perturbed features)",
        "`noise_σ`: Gaussian noise added to standardized features. "
        "`missing_r`: fraction of feature values replaced by the training mean.",
        "Clean accuracy for reference is in the table above.",
        table(results, robust_cols, pct),
        "## Cost",
        "`size` = trainable parameters (LR, FT-Transformer) or total tree nodes (RF).",
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
