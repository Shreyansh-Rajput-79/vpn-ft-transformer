"""Cross-dataset generalization: train on one dataset, test on another.

Both CSVs must use the same feature set, so for ISCX <-> VNAT build the VNAT
features with the ISCX extractor (build_vnat_dataset.py default). Labels are
VPN / Non-VPN.

Usage:
    python cross_dataset.py --train data/vpn_dataset_cleaned.csv --test data/vnat_features.csv
    python cross_dataset.py --train data/vnat_features.csv --test data/vpn_dataset_cleaned.csv \\
        --out results/cross_dataset_vnat_to_iscx
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from utils.preprocessing import (fit_standardizer, prepare_dataset,
                                 split_indices, standardize)
from utils.training import device, predict_proba, train_ft_transformer

SEEDS = [0, 1, 2]


def scores(y, proba, vpn):
    pred = proba.argmax(1)
    return {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
            "roc_auc": roc_auc_score(y, proba[:, vpn])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="results/cross_dataset")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    src, dst = prepare_dataset(args.train), prepare_dataset(args.test)
    if src["feature_set"] != dst["feature_set"]:
        raise SystemExit(f"[ERROR] Feature sets differ: {src['feature_set']} vs {dst['feature_set']}. "
                         "Build both with the same extractor.")
    if src["classes"] != ["Non-VPN", "VPN"] or dst["classes"] != ["Non-VPN", "VPN"]:
        raise SystemExit("[ERROR] Both datasets need VPN / Non-VPN labels.")
    vpn = 1
    print(f"[INFO] train {args.train}: {len(src['Z'])} flows | test {args.test}: {len(dst['Z'])} flows")

    rows = []
    for seed in SEEDS:
        tr, va, te = split_indices(src["y"], seed=seed)
        mean, std = fit_standardizer(src["Z"][tr])
        X_tr, X_va, X_te = (standardize(src["Z"][i], mean, std) for i in (tr, va, te))
        X_dst = standardize(dst["Z"], mean, std)

        ft = train_ft_transformer(X_tr, src["y"][tr], X_va, src["y"][va], 2, seed=seed, verbose=False)
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
            .fit(X_tr, src["y"][tr]).predict_proba,
            "XGBoost": XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, tree_method="hist",
                                     device=device.type, random_state=seed)
            .fit(X_tr, src["y"][tr]).predict_proba,
            "FT-Transformer": lambda X: predict_proba(ft, X),
        }
        for name, proba in models.items():
            rows.append({"seed": seed, "model": name, "evaluated_on": "same dataset (test split)",
                         **scores(src["y"][te], proba(X_te), vpn)})
            rows.append({"seed": seed, "model": name, "evaluated_on": "other dataset (all flows)",
                         **scores(dst["y"], proba(X_dst), vpn)})
            print(f"[seed {seed}] {name:15s} in-domain acc {rows[-2]['accuracy']:.4f} | "
                  f"cross-dataset acc {rows[-1]['accuracy']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}/per_seed.csv", index=False)
    fmt = lambda s: f"{s.mean()*100:.2f} ± {s.std()*100:.2f}"
    table = df.groupby(["evaluated_on", "model"], sort=False)[["accuracy", "f1_macro", "roc_auc"]].agg(fmt)
    text = (f"# Cross-dataset generalization\n\nTrained on `{args.train}`, tested on `{args.test}`. "
            f"Mean ± std over seeds {SEEDS}, in %.\n\n{table.to_markdown()}\n\n"
            "The drop from the first block to the second is the generalization gap.\n")
    with open(f"{args.out}/summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
