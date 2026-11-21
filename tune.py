"""Random hyperparameter search for the FT-Transformer (run on the GPU machine).

Every trial trains on the seed-42 training split and is scored on the
validation split only, so the test split stays untouched for the final report.

Usage:
    python tune.py --data data/vnat_features.csv --group-col source_file --trials 20
    python train.py ... --config results/tuning/best_config.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from utils.preprocessing import (LABEL_COL, fit_standardizer, prepare_dataset,
                                 split_indices, standardize)
from utils.training import predict_proba, train_ft_transformer

SEED = 42
SPACE = {
    "d_model": [32, 64, 128],
    "n_layers": [2, 3, 4],
    "n_heads": [4, 8],
    "ffn_factor": [2, 4],
    "dropout": [0.0, 0.1, 0.2],
    "lr": [1e-4, 3e-4, 1e-3],
    "batch_size": [128, 256, 512],
}


def sample(rng):
    p = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
    config = dict(d_model=int(p["d_model"]), n_heads=int(p["n_heads"]), n_layers=int(p["n_layers"]),
                  dim_feedforward=int(p["d_model"] * p["ffn_factor"]), dropout=float(p["dropout"]))
    return config, float(p["lr"]), int(p["batch_size"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/vpn_dataset_cleaned.csv")
    parser.add_argument("--label-col", default=LABEL_COL)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", default="results/tuning")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = prepare_dataset(args.data, args.label_col)
    y, Z = data["y"], data["Z"]
    groups = data["df"][args.group_col].to_numpy() if args.group_col else None
    tr, va, _ = split_indices(y, seed=SEED, groups=groups)
    mean, std = fit_standardizer(Z[tr])
    X_train, X_val = standardize(Z[tr], mean, std), standardize(Z[va], mean, std)

    rng = np.random.default_rng(SEED)
    rows = []
    for trial in range(args.trials):
        config, lr, batch_size = sample(rng)
        start = time.perf_counter()
        model = train_ft_transformer(X_train, y[tr], X_val, y[va], len(data["classes"]), seed=SEED,
                                     verbose=False, config=config, lr=lr, epochs=args.epochs,
                                     batch_size=batch_size)
        val_acc = (predict_proba(model, X_val).argmax(1) == y[va]).mean()
        rows.append({**config, "lr": lr, "batch_size": batch_size, "val_loss": model.best_val_loss,
                     "val_accuracy": val_acc, "minutes": (time.perf_counter() - start) / 60})
        print(f"[trial {trial+1}/{args.trials}] val loss {model.best_val_loss:.4f} acc {val_acc:.4f} "
              f"| {config} lr={lr} bs={batch_size}")

    trials = pd.DataFrame(rows).sort_values("val_loss")
    trials.to_csv(f"{args.out}/trials.csv", index=False)
    best = trials.iloc[0]
    best_config = {
        "model_config": {k: (float(best[k]) if k == "dropout" else int(best[k]))
                         for k in ("d_model", "n_heads", "n_layers", "dim_feedforward", "dropout")},
        "lr": float(best.lr), "batch_size": int(best.batch_size),
        "val_loss": float(best.val_loss), "val_accuracy": float(best.val_accuracy),
    }
    with open(f"{args.out}/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n[RESULT] Best: {best_config}\n[INFO] Saved to {args.out}/best_config.json")


if __name__ == "__main__":
    main()
