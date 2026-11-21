# ==========================================
# 1. Imports
# ==========================================
import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.preprocessing import (FEATURES, fit_standardizer, load_dataset,
                                 split_by_group, standardize, to_matrix)
from utils.training import MODEL_CONFIG, device, predict_proba, train_ft_transformer

SEED = 42

parser = argparse.ArgumentParser(description="Train the FT-Transformer VPN classifier.")
parser.add_argument("--data", default="data/vpn_dataset_cleaned.csv",
                    help="ISCX CSV (default) or the output of build_vnat_dataset.py")
parser.add_argument("--group-col", default=None,
                    help="split by this column (e.g. source_file) so flows from one capture "
                         "never appear in both train and test")
parser.add_argument("--model-out", default="models/model.pth")
args = parser.parse_args()

# ==========================================
# 2. Load Dataset
# ==========================================
print("[INFO] Loading dataset...")

df, labels = load_dataset(args.data)
classes = sorted(labels.unique())
y = labels.map({c: i for i, c in enumerate(classes)}).to_numpy()
X = to_matrix(df)

print(f"[INFO] {len(X)} unique flows, {X.shape[1]} features, classes: {classes}, device: {device}")

# ==========================================
# 3. Train / Validation / Test Split (70 / 15 / 15)
# ==========================================
if args.group_col:
    tr, va, te = split_by_group(df[args.group_col].to_numpy(), y, seed=SEED)
    X_train, X_val, X_test = X[tr], X[va], X[te]
    y_train, y_val, y_test = y[tr], y[va], y[te]
else:
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

# Scaling statistics come from the training split only.
mean, std = fit_standardizer(X_train)
X_train, X_val, X_test = (standardize(a, mean, std) for a in (X_train, X_val, X_test))

print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ==========================================
# 4. Training (early stopping on validation loss)
# ==========================================
print("[INFO] Training started...")

model = train_ft_transformer(X_train, y_train, X_val, y_val, len(classes), seed=SEED)

# ==========================================
# 5. Test Set Evaluation
# ==========================================
test_preds = predict_proba(model, X_test).argmax(axis=1)

print(f"\n[RESULT] Test accuracy: {(test_preds == y_test).mean():.4f}")
print(classification_report(y_test, test_preds, target_names=classes, digits=4))
print("Confusion matrix (rows = true, cols = predicted):", classes)
print(confusion_matrix(y_test, test_preds))

# ==========================================
# 6. Save Model
# ==========================================
# Only tensors / plain Python types, so it loads with torch.load(weights_only=True).
torch.save({
    "model": model.state_dict(),
    "model_config": MODEL_CONFIG,
    "features": FEATURES,
    "classes": classes,
    "mean": mean.tolist(),
    "std": std.tolist(),
}, args.model_out)

print(f"[INFO] Model saved to {args.model_out}")
