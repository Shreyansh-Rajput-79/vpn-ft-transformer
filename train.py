# ==========================================
# 1. Imports
# ==========================================
import argparse
import json

import torch
from sklearn.metrics import classification_report, confusion_matrix

from utils.narrate import class_medians
from utils.preprocessing import (LABEL_COL, fit_standardizer, prepare_dataset,
                                 split_indices, standardize)
from utils.training import (BATCH_SIZE, EPOCHS, LR, MODEL_CONFIG, device,
                            predict_proba, train_ft_transformer)

SEED = 42

parser = argparse.ArgumentParser(description="Train the FT-Transformer traffic classifier.")
parser.add_argument("--data", default="data/vpn_dataset_cleaned.csv",
                    help="ISCX CSV (default) or the output of build_vnat_dataset.py")
parser.add_argument("--label-col", default=LABEL_COL,
                    help="class1 (VPN / Non-VPN, default), or category / app for VNAT")
parser.add_argument("--group-col", default=None,
                    help="split by this column (e.g. source_file) so flows from one capture "
                         "never appear in both train and test")
parser.add_argument("--config", default=None,
                    help="JSON with model hyperparameters, e.g. results/tuning/best_config.json")
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--model-out", default="models/model.pth")
args = parser.parse_args()

# Hyperparameters: defaults, overridden by --config (from tune.py), then CLI flags.
config, lr, batch_size = dict(MODEL_CONFIG), LR, BATCH_SIZE
if args.config:
    tuned = json.load(open(args.config))
    config.update(tuned.get("model_config", {}))
    lr, batch_size = tuned.get("lr", lr), tuned.get("batch_size", batch_size)
batch_size = args.batch_size or batch_size

# ==========================================
# 2. Load Dataset
# ==========================================
print("[INFO] Loading dataset...")

data = prepare_dataset(args.data, args.label_col)
classes, y, Z = data["classes"], data["y"], data["Z"]

print(f"[INFO] {len(Z)} unique flows, {Z.shape[1]} {data['feature_set']} features, "
      f"classes: {classes}, device: {device}")

# ==========================================
# 3. Train / Validation / Test Split (70 / 15 / 15)
# ==========================================
groups = data["df"][args.group_col].to_numpy() if args.group_col else None
tr, va, te = split_indices(y, seed=SEED, groups=groups)

# Scaling statistics come from the training split only.
mean, std = fit_standardizer(Z[tr])
X_train, X_val, X_test = (standardize(Z[i], mean, std) for i in (tr, va, te))
y_train, y_val, y_test = y[tr], y[va], y[te]

print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ==========================================
# 4. Training (early stopping on validation loss)
# ==========================================
print(f"[INFO] Training started... config={config}, lr={lr}, batch_size={batch_size}")

model = train_ft_transformer(X_train, y_train, X_val, y_val, len(classes), seed=SEED,
                             config=config, lr=lr, epochs=args.epochs, batch_size=batch_size)

# ==========================================
# 5. Test Set Evaluation
# ==========================================
test_preds = predict_proba(model, X_test).argmax(axis=1)

print(f"\n[RESULT] Test accuracy: {(test_preds == y_test).mean():.4f}")
print(classification_report(y_test, test_preds, labels=range(len(classes)),
                            target_names=classes, digits=4, zero_division=0))
print("Confusion matrix (rows = true, cols = predicted):", classes)
print(confusion_matrix(y_test, test_preds, labels=range(len(classes))))

# ==========================================
# 6. Save Model
# ==========================================
# Only tensors / plain Python types, so it loads with torch.load(weights_only=True).
torch.save({
    "model": {k: v.cpu() for k, v in model.state_dict().items()},
    "model_config": config,
    "features": data["features"],
    "feature_set": data["feature_set"],
    "label_col": args.label_col,
    "classes": classes,
    "mean": mean.tolist(),
    "std": std.tolist(),
    # typical value of each feature per class, for plain-language explanations
    "class_medians": class_medians(Z[tr], y_train, len(classes)),
}, args.model_out)

print(f"[INFO] Model saved to {args.model_out}")
