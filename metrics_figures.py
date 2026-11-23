"""ROC curves, confusion-matrix heatmaps and per-class TP/FP/FN/TN for one task.

Uses exactly the models, preprocessing and split of compare_models.py for its
last seed (42), the split behind the confusion matrices in summary.md, so the
numbers here match that table.

    python metrics_figures.py --data data/vnat_nfstream.csv --group-col source_file \
        --config results/tuning/best_config.json --out results/vnat

Outputs (in --out/figures):
    roc_curves.png          ROC per model (binary) or one-vs-rest per class (multi-class)
    pr_curves.png           precision-recall curves, same layout
    confusion_matrices.png  row-normalised heatmaps with counts
    per_class_metrics.csv   precision, recall, F1, support, TP/FP/FN/TN per model and class
    metrics.md              the same as tables
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)

import compare_models as cm
from utils.plotting import AQUA, BLUE, INK_2, ORANGE, SEQUENTIAL, YELLOW, plt
from utils.preprocessing import (FEATURE_SETS, LABEL_COL, fit_standardizer,
                                 prepare_dataset, split_indices, standardize)

SEED = cm.SEEDS[-1]
COLORS = {"Logistic Regression": YELLOW, "Random Forest": ORANGE, "XGBoost": AQUA,
          "FT-Transformer": BLUE}
# Drawn in this order, thickest first, so identical curves stay visible.
WIDTHS = {"Logistic Regression": 3.0, "Random Forest": 2.4, "XGBoost": 1.8, "FT-Transformer": 1.2}


def suptitle(fig, text, width_in):
    import textwrap
    fig.suptitle(textwrap.fill(text, max(30, int(width_in * 11))), x=0.01, ha="left", fontweight="bold")


def per_class_table(name, y, pred, classes):
    m = confusion_matrix(y, pred, labels=range(len(classes)))
    rows = []
    for k, c in enumerate(classes):
        tp = m[k, k]
        fn = m[k].sum() - tp
        fp = m[:, k].sum() - tp
        tn = m.sum() - tp - fn - fp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"model": name, "class": c, "support": int(m[k].sum()),
                     "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                     "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0,
                     "FPR": fp / (fp + tn) if fp + tn else 0.0})
    return rows


def curve_panels(probas, y, classes, kind, path, title):
    binary = len(classes) == 2
    panels = [classes.index("VPN") if "VPN" in classes else 1] if binary else list(range(len(classes)))
    ncol = min(len(panels), 3)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow), squeeze=False)
    for ax, k in zip(axes.flat, panels):
        yk = (y == k).astype(int)
        for name, proba in probas.items():
            s = proba[:, k]
            if kind == "roc":
                fpr, tpr, _ = roc_curve(yk, s)
                ax.plot(fpr, tpr, color=COLORS[name], lw=WIDTHS[name], label=f"{name} (AUC {auc(fpr, tpr):.4f})")
            else:
                prec, rec, _ = precision_recall_curve(yk, s)
                ax.plot(rec, prec, color=COLORS[name], lw=WIDTHS[name],
                        label=f"{name} (AP {average_precision_score(yk, s):.4f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color=INK_2, lw=0.8)
            ax.set(xlabel="False positive rate", ylabel="True positive rate")
            # Zoom on the top-left corner, where strong models differ.
            ins = ax.inset_axes([0.42, 0.30, 0.52, 0.38])
            for name, proba in probas.items():
                fpr, tpr, _ = roc_curve(yk, proba[:, k])
                ins.plot(fpr, tpr, color=COLORS[name], lw=WIDTHS[name] * 0.8)
            ins.set(xlim=(0, 0.1), ylim=(0.9, 1.002))
            ins.tick_params(labelsize=6)
            ins.set_title("zoom: FPR 0-0.1", fontsize=6.5, fontweight="normal")
        else:
            ax.axhline(yk.mean(), ls="--", color=INK_2, lw=0.8)
            ax.set(xlabel="Recall", ylabel="Precision")
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01),
               title=f"{classes[k]} (positive class)" if binary else f"{classes[k]} vs rest")
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=7.5)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    suptitle(fig, title, 4.2 * ncol)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_panels(preds, y, classes, path, title):
    n = len(preds)
    size = 3.2 if len(classes) == 2 else 4.2
    fig, axes = plt.subplots(1, n, figsize=(size * n, size + 0.5), squeeze=False)
    for ax, (name, pred) in zip(axes.flat, preds.items()):
        m = confusion_matrix(y, pred, labels=range(len(classes)))
        norm = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{m[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set(xlabel="Predicted", ylabel="True", title=name)
    suptitle(fig, title + " (counts and % of each true class)", size * n)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--label-col", default=LABEL_COL)
    parser.add_argument("--feature-set", default=None, choices=sorted(FEATURE_SETS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True, help="task folder; figures go to <out>/figures")
    parser.add_argument("--title", default=None)
    parser.add_argument("--replot", action="store_true",
                        help="redraw from the saved probabilities (figures/probabilities.npz) without training")
    args = parser.parse_args()
    if args.config:
        cm.FTTransformerModel.tuned = json.load(open(args.config))
    out = os.path.join(args.out, "figures")
    os.makedirs(out, exist_ok=True)

    data = prepare_dataset(args.data, args.label_col, args.feature_set)
    classes, y, Z = data["classes"], data["y"], data["Z"]
    groups = data["df"][args.group_col].to_numpy() if args.group_col else None
    tr, va, te = split_indices(y, seed=SEED, groups=groups)
    mean, std = fit_standardizer(Z[tr])
    X_train, X_val, X_test = (standardize(Z[i], mean, std) for i in (tr, va, te))
    y_test = y[te]
    title = args.title or f"{os.path.basename(args.data)}, {data['feature_set']} features, label {args.label_col}"
    print(f"[INFO] {title}: {len(te)} test flows, seed {SEED}")

    store = f"{out}/probabilities.npz"
    probas, preds, rows = {}, {}, []
    if args.replot:
        saved = np.load(store)
        assert (saved["y_test"] == y_test).all(), "saved probabilities belong to a different split"
        probas = {name: saved[name] for name in cm.MODELS}
    for name, make in cm.MODELS.items():
        if name not in probas:
            model = make().fit(X_train, y[tr], X_val, y[va], len(classes), SEED)
            probas[name] = model.predict_proba(X_test)
        preds[name] = probas[name].argmax(1)
        rows += per_class_table(name, y_test, preds[name], classes)
        print(f"[INFO] {name:20s} acc {(preds[name] == y_test).mean():.4f}")
    np.savez_compressed(store, y_test=y_test, **probas)

    curve_panels(probas, y_test, classes, "roc", f"{out}/roc_curves.png", "ROC curves, " + title)
    curve_panels(probas, y_test, classes, "pr", f"{out}/pr_curves.png", "Precision-recall curves, " + title)
    confusion_panels(preds, y_test, classes, f"{out}/confusion_matrices.png", "Confusion matrices, " + title)

    table = pd.DataFrame(rows)
    table.to_csv(f"{out}/per_class_metrics.csv", index=False)
    shown = table.copy()
    for c in ("precision", "recall", "f1", "FPR"):
        shown[c] = (shown[c] * 100).map("{:.2f}".format)
    text = (f"# Per-class metrics: {title}\n\nTest split of seed {SEED} ({len(te)} flows), the split "
            f"behind the confusion matrices in summary.md. Precision, recall, F1 and FPR in %. "
            f"For each class, that class is the positive one (one-vs-rest).\n\n"
            + shown.to_markdown(index=False, disable_numparse=True)
            + "\n\nFigures: roc_curves.png, pr_curves.png, confusion_matrices.png\n")
    open(f"{out}/metrics.md", "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
