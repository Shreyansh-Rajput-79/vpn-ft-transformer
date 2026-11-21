"""Explainability report for the trained FT-Transformer.

  1. Per-flow explanations   - attention rollout: which features each decision used
  2. Global importance       - mean attention vs. permutation importance (FT and RF)
  3. Faithfulness check      - removing the most-attended features should hurt
                               the prediction more than removing random ones
  4. Example flows           - heatmap of per-flow relevance

Uses the same seed-42 test split as train.py (the split models/model.pth was
evaluated on). Outputs go to results/explain/.
"""
# ==========================================
# 1. Imports
# ==========================================
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from predict import load_model
from utils.explain import (deletion_curve, explain, permutation_importance,
                           top_features)
from utils.preprocessing import FEATURES, load_dataset, standardize, to_matrix

SEED = 42
DATA_PATH = "data/vpn_dataset_cleaned.csv"
MODEL_PATH = "models/model.pth"
OUT_DIR = "results/explain"
DELETION_KS = list(range(0, 11))

# Chart styling (reference data-viz palette, light mode)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 9, "legend.frameon": False,
})

# ==========================================
# 2. Data, model, explanations
# ==========================================
def load_split():
    df, labels = load_dataset(DATA_PATH)
    X = to_matrix(df)
    X_train, X_tmp, l_train, l_tmp = train_test_split(
        X, labels.to_numpy(), test_size=0.3, stratify=labels, random_state=SEED)
    _, X_test, _, l_test = train_test_split(
        X_tmp, l_tmp, test_size=0.5, stratify=l_tmp, random_state=SEED)
    return X_train, l_train, X_test, l_test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model, ckpt = load_model(MODEL_PATH)
    classes = ckpt["classes"]
    X_train, l_train, X_test, l_test = load_split()
    X_train = standardize(X_train, ckpt["mean"], ckpt["std"])
    X_test = standardize(X_test, ckpt["mean"], ckpt["std"])
    y_train = np.array([classes.index(c) for c in l_train])
    y_test = np.array([classes.index(c) for c in l_test])

    proba, relevance = explain(model, X_test)
    pred = proba.argmax(axis=1)
    print(f"[INFO] FT-Transformer test accuracy: {(pred == y_test).mean():.4f}")

    ft_proba = lambda X: explain(model, X)[0]
    ft_predict = lambda X: ft_proba(X).argmax(axis=1)

    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
    print(f"[INFO] Random Forest test accuracy: {rf.score(X_test, y_test):.4f}")

    # Per-flow explanations
    flows = pd.DataFrame({
        "true": [classes[i] for i in y_test],
        "prediction": [classes[i] for i in pred],
        "confidence": proba.max(axis=1).round(4),
        "top_features": top_features(relevance, FEATURES),
    })
    flows = pd.concat([flows, pd.DataFrame(relevance.round(4), columns=FEATURES)], axis=1)
    flows.to_csv(f"{OUT_DIR}/test_flow_explanations.csv", index=False)

    # Global importance
    print("[INFO] Permutation importance (FT-Transformer, Random Forest)...")
    ft_perm, ft_perm_std = permutation_importance(ft_predict, X_test, y_test, seed=SEED)
    rf_perm, rf_perm_std = permutation_importance(rf.predict, X_test, y_test, seed=SEED)
    importance = pd.DataFrame({
        "ft_attention": relevance.mean(axis=0),
        "ft_attention_non_vpn": relevance[pred == classes.index("Non-VPN")].mean(axis=0),
        "ft_attention_vpn": relevance[pred == classes.index("VPN")].mean(axis=0),
        "ft_permutation": ft_perm, "ft_permutation_std": ft_perm_std,
        "rf_permutation": rf_perm, "rf_permutation_std": rf_perm_std,
        "rf_impurity": rf.feature_importances_,
    }, index=FEATURES)
    importance.to_csv(f"{OUT_DIR}/global_importance.csv")

    # Faithfulness: deletion curves
    print("[INFO] Deletion test...")
    rng = np.random.default_rng(SEED)
    by_attention = np.argsort(-relevance, axis=1)
    curves = {
        "Most-attended first": deletion_curve(ft_proba, X_test, by_attention, DELETION_KS),
        "Random order": np.mean([
            deletion_curve(ft_proba, X_test,
                           np.argsort(rng.random(relevance.shape), axis=1), DELETION_KS)
            for _ in range(5)], axis=0),
        "Least-attended first": deletion_curve(ft_proba, X_test, by_attention[:, ::-1], DELETION_KS),
    }

    plot_global(importance)
    plot_by_class(importance)
    plot_deletion(curves)
    examples = pick_examples(y_test, pred, proba, relevance)
    plot_examples(relevance, y_test, pred, proba, examples, classes)
    write_summary(importance, curves, flows, examples)

# ==========================================
# 3. Figures
# ==========================================
def plot_global(imp):
    order = imp.sort_values("ft_permutation").index  # most important at the top
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True)
    panels = [
        ("FT-Transformer: attention", "mean share of CLS attention (%)",
         imp.loc[order, "ft_attention"] * 100, None, BLUE),
        ("FT-Transformer: permutation", "accuracy drop when shuffled (pts)",
         imp.loc[order, "ft_permutation"] * 100, imp.loc[order, "ft_permutation_std"] * 100, BLUE),
        ("Random Forest: permutation", "accuracy drop when shuffled (pts)",
         imp.loc[order, "rf_permutation"] * 100, imp.loc[order, "rf_permutation_std"] * 100, ORANGE),
    ]
    for ax, (title, xlabel, values, err, color) in zip(axes, panels):
        ax.barh(order, values, height=0.6, color=color, xerr=err,
                error_kw=dict(ecolor=INK_2, elinewidth=0.8, capsize=0))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK_2, linewidth=0.8)
    fig.suptitle("Which features drive the decision?  (sorted by FT-Transformer permutation importance)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/global_importance.png", dpi=160)
    plt.close(fig)


def plot_by_class(imp):
    order = imp.sort_values("ft_attention").index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.barh(y + 0.18, imp.loc[order, "ft_attention_non_vpn"] * 100, height=0.34,
            color=BLUE, label="Flows predicted Non-VPN")
    ax.barh(y - 0.18, imp.loc[order, "ft_attention_vpn"] * 100, height=0.34,
            color=ORANGE, label="Flows predicted VPN")
    ax.set_yticks(y, order)
    ax.set_xlabel("mean share of CLS attention (%)")
    ax.set_title("Attention by predicted class")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/attention_by_class.png", dpi=160)
    plt.close(fig)


def plot_deletion(curves):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (name, curve), color in zip(curves.items(), (BLUE, ORANGE, AQUA)):
        ax.plot(DELETION_KS, curve * 100, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.annotate(name, (DELETION_KS[-1], curve[-1] * 100), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8.5)
    ax.set_xlabel("features removed from each flow (set to training mean)")
    ax.set_ylabel("confidence in original prediction (%)")
    ax.set_title("Faithfulness check: removing attended features should hurt most")
    ax.set_xlim(-0.3, DELETION_KS[-1] + 3.2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/deletion_test.png", dpi=160)
    plt.close(fig)


def pick_examples(y, pred, proba, relevance):
    """3 correct flows per class and 2 confident errors, with distinct explanations
    (the dataset has many near-identical flows)."""
    rng = np.random.default_rng(SEED)
    conf = proba.max(axis=1)
    picks, seen = [], set()

    def take(candidates, n):
        taken = 0
        for i in candidates:
            key = tuple(np.round(relevance[i], 2))
            if key not in seen:
                seen.add(key)
                picks.append(i)
                taken += 1
                if taken == n:
                    return

    for cls in (0, 1):
        take(rng.permutation(np.where((y == cls) & (pred == cls) & (conf > 0.9))[0]), 3)
    wrong = np.where(y != pred)[0]
    take(wrong[np.argsort(-conf[wrong])], 2)
    return picks


def plot_examples(relevance, y, pred, proba, picks, classes):
    data = relevance[picks] * 100
    labels = [f"{'OK ' if y[i] == pred[i] else 'ERR'}  true {classes[y[i]]:7s} -> {classes[pred[i]]} "
              f"({proba[i].max()*100:.0f}%)" for i in picks]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    im = ax.imshow(data, cmap=SEQUENTIAL, aspect="auto", vmin=0)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right")
    ax.set_yticks(range(len(picks)), labels, fontfamily="monospace")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Mark the single most-relevant feature of each flow.
    for row, values in enumerate(data):
        col = values.argmax()
        ax.text(col, row, f"{values[col]:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if values[col] > data.max() * 0.55 else INK)
    fig.colorbar(im, ax=ax, label="share of CLS attention (%)", fraction=0.025, pad=0.01)
    ax.set_title("Per-flow explanations (attention rollout): 3 correct Non-VPN, 3 correct VPN, 2 confident errors")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/example_flows.png", dpi=160)
    plt.close(fig)

# ==========================================
# 4. Summary
# ==========================================
def write_summary(imp, curves, flows, examples):
    rho_attn_perm = spearmanr(imp.ft_attention, imp.ft_permutation).statistic
    rho_ft_rf = spearmanr(imp.ft_permutation, imp.rf_permutation).statistic
    area = {name: np.trapezoid(c, DELETION_KS) / DELETION_KS[-1] for name, c in curves.items()}

    top = lambda col: ", ".join(imp[col].sort_values(ascending=False).index[:5])
    curve_table = pd.DataFrame({name: (c * 100).round(2) for name, c in curves.items()},
                               index=pd.Index(DELETION_KS, name="features removed"))

    parts = [
        "# FT-Transformer explainability report",
        "Seed-42 test split (same as train.py), "
        f"{len(flows)} flows. Per-flow explanations: `test_flow_explanations.csv`.",
        "## Top-5 features by method",
        f"- FT attention (mean): {top('ft_attention')}\n"
        f"- FT permutation: {top('ft_permutation')}\n"
        f"- RF permutation: {top('rf_permutation')}\n"
        f"- RF impurity: {top('rf_impurity')}",
        "## Agreement between methods (Spearman rank correlation over 23 features)",
        f"- FT attention vs FT permutation: {rho_attn_perm:.3f}\n"
        f"- FT permutation vs RF permutation: {rho_ft_rf:.3f}",
        "## Faithfulness (deletion test)",
        "Mean confidence (%) in the original prediction after removing k features per flow. "
        "Lower area = removing those features hurts more.",
        curve_table.to_markdown(),
        "Area under curve (mean confidence, %): "
        + ", ".join(f"{n}: {a*100:.2f}" for n, a in area.items()),
        "## Example flows",
        flows.loc[examples, ["true", "prediction", "confidence", "top_features"]].to_markdown(index=False),
        "## Figures",
        "- `global_importance.png`\n- `attention_by_class.png`\n"
        "- `deletion_test.png`\n- `example_flows.png`",
    ]
    text = "\n\n".join(parts) + "\n"
    with open(f"{OUT_DIR}/summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
