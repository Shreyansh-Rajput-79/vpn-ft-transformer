"""Explainability report for a trained FT-Transformer, with SHAP on the Random Forest.

  1. Per-flow explanations   - attention rollout: which features each decision used
  2. Global importance       - mean attention vs. permutation importance (FT and RF)
  3. Faithfulness check      - removing the features an explanation ranks highest
                               should hurt the prediction more than removing random
                               ones. Run for FT attention AND for RF TreeSHAP, so the
                               two models' explanations are judged the same way.
  4. Example flows           - heatmap of per-flow relevance

Uses the same seed-42 split as train.py. Examples:
    python explain.py                                               # ISCX, models/model.pth
    python explain.py --data data/vnat_features.csv --group-col source_file \\
        --model models/vnat_model.pth --out results/vnat/explain
"""
# ==========================================
# 1. Imports
# ==========================================
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

from predict import load_model
from utils.explain import (deletion_curve, explain, permutation_importance,
                           top_features)
from utils.plotting import (AQUA, BLUE, INK, INK_2, MODEL_COLORS, ORANGE,
                            SEQUENTIAL)
from utils.preprocessing import (load_dataset, split_indices, standardize,
                                 to_matrix)

SEED = 42
DELETION_KS = list(range(0, 11))
SHAP_FLOWS = 500  # TreeSHAP on a 300-tree forest is slow; explain a sample of test flows

# ==========================================
# 2. Data, model, explanations
# ==========================================
def load_split(data_path, ckpt, group_col=None, label_col=None):
    """Seed-42 split of `data_path`, encoded with the checkpoint's features and classes.

    Returns (Z_train, y_train, Z_test, y_test, df_test); Z is the signed-log
    matrix before standardization.
    """
    label_col = label_col or ckpt["label_col"]
    df, labels = load_dataset(data_path, label_col)
    classes = ckpt["classes"]
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise SystemExit(f"[ERROR] Labels {unknown} are not classes of this model {classes}.")
    y = labels.map({c: i for i, c in enumerate(classes)}).to_numpy()
    Z = to_matrix(df, ckpt["features"])
    groups = df[group_col].to_numpy() if group_col else None
    tr, _, te = split_indices(y, seed=SEED, groups=groups)
    return Z[tr], y[tr], Z[te], y[te], df.iloc[te].reset_index(drop=True)


def rf_shap(rf, X):
    """|TreeSHAP| of the predicted class per flow, normalized to sum to 1, plus seconds per flow."""
    start = time.perf_counter()
    values = np.asarray(shap.TreeExplainer(rf).shap_values(X))  # (flows, features, classes)
    seconds = (time.perf_counter() - start) / len(X)
    pred = rf.predict(X)
    contrib = np.abs(values[np.arange(len(X)), :, pred])
    return contrib / np.maximum(contrib.sum(axis=1, keepdims=True), 1e-12), seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/vpn_dataset_cleaned.csv")
    parser.add_argument("--model", default="models/model.pth")
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--out", default="results/explain")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, ckpt = load_model(args.model)
    classes, features = ckpt["classes"], ckpt["features"]
    Z_train, y_train, Z_test, y_test, _ = load_split(args.data, ckpt, args.group_col)
    X_train = standardize(Z_train, ckpt["mean"], ckpt["std"])
    X_test = standardize(Z_test, ckpt["mean"], ckpt["std"])

    start = time.perf_counter()
    proba, relevance = explain(model, X_test)
    ft_seconds = (time.perf_counter() - start) / len(X_test)
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
        "top_features": top_features(relevance, features),
    })
    flows = pd.concat([flows, pd.DataFrame(relevance.round(4), columns=features)], axis=1)
    flows.to_csv(f"{args.out}/test_flow_explanations.csv", index=False)

    # Global importance
    print("[INFO] Permutation importance (FT-Transformer, Random Forest)...")
    ft_perm, ft_perm_std = permutation_importance(ft_predict, X_test, y_test, seed=SEED)
    rf_perm, rf_perm_std = permutation_importance(rf.predict, X_test, y_test, seed=SEED)
    importance = pd.DataFrame({
        "ft_attention": relevance.mean(axis=0),
        **{f"ft_attention_{c}": relevance[pred == i].mean(axis=0) if (pred == i).any()
           else np.zeros(len(features)) for i, c in enumerate(classes)},
        "ft_permutation": ft_perm, "ft_permutation_std": ft_perm_std,
        "rf_permutation": rf_perm, "rf_permutation_std": rf_perm_std,
        "rf_impurity": rf.feature_importances_,
    }, index=features)

    # SHAP on the Random Forest (sample of test flows)
    print(f"[INFO] TreeSHAP on the Random Forest ({SHAP_FLOWS} flows)...")
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X_test), size=min(SHAP_FLOWS, len(X_test)), replace=False)
    shap_rel, shap_seconds = rf_shap(rf, X_test[sample])
    importance["rf_shap"] = shap_rel.mean(axis=0)
    importance.to_csv(f"{args.out}/global_importance.csv")

    # Faithfulness: deletion curves, for each model with its own explanation
    print("[INFO] Deletion tests...")
    random_order = lambda shape: np.argsort(rng.random(shape), axis=1)
    by_attention = np.argsort(-relevance, axis=1)
    curves = {
        "Most-attended first": deletion_curve(ft_proba, X_test, by_attention, DELETION_KS),
        "Random order": np.mean([deletion_curve(ft_proba, X_test, random_order(relevance.shape),
                                                DELETION_KS) for _ in range(5)], axis=0),
        "Least-attended first": deletion_curve(ft_proba, X_test, by_attention[:, ::-1], DELETION_KS),
    }
    Xs = X_test[sample]
    faithfulness = {
        "FT-Transformer (attention)": (
            deletion_curve(ft_proba, Xs, by_attention[sample], DELETION_KS),
            np.mean([deletion_curve(ft_proba, Xs, random_order(Xs.shape), DELETION_KS)
                     for _ in range(5)], axis=0)),
        "Random Forest (TreeSHAP)": (
            deletion_curve(rf.predict_proba, Xs, np.argsort(-shap_rel, axis=1), DELETION_KS),
            np.mean([deletion_curve(rf.predict_proba, Xs, random_order(Xs.shape), DELETION_KS)
                     for _ in range(5)], axis=0)),
    }
    cost = {"FT-Transformer (attention)": ft_seconds, "Random Forest (TreeSHAP)": shap_seconds}

    plot_global(importance, args.out)
    plot_by_class(importance, classes, args.out)
    plot_deletion(curves, args.out)
    plot_faithfulness(faithfulness, args.out)
    examples = pick_examples(y_test, pred, proba, relevance, len(classes))
    plot_examples(relevance, y_test, pred, proba, examples, classes, features, args.out)
    write_summary(importance, curves, faithfulness, cost, flows, examples, features, args.out)

# ==========================================
# 3. Figures
# ==========================================
def plot_global(imp, out):
    order = imp.sort_values("ft_permutation").index  # most important at the top
    fig, axes = plt.subplots(1, 4, figsize=(16, 0.27 * len(imp) + 2), sharey=True)
    panels = [
        ("FT-Transformer: attention", "mean share of CLS attention (%)",
         imp.loc[order, "ft_attention"] * 100, None, BLUE),
        ("FT-Transformer: permutation", "accuracy drop when shuffled (pts)",
         imp.loc[order, "ft_permutation"] * 100, imp.loc[order, "ft_permutation_std"] * 100, BLUE),
        ("Random Forest: TreeSHAP", "mean share of |SHAP| (%)",
         imp.loc[order, "rf_shap"] * 100, None, ORANGE),
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
    fig.savefig(f"{out}/global_importance.png", dpi=160)
    plt.close(fig)


def plot_by_class(imp, classes, out):
    order = imp.sort_values("ft_attention").index
    if len(classes) == 2:
        y = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(7.5, 0.27 * len(order) + 1.5))
        for offset, cls, color in ((0.18, classes[0], BLUE), (-0.18, classes[1], ORANGE)):
            ax.barh(y + offset, imp.loc[order, f"ft_attention_{cls}"] * 100, height=0.34,
                    color=color, label=f"Flows predicted {cls}")
        ax.set_yticks(y, order)
        ax.set_xlabel("mean share of CLS attention (%)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
    else:
        # Many classes: a class x feature heatmap reads better than grouped bars.
        data = np.array([imp.loc[order[::-1], f"ft_attention_{c}"] * 100 for c in classes])
        fig, ax = plt.subplots(figsize=(0.3 * len(order) + 3, 0.45 * len(classes) + 2))
        im = ax.imshow(data, cmap=SEQUENTIAL, aspect="auto", vmin=0)
        ax.set_xticks(range(len(order)), order[::-1], rotation=60, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="mean share of CLS attention (%)", fraction=0.025, pad=0.01)
    ax.set_title("Attention by predicted class")
    fig.tight_layout()
    fig.savefig(f"{out}/attention_by_class.png", dpi=160)
    plt.close(fig)


def plot_deletion(curves, out):
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
    fig.savefig(f"{out}/deletion_test.png", dpi=160)
    plt.close(fig)


def plot_faithfulness(faithfulness, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (name, (top, rand)) in zip(axes, faithfulness.items()):
        color = MODEL_COLORS[name.split(" (")[0]]
        ax.plot(DELETION_KS, top * 100, color=color, linewidth=2, marker="o", markersize=4,
                label="Top-ranked features removed first")
        ax.plot(DELETION_KS, rand * 100, color=color, linewidth=2, linestyle="--",
                label="Random features removed")
        ax.fill_between(DELETION_KS, top * 100, rand * 100, color=color, alpha=0.12)
        ax.set_title(name)
        ax.set_xlabel("features removed from each flow")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower left")
    axes[0].set_ylabel("confidence in original prediction (%)")
    fig.suptitle("Explanation faithfulness: bigger gap = explanation points at what the model really uses",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out}/faithfulness_ft_vs_rf.png", dpi=160)
    plt.close(fig)


def pick_examples(y, pred, proba, relevance, num_classes):
    """Correct flows of every class and 2 confident errors, with distinct explanations
    (the dataset has many near-identical flows)."""
    rng = np.random.default_rng(SEED)
    conf = proba.max(axis=1)
    per_class = 3 if num_classes == 2 else 1
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

    for cls in range(num_classes):
        take(rng.permutation(np.where((y == cls) & (pred == cls) & (conf > 0.9))[0]), per_class)
    wrong = np.where(y != pred)[0]
    take(wrong[np.argsort(-conf[wrong])], 2)
    return picks


def plot_examples(relevance, y, pred, proba, picks, classes, features, out):
    data = relevance[picks] * 100
    width = max(len(c) for c in classes)
    labels = [f"{'OK ' if y[i] == pred[i] else 'ERR'}  true {classes[y[i]]:{width}s} -> {classes[pred[i]]} "
              f"({proba[i].max()*100:.0f}%)" for i in picks]
    fig, ax = plt.subplots(figsize=(max(12, 0.3 * len(features) + 6), 0.45 * len(picks) + 2.6))
    im = ax.imshow(data, cmap=SEQUENTIAL, aspect="auto", vmin=0)
    ax.set_xticks(range(len(features)), features, rotation=60, ha="right")
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
    ax.set_title("Per-flow explanations (attention rollout): correct flows per class and confident errors")
    fig.tight_layout()
    fig.savefig(f"{out}/example_flows.png", dpi=160)
    plt.close(fig)

# ==========================================
# 4. Summary
# ==========================================
def write_summary(imp, curves, faithfulness, cost, flows, examples, features, out):
    rho_attn_perm = spearmanr(imp.ft_attention, imp.ft_permutation).statistic
    rho_ft_rf = spearmanr(imp.ft_permutation, imp.rf_permutation).statistic
    rho_shap_perm = spearmanr(imp.rf_shap, imp.rf_permutation).statistic
    area = lambda c: np.trapezoid(c, DELETION_KS) / DELETION_KS[-1] * 100

    top = lambda col: ", ".join(imp[col].sort_values(ascending=False).index[:5])
    curve_table = pd.DataFrame({name: (c * 100).round(2) for name, c in curves.items()},
                               index=pd.Index(DELETION_KS, name="features removed"))
    faith_table = pd.DataFrame([{
        "explanation": name,
        "confidence after removing top-1 (%)": round(t[1] * 100, 2),
        "after removing random-1 (%)": round(r[1] * 100, 2),
        "AUC top-first": round(area(t), 2),
        "AUC random": round(area(r), 2),
        "faithfulness gap (AUC random - top)": round(area(r) - area(t), 2),
        "explanation cost (ms / flow)": round(cost[name] * 1000, 3),
    } for name, (t, r) in faithfulness.items()])

    parts = [
        "# Explainability report",
        f"Seed-42 test split (same as train.py), {len(flows)} flows, {len(features)} features. "
        "Per-flow explanations: `test_flow_explanations.csv`.",
        "## Top-5 features by method",
        f"- FT attention (mean): {top('ft_attention')}\n"
        f"- FT permutation: {top('ft_permutation')}\n"
        f"- RF TreeSHAP (mean |SHAP|): {top('rf_shap')}\n"
        f"- RF permutation: {top('rf_permutation')}\n"
        f"- RF impurity: {top('rf_impurity')}",
        f"## Agreement between methods (Spearman rank correlation over {len(features)} features)",
        f"- FT attention vs FT permutation: {rho_attn_perm:.3f}\n"
        f"- RF TreeSHAP vs RF permutation: {rho_shap_perm:.3f}\n"
        f"- FT permutation vs RF permutation: {rho_ft_rf:.3f}",
        "## Faithfulness (FT-Transformer, all test flows)",
        "Mean confidence (%) in the original prediction after removing k features per flow. "
        "Lower = removing those features hurts more.",
        curve_table.to_markdown(),
        f"## FT attention vs RF TreeSHAP (same {SHAP_FLOWS} sampled flows)",
        "Each model is tested with its own explanation. The gap between the random curve and "
        "the top-first curve measures how well the explanation identifies what the model relies on. "
        "Attention is produced in the same forward pass as the prediction; TreeSHAP is computed "
        "afterwards.",
        faith_table.to_markdown(index=False),
        "## Example flows",
        flows.loc[examples, ["true", "prediction", "confidence", "top_features"]].to_markdown(index=False),
        "## Figures",
        "- `global_importance.png`\n- `attention_by_class.png`\n- `deletion_test.png`\n"
        "- `faithfulness_ft_vs_rf.png`\n- `example_flows.png`",
    ]
    text = "\n\n".join(parts) + "\n"
    with open(f"{out}/summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
