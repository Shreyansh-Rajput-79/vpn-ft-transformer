"""Adversarial robustness: can a VPN user evade the classifier?

Threat model: the attacker controls their own traffic (not the model) and wants
VPN flows classified as Non-VPN. Repeated over several seeds; each seed trains a
fresh FT-Transformer and Random Forest on its own split.

  1. Traffic shaping (realistic, black-box): slow the flow down and/or pad packets.
     Each has a real cost to the attacker (latency, bandwidth).
  2. Defense: shaping augmentation. Both models are retrained with randomly
     shaped copies of the training flows and attacked again.
  3. Feature-space attacks (worst case): perturb any feature within an L-inf
     budget eps (black-box random search on both models, white-box PGD on FT).

Examples:
    python security_eval.py                                          # ISCX, 5 seeds
    python security_eval.py --data data/vnat_features.csv --group-col source_file --out results/vnat/security
    python security_eval.py --seeds 42 --skip-feature-space          # quick run

Outputs: per_seed_*.csv, summary.md and figures in --out (default results/security).
"""
# ==========================================
# 1. Imports
# ==========================================
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from utils.attacks import (augment_with_shaping, min_slowdown_to_evade,
                           pgd_attack, random_search_attack)
from utils.plotting import INK, MODEL_COLORS
from utils.preprocessing import (fit_standardizer, prepare_dataset,
                                 split_indices, standardize)
from utils.training import predict_proba, train_ft_transformer

SLOWDOWNS = np.round(np.arange(1.0, 5.0001, 0.05), 2)
BUDGETS = [1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
PADDINGS = {"Delay only": [1.0], "Delay + up to 100% padding": [1.0, 1.25, 1.5, 1.75, 2.0]}
EPSILONS = [0.05, 0.1, 0.25, 0.5, 1.0]

# ==========================================
# 2. Experiment
# ==========================================
def train_models(X_train, y_train, X_val, y_val, seed):
    ft = train_ft_transformer(X_train, y_train, X_val, y_val, 2, seed=seed, verbose=False)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed).fit(X_train, y_train)
    return {"FT-Transformer": (lambda X: predict_proba(ft, X), ft),
            "Random Forest": (rf.predict_proba, None)}


def run_seed(seed, data, groups, defense, feature_space):
    y, Z, features = data["y"], data["Z"], data["features"]
    vpn, non_vpn = data["classes"].index("VPN"), data["classes"].index("Non-VPN")
    tr, va, te = split_indices(y, seed=seed, groups=groups)

    variants = {"standard": (Z[tr], y[tr])}
    if defense:
        variants["shaping-augmented"] = augment_with_shaping(Z[tr], y[tr], features, seed=seed)

    shaping_rows, fs_rows = [], []
    for variant, (Z_train, y_train) in variants.items():
        mean, std = fit_standardizer(Z_train)
        X_train, X_val, X_test = (standardize(a, mean, std) for a in (Z_train, Z[va], Z[te]))
        models = train_models(X_train, y_train, X_val, y[va], seed)

        for name, (proba, net) in models.items():
            clean_pred = proba(X_test).argmax(1)
            detected = (y[te] == vpn) & (clean_pred == vpn)
            Z_att = Z[te][detected]
            predict = lambda Zs, proba=proba: proba(standardize(Zs, mean, std)).argmax(1)

            row = {"seed": seed, "variant": variant, "model": name,
                   "clean_accuracy": (clean_pred == y[te]).mean(),
                   "vpn_detected": int(detected.sum())}
            for mode, paddings in PADDINGS.items():
                needed = min_slowdown_to_evade(predict, Z_att, SLOWDOWNS, paddings, non_vpn, features)
                for b in BUDGETS:
                    row[f"{mode} <= {b}x"] = (needed <= b).mean()
                row[f"{mode} curve"] = [(needed <= s).mean() for s in SLOWDOWNS]
            shaping_rows.append(row)
            print(f"[seed {seed}] {variant:17s} {name:15s} clean acc {row['clean_accuracy']:.4f} | "
                  f"evasion <=1.1x: {row['Delay only <= 1.1x']*100:5.1f}% (delay), "
                  f"{row['Delay + up to 100% padding <= 1.1x']*100:5.1f}% (+padding)")

            if feature_space and variant == "standard":
                X_att = X_test[detected]
                for eps in EPSILONS:
                    adv = random_search_attack(proba, X_att, vpn, eps, seed=seed)
                    fs_rows.append({"seed": seed, "model": name, "attack": "Black-box random search",
                                    "eps": eps, "evasion": (proba(adv).argmax(1) == non_vpn).mean()})
                    if net is not None:
                        adv = pgd_attack(net, X_att, vpn, eps)
                        fs_rows.append({"seed": seed, "model": name, "attack": "White-box PGD",
                                        "eps": eps, "evasion": (proba(adv).argmax(1) == non_vpn).mean()})
    return shaping_rows, fs_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/vpn_dataset_cleaned.csv")
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
    parser.add_argument("--no-defense", action="store_true", help="skip the shaping-augmentation defense")
    parser.add_argument("--skip-feature-space", action="store_true")
    parser.add_argument("--out", default="results/security")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = prepare_dataset(args.data, "class1")
    if data["classes"] != ["Non-VPN", "VPN"]:
        raise SystemExit(f"[ERROR] Security evaluation needs VPN / Non-VPN labels, got {data['classes']}")
    groups = data["df"][args.group_col].to_numpy() if args.group_col else None
    print(f"[INFO] {len(data['Z'])} flows, {len(data['features'])} {data['feature_set']} features, "
          f"seeds {args.seeds}")

    shaping, feature_space = [], []
    for seed in args.seeds:
        s, f = run_seed(seed, data, groups, not args.no_defense, not args.skip_feature_space)
        shaping += s
        feature_space += f

    shaping = pd.DataFrame(shaping)
    shaping.drop(columns=[c for c in shaping if c.endswith("curve")]).to_csv(
        f"{args.out}/per_seed_shaping.csv", index=False)
    feature_space = pd.DataFrame(feature_space)
    if not feature_space.empty:
        feature_space.to_csv(f"{args.out}/per_seed_feature_space.csv", index=False)

    plot_shaping(shaping, args.out)
    if not feature_space.empty:
        plot_feature_space(feature_space, args.out)
    write_summary(shaping, feature_space, args)

# ==========================================
# 3. Figures
# ==========================================
def plot_shaping(shaping, out):
    variants = list(dict.fromkeys(shaping.variant))
    fig, axes = plt.subplots(len(variants), 2, figsize=(12, 4.3 * len(variants)), sharey=True, squeeze=False)
    for r, variant in enumerate(variants):
        for c, mode in enumerate(PADDINGS):
            ax = axes[r, c]
            for name, color in MODEL_COLORS.items():
                rows = shaping[(shaping.variant == variant) & (shaping.model == name)]
                if rows.empty:
                    continue
                curves = np.array(rows[f"{mode} curve"].tolist()) * 100
                mean, std = curves.mean(0), curves.std(0)
                ax.plot(SLOWDOWNS, mean, color=color, linewidth=2, label=name)
                ax.fill_between(SLOWDOWNS, np.clip(mean - std, 0, 100), np.clip(mean + std, 0, 100),
                                color=color, alpha=0.15)
                ax.annotate(f"{name} ({mean[-1]:.0f}%)", (SLOWDOWNS[-1], mean[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", color=INK, fontsize=8.5)
            ax.set_title(f"{mode} - {variant} models")
            ax.set_xlabel("slowdown factor the attacker accepts (1.5 = 50% slower)")
            ax.set_xlim(1, SLOWDOWNS[-1] + 1.6)
            ax.grid(axis="x", visible=False)
        axes[r, 0].set_ylabel("detected VPN flows that evade (%)")
    axes[0, 0].legend(loc="upper left")
    n = shaping.seed.nunique()
    fig.suptitle(f"Traffic-shaping attack (mean ± std over {n} seed{'s' if n > 1 else ''})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{out}/shaping_attack.png", dpi=160)
    plt.close(fig)


def plot_feature_space(fs, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for (name, attack), rows in fs.groupby(["model", "attack"]):
        g = rows.groupby("eps").evasion
        mean, std = g.mean() * 100, g.std().fillna(0) * 100
        style = "--" if "PGD" in attack else "-"
        ax.errorbar(mean.index, mean, yerr=std, fmt=style, color=MODEL_COLORS[name], linewidth=2,
                    marker="o", markersize=4, capsize=0, label=f"{name}, {attack}")
    ax.set_xscale("log")
    ax.set_xticks(EPSILONS, [str(e) for e in EPSILONS])
    ax.set_xlabel("perturbation budget eps (std. units per feature)")
    ax.set_ylabel("detected VPN flows that evade (%)")
    ax.set_title("Feature-space attacks (worst case)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{out}/feature_space_attacks.png", dpi=160)
    plt.close(fig)

# ==========================================
# 4. Summary
# ==========================================
def write_summary(shaping, fs, args):
    pct = lambda s: f"{s.mean()*100:.1f} ± {s.std()*100:.1f}" if len(s) > 1 else f"{s.mean()*100:.1f}"
    budget_cols = [c for c in shaping if " <= " in c]

    def table(cols):
        g = shaping.groupby(["variant", "model"], sort=False)
        return g[cols].agg(pct).to_markdown()

    parts = [
        "# Adversarial robustness: VPN evasion",
        f"Data: `{args.data}`, seeds {args.seeds}. Values: mean ± std over seeds, in %. "
        "Evasion rate = % of VPN flows the model detected that are classified Non-VPN after the attack.",
        "## Clean accuracy",
        table(["clean_accuracy"]),
        "## Traffic shaping: delay only",
        "Evasion rate within each slowdown budget.",
        table([c for c in budget_cols if c.startswith("Delay only")]),
        "## Traffic shaping: delay + up to 100% padding",
        table([c for c in budget_cols if c.startswith("Delay +")]),
        "`shaping-augmented` = models retrained with one randomly shaped copy of every training "
        "flow (slowdown 1-3x, padding 1-2x, both classes, same label).",
    ]
    if not fs.empty:
        t = fs.groupby(["attack", "model", "eps"]).evasion.agg(pct).unstack("eps")
        parts += ["## Feature-space attacks (standard models)",
                  "L-inf budget eps on standardized features.", t.to_markdown()]
    parts += [
        "## Caveats",
        "- Traffic shaping is simulated on flow features, not re-captured packets; stretched gaps "
        "crossing the 1 s activity threshold would also create idle periods in a real capture.\n"
        "- Feature-space attacks ignore whether the perturbed feature vector is physically "
        "achievable: they are an upper bound on attacker power.\n"
        "- The defense is evaluated against the same shaping family it was trained on, up to 3x "
        "slowdown; budgets beyond 3x are outside its training range.",
    ]
    text = "\n\n".join(parts) + "\n"
    with open(f"{args.out}/summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
