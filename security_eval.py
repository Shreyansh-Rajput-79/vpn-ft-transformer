"""Adversarial robustness: can a VPN user evade the classifier?

Threat model: the attacker controls their own traffic (not the model) and wants
VPN flows classified as Non-VPN. Both models are attacked on the same VPN test
flows (seed-42 split, same as train.py / explain.py).

  1. Traffic shaping (realistic, black-box): slow the flow down and/or pad packets.
     Each has a real cost to the attacker (latency, bandwidth).
  2. Feature-space attacks (worst case): perturb any feature within an L-inf
     budget eps on standardized features.
       - black-box random search on both models (same attack, fair comparison)
       - white-box PGD on the FT-Transformer (strongest attacker)
       - transfer: adversarial flows crafted on one model, replayed on the other

Outputs go to results/security/.
"""
# ==========================================
# 1. Imports
# ==========================================
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from explain import BLUE, INK, INK_2, ORANGE, SEED, load_split
from predict import load_model
from utils.attacks import (min_slowdown_to_evade, pgd_attack,
                           random_search_attack)
from utils.explain import explain
from utils.preprocessing import standardize

MODEL_PATH = "models/model.pth"
OUT_DIR = "results/security"

SLOWDOWNS = np.round(np.arange(1.0, 5.0001, 0.05), 2)
PADDINGS = {"Delay only": [1.0], "Delay + up to 100% padding": [1.0, 1.25, 1.5, 1.75, 2.0]}
EPSILONS = [0.05, 0.1, 0.25, 0.5, 1.0]

# ==========================================
# 2. Experiment
# ==========================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model, ckpt = load_model(MODEL_PATH)
    classes = ckpt["classes"]
    vpn, non_vpn = classes.index("VPN"), classes.index("Non-VPN")
    mean, std = ckpt["mean"], ckpt["std"]

    Z_train, l_train, Z_test, l_test = load_split()
    y_train = np.array([classes.index(c) for c in l_train])
    y_test = np.array([classes.index(c) for c in l_test])
    X_train, X_test = standardize(Z_train, mean, std), standardize(Z_test, mean, std)

    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_train, y_train)

    # Each model maps standardized features -> probabilities.
    probas = {
        "FT-Transformer": lambda X: explain(model, X)[0],
        "Random Forest": rf.predict_proba,
    }
    colors = {"FT-Transformer": BLUE, "Random Forest": ORANGE}

    # Attack targets: VPN flows that the model gets right (per model, so each
    # model is only credited for flows it actually detects).
    targets = {}
    for name, proba in probas.items():
        detected = (y_test == vpn) & (proba(X_test).argmax(1) == vpn)
        targets[name] = np.where(detected)[0]
        print(f"[INFO] {name}: {len(targets[name])} / {(y_test == vpn).sum()} VPN test flows detected")

    # ---- 1. Traffic shaping ------------------------------------------------
    print("[INFO] Traffic-shaping attack...")
    shaping = {}
    for name, proba in probas.items():
        Z = Z_test[targets[name]]
        predict = lambda Zs, proba=proba: proba(standardize(Zs, mean, std)).argmax(1)
        for mode, paddings in PADDINGS.items():
            shaping[(name, mode)] = min_slowdown_to_evade(predict, Z, SLOWDOWNS, paddings, non_vpn)

    # ---- 2. Feature-space attacks -----------------------------------------
    print("[INFO] Feature-space attacks...")
    rows = []
    for eps in EPSILONS:
        adv = {}
        for name, proba in probas.items():
            X = X_test[targets[name]]
            adv[name] = random_search_attack(proba, X, vpn, eps, seed=SEED)
            rows.append({"eps": eps, "attack": "Black-box random search", "model": name,
                         "evasion_rate": (proba(adv[name]).argmax(1) == non_vpn).mean()})

        X = X_test[targets["FT-Transformer"]]
        pgd = pgd_attack(model, X, vpn, eps)
        rows.append({"eps": eps, "attack": "White-box PGD", "model": "FT-Transformer",
                     "evasion_rate": (probas["FT-Transformer"](pgd).argmax(1) == non_vpn).mean()})

        # Transfer: craft on one model, replay on the other (flows both detect).
        both = np.intersect1d(targets["FT-Transformer"], targets["Random Forest"])
        for src, dst in (("FT-Transformer", "Random Forest"), ("Random Forest", "FT-Transformer")):
            crafted = random_search_attack(probas[src], X_test[both], vpn, eps, seed=SEED)
            rows.append({"eps": eps, "attack": f"Transfer from {src}", "model": dst,
                         "evasion_rate": (probas[dst](crafted).argmax(1) == non_vpn).mean()})
        pgd_both = pgd_attack(model, X_test[both], vpn, eps)
        rows.append({"eps": eps, "attack": "Transfer from FT-Transformer (PGD)", "model": "Random Forest",
                     "evasion_rate": (probas["Random Forest"](pgd_both).argmax(1) == non_vpn).mean()})
        print(f"  eps={eps}: done")

    feature_space = pd.DataFrame(rows)
    feature_space.to_csv(f"{OUT_DIR}/feature_space_attacks.csv", index=False)

    plot_shaping(shaping, colors)
    plot_feature_space(feature_space, colors)
    write_summary(shaping, feature_space, targets)

# ==========================================
# 3. Figures
# ==========================================
def plot_shaping(shaping, colors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mode in zip(axes, PADDINGS):
        for name, color in colors.items():
            needed = shaping[(name, mode)]
            rate = [(needed <= s).mean() * 100 for s in SLOWDOWNS]
            ax.plot(SLOWDOWNS, rate, color=color, linewidth=2, label=name)
            ax.annotate(f"{name} ({rate[-1]:.0f}%)", (SLOWDOWNS[-1], rate[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=8.5)
        ax.set_title(mode)
        ax.set_xlabel("slowdown factor the attacker accepts (1.5 = 50% slower)")
        ax.set_xlim(1, SLOWDOWNS[-1] + 1.6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("detected VPN flows that evade (%)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Traffic-shaping attack: how much slowdown does evasion cost?",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/shaping_attack.png", dpi=160)
    plt.close(fig)


def plot_feature_space(df, colors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax = axes[0]
    for (attack, name), style in ((("Black-box random search", "FT-Transformer"), "-"),
                                  (("Black-box random search", "Random Forest"), "-"),
                                  (("White-box PGD", "FT-Transformer"), "--")):
        d = df[(df.attack == attack) & (df.model == name)]
        label = f"{name}, {'black-box' if 'Black' in attack else 'white-box PGD'}"
        ax.plot(d.eps, d.evasion_rate * 100, style, color=colors[name], linewidth=2,
                marker="o", markersize=4, label=label)
    ax.set_title("Direct attack")

    ax = axes[1]
    for attack, name, style in (("Transfer from Random Forest", "FT-Transformer", "-"),
                                ("Transfer from FT-Transformer", "Random Forest", "-"),
                                ("Transfer from FT-Transformer (PGD)", "Random Forest", "--")):
        d = df[(df.attack == attack) & (df.model == name)]
        ax.plot(d.eps, d.evasion_rate * 100, style, color=colors[name], linewidth=2,
                marker="o", markersize=4, label=f"on {name}, {attack.lower()}")
    ax.set_title("Transfer attack (crafted on the other model)")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(EPSILONS, [str(e) for e in EPSILONS])
        ax.set_xlabel("perturbation budget eps (std. units per feature)")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("detected VPN flows that evade (%)")
    fig.suptitle("Feature-space attacks (worst case)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/feature_space_attacks.png", dpi=160)
    plt.close(fig)

# ==========================================
# 4. Summary
# ==========================================
def write_summary(shaping, feature_space, targets):
    budgets = [1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
    shaping_table = pd.DataFrame(
        {f"{name} / {mode}": [(shaping[(name, mode)] <= b).mean() * 100 for b in budgets]
         for (name, mode) in shaping},
        index=pd.Index([f"<= {b}x" for b in budgets], name="slowdown")).round(1)

    median = {k: np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan
              for k, v in shaping.items()}

    fs = feature_space.assign(evasion_rate=feature_space.evasion_rate * 100)
    fs_table = fs.pivot_table(index=["attack", "model"], columns="eps", values="evasion_rate").round(1)

    parts = [
        "# Adversarial robustness: VPN evasion",
        "Attacker goal: make detected VPN flows look Non-VPN. Seed-42 test split. "
        "Evasion rate = % of VPN flows the model detected that are classified Non-VPN after the attack. "
        + ", ".join(f"{n}: {len(t)} detected flows attacked" for n, t in targets.items()) + ".",
        "## 1. Traffic shaping (realistic, black-box)",
        "Attacker slows the flow down (every gap stretched) and optionally pads packets "
        "(byte overhead up to 100%). Evasion rate (%) within each slowdown budget:",
        shaping_table.to_markdown(),
        "Median slowdown needed among flows that can evade within 5x: "
        + "; ".join(f"{n} / {m}: {v:.2f}x" for (n, m), v in median.items()),
        "## 2. Feature-space attacks (worst case)",
        "L-inf budget eps on standardized features (1.0 = one standard deviation per feature). "
        "Evasion rate (%):",
        fs_table.to_markdown(),
        "## Figures",
        "- `shaping_attack.png`\n- `feature_space_attacks.png`",
        "## Caveats",
        "- Traffic shaping is simulated on flow features, not re-captured packets; stretched gaps "
        "crossing the 1 s activity threshold would also create idle periods in a real capture.\n"
        "- Feature-space attacks ignore whether the perturbed feature vector is physically "
        "achievable: they are an upper bound on attacker power, not a realistic attack.",
    ]
    text = "\n\n".join(parts) + "\n"
    with open(f"{OUT_DIR}/summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
