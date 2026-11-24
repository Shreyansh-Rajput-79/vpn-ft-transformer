"""Live classroom demo: slow VPN traffic down and see which model is fooled.

Uses the shipped FT-Transformer (models/vnat_timing_model.pth, 23 timing features)
and trains a Random Forest on exactly the same training split (seed 42, split by
capture file), so both models are compared on equal terms. Then it takes VPN
flows from the held-out test recordings that BOTH models detect, applies the same
traffic shaping as security_eval.py (every gap between packets stretched), and
counts how many flows each model now calls Non-VPN.

Run from the project folder:
    python tools/demo_shaping.py                 # 2x slowdown
    python tools/demo_shaping.py --slowdown 1.5
    python tools/demo_shaping.py --slowdown 3

Single split, so the numbers differ a little from the 5-seed averages in
results/FINAL_RESULTS.md section 6.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from predict import load_model
from utils.attacks import shape_traffic
from utils.preprocessing import from_matrix, prepare_dataset, split_indices, standardize
from utils.training import predict_proba

# feature -> (label, scale): time features are stored in microseconds, shown in seconds
SHOW = {"duration": ("flow duration (s)", 1e-6),
        "mean_flowiat": ("avg gap between packets (s)", 1e-6),
        "flowPktsPerSecond": ("packets per second", 1),
        "flowBytesPerSecond": ("bytes per second", 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slowdown", type=float, default=2.0, help="2 = traffic runs at half speed")
    ap.add_argument("--data", default="data/vnat_features.csv")
    ap.add_argument("--model", default="models/vnat_timing_model.pth")
    args = ap.parse_args()

    print("Loading the FT-Transformer and the VNAT timing features...")
    ft, ckpt = load_model(args.model)
    data = prepare_dataset(args.data, "class1")
    Z, y, feats = data["Z"], data["y"], data["features"]
    vpn, non_vpn = data["classes"].index("VPN"), data["classes"].index("Non-VPN")
    groups = data["df"]["source_file"].to_numpy()
    tr, _, te = split_indices(y, seed=42, groups=groups)   # same split the model was trained on
    mean, std = ckpt["mean"], ckpt["std"]

    print("Training a Random Forest on the same training recordings (a few seconds)...")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(standardize(Z[tr], mean, std), y[tr])

    models = {
        "Random Forest": lambda Zs: rf.predict_proba(standardize(Zs, mean, std)),
        "FT-Transformer": lambda Zs: predict_proba(ft, standardize(Zs, mean, std)),
    }

    # VPN flows from unseen test recordings that both models catch before the attack
    Zt = Z[te][y[te] == vpn]
    caught = np.ones(len(Zt), bool)
    for proba in models.values():
        caught &= proba(Zt).argmax(1) == vpn
    Zv = Zt[caught]
    Zs = shape_traffic(Zv, args.slowdown, 1.0, feats)
    print(f"\n{len(Zv)} VPN flows from unseen test recordings, caught by BOTH models.")
    print(f"Attack: every gap between packets stretched {args.slowdown:g}x "
          f"(the connection runs at {100 / args.slowdown:.0f}% speed). Nothing else changes.\n")

    after = {}
    print(f"{'Model':16s} {'still caught as VPN':>22s} {'slipped through as Non-VPN':>28s}")
    for name, proba in models.items():
        pred = proba(Zs).argmax(1)
        after[name] = pred
        n_ev = int((pred == non_vpn).sum())
        print(f"{name:16s} {len(Zv) - n_ev:>14d} ({100 * (1 - n_ev / len(Zv)):5.1f}%) "
              f"{n_ev:>18d} ({100 * n_ev / len(Zv):5.1f}%)")

    # One concrete flow: fools the Random Forest, not the FT-Transformer
    pick = np.where((after["Random Forest"] == non_vpn) & (after["FT-Transformer"] == vpn))[0]
    if len(pick) == 0:
        print("\nNo single flow fooled only the Random Forest at this slowdown.")
        return
    i = pick[len(pick) // 2]
    before_raw, after_raw = from_matrix(Zv[i:i + 1])[0], from_matrix(Zs[i:i + 1])[0]
    print(f"\nExample flow ({len(pick)} flows behave like this one):")
    print(f"  {'':30s} {'before':>10s} {'after':>10s}")
    for f, (label, scale) in SHOW.items():
        if f in feats:
            j = feats.index(f)
            print(f"  {label:30s} {before_raw[j] * scale:10.2f} {after_raw[j] * scale:10.2f}")
    for name, proba in models.items():
        p0, p1 = proba(Zv[i:i + 1])[0], proba(Zs[i:i + 1])[0]
        print(f"  {name:16s} before: {data['classes'][p0.argmax()]:8s} after: {data['classes'][p1.argmax()]}")


if __name__ == "__main__":
    main()
