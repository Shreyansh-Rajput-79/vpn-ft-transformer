"""Classify flows as VPN / Non-VPN.

Usage:
    python predict.py capture.pcap               # raw capture (.pcap / .pcapng)
    python predict.py flows.csv                  # ISCX-style feature CSV
    python predict.py capture.pcap -o out.csv    # also save per-flow results
    python predict.py capture.pcap --explain     # add the top features behind each decision
"""
# ==========================================
# 1. Imports
# ==========================================
import argparse
from pathlib import Path

import pandas as pd
import torch

from models.ft_transformer import FTTransformer
from utils.explain import explain, top_features
from utils.preprocessing import (LABEL_COL, LABEL_MAP, fix_iscx_headers,
                                 standardize, to_matrix)

# ==========================================
# 2. Load Saved Model
# ==========================================
def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = FTTransformer(len(ckpt["features"]), len(ckpt["classes"]), **ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt

# ==========================================
# 3. Prediction
# ==========================================
def predict_df(df, model, ckpt, with_explanations=False):
    """Adds `prediction` and `confidence` columns to a DataFrame of flow features.

    with_explanations also adds `top_features`: the three features the model
    attended to most for that flow (attention rollout).
    """
    X = standardize(to_matrix(df), ckpt["mean"], ckpt["std"])
    probs, relevance = explain(model, X)

    df = df.copy()
    df["prediction"] = [ckpt["classes"][i] for i in probs.argmax(axis=1)]
    df["confidence"] = probs.max(axis=1).round(4)
    if with_explanations:
        df["top_features"] = top_features(relevance, ckpt["features"])
    return df


def load_input(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".pcap", ".pcapng", ".cap"):
        from utils.pcap_processing import pcap_to_df
        df = pcap_to_df(path)
        print(f"[INFO] Extracted {len(df)} TCP/UDP flows from {path}")
    elif suffix == ".csv":
        df = fix_iscx_headers(pd.read_csv(path))
        print(f"[INFO] Loaded {len(df)} rows from {path}")
    else:
        raise SystemExit(f"[ERROR] Unsupported file type '{suffix}', expected .csv or .pcap/.pcapng")
    return df

# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help=".csv of ISCX features or a .pcap/.pcapng capture")
    parser.add_argument("-m", "--model", default="models/model.pth")
    parser.add_argument("-o", "--output", help="write per-flow predictions to this CSV")
    parser.add_argument("--explain", action="store_true", help="add the top features behind each prediction")
    args = parser.parse_args()

    model, ckpt = load_model(args.model)
    df = load_input(args.input)
    if df.empty:
        raise SystemExit("[ERROR] No flows found in input.")

    results = predict_df(df, model, ckpt, with_explanations=args.explain)

    print("\n[RESULT] Flows per class:")
    print(results["prediction"].value_counts().to_string())

    if LABEL_COL in results.columns:
        truth = results[LABEL_COL].astype(str).str.strip().replace(LABEL_MAP)
        print(f"\n[RESULT] Accuracy vs. '{LABEL_COL}' column: {(truth == results['prediction']).mean():.4f}")

    shown = [c for c in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol", "packets")
             if c in results.columns] + ["prediction", "confidence"]
    shown += ["top_features"] if args.explain else []
    print("\n[RESULT] First flows:")
    print(results[shown].head(10).to_string(index=False))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n[INFO] Saved predictions to {args.output}")
