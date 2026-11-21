"""Classify flows as VPN / Non-VPN (or by category / app, depending on the model).

Usage:
    python predict.py capture.pcap               # raw capture (.pcap / .pcapng)
    python predict.py flows.csv                  # feature CSV
    python predict.py capture.pcap -o out.csv    # also save per-flow results
    python predict.py capture.pcap --explain     # add the top features behind each decision
    python predict.py hotspot.pcap --by-device   # per-device and device-to-device summary
"""
# ==========================================
# 1. Imports
# ==========================================
import argparse
import ipaddress
from pathlib import Path

import pandas as pd
import torch

from models.ft_transformer import FTTransformer
from utils.explain import explain, top_features
from utils.narrate import reasons
from utils.preprocessing import (LABEL_MAP, fix_iscx_headers, standardize,
                                 to_matrix)

# ==========================================
# 2. Load Saved Model
# ==========================================
def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt.setdefault("feature_set", "iscx")   # models saved before NFStream support
    ckpt.setdefault("label_col", "class1")
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
    attended to most for that flow (attention rollout), and `reason`: the same
    in plain language, compared with typical values of each class.
    """
    Z = to_matrix(df, ckpt["features"])
    X = standardize(Z, ckpt["mean"], ckpt["std"])
    probs, relevance = explain(model, X)

    df = df.copy()
    df["prediction"] = [ckpt["classes"][i] for i in probs.argmax(axis=1)]
    df["confidence"] = probs.max(axis=1).round(4)
    if with_explanations:
        df["top_features"] = top_features(relevance, ckpt["features"])
        df["reason"] = reasons(Z, relevance, probs, ckpt["features"], ckpt["classes"],
                               ckpt.get("class_medians"))
    return df


def load_input(path, feature_set):
    suffix = Path(path).suffix.lower()
    if suffix in (".pcap", ".pcapng", ".cap"):
        from utils.pcap_processing import EXTRACTORS
        df = EXTRACTORS[feature_set](path)
        print(f"[INFO] Extracted {len(df)} TCP/UDP flows ({feature_set} features) from {path}")
    elif suffix == ".csv":
        df = fix_iscx_headers(pd.read_csv(path))
        print(f"[INFO] Loaded {len(df)} rows from {path}")
    else:
        raise SystemExit(f"[ERROR] Unsupported file type '{suffix}', expected .csv or .pcap/.pcapng")
    return df

# ==========================================
# 4. Per-device summary (e.g. devices on your own hotspot)
# ==========================================
def _is_local(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def device_summary(results):
    """One row per local device: flows, share of each predicted class, verdict."""
    local_src = results.src_ip.map(_is_local)
    device = results.src_ip.where(local_src, results.dst_ip)
    shares = pd.crosstab(device, results.prediction, normalize="index").mul(100).round(1)
    table = pd.concat([device.value_counts().rename("flows"), shares], axis=1).fillna(0)
    table["verdict"] = shares.idxmax(axis=1)
    table.index.name = "device"
    return table.sort_values("flows", ascending=False)


def pair_summary(results):
    """Flows directly between two local devices, grouped by device pair."""
    local = results[results.src_ip.map(_is_local) & results.dst_ip.map(_is_local)]
    if local.empty:
        return None
    pair = local.apply(lambda r: " <-> ".join(sorted([r.src_ip, r.dst_ip])), axis=1)
    shares = pd.crosstab(pair, local.prediction, normalize="index").mul(100).round(1)
    table = pd.concat([pair.value_counts().rename("flows"), shares], axis=1).fillna(0)
    table["verdict"] = shares.idxmax(axis=1)
    table.index.name = "device pair"
    return table.sort_values("flows", ascending=False)

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="feature .csv or a .pcap/.pcapng capture")
    parser.add_argument("-m", "--model", default="models/model.pth")
    parser.add_argument("-o", "--output", help="write per-flow predictions to this CSV")
    parser.add_argument("--explain", action="store_true", help="add the top features behind each prediction")
    parser.add_argument("--by-device", action="store_true",
                        help="summarize per local device and per device pair (pcap input)")
    args = parser.parse_args()

    model, ckpt = load_model(args.model)
    df = load_input(args.input, ckpt["feature_set"])
    if df.empty:
        raise SystemExit("[ERROR] No flows found in input.")

    results = predict_df(df, model, ckpt, with_explanations=args.explain)

    print("\n[RESULT] Flows per class:")
    print(results["prediction"].value_counts().to_string())

    label_col = ckpt["label_col"]
    if label_col in results.columns:
        truth = results[label_col].astype(str).str.strip()
        if label_col == "class1":
            truth = truth.replace(LABEL_MAP)
        print(f"\n[RESULT] Accuracy vs. '{label_col}' column: {(truth == results['prediction']).mean():.4f}")

    shown = [c for c in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol", "packets")
             if c in results.columns] + ["prediction", "confidence"]
    shown += ["top_features"] if args.explain else []
    print("\n[RESULT] First flows:")
    print(results[shown].head(10).to_string(index=False))
    if args.explain:
        print("\n[RESULT] Reasons (first 5 flows):")
        for i, text in enumerate(results["reason"].head(5), 1):
            print(f"  flow {i}: {text}")

    if args.by_device:
        if "src_ip" not in results.columns:
            raise SystemExit("[ERROR] --by-device needs a pcap input (flow addresses).")
        print("\n[RESULT] Per device (% of its flows in each class):")
        print(device_summary(results).to_string())
        pairs = pair_summary(results)
        print("\n[RESULT] Device-to-device flows:")
        print(pairs.to_string() if pairs is not None else "  none (no flows between two local addresses)")

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n[INFO] Saved predictions to {args.output}")
