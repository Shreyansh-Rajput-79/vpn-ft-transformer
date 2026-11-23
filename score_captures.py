"""Measure accuracy on your own labelled captures (in-the-wild evaluation).

Record traffic while you know the truth, e.g. 5-10 min of normal use with the VPN
off and the same with it on, then score one or more models on it:

    python predict_live.py -i Wi-Fi --seconds 600 -o novpn.csv --label Non-VPN
    python predict_live.py -i Wi-Fi --seconds 600 -o vpn.csv   --label VPN
    python score_captures.py Non-VPN=novpn.csv VPN=vpn.csv \
        -m models/vnat_model.pth models/vnat_noflags_model.pth -o results/in_the_wild.md

Inputs are LABEL=path. A path can be a CSV from predict_live.py, any CSV with the
model's feature columns, or a .pcap/.pcapng (e.g. recorded with Wireshark).
Every model re-classifies every flow, so models are compared on identical flows.
Background flows (DNS, DHCP, ...) are dropped, as in training.

With the VPN on, not every flow goes through the tunnel (split tunnelling, LAN
traffic, the VPN app itself). Those flows really are non-VPN, so counting them as
"VPN" would blame the model for a wrong label. Pass --vpn-server with your VPN
server's IP and/or port: VPN-labelled flows that do not touch it are reported as
"bypassed the tunnel" and left out of the score.

    python score_captures.py Non-VPN=novpn.csv VPN=vpn.csv --vpn-server 1195 203.0.113.7
"""
import ipaddress
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from build_vnat_dataset import BACKGROUND_PORTS
from predict import load_input, load_model, predict_df
from utils.preprocessing import LABEL_MAP


def drop_background(df):
    if not {"src_port", "dst_port", "protocol"} <= set(df.columns):
        return df, 0
    ports = df[["src_port", "dst_port"]].apply(pd.to_numeric, errors="coerce")
    keep = df["protocol"].astype(str).isin(["TCP", "UDP"]) & ~ports.isin(BACKGROUND_PORTS).any(axis=1)
    return df[keep].reset_index(drop=True), int((~keep).sum())


def parse_endpoints(values):
    """--vpn-server values -> (set of ports, set of IPs)."""
    ports, ips = set(), set()
    for v in values or []:
        if v.isdigit():
            ports.add(int(v))
        else:
            try:
                ips.add(str(ipaddress.ip_address(v)))
            except ValueError:
                raise SystemExit(f"[ERROR] --vpn-server '{v}' is neither a port nor an IP address")
    return ports, ips


def through_tunnel(df, ports, ips):
    hit = pd.Series(False, index=df.index)
    if ports:
        hit |= df[["src_port", "dst_port"]].apply(pd.to_numeric, errors="coerce").isin(ports).any(axis=1)
    if ips:
        hit |= df["src_ip"].astype(str).isin(ips) | df["dst_ip"].astype(str).isin(ips)
    return hit


def load_labelled(specs, feature_set, vpn_server=None):
    ports, ips = parse_endpoints(vpn_server)
    frames = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"[ERROR] '{spec}': use LABEL=path, e.g. VPN=vpn.csv")
        label, path = spec.split("=", 1)
        df = load_input(path, feature_set)
        df, dropped = drop_background(df)
        label = LABEL_MAP.get(label, label)
        df["true_label"] = label
        df["source"] = Path(path).name
        note = f"{dropped} background flows dropped"
        if label == "VPN" and (ports or ips):
            tunnel = through_tunnel(df, ports, ips)
            note += f", {int((~tunnel).sum())} flows bypassed the tunnel and are not scored"
            df = df[tunnel].reset_index(drop=True)
        elif label == "VPN":
            note += ("; no --vpn-server given, so flows that bypassed the tunnel count as "
                     "VPN and any 'Non-VPN' answer for them is scored as a mistake")
        print(f"[INFO] {path}: {len(df)} flows labelled {label} ({note})")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def score(df, classes):
    correct = df.prediction == df.true_label
    per_source = df.assign(correct=correct).groupby(["source", "true_label"]).agg(
        flows=("correct", "size"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
    labels = sorted(set(classes) | set(df.true_label))
    cm = pd.DataFrame(confusion_matrix(df.true_label, df.prediction, labels=labels),
                      index=[f"true {l}" for l in labels], columns=[f"pred {l}" for l in labels])
    confidence = df.confidence.astype(float)
    pct = lambda s: f"{s.mean():.1%}" if len(s) else "n/a"
    conf = {"confidence when right": pct(confidence[correct]),
            "confidence when wrong": pct(confidence[~correct]),
            "wrong but >= 90% confident": int(((~correct) & (confidence >= 0.9)).sum())}
    return correct.mean(), per_source, cm, conf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="LABEL=path, e.g. VPN=vpn.csv Non-VPN=novpn.pcap")
    parser.add_argument("-m", "--models", nargs="+", default=["models/vnat_model.pth"])
    parser.add_argument("-o", "--output", help="write the report as Markdown")
    parser.add_argument("--vpn-server", nargs="+", metavar="IP_OR_PORT",
                        help="your VPN server's IP and/or port: only VPN-labelled flows that touch "
                             "it are scored (others bypassed the tunnel)")
    args = parser.parse_args()

    parts = ["# In-the-wild evaluation", f"Inputs: {', '.join(args.inputs)}"]
    summary = []
    for model_path in args.models:
        model, ckpt = load_model(model_path)
        df = load_labelled(args.inputs, ckpt["feature_set"], args.vpn_server)
        unknown = sorted(set(df.true_label) - set(ckpt["classes"]))
        if unknown:
            print(f"[WARN] Labels {unknown} are not classes of {model_path} ({ckpt['classes']}); "
                  "those flows can only be wrong.")
        res = predict_df(df, model, ckpt, with_explanations=True)
        acc, per_source, cm, conf = score(res, ckpt["classes"])
        summary.append({"model": model_path, "features": ckpt["feature_set"], "flows": len(res),
                        "accuracy": f"{acc:.1%}", **conf})
        wrong = res[res.prediction != res.true_label]
        parts += [f"## {model_path} ({ckpt['feature_set']} features)",
                  f"Accuracy: **{acc:.1%}** over {len(res)} flows.",
                  per_source.to_markdown(floatfmt=".3f"), cm.to_markdown(),
                  "Example mistakes:\n\n" + ("\n".join(
                      f"- true {r.true_label}, predicted {r.prediction}: {r.reason}"
                      for r in wrong.head(5).itertuples()) or "- none")]

    parts.insert(2, "## Summary\n\n" + pd.DataFrame(summary).to_markdown(index=False))
    text = "\n\n".join(parts) + "\n"
    print("\n" + text.encode("ascii", "replace").decode())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"[INFO] Report saved to {args.output}")


if __name__ == "__main__":
    main()
