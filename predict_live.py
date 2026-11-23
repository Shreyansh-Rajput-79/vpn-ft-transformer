"""Live classification: capture from a network interface and explain each flow as it ends.

Each flow is classified (VPN / Non-VPN, or category, depending on the model) with
a plain-language reason, about 0-30 s after its traffic: a flow is judged once it
reaches the 15 s cut used in training or goes quiet for 15 s.

Needs an NFStream-feature model (e.g. models/vnat_model.pth) and, on Windows, Npcap.

Usage:
    python predict_live.py --list                             # show interfaces
    python predict_live.py -i Wi-Fi                           # VPN / Non-VPN, until Ctrl+C
    python predict_live.py -i Wi-Fi -m models/vnat_noflags_model.pth   # without the TCP-flag shortcut
    python predict_live.py -i Wi-Fi -m models/vnat_category_model.pth
    python predict_live.py -i Wi-Fi --seconds 120 -o live.csv
    python predict_live.py -i Wi-Fi --keep-background         # also judge DNS, DHCP, ...
    python predict_live.py -i Wi-Fi -m models/vnat_noflags_model.pth --trace 2   # show every stage for 2 flows

Background flows (DNS, DHCP, NTP, NetBIOS, SMB, LDAP, discovery, non-TCP/UDP) are
skipped by default: the model was trained without them (build_vnat_dataset.clean_flows),
so its answer for them would be meaningless.
"""
import argparse
import json
import os
import subprocess
import time

import pandas as pd

from build_vnat_dataset import BACKGROUND_PORTS
from predict import load_model, predict_df
from utils.pcap_processing import FLOW_TIMEOUT_US, NFSTREAM_METERS
from utils.preprocessing import NFSTREAM_FEATURES
from utils.trace import trace_flow

CUT_S = FLOW_TIMEOUT_US // 1_000_000  # 15 s, as in training
PROTOCOLS = {6: "TCP", 17: "UDP"}


# ------------------------------------------
# Interfaces
# ------------------------------------------
def windows_adapters():
    """[(friendly name, Npcap device name, description)] for adapters that are up."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-NetAdapter | Where-Object Status -eq Up | "
         "Select-Object Name, InterfaceGuid, InterfaceDescription | ConvertTo-Json"],
        capture_output=True, text=True, check=True).stdout
    rows = json.loads(out) if out.strip() else []
    rows = rows if isinstance(rows, list) else [rows]
    return [(r["Name"], rf"\Device\NPF_{r['InterfaceGuid']}", r["InterfaceDescription"]) for r in rows]


def resolve_interface(name):
    """NFStream on Windows needs the Npcap device name; accept the friendly name too."""
    if os.name != "nt" or name.startswith("\\Device\\"):
        return name
    for friendly, device, _ in windows_adapters():
        if friendly.lower() == name.lower():
            return device
    raise SystemExit(f"[ERROR] No active interface called '{name}'. Run with --list.")


# ------------------------------------------
# Flows
# ------------------------------------------
def is_background(flow):
    return (flow.protocol not in PROTOCOLS
            or flow.src_port in BACKGROUND_PORTS or flow.dst_port in BACKGROUND_PORTS)


def flow_row(flow):
    row = {
        "src_ip": flow.src_ip, "src_port": flow.src_port,
        "dst_ip": flow.dst_ip, "dst_port": flow.dst_port,
        "protocol": PROTOCOLS.get(flow.protocol, str(flow.protocol)),
        "start_time": pd.to_datetime(flow.bidirectional_first_seen_ms, unit="ms"),
        "packets": flow.bidirectional_packets, "bytes": flow.bidirectional_bytes,
    }
    row.update({f: getattr(flow, f) for f in NFSTREAM_FEATURES})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--interface", help="interface name, e.g. Wi-Fi (see --list)")
    parser.add_argument("--list", action="store_true", help="list active interfaces and exit")
    parser.add_argument("-m", "--model", default="models/vnat_model.pth")
    parser.add_argument("--seconds", type=float, default=None, help="stop after this long (default: Ctrl+C)")
    parser.add_argument("-o", "--output", help="append every judged flow to this CSV")
    parser.add_argument("--label", default=None,
                        help="true class of everything captured (e.g. VPN while your VPN is on); "
                             "saved as a column so score_captures.py can measure accuracy")
    parser.add_argument("--keep-background", action="store_true",
                        help="also classify DNS, DHCP and other housekeeping flows")
    parser.add_argument("--bpf", default=None, help="capture filter, e.g. \"host 192.168.1.20\"")
    parser.add_argument("--technical", action="store_true",
                        help="print the compact technical reason instead of the plain-language one")
    parser.add_argument("--trace", type=int, default=0, metavar="N",
                        help="print every pipeline stage (capture, flow, features, preprocessing, "
                             "model, explanation) for the first N judged flows")
    args = parser.parse_args()

    if args.list:
        if os.name != "nt":
            raise SystemExit("Use the interface name from `ip link` (e.g. wlan0).")
        for friendly, device, desc in windows_adapters():
            print(f"{friendly:20s} {desc}\n{'':20s} {device}")
        return
    if not args.interface:
        parser.error("--interface is required (see --list)")

    model, ckpt = load_model(args.model)
    if not ckpt["feature_set"].startswith("nfstream"):
        raise SystemExit(f"[ERROR] {args.model} uses '{ckpt['feature_set']}' features; live mode "
                         "needs an NFStream-feature model (e.g. models/vnat_model.pth).")

    from nfstream import NFStreamer
    # Flows are cut 15 s after their first packet, as in training. A 15 s idle
    # timeout gives the same flows as training's 120 s (a flow of at most 15 s
    # cannot contain a longer gap) but reports quiet flows sooner.
    streamer = NFStreamer(source=resolve_interface(args.interface), statistical_analysis=True,
                          n_meters=NFSTREAM_METERS,
                          active_timeout=CUT_S, idle_timeout=CUT_S, bpf_filter=args.bpf)

    print(f"[INFO] Model {args.model} ({ckpt['label_col']}: {', '.join(ckpt['classes'])}). "
          f"Capturing on {args.interface}; flows are judged when they end (<= {CUT_S} s) "
          f"or go quiet. Ctrl+C to stop.\n")
    counts, skipped, start, header = {}, 0, time.time(), True
    try:
        for flow in streamer:
            if args.seconds and time.time() - start > args.seconds:
                break
            if not args.keep_background and is_background(flow):
                skipped += 1
                continue
            row = flow_row(flow)
            result = predict_df(pd.DataFrame([row]), model, ckpt, with_explanations=True).iloc[0]
            counts[result.prediction] = counts.get(result.prediction, 0) + 1
            if sum(counts.values()) <= args.trace:
                trace_flow(sum(counts.values()), row, model, ckpt)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] {result.protocol} {result.src_ip}:{result.src_port} -> "
                      f"{result.dst_ip}:{result.dst_port} ({result.packets} pkts)\n  "
                      + (result.reason if args.technical else result.explanation_text) + "\n", flush=True)
            if args.output:
                if args.label:
                    result["true_label"] = args.label
                result.drop("explanation_text").to_frame().T.to_csv(args.output, mode="a", header=header and not os.path.exists(args.output),
                                           index=False)
                header = False
    except KeyboardInterrupt:
        pass

    total = sum(counts.values())
    print(f"\n[SUMMARY] {total} flows judged in {time.time() - start:.0f} s, "
          f"{skipped} background flows skipped.")
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {n} ({n / total:.0%})")


if __name__ == "__main__":
    main()
