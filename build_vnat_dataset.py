"""Turn the VNAT pcaps into a labelled feature CSV that train.py can use.

VNAT file names carry the label: vpn_*.pcap -> VPN, nonvpn_*.pcap -> Non-VPN
(e.g. vpn_netflix_A.pcap, nonvpn_ssh_capture1.pcap).

Usage:
    python build_vnat_dataset.py path/to/VNAT -o data/vnat_features.csv -j 8

Then:
    python train.py --data data/vnat_features.csv --group-col source_file
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from utils.pcap_processing import pcap_to_df
from utils.preprocessing import LABEL_COL

PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}


def label_from_name(path):
    name = path.name.lower()
    if name.startswith("nonvpn"):
        return "Non-VPN"
    if name.startswith("vpn"):
        return "VPN"
    return None


def process(path):
    df = pcap_to_df(str(path))
    df[LABEL_COL] = label_from_name(path)
    df["source_file"] = path.name
    return path, df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vnat_dir", help="folder containing the VNAT pcaps (searched recursively)")
    parser.add_argument("-o", "--output", default="data/vnat_features.csv")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="parallel processes")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.vnat_dir).rglob("*") if p.suffix.lower() in PCAP_SUFFIXES)
    skipped = [p.name for p in files if label_from_name(p) is None]
    files = [p for p in files if label_from_name(p) is not None]
    if skipped:
        print(f"[WARN] Skipping {len(skipped)} files without a vpn_/nonvpn_ prefix: {skipped[:5]}...")
    if not files:
        raise SystemExit(f"[ERROR] No labelled pcaps found under {args.vnat_dir}")

    print(f"[INFO] Extracting flows from {len(files)} pcaps with {args.jobs} processes...")
    frames = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(process, p) for p in files]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                path, df = future.result()
            except Exception as e:  # a corrupt capture should not kill the whole run
                print(f"[WARN] Failed: {e}")
                continue
            frames.append(df)
            print(f"[{done}/{len(files)}] {path.name}: {len(df)} flows")

    data = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)

    print(f"\n[INFO] Saved {len(data)} flows to {args.output}")
    print(data[LABEL_COL].value_counts().to_string())


if __name__ == "__main__":
    main()
