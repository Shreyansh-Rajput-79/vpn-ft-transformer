"""Turn the VNAT pcaps into a labelled feature CSV that train.py can use.

VNAT names its captures vpn_<app>_capture<#>.pcap / nonvpn_<app>_capture<#>.pcap
(MIT Lincoln Laboratory). Every flow gets three labels:
    class1    VPN / Non-VPN
    app       netflix, youtube, vimeo, voip, skype-chat, ssh, rdp, sftp, rsync, scp
    category  Streaming, VoIP, Chat, C2, File Transfer

Usage:
    python build_vnat_dataset.py path/to/VNAT -o data/vnat_features.csv                  # 23 ISCX features
    python build_vnat_dataset.py path/to/VNAT -o data/vnat_nfstream.csv --extractor nfstream

Then e.g.:
    python train.py --data data/vnat_features.csv --group-col source_file
    python train.py --data data/vnat_features.csv --group-col source_file --label-col category
"""
import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from utils.pcap_processing import EXTRACTORS
from utils.preprocessing import LABEL_COL

PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}

# VNAT's filename keywords -> traffic category (from the MIT LL dataset page).
APP_CATEGORY = {
    "netflix": "Streaming", "youtube": "Streaming", "vimeo": "Streaming",
    "voip": "VoIP",
    "skype-chat": "Chat",
    "ssh": "C2", "rdp": "C2",
    "sftp": "File Transfer", "rsync": "File Transfer", "scp": "File Transfer",
}

_NAME = re.compile(r"^(nonvpn|vpn)_(.+?)(?:_capture\d*|_[a-z])?$")


def parse_name(path):
    """(VPN label, app, category) from a VNAT file name, or None if it doesn't match."""
    m = _NAME.match(path.stem.lower())
    if not m:
        return None
    vpn = "VPN" if m.group(1) == "vpn" else "Non-VPN"
    app = m.group(2)
    return vpn, app, APP_CATEGORY.get(app, "Other")


def process(path, extractor):
    df = EXTRACTORS[extractor](str(path))
    df[LABEL_COL], df["app"], df["category"] = parse_name(path)
    df["source_file"] = path.name
    return path, df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vnat_dir", help="folder containing the VNAT pcaps (searched recursively)")
    parser.add_argument("-o", "--output", default="data/vnat_features.csv")
    parser.add_argument("--extractor", choices=list(EXTRACTORS), default="iscx",
                        help="iscx: 23 ISCX time-based features (comparable with the ISCX CSV); "
                             "nfstream: 57 NFStream behavioural features")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="parallel processes")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.vnat_dir).rglob("*") if p.suffix.lower() in PCAP_SUFFIXES)
    skipped = [p.name for p in files if parse_name(p) is None]
    files = [p for p in files if parse_name(p) is not None]
    if skipped:
        print(f"[WARN] Skipping {len(skipped)} files without a vpn_/nonvpn_ prefix: {skipped[:5]}...")
    unknown = sorted({parse_name(p)[1] for p in files if parse_name(p)[2] == "Other"})
    if unknown:
        print(f"[WARN] Apps without a known category (labelled 'Other'): {unknown}")
    if not files:
        raise SystemExit(f"[ERROR] No labelled pcaps found under {args.vnat_dir}")

    print(f"[INFO] Extracting '{args.extractor}' features from {len(files)} pcaps "
          f"with {args.jobs} processes...")
    frames = []

    def collect(done, name, result):
        try:
            path, df = result()
        except Exception as e:  # a corrupt capture should not kill the whole run
            print(f"[WARN] Failed on {name}: {e}")
            return
        frames.append(df)
        print(f"[{done}/{len(files)}] {path.name}: {len(df)} flows")

    if args.extractor == "nfstream":
        # NFStream starts its own worker processes; nesting it inside a process
        # pool hangs on Windows. It is fast (C engine), so run files one by one.
        for done, p in enumerate(files, 1):
            collect(done, p.name, lambda p=p: process(p, args.extractor))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process, p, args.extractor): p for p in files}
            for done, future in enumerate(as_completed(futures), 1):
                collect(done, futures[future].name, future.result)

    data = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)

    print(f"\n[INFO] Saved {len(data)} flows to {args.output}")
    print(pd.crosstab(data["category"], data[LABEL_COL], margins=True).to_string())


if __name__ == "__main__":
    main()
