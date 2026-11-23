"""Print every pipeline stage for one flow: capture -> flow -> features -> preprocessing
-> FT-Transformer -> explanation. Used by predict_live.py and predict.py (--trace N).

Uses the same functions as predict.predict_df, so the verdict is identical to the
normal output; it only shows the intermediate values.
"""
import time

import numpy as np
import pandas as pd
import torch

from .explain import explain
from .narrate import describe, plain_reason, reasons
from .preprocessing import standardize, to_matrix

CUT_S = 15


def trace_flow(n, row, model, ckpt, source="the network interface (Npcap)"):
    feats, classes, cfg = ckpt["features"], ckpt["classes"], ckpt["model_config"]
    df = pd.DataFrame([row])
    bar = "=" * 78
    print(f"\n{bar}\n FLOW {n}: full pipeline trace\n{bar}")

    print(f"[1] CAPTURE    {row['protocol']} {row['src_ip']}:{row['src_port']} -> {row['dst_ip']}:{row['dst_port']}\n"
          f"               {row['packets']} packets, {int(row['bytes']):,} bytes read from {source}")
    if "src2dst_packets" in row:
        print(f"[2] FLOW       NFStream joined both directions into one flow, cut at {CUT_S} s:\n"
              f"               duration {row['bidirectional_duration_ms'] / 1000:.2f} s, "
              f"{row['src2dst_packets']} client->server + {row['dst2src_packets']} server->client packets")

    Z = to_matrix(df, feats)                           # signed log
    X = standardize(Z, ckpt["mean"], ckpt["std"])      # training mean / std
    t0 = time.perf_counter()
    probs, rel = explain(model, X)
    ms = (time.perf_counter() - t0) * 1000
    top = set(np.argsort(-rel[0])[:3])

    print(f"[3] FEATURES   identity dropped (IP addresses, ports, timestamps): never model input.\n"
          f"               {len(feats)} behavioural features kept ({ckpt['feature_set']}):")
    print("[4] PREPROCESS raw value -> sign*log(1+|x|) -> standardised with the TRAINING mean/std\n")
    print(f"     {'feature':30s} {'raw':>12s} {'log':>8s} {'scaled':>8s} {'attention':>10s}")
    for j, f in enumerate(feats):
        mark = " <- top-3" if j in top else ""
        print(f"     {f:30s} {float(df[f].iloc[0]):12.2f} {Z[0, j]:8.3f} {X[0, j]:8.3f} {rel[0, j] * 100:9.1f}%{mark}")

    with torch.no_grad():
        logits = model(torch.tensor(X)).float().numpy()[0]
    print(f"\n[5] MODEL      FT-Transformer: each of the {len(feats)} features -> a {cfg.get('d_model')}-number token, "
          f"plus [CLS] = {len(feats) + 1} tokens\n"
          f"               {cfg.get('n_layers')} encoder layers x {cfg.get('n_heads')} attention heads; "
          f"[CLS] -> linear layer -> softmax")
    print("               logits:        " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(classes, logits)))
    print("               probabilities: " + ", ".join(f"{c} {p * 100:.2f}%" for c, p in zip(classes, probs[0])))
    print(f"               verdict: {classes[int(probs[0].argmax())]}   (model + explanation took {ms:.1f} ms)")

    medians = ckpt.get("class_medians")
    pred, conf = int(probs[0].argmax()), float(probs[0].max())
    print("[6] EXPLAIN    attention rollout from [CLS] -> the 3 features the model relied on most,\n"
          "               each compared with typical values of each class in the training data:\n")
    print("  " + plain_reason(Z[0], rel[0], pred, conf, feats, classes, medians).replace("\n", "\n  "))
    print("\n  Technical form: " + reasons(Z, rel, probs, feats, classes, medians)[0])
    print(bar + "\n", flush=True)
