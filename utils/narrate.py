"""Plain-language reasons for individual predictions.

For each flow, takes the features the model relied on most (attention rollout)
and compares the flow's value with the typical (median) value of each class in
the training data, e.g.

    Classified as VPN (97% confident). Main reasons: (1) the average gap between
    packets was 2.1 ms, typical of VPN traffic (VPN median 1.9 ms, Non-VPN 48 ms); ...

Which features are named comes from the model (faithful, see the deletion
test). The class comparison is descriptive context from the training data.
"""
import re

import numpy as np

from .preprocessing import from_matrix

_STAT = {"min": "shortest", "max": "longest", "mean": "average", "std": "variation in the",
         "stddev": "variation in the"}
_SIZE_STAT = {"min": "smallest", "max": "largest", "mean": "average", "stddev": "variation in the"}
_DIR = {"bidirectional": "", "src2dst": "client->server ", "dst2src": "server->client "}


def class_medians(Z, y, num_classes):
    """Per-class median of each feature (signed-log space), as nested lists for the checkpoint."""
    return [np.median(Z[y == c], axis=0).tolist() if (y == c).any() else [0.0] * Z.shape[1]
            for c in range(num_classes)]

# ------------------------------------------
# Feature names -> words, values -> units
# ------------------------------------------
def describe(name):
    """(phrase, unit) for a feature. unit: 'us', 'ms', 'bytes', 'count', 'pps', 'Bps'."""
    m = re.fullmatch(r"(min|max|mean|std)_(fiat|biat|flowiat|active|idle)", name)
    if m:
        stat, kind = _STAT[m[1]], m[2]
        what = {"fiat": "gap between client->server packets",
                "biat": "gap between server->client packets",
                "flowiat": "gap between packets",
                "active": "burst of activity", "idle": "idle period"}[kind]
        return f"the {stat} {what}", "us"
    iscx = {"duration": ("the flow duration", "us"),
            "flowPktsPerSecond": ("the packet rate", "pps"),
            "flowBytesPerSecond": ("the data rate", "Bps")}
    if name in iscx:
        return iscx[name]

    m = re.fullmatch(r"(bidirectional|src2dst|dst2src)_(.+)", name)
    if m:
        d, rest = _DIR[m[1]], m[2]
        if rest == "duration_ms":
            return f"the {d or 'flow '}duration", "ms"
        if rest == "packets":
            return f"the number of {d}packets", "count"
        if rest == "bytes":
            return f"the {d}data volume", "bytes"
        s = re.fullmatch(r"(min|max|mean|stddev)_ps", rest)
        if s:
            return f"the {_SIZE_STAT[s[1]]} {d}packet size", "bytes"
        s = re.fullmatch(r"(min|max|mean|stddev)_piat_ms", rest)
        if s:
            return f"the {_STAT[s[1]]} gap between {d}packets", "ms"
        s = re.fullmatch(r"(\w+)_packets", rest)
        if s:
            return f"the number of {d}{s[1].upper()}-flag packets", "count"
    return name, "count"


def fmt(value, unit):
    if unit in ("us", "ms"):
        sec = value / (1e6 if unit == "us" else 1e3)
        if abs(sec) >= 1:
            return f"{sec:.3g} s"
        if abs(sec) >= 1e-3:
            return f"{sec*1e3:.3g} ms"
        return f"{sec*1e6:.3g} us"
    if unit == "bytes":
        for size, suffix in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
            if abs(value) >= size:
                return f"{value/size:.3g} {suffix}"
        return f"{value:.0f} bytes"
    if unit == "pps":
        return f"{value:.3g} packets/s"
    if unit == "Bps":
        return fmt(value, "bytes") + "/s"
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"

# ------------------------------------------
# Sentences
# ------------------------------------------
def reason(z_row, relevance_row, pred, confidence, features, classes, medians, k=3):
    """One plain-language explanation for one flow.

    z_row: signed-log feature values; medians: class_medians(...) or None.
    """
    medians = None if medians is None else np.asarray(medians)
    raw = from_matrix(np.asarray(z_row))
    parts, compared, supports = [], 0, 0
    for j in np.argsort(-relevance_row)[:k]:
        phrase, unit = describe(features[j])
        sentinel = bool(re.match(r"(min|max)_", features[j]))   # ISCX: -1 = not measurable
        show = lambda v: "usually not measurable" if sentinel and np.isclose(v, -1) else fmt(v, unit)
        if sentinel and np.isclose(raw[j], -1):
            parts.append(f"({len(parts)+1}) {phrase} could not be measured (too few packets)")
            continue
        text = f"{phrase} was {fmt(raw[j], unit)}"
        if medians is not None:
            extra, closest = _compare(medians[:, j], z_row[j], pred, classes, unit, show)
            text += extra
            if closest is not None:
                compared += 1
                supports += closest == pred
        parts.append(f"({len(parts)+1}) {text}")
    out = (f"Classified as {classes[pred]} ({confidence*100:.0f}% confident). "
           f"Main reasons: " + "; ".join(parts) + ".")
    if compared and not supports:
        out += (f" None of these values on its own is typical of {classes[pred]}; "
                "the decision comes from how the features combine.")
    return out


TYPICAL = 0.5  # within ~1.6x of a class median (signed-log units) counts as "typical"


def _compare(med_j, z, pred, classes, unit, show):
    """', typical of VPN traffic (VPN median 1.9 ms, Non-VPN median 48 ms)' and variants."""
    typical = from_matrix(med_j)
    refs = _reference_classes(med_j, z, pred, len(classes))
    ref = ", ".join(f"{classes[c]} median {show(typical[c])}" for c in refs)
    if np.allclose(med_j[refs], med_j[refs[0]]):
        both = show(typical[refs[0]])
        if both == "usually not measurable":
            return " (usually not measurable for either class)", None
        return f" (both {classes[refs[0]]} and {classes[refs[1]]} typically {both})", None
    closest = int(np.argmin(np.abs(med_j - z)))
    other = refs[1] if closest == refs[0] else refs[0]
    d, gap = z - med_j[closest], med_j[other] - med_j[closest]
    name = classes[closest]
    if abs(d) <= TYPICAL:
        verdict = f"typical of {name} traffic"
    elif np.sign(d) == -np.sign(gap):
        verdict = f"{'higher' if d > 0 else 'lower'} than usual for either class, on the {name} side"
    else:
        verdict = f"closer to {name} than {classes[other]} traffic"
    return f", {verdict} ({ref})", closest


def _reference_classes(med_j, z, pred, n):
    """Predicted class plus the closest other class (all classes when binary)."""
    if n <= 2:
        return [pred] + [c for c in range(n) if c != pred]
    others = sorted((c for c in range(n) if c != pred), key=lambda c: abs(med_j[c] - z))
    return [pred, others[0]]


def reasons(Z, relevance, proba, features, classes, medians, k=3):
    pred, conf = proba.argmax(axis=1), proba.max(axis=1)
    return [reason(Z[i], relevance[i], pred[i], conf[i], features, classes, medians, k)
            for i in range(len(Z))]
