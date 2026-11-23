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


# ------------------------------------------
# Plain language: what each value means, not just the number
# ------------------------------------------
# For every top feature: the value, what that behaviour usually means, and which
# class it points to (closest class median in the training data). For VPN /
# Non-VPN models it also says why a VPN changes that behaviour.

_VPN_WHY = {
    ("size", "min", "low"): "A VPN adds its own headers and encryption data to every packet, "
                            "so VPN traffic has no packets this small.",
    ("size", "min", "high"): "That is what a VPN does: it wraps every packet, even a bare "
                             "acknowledgement, in extra headers and encryption data.",
    ("size", "mean", "low"): "Nothing suggests the extra wrapping a VPN adds to each packet.",
    ("size", "mean", "high"): "This fits the extra wrapping a VPN adds to each packet "
                              "(or simply a large data transfer).",
    ("flags", None, "high"): "The VPN in the training data runs over UDP, which has no such signals, "
                             "so this mainly tells the model the protocol (a shortcut of this dataset).",
    ("flags", None, "low"): "The VPN in the training data runs over UDP, which has no such signals.",
}

_MEANING = {
    ("size", "min", "low"): "Its smallest packets were tiny: bare control messages, such as \"got it\" "
                            "acknowledgements, sent as they are.",
    ("size", "min", "high"): "Even its smallest packet was large: every packet seems to carry extra bytes.",
    ("size", "mean", "low"): "Its packets were smaller than usual.",
    ("size", "mean", "high"): "Its packets were larger than usual.",
    ("size", "max", "low"): "Even its largest packet was small: only short messages were exchanged.",
    ("size", "max", "high"): "It had some large, full-size data packets.",
    ("size", "std", "low"): "Its packet sizes were more uniform than usual: a regular, repetitive pattern.",
    ("size", "std", "high"): "Its packet sizes varied more than usual: a mix of small control packets "
                             "and larger data packets.",
    ("gap", "mean", "low"): "Its packets came faster than usual: the connection was busy.",
    ("gap", "mean", "high"): "Its packets came slower than usual: the connection was mostly waiting.",
    ("gap", "min", "low"): "Some of its packets came back-to-back.",
    ("gap", "min", "high"): "Even its shortest pause between packets was long.",
    ("gap", "max", "low"): "Its longest pause was shorter than usual: the connection stayed fairly busy.",
    ("gap", "max", "high"): "Its longest pause was longer than usual: the connection sat idle at times.",
    ("gap", "std", "low"): "Its packets came at a steadier rhythm than usual.",
    ("gap", "std", "high"): "Its packets came in more irregular bursts than usual.",
    ("count", None, "low"): "Fewer packets than usual were exchanged.",
    ("count", None, "high"): "More packets than usual were exchanged: a busy connection.",
    ("volume", None, "low"): "Less data than usual was transferred.",
    ("volume", None, "high"): "More data than usual was transferred.",
    ("duration", None, "low"): "The connection was shorter than usual.",
    ("duration", None, "high"): "The connection lasted longer than usual.",
    ("flags", None, "low"): "Few or no TCP control signals (SYN, ACK, PSH, ...) were used.",
    ("flags", None, "high"): "The connection used TCP control signals (SYN, ACK, PSH, ...).",
    ("rate", None, "low"): "The traffic was slower than usual.",
    ("rate", None, "high"): "The traffic was faster than usual.",
    ("active", None, "low"): "Its bursts of activity were shorter than usual.",
    ("active", None, "high"): "Its bursts of activity were longer than usual.",
    ("idle", None, "low"): "Its pauses between bursts were shorter than usual.",
    ("idle", None, "high"): "Its pauses between bursts were longer than usual.",
}
WEAK = "Close to typical traffic of both classes, so on its own this is a weak clue."
CLEAR_GAP = 0.2    # class medians this far apart (signed-log units, ~22%) differ clearly
MARGIN = 0.1       # a value this far outside both medians is "higher / lower than usual"


def _side(z, med):
    """'low' / 'high' if the value clearly sits on one side, None if it is a weak clue."""
    lo, hi = med.min(), med.max()
    if z < lo - MARGIN:
        return "low"
    if z > hi + MARGIN:
        return "high"
    if hi - lo >= CLEAR_GAP:
        return "low" if z < med.mean() else "high"
    return None


def _kind(name):
    """(kind, stat): kind in size/gap/count/volume/duration/flags/rate/active/idle."""
    stat_of = lambda s: {"stddev": "std"}.get(s, s)
    m = re.fullmatch(r"(min|max|mean|std)_(fiat|biat|flowiat|active|idle)", name)
    if m:
        return {"active": "active", "idle": "idle"}.get(m[2], "gap"), stat_of(m[1])
    if name == "duration":
        return "duration", None
    if name in ("flowPktsPerSecond", "flowBytesPerSecond"):
        return "rate", None
    m = re.fullmatch(r"(?:bidirectional|src2dst|dst2src)_(.+)", name)
    if m:
        rest = m[1]
        s = re.fullmatch(r"(min|max|mean|stddev)_ps", rest)
        if s:
            return "size", stat_of(s[1])
        s = re.fullmatch(r"(min|max|mean|stddev)_piat_ms", rest)
        if s:
            return "gap", stat_of(s[1])
        if rest == "duration_ms":
            return "duration", None
        if rest == "packets":
            return "count", None
        if rest == "bytes":
            return "volume", None
        if rest.endswith("_packets"):
            return "flags", None
    return None, None


def _label(name):
    phrase = describe(name)[0]
    phrase = phrase[4:] if phrase.startswith("the ") else phrase
    return phrase[0].upper() + phrase[1:]


def plain_reason(z_row, relevance_row, pred, confidence, features, classes, medians, k=3,
                 multiline=True):
    """Explanation for a non-specialist: for each top feature the value, what it means
    and which class it points to, then a one-line summary."""
    medians = None if medians is None else np.asarray(medians)
    z_row = np.asarray(z_row)
    raw = from_matrix(z_row)
    vpn_task = set(classes) == {"VPN", "Non-VPN"}
    top = list(np.argsort(-relevance_row)[:k])
    lines, votes = [], []
    for n, j in enumerate(top, 1):
        head, meaning, vote = _clue(j, z_row, raw, pred, features, classes, medians, vpn_task)
        lines.append((f"{n}. {head}", meaning))
        if vote is not None:
            votes.append(vote)

    verdict = f"{classes[pred]} ({confidence * 100:.0f}% confident)"
    if not votes:
        summary = ("The features the model looked at most are weak clues on their own; "
                   "the decision comes from all features together.")
    elif all(votes):
        summary = f"{'All ' if len(votes) > 1 else ''}{len(votes)} clear clue{'s' if len(votes) > 1 else ''} " \
                  f"point{'' if len(votes) > 1 else 's'} to {classes[pred]}."
    elif any(votes):
        summary = (f"{sum(votes)} of {len(votes)} clear clues point to {classes[pred]}; "
                   "the model weighed them together with the other features.")
    else:
        summary = (f"None of these clues on its own points to {classes[pred]}: "
                   "the decision comes from how the features combine, so treat it with care.")

    # Clearest sign in the data: the feature where this flow sits most clearly on the
    # predicted class's side. Context from the training data, NOT what the model attended to.
    sign = None
    if medians is not None and not all(votes or [False]):
        best, best_gap = None, CLEAR_GAP
        for j in range(len(features)):
            if j in top or _kind(features[j])[0] in (None, "flags"):
                continue
            refs = _reference_classes(medians[:, j], z_row[j], pred, len(classes))
            m_pred, m_other = medians[refs[0], j], medians[refs[1], j]
            gap = abs(m_pred - m_other)
            if gap > best_gap and abs(z_row[j] - m_pred) < abs(z_row[j] - m_other) \
                    and abs(z_row[j] - m_pred) <= gap:
                best, best_gap = j, gap
        if best is not None:
            head, meaning, _ = _clue(best, z_row, raw, pred, features, classes, medians, vpn_task)
            sign = (head, meaning)

    if multiline:
        body = "\n".join(f"  {h}\n     {m}" if m else f"  {h}" for h, m in lines)
        out = f"{verdict}. What the model looked at most:\n{body}\n  In short: {summary}"
        if sign:
            out += (f"\n  Clearest sign of {classes[pred]} in this flow (from the training data, "
                    f"not the model's attention):\n     {sign[0]}\n     {sign[1]}")
        return out
    out = f"{verdict}. " + " ".join(f"{h}. {m}" for h, m in lines) + f" In short: {summary}"
    if sign:
        out += f" Clearest sign of {classes[pred]} (training data, not attention): {sign[0]}. {sign[1]}"
    return out


def _clue(j, z_row, raw, pred, features, classes, medians, vpn_task):
    """(headline, meaning, vote) for one feature; vote is None for a weak clue."""
    name, z = features[j], z_row[j]
    unit = describe(name)[1]
    head = f"{_label(name)}: {fmt(raw[j], unit)}"
    if medians is None:
        return head, "", None
    kind, stat = _kind(name)
    refs = _reference_classes(medians[:, j], z, pred, len(classes))
    med = medians[refs, j]
    ref_text = ", ".join(f"{classes[c]} {fmt(t, unit)}" for c, t in zip(refs, from_matrix(med)))
    side = _side(z, med)
    if side is None:
        return f"{head}  -> weak clue  (typical: {ref_text})", WEAK, None
    if med.max() - med.min() < CLEAR_GAP:     # classes look alike on this feature
        return (f"{head}  -> no class signal  (typical: {ref_text})",
                _MEANING.get((kind, stat, side), "") + " Both classes are alike here, so this "
                "says little about the class.", None)
    closest = refs[int(np.argmin(np.abs(med - z)))]
    outside = z < med.min() - TYPICAL or z > med.max() + TYPICAL
    head += (f"  -> {'unusual for both, nearer' if outside else 'points to'} "
             f"{classes[closest]}  (typical: {ref_text})")
    meaning = _MEANING.get((kind, stat, side), "")
    if vpn_task and not (kind == "size" and stat in ("max", "std")):
        meaning = (meaning + " " + _VPN_WHY.get((kind, None if kind == "flags" else stat, side), "")).strip()
    return head, meaning, closest == pred


def plain_reasons(Z, relevance, proba, features, classes, medians, k=3, multiline=False):
    pred, conf = proba.argmax(axis=1), proba.max(axis=1)
    return [plain_reason(Z[i], relevance[i], pred[i], conf[i], features, classes, medians, k, multiline)
            for i in range(len(Z))]
