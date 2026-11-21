"""Evasion attacks against the VPN classifier.

All attacks push VPN flows towards the Non-VPN class. They work on the
signed-log feature matrix from to_matrix() unless stated otherwise.
"""
import numpy as np
import torch
import torch.nn.functional as F

from utils.preprocessing import FEATURES, from_matrix

def _shaping_columns(features):
    """Which columns a traffic-shaping sender changes, for either feature set."""
    time = np.array([f.endswith("_ms") or f in ("duration",) or "iat" in f
                     or f.split("_")[-1] in ("active", "idle") for f in features])
    size = np.array([f.endswith("_ps") or f.endswith("_bytes") for f in features])
    pkts_rate = np.array([f == "flowPktsPerSecond" for f in features])
    bytes_rate = np.array([f == "flowBytesPerSecond" for f in features])
    return time, size, pkts_rate, bytes_rate


# ==========================================
# 1. Traffic shaping (problem-space, realistic)
# ==========================================
def shape_traffic(Z, slowdown, padding, features=FEATURES):
    """Feature vector of the same flow after the sender

    * slows it down by `slowdown` (every gap stretched, e.g. 1.5 = 50% slower), and
    * pads packets so the flow carries `padding` x the bytes (1.25 = 25% overhead).

    slowdown / padding are scalars or one value per flow. Times scale by
    slowdown, packet sizes and byte counts by padding, packet rate by
    1/slowdown and byte rate by padding/slowdown. Packet counts and TCP flags
    are unchanged. Absent ISCX statistics (-1) stay absent. Approximation: in a
    real capture, stretched gaps that cross the 1 s activity threshold would
    also create new idle periods.
    """
    time, size, pkts_rate, bytes_rate = _shaping_columns(features)
    s = np.broadcast_to(np.asarray(slowdown, dtype=np.float64).reshape(-1, 1), (len(Z), 1))
    p = np.broadcast_to(np.asarray(padding, dtype=np.float64).reshape(-1, 1), (len(Z), 1))

    raw = from_matrix(Z).copy()
    present = ~np.isclose(raw, -1)
    factor = np.ones_like(raw)
    factor[:, time] = s
    factor[:, size] = p
    factor[:, pkts_rate] = 1 / s
    factor[:, bytes_rate] = p / s
    raw = np.where(present, raw * factor, raw)
    return np.sign(raw) * np.log1p(np.abs(raw))


def augment_with_shaping(Z, y, features=FEATURES, max_slowdown=3.0, max_padding=2.0, seed=0):
    """Defense: one randomly shaped copy of every training flow, same label.

    Shaping both classes (not only VPN) teaches the model that slowing down or
    padding traffic does not change what it is, instead of "slow means VPN".
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(1.0, max_slowdown, len(Z))
    p = rng.uniform(1.0, max_padding, len(Z))
    return np.concatenate([Z, shape_traffic(Z, s, p, features)]), np.concatenate([y, y])


def min_slowdown_to_evade(predict, Z, slowdowns, paddings, target, features=FEATURES):
    """For each flow, the smallest slowdown (with any padding <= max) that makes
    `predict` output `target`. np.inf when no budget in the grid works."""
    best = np.full(len(Z), np.inf)
    for s in slowdowns:
        for p in paddings:
            evaded = predict(shape_traffic(Z, s, p, features)) == target
            best = np.where(evaded & np.isinf(best), s, best)
    return best


# ==========================================
# 2. Feature-space attacks (worst case, L-inf budget on standardized features)
# ==========================================
def random_search_attack(proba, X, source, eps, iters=300, seed=0):
    """Black-box L-inf random search (Square-attack style for tabular data).

    Only needs predicted probabilities, so it is applied identically to any
    model. Minimizes the probability of the `source` class.
    """
    rng = np.random.default_rng(seed)
    delta = rng.choice([-eps, eps], size=X.shape).astype(np.float32)
    best = proba(X + delta)[:, source]
    for it in range(iters):
        frac = 0.3 * (1 - it / iters) + 0.02  # change many features early, few later
        cand = delta.copy()
        mask = rng.random(X.shape) < frac
        cand[mask] = rng.choice([-eps, eps], size=mask.sum())
        p = proba(X + cand)[:, source]
        better = p < best
        delta[better], best[better] = cand[better], p[better]
    return X + delta


def pgd_attack(model, X, source, eps, steps=50):
    """White-box L-inf PGD on the FT-Transformer: maximizes the loss of `source`."""
    model_device = next(model.parameters()).device
    x0 = torch.tensor(X).to(model_device)
    target = torch.full((len(X),), source, dtype=torch.long, device=model_device)
    step = 2.5 * eps / steps
    x = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    for _ in range(steps):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), target)
        grad, = torch.autograd.grad(loss, x)
        x = x.detach() + step * grad.sign()
        x = torch.min(torch.max(x, x0 - eps), x0 + eps)
    return x.detach().cpu().numpy()
