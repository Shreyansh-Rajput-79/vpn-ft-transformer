"""Evasion attacks against the VPN classifier.

All attacks push VPN flows towards the Non-VPN class. They work on the
signed-log feature matrix from to_matrix() unless stated otherwise.
"""
import numpy as np
import torch
import torch.nn.functional as F

from utils.preprocessing import FEATURES, from_matrix

RATE_FEATURES = ["flowPktsPerSecond", "flowBytesPerSecond"]
TIME_FEATURES = [f for f in FEATURES if f not in RATE_FEATURES]
_TIME = np.array([f in TIME_FEATURES for f in FEATURES])
_PKTS = FEATURES.index("flowPktsPerSecond")
_BYTES = FEATURES.index("flowBytesPerSecond")


# ==========================================
# 1. Traffic shaping (problem-space, realistic)
# ==========================================
def shape_traffic(Z, slowdown, padding):
    """Feature vector of the same flow after the sender

    * slows it down by `slowdown` (every gap stretched, e.g. 1.5 = 50% slower), and
    * pads packets so the flow carries `padding` x the bytes (1.25 = 25% overhead).

    Times scale by `slowdown`, packet rate by 1/slowdown and byte rate by
    padding/slowdown. Absent statistics (-1) stay absent. Approximation: in a
    real capture, stretched gaps that cross the 1 s activity threshold would
    also create new idle periods.
    """
    raw = from_matrix(Z)
    present = ~np.isclose(raw, -1)
    raw = raw.copy()
    raw[:, _TIME] = np.where(present[:, _TIME], raw[:, _TIME] * slowdown, raw[:, _TIME])
    raw[:, _PKTS] /= slowdown
    raw[:, _BYTES] *= padding / slowdown
    return np.sign(raw) * np.log1p(np.abs(raw))


def min_slowdown_to_evade(predict, Z, slowdowns, paddings, target):
    """For each flow, the smallest slowdown (with any padding <= max) that makes
    `predict` output `target`. np.inf when no budget in the grid works."""
    best = np.full(len(Z), np.inf)
    for s in slowdowns:
        for p in paddings:
            evaded = predict(shape_traffic(Z, s, p)) == target
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
