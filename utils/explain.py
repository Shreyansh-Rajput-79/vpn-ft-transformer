import numpy as np
import torch


def attention_rollout(attn):
    """Attention rollout (Abnar & Zuidema, 2020).

    attn: [batch, layers, tokens, tokens] head-averaged attention.
    Returns [batch, features]: how much the CLS token (the one the classifier
    reads) draws from each feature token across all layers. Rows sum to 1.
    """
    batch, layers, tokens, _ = attn.shape
    eye = torch.eye(tokens)
    rollout = eye.expand(batch, tokens, tokens)
    for layer in range(layers):
        # Account for the residual connection, then re-normalize.
        a = 0.5 * attn[:, layer] + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    relevance = rollout[:, 0, 1:]
    return (relevance / relevance.sum(dim=-1, keepdim=True)).numpy()


def explain(model, X, batch_size=1024):
    """Returns (probabilities, per-feature relevance) for every row of X."""
    model_device = next(model.parameters()).device
    probs, relevance = [], []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size]).to(model_device)
        logits, attn = model.forward_with_attention(xb)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        relevance.append(attention_rollout(attn.float().cpu()))
    return np.concatenate(probs), np.concatenate(relevance)


def top_features(relevance, names, k=3):
    """'feat_a (31%), feat_b (12%), ...' for each row."""
    order = np.argsort(-relevance, axis=1)[:, :k]
    return [", ".join(f"{names[j]} ({row[j]*100:.0f}%)" for j in idx)
            for row, idx in zip(relevance, order)]


def permutation_importance(predict, X, y, n_repeats=5, seed=0):
    """Mean accuracy drop when each feature is shuffled. Model-agnostic."""
    rng = np.random.default_rng(seed)
    base = (predict(X) == y).mean()
    drops = np.zeros((n_repeats, X.shape[1]))
    for r in range(n_repeats):
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops[r, j] = base - (predict(Xp) == y).mean()
    return drops.mean(axis=0), drops.std(axis=0)


def deletion_curve(predict_proba, X, ranking, ks, seed=0):
    """Confidence in the original prediction after removing the k top-ranked
    features of each flow (set to the training mean, i.e. 0 after scaling).

    ranking: [rows, features] per-flow feature order, most important first.
    If the ranking is faithful, confidence should fall faster than for a
    random ranking.
    """
    original = predict_proba(X)
    pred = original.argmax(axis=1)
    rows = np.arange(len(X))
    curve = []
    for k in ks:
        Xk = X.copy()
        for col in range(k):
            Xk[rows, ranking[:, col]] = 0
        curve.append(predict_proba(Xk)[rows, pred].mean())
    return np.array(curve)
