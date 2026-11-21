import numpy as np
import pandas as pd

LABEL_COL = "class1"

# The ISCX VPN-nonVPN 2016 time-based CSVs have shifted headers in the fiat/biat
# block: the column named "total_fiat" actually holds min_fiat, "min_fiat" holds
# max_fiat, "max_fiat" holds mean_fiat and "mean_fiat" holds std_fiat (same for biat).
# Verified on the data: "total_fiat" <= "min_fiat" in 100% of rows, and the values
# are consistent with min / max / mean / std of the inter-arrival times.
ISCX_HEADER_FIX = {
    "total_fiat": "min_fiat", "total_biat": "min_biat",
    "min_fiat": "max_fiat", "min_biat": "max_biat",
    "max_fiat": "mean_fiat", "max_biat": "mean_biat",
    "mean_fiat": "std_fiat", "mean_biat": "std_biat",
}

# Model input, in order. All times are in microseconds.
# Missing statistics follow the ISCX convention: min/max = -1, mean/std = 0.
FEATURES = [
    "duration",
    "min_fiat", "max_fiat", "mean_fiat", "std_fiat",
    "min_biat", "max_biat", "mean_biat", "std_biat",
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle",
]

LABEL_MAP = {
    "NonVPN": "Non-VPN",
    "BENIGN": "Non-VPN",
    "Benign": "Non-VPN",
}


def fix_iscx_headers(df):
    """Rename the shifted ISCX columns to what they actually contain."""
    if "total_fiat" in df.columns:
        df = df.rename(columns=ISCX_HEADER_FIX)
    return df


def load_dataset(path):
    """Load a labelled ISCX CSV. Returns (features DataFrame, label Series)."""
    df = fix_iscx_headers(pd.read_csv(path))
    df = df.drop_duplicates().reset_index(drop=True)

    y = df[LABEL_COL].astype(str).str.strip().replace(LABEL_MAP)
    return df, y


def to_matrix(df):
    """Select FEATURES in training order and compress their range."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float64)

    # Timing features span 0 .. 6e8 us. A signed log keeps the -1 "absent"
    # sentinel distinct from 0 while taming the heavy tails.
    return np.sign(X) * np.log1p(np.abs(X))


def from_matrix(Z):
    """Inverse of the signed log in to_matrix: back to raw feature units."""
    return np.sign(Z) * np.expm1(np.abs(Z))


def fit_standardizer(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize(X, mean, std):
    return ((X - np.asarray(mean)) / np.asarray(std)).astype(np.float32)


def split_by_group(groups, y, seed=42):
    """70/15/15 split by whole groups (capture files), per class so every split has
    both classes. Flows from one capture never land in both train and test.

    Returns (train_idx, val_idx, test_idx).
    """
    rng = np.random.default_rng(seed)
    split = np.empty(len(y), dtype=object)
    for cls in np.unique(y):
        files = rng.permutation(np.unique(groups[y == cls]))
        if len(files) < 3:
            raise SystemExit(f"[ERROR] Class index {cls} has only {len(files)} capture files; need >= 3.")
        n_test = n_val = max(1, round(0.15 * len(files)))
        for name, part in (("test", files[:n_test]), ("val", files[n_test:n_test + n_val]),
                           ("train", files[n_test + n_val:])):
            split[np.isin(groups, part) & (y == cls)] = name
    return [np.where(split == s)[0] for s in ("train", "val", "test")]
