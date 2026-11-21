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

# NFStream (statistical_analysis=True) behavioural features. Deliberately left
# out because they identify endpoints or leak the answer instead of describing
# behaviour: id, IPs, MACs, OUIs, ports, protocol, ip_version, vlan/tunnel ids,
# *_first_seen_ms / *_last_seen_ms timestamps, and every nDPI field
# (application_name, application_category_name, requested_server_name,
# fingerprints, user_agent, content_type...). nDPI would simply report "OpenVPN".
_DIRS = ("bidirectional", "src2dst", "dst2src")
NFSTREAM_FEATURES = (
    [f"{d}_{m}" for d in _DIRS for m in ("duration_ms", "packets", "bytes")]
    + [f"{d}_{s}_ps" for d in _DIRS for s in ("min", "mean", "stddev", "max")]
    + [f"{d}_{s}_piat_ms" for d in _DIRS for s in ("min", "mean", "stddev", "max")]
    + [f"{d}_{flag}_packets" for d in _DIRS
       for flag in ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin")]
)

FEATURE_SETS = {"iscx": FEATURES, "nfstream": NFSTREAM_FEATURES}

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


def detect_feature_set(df):
    """'iscx' or 'nfstream', whichever feature set the DataFrame contains."""
    for name, features in FEATURE_SETS.items():
        if all(c in df.columns for c in features):
            return name
    raise ValueError("Input has neither the ISCX nor the NFStream feature columns.")


def load_dataset(path, label_col=LABEL_COL):
    """Load a labelled feature CSV (ISCX, or build_vnat_dataset.py output).

    label_col: "class1" for VPN / Non-VPN, or "category" / "app" for VNAT.
    Returns (DataFrame, label Series).
    """
    df = fix_iscx_headers(pd.read_csv(path))
    df = df.drop_duplicates().reset_index(drop=True)
    if label_col not in df.columns:
        raise SystemExit(f"[ERROR] Label column '{label_col}' not in {path}. "
                         f"Columns: {list(df.columns)[:10]}...")

    y = df[label_col].astype(str).str.strip()
    if label_col == LABEL_COL:
        y = y.replace(LABEL_MAP)
    return df, y


def to_matrix(df, features=FEATURES):
    """Select `features` in training order and compress their range."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float64)

    # Timing / byte features span many orders of magnitude. A signed log keeps the -1 "absent"
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
    every class. Flows from one capture never land in both train and test.

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


def split_indices(y, seed=42, groups=None):
    """70/15/15 train/val/test indices: by capture file when `groups` is given,
    otherwise stratified by flow."""
    if groups is not None:
        return split_by_group(np.asarray(groups), y, seed=seed)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    return tr, va, te


def prepare_dataset(path, label_col=LABEL_COL):
    """Load a feature CSV and everything the scripts need from it.

    Returns a dict with: df, classes, y (class indices), features, feature_set,
    Z (signed-log feature matrix, not yet standardized).
    """
    df, labels = load_dataset(path, label_col)
    feature_set = detect_feature_set(df)
    features = FEATURE_SETS[feature_set]
    classes = sorted(labels.unique())
    y = labels.map({c: i for i, c in enumerate(classes)}).to_numpy()
    return dict(df=df, classes=classes, y=y, features=features,
                feature_set=feature_set, Z=to_matrix(df, features))
