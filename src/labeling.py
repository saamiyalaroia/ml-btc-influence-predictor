"""
v5 labeling — assigns up/flat/down labels to each tweet from the BTC
4-hour forward return, using only information available at tweet time.

Why this is a separate module from data_loading
-----------------------------------------------
The original loader used `np.searchsorted(side='left')`, which can pick a
candle that *starts* at or after the tweet timestamp. In the worst case
that means the chosen `p0` is from a candle that closes 1h *after* the
tweet — i.e. the model gets to peek at the future. In academic terms this
is lookahead bias.

v5 uses `pd.merge_asof(direction='backward')` so `p0` is the most recent
1-hour close that has already occurred at tweet time. `p4` is the close
of the candle finishing 4h after the tweet, which is what the label
represents and is the same thing the live system would observe four
hours later.

Output schema
-------------
The returned frame is the input plus columns:
    p0_time, p0_price          (most recent completed close ≤ tweet time)
    p4_time, p4_price          (most recent completed close ≤ tweet+4h)
    return_4h                  (p4-p0)/p0 as float
    label                      0/1/2
    label_name                 'down'/'flat'/'up'

Rows that cannot be labeled (missing price either side) are dropped, and
the count is logged.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd

from config import (
    DATA_DIR,
    DOWN_THRESHOLD,
    ID2LABEL,
    UP_THRESHOLD,
)


def _load_btc_prices(price_path: str) -> pd.DataFrame:
    df = pd.read_csv(price_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "close"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    return df[["timestamp", "close"]]


def assign_btc_labels(
    tweets_df: pd.DataFrame,
    price_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Attach (p0_time, p0_price, p4_time, p4_price, return_4h, label,
    label_name) to every tweet using a backward-asof join against hourly
    BTC closes.

    Parameters
    ----------
    tweets_df  : DataFrame with at least 'created_at' (UTC datetime).
    price_path : Path to btc_prices_1h.csv. Defaults to data/btc_prices_1h.csv.

    Returns
    -------
    labeled_df : copy of tweets_df with the new columns appended,
                 restricted to rows that received a valid label.
    report     : dict suitable for json-dumping; counts by stage.
    """
    if price_path is None:
        price_path = os.path.join(DATA_DIR, "btc_prices_1h.csv")
    if not os.path.exists(price_path):
        raise FileNotFoundError(
            f"BTC price cache not found at {price_path}; "
            "run main.py load to fetch from Coinbase first."
        )

    btc = _load_btc_prices(price_path)

    df = tweets_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    n_in = len(df)
    df = df.dropna(subset=["created_at"]).reset_index(drop=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    df["t4"] = df["created_at"] + timedelta(hours=4)

    # ---- backward-asof joins -------------------------------------------
    # `direction='backward'` gives the most recent timestamp ≤ left key.
    p0 = pd.merge_asof(
        df[["created_at"]].rename(columns={"created_at": "key"}),
        btc.rename(columns={"timestamp": "key", "close": "p0_price"}),
        on="key",
        direction="backward",
    )
    p4 = pd.merge_asof(
        df[["t4"]].rename(columns={"t4": "key"}),
        btc.rename(columns={"timestamp": "key", "close": "p4_price"}),
        on="key",
        direction="backward",
    )
    df["p0_time"]  = p0["key"]
    df["p0_price"] = p0["p0_price"]
    df["p4_time"]  = p4["key"]
    df["p4_price"] = p4["p4_price"]

    # ---- drop rows without both prices ---------------------------------
    valid = df["p0_price"].notna() & df["p4_price"].notna() & (df["p0_price"] > 0)
    n_after_prices = int(valid.sum())
    df = df[valid].reset_index(drop=True)

    # ---- compute return + label ----------------------------------------
    df["return_4h"] = (df["p4_price"] - df["p0_price"]) / df["p0_price"]

    def _to_label(r: float) -> int:
        if r >  UP_THRESHOLD:   return 2
        if r <  DOWN_THRESHOLD: return 0
        return 1

    df["label"]      = df["return_4h"].apply(_to_label).astype(int)
    df["label_name"] = df["label"].map(ID2LABEL)

    # ---- assertions -----------------------------------------------------
    # If the loader is sane, p0_time ≤ created_at and p4_time ≤ t4.
    bad_p0 = (df["p0_time"] > df["created_at"]).sum()
    bad_p4 = (df["p4_time"] > df["t4"]).sum()
    assert bad_p0 == 0, f"lookahead detected on {bad_p0} rows for p0"
    assert bad_p4 == 0, f"lookahead detected on {bad_p4} rows for p4"

    df = df.drop(columns=["t4"])

    report = {
        "n_input":             n_in,
        "n_after_required":    int(len(tweets_df.dropna(subset=["created_at"]))),
        "n_after_price_match": n_after_prices,
        "n_final":             int(len(df)),
        "label_distribution": {
            ID2LABEL[k]: int((df["label"] == k).sum()) for k in (0, 1, 2)
        },
        "thresholds": {"up": UP_THRESHOLD, "down": DOWN_THRESHOLD},
        "lookahead_check": {
            "rows_with_p0_after_tweet": int(bad_p0),
            "rows_with_p4_after_t4":    int(bad_p4),
        },
    }
    return df, report
