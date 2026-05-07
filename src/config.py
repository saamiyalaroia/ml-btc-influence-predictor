"""
v5 configuration — single source of truth for paths, data sources,
labeling thresholds, model hyperparameters, and the crypto-content filter.

Compared to the original config.py, v5 adds:
  * Typed `DataSource` dataclass for each input file (no more relying on
    filename-substring tricks scattered across the loader).
  * Centralized crypto-keyword filter so train and backtest cannot diverge.
  * `FILTER_MODE` flag enforced as a contract between training and
    evaluation; both modules assert that the value matches what's stored
    on disk in the run config.
  * `chronological_split()` helper — the only supported split.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# v5 lives inside the parent project; raw inputs are shared via symlinks
# under v5/data/mendeley/, while v5 writes its own outputs into v5/data/,
# v5/models/, v5/results/, v5/plots/.

V5_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(V5_DIR, "data")
MENDELEY_DIR = os.path.join(DATA_DIR, "mendeley")
MODEL_DIR    = os.path.join(V5_DIR, "models")
RESULTS_DIR  = os.path.join(V5_DIR, "results")
PLOTS_DIR    = os.path.join(V5_DIR, "plots")
RUNS_DIR     = os.path.join(RESULTS_DIR, "runs")
WEIGHTS_FILE = os.path.join(DATA_DIR, "influence_weights.json")

for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR, PLOTS_DIR, RUNS_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Accounts and data sources
# ---------------------------------------------------------------------------
# Display name lookup. Keys are the canonical username we store in the
# `username` column after loading.
ACCOUNTS = {
    "realDonaldTrump": {"name": "Donald Trump"},
    "elonmusk":        {"name": "Elon Musk"},
    "cz_binance":      {"name": "Changpeng Zhao"},
    "VitalikButerin":  {"name": "Vitalik Buterin"},
    "saylor":          {"name": "Michael Saylor"},
}


@dataclass(frozen=True)
class DataSource:
    """
    Declarative description of one raw input file.

    Attributes
    ----------
    canonical_username : Username we want every row to be tagged with after
        loading. Used both for the `username` column and (when missing in
        the file) as a fallback inferred from this field.
    filename : Path under MENDELEY_DIR.
    file_kind : One of 'xlsx', 'xls', 'csv'. Selects the read backend.
    column_map : Map from {target field} → {column name in this file}.
        Targets are: tweet_id / username / text / created_at / likes / retweets.
        A target may be omitted; for `username` this triggers the
        canonical-username fallback.
    note : Short free-text comment for the data-quality report.
    """
    canonical_username: str
    filename:           str
    file_kind:          str
    column_map:         dict
    note:               str = ""


# Order is significant only for reporting; functionally each source is
# independent and concatenated.
DATA_SOURCES: list[DataSource] = [
    DataSource(
        canonical_username="elonmusk",
        filename="elonmusk_posts.csv",
        file_kind="csv",
        column_map={
            "tweet_id":   "id",
            "text":       "fullText",
            "created_at": "createdAt",
            "likes":      "likeCount",
            "retweets":   "retweetCount",
        },
        note="Kaggle 'all_musk_posts'; lacks a username column — populated "
             "from canonical_username at load time.",
    ),
    DataSource(
        canonical_username="realDonaldTrump",
        filename="realDonaldTrump_posts.csv",
        file_kind="csv",
        column_map={
            "tweet_id":   "id",
            "username":   "handle",
            "text":       "text",
            "created_at": "date",
            "likes":      "favorite_count",
            "retweets":   "repost_count",
        },
        note="Kaggle 'djt_posts_dec2025'. Mixes X and Truth Social posts; "
             "Truth Social rows are kept because the model treats the text "
             "as content, not platform-tagged.",
    ),
    DataSource(
        canonical_username="cz_binance",
        filename="cz_binance.xlsx",
        file_kind="xlsx",
        column_map={
            "tweet_id":   "postid",
            "username":   "screen_name",
            "text":       "标题",
            "created_at": "发帖时间",
            "likes":      "喜欢",
            "retweets":   "转发",
        },
        note="Apify scrape; Chinese column headers from the scraper's locale.",
    ),
    DataSource(
        canonical_username="VitalikButerin",
        filename="VitalikButerin.xlsx",
        file_kind="xlsx",
        column_map={
            "tweet_id":   "postid",
            "username":   "screen_name",
            "text":       "标题",
            "created_at": "发帖时间",
            "likes":      "喜欢",
            "retweets":   "转发",
        },
        note="Apify scrape; small sample (only ~275 rows in 2-year window).",
    ),
    DataSource(
        canonical_username="saylor",
        filename="saylor.xlsx",
        file_kind="xlsx",
        column_map={
            "tweet_id":   "postid",
            "username":   "screen_name",
            "text":       "标题",
            "created_at": "发帖时间",
            "likes":      "喜欢",
            "retweets":   "转发",
        },
        note="Apify scrape; ~1.6k rows.",
    ),
]


# ---------------------------------------------------------------------------
# Time window, labeling, model
# ---------------------------------------------------------------------------
START_DATE = "2023-03-24"
END_DATE   = "2025-03-24"

# Approximate chronological-split boundaries for 70/10/20 over the 2-year
# window. Used by visualize.py for plot annotations only — the actual
# split is computed at training time by `chronological_split()`.
TRAIN_END_DATE  = "2024-08-16"   # ~70 % of [START, END]
TEST_START_DATE = "2024-10-29"   # ~80 % of [START, END]

# 4-hour BTC return thresholds for label assignment.
UP_THRESHOLD   =  0.01
DOWN_THRESHOLD = -0.01

LABEL2ID = {"down": 0, "flat": 1, "up": 2}
ID2LABEL = {0: "down", 1: "flat", 2: "up"}

BERT_MODEL    = "bert-base-uncased"
MAX_LENGTH    = 128
BATCH_SIZE    = 16
NUM_EPOCHS    = 5
LEARNING_RATE = 1e-5
DROPOUT       = 0.3
SEED          = 42

# Validation gating for early stopping. Patience counts epochs without
# improvement on the chosen metric; min_delta is the minimum improvement
# considered "real."
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-3
MODEL_SELECTION_METRIC  = "val_macro_f1"  # not val_loss, not val_accuracy


# ---------------------------------------------------------------------------
# Crypto-content filter (single source of truth for train + backtest)
# ---------------------------------------------------------------------------
# `FILTER_MODE` is a contract: training writes the value into the run
# directory, backtest reads it back and asserts equality. Anything else is
# considered a configuration error.
#   "crypto_only" — only tweets matching CRYPTO_KEYWORDS are kept, both at
#                   training and at evaluation time.
#   "all"         — no filter; train and evaluate on the full set.
FILTER_MODE = "crypto_only"
assert FILTER_MODE in {"crypto_only", "all"}, FILTER_MODE

CRYPTO_KEYWORDS = (
    "bitcoin", "btc", "crypto", "cryptocurrency", "blockchain",
    "ethereum", "eth", "binance", "bnb", "satoshi", "hodl",
    "defi", "nft", "web3", "altcoin", "coinbase", "token", "coin",
    "doge", "dogecoin", "xrp", "ripple", "solana", "sol",
    "trading", "market", "price", "bull", "bear", "pump", "dump",
    "wallet", "exchange", "investment", "financial", "money",
)


def is_crypto_related(text: object) -> bool:
    """Substring keyword filter on lower-cased tweet text. NaN → False."""
    if text is None:
        return False
    s = str(text).lower()
    return any(kw in s for kw in CRYPTO_KEYWORDS)


def add_crypto_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `is_crypto` boolean column without mutating the input frame."""
    df = df.copy()
    df["is_crypto"] = df["text"].apply(is_crypto_related)
    return df


def apply_filter_mode(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """
    Apply the configured filter mode. The returned frame is a copy.

    A small wrapper over `is_crypto_related` whose only job is to enforce
    that nobody calls it with an unknown mode. `mode=None` falls back to the
    module-level `FILTER_MODE`.
    """
    mode = mode or FILTER_MODE
    if mode == "all":
        return df.copy()
    if mode == "crypto_only":
        out = df[df["text"].apply(is_crypto_related)].reset_index(drop=True)
        return out
    raise ValueError(f"Unknown FILTER_MODE: {mode!r}")


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
# The financial setting requires that train < val < test in time. Random
# splits leak future information into validation and inflate metrics; we
# therefore only support time-ordered splitting.

TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.10
TEST_FRACTION  = 0.20
assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-9


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
    val_frac:   float = VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Sort by `created_at` and slice into (train, val, test).

    Parameters are explicit so we can override in tests if needed; the
    defaults match the values declared above.

    Returns
    -------
    train_df, val_df, test_df, report
        `report` is a dict suitable for json-dumping to split_report.json.
    """
    if "created_at" not in df.columns:
        raise ValueError("chronological_split requires a 'created_at' column")

    sorted_df = (
        df.dropna(subset=["created_at"])
          .sort_values("created_at")
          .reset_index(drop=True)
    )
    n = len(sorted_df)
    n_train = int(round(n * train_frac))
    n_val   = int(round(n * val_frac))

    train_df = sorted_df.iloc[: n_train].reset_index(drop=True)
    val_df   = sorted_df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df  = sorted_df.iloc[n_train + n_val :].reset_index(drop=True)

    def _bounds(d):
        if d.empty:
            return None, None
        return str(d["created_at"].iloc[0]), str(d["created_at"].iloc[-1])

    report = {
        "n_total": int(n),
        "fractions": {
            "train": float(train_frac),
            "val":   float(val_frac),
            "test":  float(1 - train_frac - val_frac),
        },
        "n": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "ranges": {
            "train": _bounds(train_df),
            "val":   _bounds(val_df),
            "test":  _bounds(test_df),
        },
    }
    return train_df, val_df, test_df, report


# ---------------------------------------------------------------------------
# Convenience: dump a snapshot of the static config to disk for run logs.
# ---------------------------------------------------------------------------

def config_snapshot() -> dict:
    """Plain-data dict of every config knob a downstream consumer might need."""
    return {
        "accounts":         list(ACCOUNTS),
        "data_sources":     [asdict(s) for s in DATA_SOURCES],
        "start_date":       START_DATE,
        "end_date":         END_DATE,
        "up_threshold":     UP_THRESHOLD,
        "down_threshold":   DOWN_THRESHOLD,
        "bert_model":       BERT_MODEL,
        "max_length":       MAX_LENGTH,
        "batch_size":       BATCH_SIZE,
        "num_epochs":       NUM_EPOCHS,
        "learning_rate":    LEARNING_RATE,
        "dropout":          DROPOUT,
        "seed":             SEED,
        "filter_mode":      FILTER_MODE,
        "split":            {
            "train_frac": TRAIN_FRACTION,
            "val_frac":   VAL_FRACTION,
            "test_frac":  TEST_FRACTION,
        },
        "early_stopping": {
            "patience":      EARLY_STOPPING_PATIENCE,
            "min_delta":     EARLY_STOPPING_MIN_DELTA,
            "metric":        MODEL_SELECTION_METRIC,
        },
    }
