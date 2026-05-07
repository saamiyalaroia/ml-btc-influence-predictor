"""
v5 data loading: read raw inputs (Kaggle CSVs + Apify .xlsx files), normalize
column names, deduplicate, time-window-trim, and produce both
`data/all_tweets.csv` and a structured data-quality report at
`results/data_quality_report.json`.

Engineering goals:
  * Each input file is described declaratively by `DataSource` in config.
  * The loader is pure: same input → same output, no hidden global state.
  * Failure modes are loud: a missing required column raises rather than
    silently dropping a whole source.
  * Every sanitization step is recorded in the data-quality report so a
    reviewer can audit dataset construction.
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any

import pandas as pd

from config import (
    ACCOUNTS,
    DATA_DIR,
    DATA_SOURCES,
    END_DATE,
    MENDELEY_DIR,
    RESULTS_DIR,
    START_DATE,
    DataSource,
    add_crypto_filter_columns,
)


REQUIRED_FIELDS = ("text", "created_at")          # username may be filled in
OUTPUT_COLUMNS  = ("tweet_id", "username", "text", "created_at",
                   "likes", "retweets")


# ---------------------------------------------------------------------------
# Per-source loader
# ---------------------------------------------------------------------------

def _read_raw(path: str, kind: str) -> pd.DataFrame:
    """Backend-aware read; .xls falls back to calamine for Mendeley files."""
    if kind == "csv":
        return pd.read_csv(path, low_memory=False)
    if kind == "xlsx":
        # Apify xlsx files use openpyxl by default — fine.
        return pd.read_excel(path)
    if kind == "xls":
        # Legacy .xls (Mendeley dump) needs python-calamine.
        return pd.read_excel(path, engine="calamine")
    raise ValueError(f"Unknown file_kind: {kind!r}")


def _normalize_text(s: object) -> str:
    """NFKC-normalize unicode (handles fullwidth / weird chars in tweets)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return unicodedata.normalize("NFKC", str(s)).strip()


def load_and_clean_source(source: DataSource) -> tuple[pd.DataFrame, dict]:
    """
    Read one DataSource and return:
      * a DataFrame with exactly OUTPUT_COLUMNS and `is_crypto` bool;
      * a per-source diagnostics dict (rows in / out / dropped at each step).
    """
    diag: dict[str, Any] = {
        "source":             source.canonical_username,
        "filename":           source.filename,
        "kind":               source.file_kind,
        "note":               source.note,
        "rows_raw":           0,
        "rows_after_required":0,
        "rows_after_dedup":   0,
        "rows_after_window":  0,
        "rows_final":         0,
        "warnings":           [],
    }

    path = os.path.join(MENDELEY_DIR, source.filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")

    raw = _read_raw(path, source.file_kind)
    diag["rows_raw"] = len(raw)

    # ---- column rename --------------------------------------------------
    rename = {src: tgt for tgt, src in source.column_map.items() if src in raw.columns}
    missing = [tgt for tgt, src in source.column_map.items() if src not in raw.columns]
    if missing:
        diag["warnings"].append(f"declared but missing columns: {missing}")
    df = raw.rename(columns=rename)

    # Drop everything we didn't ask for; this prevents downstream surprises
    # when two sources share a column name with different semantics.
    keep_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[keep_cols].copy()

    # ---- fill in missing optional columns ------------------------------
    if "username" not in df.columns:
        df["username"] = source.canonical_username
    if "tweet_id" not in df.columns:
        df["tweet_id"] = ""
    if "likes" not in df.columns:
        df["likes"] = 0
    if "retweets" not in df.columns:
        df["retweets"] = 0

    # ---- type & sanitization -------------------------------------------
    df["tweet_id"]   = df["tweet_id"].astype(str)
    df["username"]   = df["username"].astype(str).str.lstrip("@")
    df["text"]       = df["text"].apply(_normalize_text)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    for c in ("likes", "retweets"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # ---- drop rows missing required fields -----------------------------
    df = df.dropna(subset=list(REQUIRED_FIELDS))
    df = df[df["text"].str.len() > 0]
    diag["rows_after_required"] = len(df)

    # ---- canonicalize username casing to ACCOUNTS keys ------------------
    canon = {u.lower(): u for u in ACCOUNTS}
    df["username"] = df["username"].str.lower().map(canon).fillna(df["username"])
    # Sources we declared should only emit one canonical username; warn if
    # other usernames sneak in (rare, but observed in the Trump archive).
    foreign = (df["username"] != source.canonical_username).sum()
    if foreign:
        diag["warnings"].append(
            f"{foreign} rows had a username different from "
            f"canonical_username; coerced to {source.canonical_username!r}"
        )
        df["username"] = source.canonical_username

    # ---- per-source dedup ----------------------------------------------
    # tweet_id alone is unreliable (Excel float-precision loss), so we use
    # tweet_id when populated, otherwise fall back to (username, text,
    # created_at). Doing it per-source first keeps the global dedup cheap.
    has_id = df["tweet_id"].str.len() > 0
    if has_id.any():
        with_id    = df[has_id ].drop_duplicates(subset=["tweet_id"])
        without_id = df[~has_id].drop_duplicates(subset=["username", "text", "created_at"])
        df = pd.concat([with_id, without_id], ignore_index=True)
    else:
        df = df.drop_duplicates(subset=["username", "text", "created_at"]).reset_index(drop=True)
    diag["rows_after_dedup"] = len(df)

    # ---- time window ---------------------------------------------------
    start = pd.Timestamp(START_DATE, tz="UTC")
    end   = pd.Timestamp(END_DATE,   tz="UTC")
    df = df[(df["created_at"] >= start) & (df["created_at"] <= end)].reset_index(drop=True)
    diag["rows_after_window"] = len(df)

    # ---- final ordering / column shape --------------------------------
    df = df[list(OUTPUT_COLUMNS)].sort_values("created_at").reset_index(drop=True)
    diag["rows_final"] = len(df)
    return df, diag


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all_sources() -> tuple[pd.DataFrame, dict]:
    """
    Load every source listed in `DATA_SOURCES`, concatenate, perform a
    final cross-source dedup, and produce a data-quality report.

    Returns
    -------
    combined : DataFrame with OUTPUT_COLUMNS + `is_crypto` bool.
    report   : dict (also written to results/data_quality_report.json).
    """
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict]    = []

    for src in DATA_SOURCES:
        df, diag = load_and_clean_source(src)
        frames.append(df)
        diagnostics.append(diag)

    combined = pd.concat(frames, ignore_index=True)
    n_pre_global = len(combined)

    # Global dedup mirrors the per-source rule: tweet_id when present,
    # otherwise (username, text, created_at).
    has_id = combined["tweet_id"].str.len() > 0
    if has_id.any():
        with_id    = combined[has_id ].drop_duplicates(subset=["tweet_id"])
        without_id = combined[~has_id].drop_duplicates(subset=["username", "text", "created_at"])
        combined = pd.concat([with_id, without_id], ignore_index=True)
    else:
        combined = combined.drop_duplicates(subset=["username", "text", "created_at"])
    combined = combined.sort_values("created_at").reset_index(drop=True)

    # ---- crypto filter column (advisory, not enforced here) ------------
    combined = add_crypto_filter_columns(combined)

    # ---- per-account totals --------------------------------------------
    per_account = (
        combined.groupby("username").size().to_dict()
        if not combined.empty else {}
    )
    crypto_per_account = (
        combined.groupby("username")["is_crypto"].sum().to_dict()
        if not combined.empty else {}
    )

    report = {
        "config": {
            "start_date":      START_DATE,
            "end_date":        END_DATE,
            "n_sources":       len(DATA_SOURCES),
        },
        "per_source":          diagnostics,
        "global_dedup": {
            "rows_pre":        int(n_pre_global),
            "rows_post":       int(len(combined)),
            "rows_dropped":    int(n_pre_global - len(combined)),
        },
        "per_account_totals":  {k: int(v) for k, v in per_account.items()},
        "per_account_crypto":  {k: int(v) for k, v in crypto_per_account.items()},
        "crypto_share_overall": (
            float(combined["is_crypto"].mean()) if len(combined) else 0.0
        ),
    }

    out_path = os.path.join(RESULTS_DIR, "data_quality_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[load] data quality report → {out_path}")

    return combined, report


# ---------------------------------------------------------------------------
# Entry helper used by main.py
# ---------------------------------------------------------------------------

def write_all_tweets_csv(df: pd.DataFrame) -> str:
    out_path = os.path.join(DATA_DIR, "all_tweets.csv")
    # `is_crypto` is a derived field, but persisting it makes downstream
    # code cheaper and lets reviewers verify the filter offline.
    cols = list(OUTPUT_COLUMNS) + ["is_crypto"]
    df[cols].to_csv(out_path, index=False)
    print(f"[load] {len(df):,} rows → {out_path}")
    return out_path
