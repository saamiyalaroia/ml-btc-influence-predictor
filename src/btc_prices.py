"""
BTC price fetcher (Coinbase public REST). Hourly candles only.

Coinbase is used instead of Binance because Binance's API blocks US IPs.
The endpoint is public and unauthenticated; we paginate in 300-candle
windows (the API maximum) and respect a polite request rate.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from tqdm import tqdm

from config import DATA_DIR, END_DATE, START_DATE


_COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRANULARITY_SEC = 3600


def _fetch_batch(start_iso: str, end_iso: str) -> list:
    resp = requests.get(
        _COINBASE_URL,
        params={"start": start_iso, "end": end_iso, "granularity": GRANULARITY_SEC},
        headers={"User-Agent": "btc-tweet-analysis/v5"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_hourly_btc(
    start_date: str = START_DATE,
    end_date:   str = END_DATE,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Return BTC/USD hourly candles in [start_date, end_date]. Cached.

    Schema: timestamp (UTC datetime) / open / high / low / close / volume.
    """
    if cache_path is None:
        cache_path = os.path.join(DATA_DIR, "btc_prices_1h.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(end_date  ).replace(tzinfo=timezone.utc)

    window = timedelta(seconds=GRANULARITY_SEC * 300)
    rows: list[list] = []
    current = start_dt
    n_total = max(1, int((end_dt - start_dt) / window) + 1)
    pbar = tqdm(total=n_total, desc="Fetching BTC candles", unit="win")
    consecutive_errors = 0
    while current < end_dt:
        batch_end = min(current + window, end_dt)
        try:
            batch = _fetch_batch(
                current.isoformat().replace("+00:00", "Z"),
                batch_end.isoformat().replace("+00:00", "Z"),
            )
            consecutive_errors = 0
        except requests.RequestException as exc:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                raise RuntimeError(
                    f"Coinbase fetch failed 5× in a row: {exc}"
                )
            time.sleep(5)
            continue
        if batch:
            rows.extend(batch)
        current = batch_end
        pbar.update(1)
        time.sleep(0.12)
    pbar.close()

    df = pd.DataFrame(rows, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
          .drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    df.to_csv(cache_path, index=False)
    return df
