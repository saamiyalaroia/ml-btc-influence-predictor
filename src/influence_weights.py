"""
Historical influence weight computation.

For each of the five accounts:
  1. Filter the labeled dataset to crypto-related tweets only.
  2. Compute the absolute 4-hour BTC return for each such tweet.
  3. Average those absolute returns -> "avg_impact" for the person.
  4. Normalize so the person with the largest avg_impact = 1.0 (100 %).

The proposal notes: "We assume Elon Musk may have the strongest influence,
so his weight can be set as the reference (100 %)." The normalization below
implements exactly this: whichever person has the highest measured impact
is scaled to 100 %; everyone else is relative to that.
"""

import os
import json

import numpy as np
import pandas as pd

from config import (
    ACCOUNTS,
    CRYPTO_KEYWORDS,
    DATA_DIR,
    WEIGHTS_FILE,
    is_crypto_related,
)

# Back-compat aliases for callers that import these names directly.
_CRYPTO_KEYWORDS   = CRYPTO_KEYWORDS
_is_crypto_related = is_crypto_related


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_influence_weights(
    labeled_df: pd.DataFrame = None,
) -> dict[str, float]:
    """
    Compute and save historical influence weights for the five accounts.

    Parameters
    ----------
    labeled_df : If None, loads data/labeled_tweets.csv automatically.

    Returns
    -------
    weights : dict  username -> normalized weight  (0.0 – 1.0)
              The highest-impact person receives weight 1.0 (100 %).
    """
    if labeled_df is None:
        csv_path = os.path.join(DATA_DIR, "labeled_tweets.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Labeled dataset not found at {csv_path}.\n"
                "Run  data_collection.prepare_dataset()  first."
            )
        labeled_df = pd.read_csv(csv_path)

    # Filter to crypto-related tweets only.
    crypto_mask = labeled_df["text"].apply(_is_crypto_related)
    crypto_df   = labeled_df[crypto_mask].copy()

    print(f"Total labeled tweets        : {len(labeled_df)}")
    print(f"Crypto-related tweets only  : {len(crypto_df)}")
    print()

    # ------------------------------------------------------------------
    # Average absolute 4-hour return per person
    # ------------------------------------------------------------------
    avg_impacts: dict[str, float] = {}

    for username in ACCOUNTS:
        person_df   = crypto_df[crypto_df["username"] == username]
        abs_returns = person_df["return_4h"].abs().dropna()

        if abs_returns.empty:
            print(f"  WARNING: No crypto tweets with price data for @{username}")
            avg_impacts[username] = 0.0
        else:
            avg_impacts[username] = float(abs_returns.mean())

        pct = avg_impacts[username] * 100
        print(
            f"  @{username:<20s}  "
            f"crypto tweets={len(person_df):5d}   "
            f"avg |4h return| = {pct:.3f} %"
        )

    # ------------------------------------------------------------------
    # Normalize: highest impact -> 1.0 (100 %)
    # ------------------------------------------------------------------
    max_impact = max(avg_impacts.values()) if avg_impacts else 0.0

    if max_impact == 0.0:
        # Edge case: no signal found – assign equal weights.
        weights = {k: 1.0 for k in avg_impacts}
    else:
        weights = {k: v / max_impact for k, v in avg_impacts.items()}

    print("\nNormalized influence weights:")
    for username, w in sorted(weights.items(), key=lambda x: -x[1]):
        name = ACCOUNTS[username]["name"]
        print(f"  {name} (@{username}) : {w * 100:.1f} %")

    # ------------------------------------------------------------------
    # Persist to disk
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(WEIGHTS_FILE) or ".", exist_ok=True)
    payload = {
        "weights":     weights,
        "avg_impacts": avg_impacts,
        "max_impact":  max_impact,
    }
    with open(WEIGHTS_FILE, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nInfluence weights saved to {WEIGHTS_FILE}")

    return weights


# ---------------------------------------------------------------------------
# Loader (used by inference.py)
# ---------------------------------------------------------------------------

def load_influence_weights() -> dict[str, float]:
    """
    Load pre-computed influence weights from disk.

    Returns
    -------
    weights : dict  username -> float (0.0 – 1.0)
    """
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            f"Weights file not found at {WEIGHTS_FILE}.\n"
            "Run  influence_weights.compute_influence_weights()  first."
        )
    with open(WEIGHTS_FILE) as fh:
        data = json.load(fh)
    return data["weights"]
