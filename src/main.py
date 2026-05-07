"""
v5 CLI entry point. Same commands as the original main.py:

    python main.py load        # multi-source ingest + label assignment
    python main.py weights     # historical influence weights
    python main.py train       # BERT + MLP fine-tune
    python main.py backtest    # walk-forward evaluation on test split
    python main.py visualize   # 13-plot bundle
    python main.py report      # backtest + visualize
    python main.py all         # load → weights → train → backtest → visualize
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from config import (
    DATA_DIR,
    FILTER_MODE,
    MODEL_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    apply_filter_mode,
)


# ---------------------------------------------------------------------------
# Step 1 — load + label
# ---------------------------------------------------------------------------

def cmd_load() -> pd.DataFrame:
    from data_loading import load_all_sources, write_all_tweets_csv
    from labeling     import assign_btc_labels
    from btc_prices   import fetch_hourly_btc

    print("=" * 60)
    print("STEP 1  —  Load & clean sources")
    print("=" * 60)
    combined, _ = load_all_sources()
    write_all_tweets_csv(combined)

    print("\n" + "=" * 60)
    print("STEP 2  —  BTC hourly candles (Coinbase)")
    print("=" * 60)
    fetch_hourly_btc()

    print("\n" + "=" * 60)
    print("STEP 3  —  Label by 4h forward return (no lookahead)")
    print("=" * 60)
    labeled, report = assign_btc_labels(combined)
    out_path = os.path.join(DATA_DIR, "labeled_tweets.csv")
    labeled.to_csv(out_path, index=False)
    print(f"[load] {len(labeled):,} labeled rows → {out_path}")
    print(f"[load] labeling report:\n{json.dumps(report, indent=2, default=str)}")
    return labeled


# ---------------------------------------------------------------------------
# Step 2 — influence weights
# ---------------------------------------------------------------------------

def cmd_weights(labeled_df: pd.DataFrame | None = None) -> dict:
    from influence_weights import compute_influence_weights
    return compute_influence_weights(labeled_df)


# ---------------------------------------------------------------------------
# Step 3 — train
# ---------------------------------------------------------------------------

def cmd_train(data_path: str | None = None):
    from train import train
    return train(data_path=data_path)


# ---------------------------------------------------------------------------
# Step 4 — backtest
# ---------------------------------------------------------------------------

def _load_checkpoint():
    """Reload best checkpoint and tokenizer."""
    import torch
    from transformers import BertTokenizer
    from config import BERT_MODEL
    from model import BertMLP

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    ckpt_path      = os.path.join(MODEL_DIR, "best_model.pt")
    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    tokenizer = (
        BertTokenizer.from_pretrained(tokenizer_path)
        if os.path.isdir(tokenizer_path)
        else BertTokenizer.from_pretrained(BERT_MODEL)
    )
    model = BertMLP(num_classes=3).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


def cmd_backtest(
    labeled_df: pd.DataFrame | None = None,
    model = None,
    tokenizer = None,
    weights:    dict | None = None,
) -> dict:
    from backtest import (
        _enforce_filter_contract,
        compute_conditional_returns,
        compute_backtest_metrics,
        compute_influence_by_period,
        run_backtest,
    )
    from influence_weights import load_influence_weights

    _enforce_filter_contract()

    if labeled_df is None:
        labeled_df = pd.read_csv(os.path.join(DATA_DIR, "labeled_tweets.csv"))
    labeled_df["created_at"] = pd.to_datetime(
        labeled_df["created_at"], utc=True, errors="coerce"
    )

    # Use the test split that train.py wrote (already filter-applied and
    # chronologically split). This is the ground-truth contract.
    test_split_path = os.path.join(DATA_DIR, "test_split.csv")
    if os.path.exists(test_split_path):
        test_df = pd.read_csv(test_split_path)
        test_df["created_at"] = pd.to_datetime(test_df["created_at"], utc=True, errors="coerce")
        print(f"[backtest] test split loaded from {test_split_path}: "
              f"{len(test_df):,} rows")
    else:
        # Fallback: rebuild the split using the same logic. This branch
        # exists for completeness but should not normally fire because
        # train.py always writes the file.
        from config import chronological_split
        df = apply_filter_mode(labeled_df)
        _, _, test_df, _ = chronological_split(df)
        print(f"[backtest] test_split.csv missing; rebuilt {len(test_df):,} rows")

    if model is None or tokenizer is None:
        model, tokenizer = _load_checkpoint()
    if weights is None:
        weights = load_influence_weights()

    cond_returns = compute_conditional_returns(labeled_df)
    backtest_df  = run_backtest(
        test_df=test_df, model=model, tokenizer=tokenizer,
        weights=weights, conditional_returns=cond_returns,
    )
    metrics   = compute_backtest_metrics(backtest_df)
    period_df = compute_influence_by_period(labeled_df)

    return {
        "backtest_df":         backtest_df,
        "metrics":             metrics,
        "conditional_returns": cond_returns,
        "y_true":              backtest_df["true_label"].tolist(),
        "y_pred":              backtest_df["pred_label"].tolist(),
        "test_report":         metrics["classification_report"],
        "period_df":           period_df,
        "labeled_df":          labeled_df,
        "weights":             weights,
    }


# ---------------------------------------------------------------------------
# Step 5 — visualize
# ---------------------------------------------------------------------------

def cmd_visualize(backtest_artifacts: dict | None = None) -> list:
    import visualize
    from influence_weights import load_influence_weights
    from sklearn.metrics import classification_report
    from config import ID2LABEL

    btc_path      = os.path.join(DATA_DIR, "btc_prices_1h.csv")
    tweets_path   = os.path.join(DATA_DIR, "all_tweets.csv")
    labeled_path  = os.path.join(DATA_DIR, "labeled_tweets.csv")
    backtest_path = os.path.join(RESULTS_DIR, "backtest_results.csv")
    period_path   = os.path.join(RESULTS_DIR, "influence_by_period.csv")
    weights_path  = os.path.join(DATA_DIR, "influence_weights.json")

    btc_df     = pd.read_csv(btc_path)     if os.path.exists(btc_path)     else pd.DataFrame()
    tweets_df  = pd.read_csv(tweets_path)  if os.path.exists(tweets_path)  else pd.DataFrame()
    labeled_df = pd.read_csv(labeled_path) if os.path.exists(labeled_path) else pd.DataFrame()

    if backtest_artifacts:
        backtest_df  = backtest_artifacts["backtest_df"]
        y_true       = backtest_artifacts["y_true"]
        y_pred       = backtest_artifacts["y_pred"]
        test_report  = backtest_artifacts["test_report"]
        cond_returns = backtest_artifacts["conditional_returns"]
        period_df    = backtest_artifacts.get("period_df", pd.DataFrame())
        weights      = backtest_artifacts["weights"]
    else:
        backtest_df = pd.read_csv(backtest_path) if os.path.exists(backtest_path) else pd.DataFrame()
        period_df   = pd.read_csv(period_path)   if os.path.exists(period_path)   else pd.DataFrame()
        weights     = load_influence_weights()   if os.path.exists(weights_path)  else {}
        if not backtest_df.empty:
            y_true = backtest_df["true_label"].tolist()
            y_pred = backtest_df["pred_label"].tolist()
        else:
            y_true, y_pred = [], []
        test_report_path = os.path.join(MODEL_DIR, "test_report.json")
        if os.path.exists(test_report_path):
            with open(test_report_path) as fh:
                test_report = json.load(fh)
        elif y_true:
            test_report = classification_report(
                y_true, y_pred,
                target_names=[ID2LABEL[i] for i in range(3)],
                output_dict=True, zero_division=0,
            )
        else:
            test_report = {}
        cond_returns = {"up": 0.015, "flat": 0.0, "down": -0.015}

    return visualize.plot_all(
        btc_df=btc_df,
        tweets_df=tweets_df,
        labeled_df=labeled_df,
        backtest_df=backtest_df,
        weights=weights,
        period_df=period_df,
        y_true_test=y_true,
        y_pred_test=y_pred,
        test_report=test_report,
        weighted_up_prob=0.5,
        expected_return_pct=0.0,
        conditional_returns=cond_returns,
    )


# ---------------------------------------------------------------------------
# Convenience: report = backtest + visualize
# ---------------------------------------------------------------------------

def cmd_report(*, labeled_df=None, model=None, tokenizer=None, weights=None):
    artifacts = cmd_backtest(labeled_df=labeled_df, model=model,
                             tokenizer=tokenizer, weights=weights)
    if artifacts:
        cmd_visualize(artifacts)


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------

def cmd_all() -> None:
    print(f"[all] FILTER_MODE = {FILTER_MODE!r}")
    df = cmd_load()
    weights = cmd_weights(df)
    model, tokenizer, _ = cmd_train()
    artifacts = cmd_backtest(labeled_df=df, model=model,
                             tokenizer=tokenizer, weights=weights)
    cmd_visualize(artifacts)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["load", "weights", "train", "backtest",
                 "visualize", "report", "all"],
    )
    args = parser.parse_args()
    {
        "load":      cmd_load,
        "weights":   cmd_weights,
        "train":     cmd_train,
        "backtest":  cmd_backtest,
        "visualize": cmd_visualize,
        "report":    cmd_report,
        "all":       cmd_all,
    }[args.command]()
    print("\n[done]")


if __name__ == "__main__":
    main()
