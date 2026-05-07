"""
v5 backtest — walk-forward evaluation that strictly mirrors the training
pipeline.

Guarantees:
  1. The crypto filter mode used here is read from `models/run_config.json`
     and asserted equal to `config.FILTER_MODE`. Mismatch raises.
  2. The test split is loaded directly from `data/test_split.csv` (which
     training writes after applying its filter and chronological split),
     so we cannot accidentally evaluate on tweets the trainer never saw
     in expectation.
  3. The expanded `backtest_results.csv` exposes per-row probabilities,
     argmax label, ground-truth label, expected return, true return, and
     a `confidence` column for downstream analysis.
  4. Metrics include macro-F1 (the model-selection metric), per-account
     accuracy/F1, and quarterly aggregates.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import BertTokenizer

from config import (
    ACCOUNTS,
    BERT_MODEL,
    DATA_DIR,
    FILTER_MODE,
    ID2LABEL,
    MAX_LENGTH,
    MODEL_DIR,
    RESULTS_DIR,
    apply_filter_mode,
)
from model import BertMLP


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)


# ---------------------------------------------------------------------------
# Conditional return means (used to translate class probs → expected return)
# ---------------------------------------------------------------------------

def compute_conditional_returns(labeled_df: pd.DataFrame) -> dict:
    """Average 4-h return per label class on the chronological train slice."""
    from config import chronological_split
    df = labeled_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at", "return_4h"])
    df = apply_filter_mode(df)
    train_df, *_ = chronological_split(df)
    out = {}
    for lbl, name in ID2LABEL.items():
        sub = train_df[train_df["label"] == lbl]["return_4h"].dropna()
        out[name] = float(sub.mean()) if len(sub) else 0.0
    print("[backtest] conditional means: "
          + ", ".join(f"{k}={v*100:+.3f}%" for k, v in out.items()))
    return out


# ---------------------------------------------------------------------------
# Filter-mode contract check
# ---------------------------------------------------------------------------

def _enforce_filter_contract() -> dict:
    """
    Read training's run_config.json and assert that its filter_mode equals
    the current `config.FILTER_MODE`. Returns the run_config dict for
    optional logging.
    """
    cfg_path = os.path.join(MODEL_DIR, "run_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"run_config.json not found at {cfg_path}; "
            "you must run `main.py train` first."
        )
    with open(cfg_path) as fh:
        run_config = json.load(fh)
    train_mode = run_config.get("filter_mode")
    if train_mode != FILTER_MODE:
        raise RuntimeError(
            f"FILTER_MODE mismatch — config.FILTER_MODE={FILTER_MODE!r} "
            f"but training was run with filter_mode={train_mode!r}. "
            "Re-run training, or change config.FILTER_MODE to match."
        )
    print(f"[backtest] filter mode contract OK ({FILTER_MODE!r})")
    return run_config


# ---------------------------------------------------------------------------
# Walk-forward inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_proba(text: str, model: BertMLP, tokenizer: BertTokenizer) -> dict:
    enc = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    probs = model.predict_proba(
        enc["input_ids"].to(DEVICE),
        enc["attention_mask"].to(DEVICE),
        enc["token_type_ids"].to(DEVICE),
    ).cpu().numpy()[0]
    return {"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])}


def predict_expected_return(p_up, p_flat, p_down, cond_returns) -> float:
    return (p_up   * cond_returns.get("up",   0.0)
            + p_flat * cond_returns.get("flat", 0.0)
            + p_down * cond_returns.get("down", 0.0))


def run_backtest(
    test_df:             pd.DataFrame,
    model:               BertMLP,
    tokenizer:           BertTokenizer,
    weights:             dict,
    conditional_returns: dict,
) -> pd.DataFrame:
    """Predict every test row and return an expanded dataframe."""
    model.eval()
    rows = []
    test_df = test_df.copy()
    test_df["created_at"] = pd.to_datetime(test_df["created_at"], utc=True, errors="coerce")
    test_df = test_df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Backtesting"):
        text = str(row["text"])
        true_label = int(row["label"])
        username   = str(row.get("username", ""))

        probs = _predict_proba(text, model, tokenizer)
        ordered = [probs["down"], probs["flat"], probs["up"]]
        pred_label = int(np.argmax(ordered))
        confidence = float(max(ordered))
        exp_ret    = predict_expected_return(probs["up"], probs["flat"], probs["down"],
                                             conditional_returns)
        actual_ret = float(row.get("return_4h", np.nan))

        ts = row["created_at"]
        quarter = f"Q{ts.quarter} {ts.year}"
        rows.append({
            "created_at":          ts,
            "username":            username,
            "text":                text[:200],
            "true_label":          true_label,
            "true_label_name":     ID2LABEL.get(true_label, "?"),
            "pred_label":          pred_label,
            "pred_label_name":     ID2LABEL.get(pred_label, "?"),
            "correct":             int(pred_label == true_label),
            "down_prob":           probs["down"],
            "flat_prob":           probs["flat"],
            "up_prob":             probs["up"],
            "confidence":          confidence,
            "expected_return_pct": exp_ret * 100,
            "actual_return_pct":   actual_ret * 100,
            "quarter":             quarter,
            "year":                int(ts.year),
            "influence_weight":    weights.get(username, 0.5),
        })

    backtest_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "backtest_results.csv")
    backtest_df.to_csv(out_path, index=False)
    print(f"[backtest] {len(backtest_df):,} rows → {out_path}")
    return backtest_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_backtest_metrics(backtest_df: pd.DataFrame) -> dict:
    """Detailed per-class, per-account, per-quarter, high-confidence metrics."""
    y_true = backtest_df["true_label"].values
    y_pred = backtest_df["pred_label"].values

    overall_acc = float(backtest_df["correct"].mean())
    majority_label = int(pd.Series(y_true).value_counts().idxmax())
    baseline_acc   = float((y_true == majority_label).mean())
    macro_f1       = float(f1_score(y_true, y_pred, labels=[0, 1, 2],
                                    average="macro", zero_division=0))

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(3)],
        output_dict=True, zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    # Per-account accuracy + macro-F1 (where there are enough rows).
    per_person = {}
    for u, sub in backtest_df.groupby("username"):
        if len(sub) < 5:
            per_person[u] = {"n": len(sub), "accuracy": float(sub["correct"].mean()),
                             "macro_f1": None}
            continue
        per_person[u] = {
            "n":        int(len(sub)),
            "accuracy": float(sub["correct"].mean()),
            "macro_f1": float(f1_score(sub["true_label"], sub["pred_label"],
                                       labels=[0, 1, 2], average="macro",
                                       zero_division=0)),
        }

    quarterly = (
        backtest_df.groupby("quarter")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_tweets"})
        .reset_index()
        .to_dict(orient="records")
    )
    yearly = (
        backtest_df.groupby("year")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_tweets"})
        .reset_index()
        .to_dict(orient="records")
    )

    # High-confidence subsets — useful for showing whether the model is
    # ever actually informative when it's "sure."
    high_conf = backtest_df[backtest_df["confidence"] >= 0.5]
    very_conf = backtest_df[backtest_df["confidence"] >= 0.7]

    metrics = {
        "n_total":          int(len(backtest_df)),
        "overall_accuracy": overall_acc,
        "baseline_accuracy": baseline_acc,
        "lift_over_baseline": overall_acc - baseline_acc,
        "macro_f1":         macro_f1,
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "per_person":       per_person,
        "quarterly":        quarterly,
        "yearly":           yearly,
        "high_confidence_subset": {
            "threshold_0.5": {
                "n":        int(len(high_conf)),
                "accuracy": float(high_conf["correct"].mean()) if len(high_conf) else None,
            },
            "threshold_0.7": {
                "n":        int(len(very_conf)),
                "accuracy": float(very_conf["correct"].mean()) if len(very_conf) else None,
            },
        },
    }

    out_path = os.path.join(RESULTS_DIR, "backtest_metrics.json")
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(f"[backtest] metrics → {out_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  n_total            : {metrics['n_total']:,}")
    print(f"  overall accuracy   : {overall_acc*100:.2f} %")
    print(f"  majority baseline  : {baseline_acc*100:.2f} %")
    print(f"  lift over baseline : {(overall_acc - baseline_acc)*100:+.2f} pp")
    print(f"  macro-F1           : {macro_f1:.4f}")
    print("  per-class:")
    for cls in ("down", "flat", "up"):
        m = report.get(cls, {})
        print(f"    {cls:4s}  P={m.get('precision', 0):.3f} "
              f"R={m.get('recall', 0):.3f} F1={m.get('f1-score', 0):.3f} "
              f"(n={int(m.get('support', 0))})")
    print("  per-person:")
    for u, s in sorted(per_person.items(), key=lambda kv: -kv[1]["accuracy"]):
        name = ACCOUNTS.get(u, {}).get("name", u)
        f1s  = f"{s['macro_f1']:.3f}" if s.get("macro_f1") is not None else "n/a"
        print(f"    {name:18s}  acc={s['accuracy']*100:5.1f} %  "
              f"macro_f1={f1s}  (n={s['n']})")
    return metrics


# ---------------------------------------------------------------------------
# Period influence (kept for visualize.py compatibility)
# ---------------------------------------------------------------------------

def compute_influence_by_period(labeled_df: pd.DataFrame) -> pd.DataFrame:
    df = labeled_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at", "return_4h"])
    df["abs_return"]  = df["return_4h"].abs()
    df["year"]        = df["created_at"].dt.year
    df["quarter_num"] = df["created_at"].dt.quarter
    df["quarter_str"] = df.apply(lambda r: f"Q{r['quarter_num']} {r['year']}", axis=1)
    rows = []
    for (qstr, uname), grp in df.groupby(["quarter_str", "username"]):
        if grp.empty:
            continue
        rows.append({
            "quarter_str":        qstr,
            "year":               int(grp["year"].iloc[0]),
            "quarter_num":        int(grp["quarter_num"].iloc[0]),
            "username":           uname,
            "name":               ACCOUNTS.get(uname, {}).get("name", uname),
            "avg_abs_return_pct": float(grp["abs_return"].mean() * 100),
            "n_tweets":           int(len(grp)),
        })
    out = (pd.DataFrame(rows)
             .sort_values(["year", "quarter_num", "username"])
             .reset_index(drop=True))
    out_path = os.path.join(RESULTS_DIR, "influence_by_period.csv")
    out.to_csv(out_path, index=False)
    print(f"[backtest] influence-by-period → {out_path}")
    return out
