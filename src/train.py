"""
v5 training — BERT-base-uncased + 2-layer MLP, fine-tuned on the labeled
tweets dataset.

Engineering improvements over v4:
  * `set_seed(SEED)` covers Python / NumPy / Torch / CUDA / MPS so reruns
    produce the same numbers.
  * Splits are time-ordered via `chronological_split()`; the boundaries
    are persisted to `results/split_report.json`.
  * Class weights are inverse-frequency, normalized so the average weight
    is 1.0 (keeps absolute loss scale comparable to the unweighted run).
  * Best checkpoint is selected by val macro-F1 (not val_loss / val_acc).
  * Early stopping with patience.
  * Per-epoch metrics + final config saved under `results/runs/<id>/`.
  * The crypto filter mode is recorded in `run_config.json`; `backtest.py`
    reads that file and asserts the same value before evaluating.

Output paths used by the rest of the pipeline are unchanged:
  models/best_model.pt         — checkpoint
  models/tokenizer/            — tokenizer for downstream loaders
  models/training_history.json — per-epoch numbers
  models/test_report.json      — sklearn classification report
  data/test_split.csv          — the test slice (already filter-applied)
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BERT_MODEL,
    DATA_DIR,
    DROPOUT,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    FILTER_MODE,
    ID2LABEL,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DIR,
    MODEL_SELECTION_METRIC,
    NUM_EPOCHS,
    RUNS_DIR,
    SEED,
    apply_filter_mode,
    chronological_split,
    config_snapshot,
)
from dataset import TweetDataset
from model import BertMLP


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS: torch.manual_seed already covers MPS RNG in modern PyTorch.


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        ttype = batch["token_type_ids"].to(device)
        y     = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask, ttype)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, device, split_name: str) -> tuple[float, dict, list, list]:
    model.eval()
    total = 0.0
    all_y, all_yhat = [], []
    for batch in tqdm(loader, desc=f"  {split_name}", leave=False):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        ttype = batch["token_type_ids"].to(device)
        y     = batch["label"].to(device)
        logits = model(ids, mask, ttype)
        total += criterion(logits, y).item()
        yhat = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        all_y.extend(y.cpu().numpy().tolist())
        all_yhat.extend(yhat)
    avg_loss = total / max(len(loader), 1)
    report = classification_report(
        all_y, all_yhat,
        labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )
    return avg_loss, report, all_y, all_yhat


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    data_path: str | None = None,
    save_dir:  str | None = None,
) -> tuple["BertMLP", "BertTokenizer", pd.DataFrame]:
    """Run training and return (model, tokenizer, test_df)."""
    set_seed(SEED)
    device = select_device()
    print(f"[train] device={device}, seed={SEED}")

    # ---- Run directory --------------------------------------------------
    run_id  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    save_dir = save_dir or MODEL_DIR
    data_path = data_path or os.path.join(DATA_DIR, "labeled_tweets.csv")

    # ---- Load labeled data ---------------------------------------------
    df = pd.read_csv(data_path)
    df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"]).reset_index(drop=True)
    print(f"[train] labeled rows (raw)            : {len(df):,}")

    # ---- Crypto filter (if enabled) ------------------------------------
    df = apply_filter_mode(df)
    print(f"[train] filter mode = {FILTER_MODE!r}, rows after filter: {len(df):,}")

    # ---- Chronological split -------------------------------------------
    train_df, val_df, test_df, split_report = chronological_split(df)
    with open(os.path.join(run_dir, "split_report.json"), "w") as fh:
        json.dump(split_report, fh, indent=2)
    # Also write at canonical path so visualize.py / backtest.py can find it.
    with open(os.path.join(MODEL_DIR, "split_report.json"), "w") as fh:
        json.dump(split_report, fh, indent=2)
    print(f"[train] split (chrono)                : "
          f"train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")

    if len(train_df) == 0:
        raise RuntimeError("Empty training split — check data window / filter mode.")

    test_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)

    # ---- Class distribution + weights ----------------------------------
    counts = Counter(train_df["label"].tolist())
    print("[train] train label counts:")
    for k in (0, 1, 2):
        print(f"          {ID2LABEL[k]:4s}: {counts[k]:,}")
    weights = np.array([1.0 / max(counts[i], 1) for i in range(3)], dtype=np.float64)
    weights = weights / weights.sum() * 3.0  # average weight = 1
    class_weights = torch.tensor(weights, device=device, dtype=torch.float)
    print(f"[train] class weights: down={weights[0]:.3f}, "
          f"flat={weights[1]:.3f}, up={weights[2]:.3f}")

    # ---- Tokenizer + Datasets ------------------------------------------
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    train_ds = TweetDataset(train_df, tokenizer)
    val_ds   = TweetDataset(val_df,   tokenizer)
    test_ds  = TweetDataset(test_df,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ---- Model + optimizer + scheduler ---------------------------------
    model     = BertMLP(num_classes=3, dropout=DROPOUT).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    no_decay = {"bias", "LayerNorm.weight"}
    optimizer = AdamW(
        [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters()
                        if     any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7,
    )

    # ---- Training loop with early stopping (on val_macro_f1) -----------
    history          = []
    best_metric      = -float("inf")
    best_epoch       = -1
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_report, _, _ = _evaluate(model, val_loader, criterion, device, "val")

        val_acc      = val_report["accuracy"]
        val_macro_f1 = val_report["macro avg"]["f1-score"]
        print(f"  train_loss = {train_loss:.4f}")
        print(f"  val_loss   = {val_loss:.4f}   "
              f"val_acc = {val_acc:.4f}   val_macro_f1 = {val_macro_f1:.4f}")
        for cls in ("down", "flat", "up"):
            m = val_report.get(cls, {})
            print(f"    {cls:4s} P={m.get('precision', 0):.3f} "
                  f"R={m.get('recall', 0):.3f} F1={m.get('f1-score', 0):.3f}")

        scheduler.step(val_macro_f1)
        history.append({
            "epoch":         epoch,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "val_accuracy":  val_acc,
            "val_macro_f1":  val_macro_f1,
            "lr":            optimizer.param_groups[0]["lr"],
        })

        # ---- model selection by chosen metric ---------------------------
        chosen = {"val_macro_f1": val_macro_f1, "val_accuracy": val_acc,
                  "val_loss": -val_loss}[MODEL_SELECTION_METRIC]
        improved = chosen > best_metric + EARLY_STOPPING_MIN_DELTA
        if improved:
            best_metric = chosen
            best_epoch  = epoch
            epochs_no_improve = 0
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":         val_loss,
                "val_accuracy":     val_acc,
                "val_macro_f1":     val_macro_f1,
            }, ckpt_path)
            tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
            print(f"  → saved best (epoch {epoch}, "
                  f"{MODEL_SELECTION_METRIC}={best_metric:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  no improvement on {MODEL_SELECTION_METRIC} "
                  f"({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("  → early stopping triggered")
                break

    elapsed = time.time() - start_time
    print(f"\n[train] training finished in {elapsed:.1f}s, "
          f"best epoch = {best_epoch}, best {MODEL_SELECTION_METRIC} = {best_metric:.4f}")

    # ---- Reload best checkpoint and evaluate on test set ---------------
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_report, y_true, y_pred = _evaluate(
        model, test_loader, criterion, device, "test"
    )
    test_macro_f1 = test_report["macro avg"]["f1-score"]
    print(f"\n[train] TEST  loss={test_loss:.4f}  "
          f"acc={test_report['accuracy']:.4f}  macro_f1={test_macro_f1:.4f}")
    for cls in ("down", "flat", "up"):
        m = test_report.get(cls, {})
        print(f"  {cls:4s}  P={m.get('precision',0):.3f} "
              f"R={m.get('recall',0):.3f} F1={m.get('f1-score',0):.3f}")

    # ---- Persist artifacts ---------------------------------------------
    with open(os.path.join(save_dir, "training_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    with open(os.path.join(save_dir, "test_report.json"), "w") as fh:
        json.dump(test_report, fh, indent=2)
    with open(os.path.join(run_dir, "test_report.json"), "w") as fh:
        json.dump(test_report, fh, indent=2)
    with open(os.path.join(run_dir, "training_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    # `run_config.json` is the contract that `backtest.py` reads to verify
    # train/eval consistency. Keep it small and machine-friendly.
    run_config = {
        "run_id":        run_id,
        "filter_mode":   FILTER_MODE,
        "best_epoch":    best_epoch,
        "best_metric":   {MODEL_SELECTION_METRIC: float(best_metric)},
        "device":        str(device),
        "elapsed_sec":   float(elapsed),
        "config":        config_snapshot(),
        "split_report":  split_report,
        "test_metrics": {
            "loss":     float(test_loss),
            "accuracy": float(test_report["accuracy"]),
            "macro_f1": float(test_macro_f1),
        },
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)
    with open(os.path.join(MODEL_DIR, "run_config.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)
    print(f"[train] run artifacts → {run_dir}")
    return model, tokenizer, test_df
