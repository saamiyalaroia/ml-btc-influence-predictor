"""
Visualization suite — generates all 13 plots and saves them to plots/.

Plot inventory
--------------
01  btc_price_history.png         BTC/USD 2018-2025, event annotations, train/test shading
02  btc_price_with_tweets.png     BTC price + per-person tweet event markers
03  training_history.png          Loss + accuracy curves per epoch
04  confusion_matrix_test.png     Normalized confusion matrix (test set 2023-2025)
05  classification_report_test.png Precision / Recall / F1 grouped bar chart
06  label_distribution.png        Label counts per person + 4h-return KDE by label
07  influence_weights.png         Horizontal bar chart of normalized influence weights
08  influence_over_time.png       Per-person avg |4h return| by calendar quarter
09  backtest_accuracy_over_time.png Rolling 90-day accuracy vs 33.3 % baseline
10  backtest_heatmap_accuracy.png  Year × Quarter accuracy heatmap
11  per_person_accuracy.png       Per-person accuracy on test set
12  quantified_return_scatter.png  Predicted expected return vs actual return
13  future_price_projection.png    Fan chart: last 90 days + 4-hour forward projection
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    ACCOUNTS, DATA_DIR, MODEL_DIR, RESULTS_DIR, PLOTS_DIR,
    TRAIN_END_DATE, TEST_START_DATE, ID2LABEL,
)

os.makedirs(PLOTS_DIR, exist_ok=True)

# Color palette for accounts (consistent across all plots)
ACCOUNT_COLORS = {
    "realDonaldTrump": "#E63946",   # red
    "elonmusk":        "#457B9D",   # blue
    "cz_binance":      "#F4A261",   # orange
    "VitalikButerin":  "#2A9D8F",   # teal
    "saylor":          "#8338EC",   # purple
}

LABEL_COLORS = {
    "up":   "#2DC653",
    "flat": "#AAAAAA",
    "down": "#E63946",
}

sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {path}")
    return path


# ---------------------------------------------------------------------------
# 01  BTC price history with event annotations
# ---------------------------------------------------------------------------

_BTC_EVENTS = [
    ("2018-01-01", "2018 peak\n~$19k",      "top"),
    ("2018-12-15", "2018 crash\n~$3.2k",    "bottom"),
    ("2020-03-13", "COVID crash\n~$3.8k",   "bottom"),
    ("2020-10-01", "Institutional\nbuying",  "top"),
    ("2021-04-14", "ATH $64k",              "top"),
    ("2021-11-10", "ATH $69k",              "top"),
    ("2022-05-12", "LUNA collapse",         "bottom"),
    ("2022-11-11", "FTX collapse",          "bottom"),
    ("2023-01-01", "Recovery",              "top"),
    ("2024-01-10", "Spot ETF\napproval",    "top"),
    ("2024-04-20", "Halving",               "top"),
    ("2025-01-01", "New cycle",             "top"),
]


def plot_btc_price_history(btc_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    BTC/USD full history 2018-2025 with:
      - Log-scale y-axis
      - Shaded train (2018-2023) and test (2023-2025) periods
      - Annotated key events
    """
    df = btc_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(df["timestamp"], df["close"], color="#1A1A2E", linewidth=0.8, alpha=0.9, zorder=3)

    # Shade training and test periods
    train_end  = pd.Timestamp(TRAIN_END_DATE,  tz="UTC")
    test_start = pd.Timestamp(TEST_START_DATE, tz="UTC")
    xmin = df["timestamp"].min()
    xmax = df["timestamp"].max()

    ax.axvspan(xmin,       train_end, alpha=0.08, color="steelblue",  label="Train (2018-2023)")
    ax.axvspan(test_start, xmax,      alpha=0.12, color="darkorange", label="Test  (2023-2025)")
    ax.axvline(train_end, color="steelblue",  linestyle="--", linewidth=1.2, alpha=0.7)

    # Major event annotations
    y_log_range = np.log10(df["close"].max()) - np.log10(df["close"].min())
    for date_str, label, pos in _BTC_EVENTS:
        ts = pd.Timestamp(date_str, tz="UTC")
        if ts < xmin or ts > xmax:
            continue
        # Nearest price
        idx   = (df["timestamp"] - ts).abs().idxmin()
        price = df.loc[idx, "close"]
        offset = 1.3 if pos == "top" else 0.6
        ax.annotate(
            label,
            xy=(ts, price),
            xytext=(ts, price * offset),
            fontsize=7,
            ha="center",
            color="#444444",
            arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("BTC / USD  (log scale)", fontsize=12)
    ax.set_title("Bitcoin Price History  2018 – 2025\nOther factors: macroeconomics, regulation, "
                 "institutional adoption, technological events", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()

    if save:
        _save(fig, "01_btc_price_history.png")
    return fig


# ---------------------------------------------------------------------------
# 02  BTC price + tweet event markers
# ---------------------------------------------------------------------------

def plot_btc_with_tweet_events(
    btc_df:    pd.DataFrame,
    tweets_df: pd.DataFrame,
    save:      bool = True,
) -> plt.Figure:
    """BTC price as background + vertical tick marks per person's tweets."""
    btc = btc_df.copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    btc = btc.sort_values("timestamp")

    tw = tweets_df.copy()
    tw["created_at"] = pd.to_datetime(tw["created_at"], utc=True, errors="coerce")
    tw = tw.dropna(subset=["created_at"])

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(btc["timestamp"], btc["close"], color="#1A1A2E", linewidth=0.7, alpha=0.8, zorder=1,
            label="BTC/USD")

    ymin, ymax = btc["close"].min() * 0.9, btc["close"].max() * 1.1

    for username, color in ACCOUNT_COLORS.items():
        subset = tw[tw["username"] == username]
        name   = ACCOUNTS.get(username, {}).get("name", username)
        for ts in subset["created_at"]:
            ax.axvline(ts, color=color, alpha=0.3, linewidth=0.6, zorder=2)
        # Dummy line for legend
        ax.plot([], [], color=color, linewidth=2, label=name)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("BTC / USD  (log scale)", fontsize=12)
    ax.set_title("BTC Price vs. Influential Tweets  (2018 – 2025)\n"
                 "Vertical lines mark individual tweet timestamps", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    if save:
        _save(fig, "02_btc_price_with_tweets.png")
    return fig


# ---------------------------------------------------------------------------
# 03  Training history: loss + accuracy curves
# ---------------------------------------------------------------------------

def plot_training_history(
    history_path: str = None,
    save:         bool = True,
) -> plt.Figure:
    """Two subplots: (left) train/val loss; (right) val accuracy per epoch."""
    history_path = history_path or os.path.join(MODEL_DIR, "training_history.json")
    with open(history_path) as fh:
        history = json.load(fh)

    epochs     = [h["epoch"]        for h in history]
    train_loss = [h["train_loss"]   for h in history]
    val_loss   = [h["val_loss"]     for h in history]
    val_acc    = [h["val_accuracy"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    axes[0].plot(epochs, train_loss, "o-", color="#E63946", label="Train loss")
    axes[0].plot(epochs, val_loss,   "s--", color="#457B9D",  label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in val_acc], "o-", color="#2DC653")
    axes[1].axhline(33.3, linestyle="--", color="#AAAAAA", linewidth=1, label="Random baseline")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].set_ylim(0, 105)

    fig.suptitle("Model Training History  (BERT + MLP)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        _save(fig, "03_training_history.png")
    return fig


# ---------------------------------------------------------------------------
# 04  Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true:      list,
    y_pred:      list,
    split_label: str  = "test",
    save:        bool = True,
) -> plt.Figure:
    """Seaborn heatmap of normalized confusion matrix."""
    labels     = [0, 1, 2]
    label_names = [ID2LABEL[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    overall_acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "% of actual class"},
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("Actual label",    fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {split_label.title()} set\n"
        f"Overall accuracy: {overall_acc:.1f} %",
        fontsize=13,
    )
    fig.tight_layout()

    if save:
        _save(fig, f"04_confusion_matrix_{split_label}.png")
    return fig


# ---------------------------------------------------------------------------
# 05  Classification report bars
# ---------------------------------------------------------------------------

def plot_classification_report_bars(
    report_dict: dict,
    split_label: str  = "test",
    save:        bool = True,
) -> plt.Figure:
    """Grouped bar chart: Precision / Recall / F1 per class."""
    classes  = ["down", "flat", "up"]
    metrics  = ["precision", "recall", "f1-score"]
    colors   = ["#E63946", "#457B9D", "#2DC653"]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report_dict.get(cls, {}).get(metric, 0) * 100 for cls in classes]
        bars = ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    acc = report_dict.get("accuracy", 0) * 100
    ax.axhline(acc, linestyle="--", color="black", linewidth=1.2,
               label=f"Overall accuracy: {acc:.1f} %")

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(f"Classification Report — {split_label.title()} set\n"
                 f"(Overall accuracy: {acc:.1f} %)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save:
        _save(fig, f"05_classification_report_{split_label}.png")
    return fig


# ---------------------------------------------------------------------------
# 06  Label distribution + return KDE
# ---------------------------------------------------------------------------

def plot_label_distribution(
    labeled_df: pd.DataFrame,
    save:       bool = True,
) -> plt.Figure:
    """Stacked bar of label counts per person + KDE of 4h returns by label."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: stacked bar per person
    usernames = list(ACCOUNTS.keys())
    names     = [ACCOUNTS[u]["name"] for u in usernames]
    label_counts = {lbl_name: [] for lbl_name in ID2LABEL.values()}
    for u in usernames:
        sub = labeled_df[labeled_df["username"] == u]
        for lbl_id, lbl_name in ID2LABEL.items():
            label_counts[lbl_name].append((sub["label"] == lbl_id).sum())

    bottom = np.zeros(len(usernames))
    for lbl_name, color in LABEL_COLORS.items():
        counts = np.array(label_counts[lbl_name])
        axes[0].bar(names, counts, bottom=bottom, label=lbl_name.capitalize(),
                    color=color, alpha=0.85)
        bottom += counts

    axes[0].set_xlabel("Account", fontsize=11)
    axes[0].set_ylabel("Number of tweets", fontsize=11)
    axes[0].set_title("Tweet Label Distribution\nby Account", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(axis="x", rotation=20)

    # Right: KDE of 4h returns colored by label
    ret_col = labeled_df["return_4h"].dropna()
    clip_lo  = ret_col.quantile(0.01)
    clip_hi  = ret_col.quantile(0.99)
    for lbl_id, lbl_name in ID2LABEL.items():
        sub = labeled_df[(labeled_df["label"] == lbl_id)]["return_4h"].dropna()
        sub = sub.clip(clip_lo, clip_hi)
        if len(sub) > 10:
            sub.plot.kde(ax=axes[1], label=lbl_name.capitalize(),
                         color=LABEL_COLORS[lbl_name], linewidth=2)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set_xlabel("4-hour BTC return", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Distribution of 4-hour BTC Returns\nby Tweet Label", fontsize=12)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))
    axes[1].legend(fontsize=10)

    fig.suptitle("Dataset Label Distribution  (2018 – 2025)", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        _save(fig, "06_label_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 07  Influence weights bar chart
# ---------------------------------------------------------------------------

def plot_influence_weights(weights: dict, save: bool = True) -> plt.Figure:
    """Horizontal bar chart of normalized influence weights for all 5 accounts."""
    sorted_items = sorted(weights.items(), key=lambda x: x[1])
    usernames    = [u for u, _ in sorted_items]
    values       = [v * 100 for _, v in sorted_items]
    names        = [ACCOUNTS.get(u, {}).get("name", u) for u in usernames]
    colors       = [ACCOUNT_COLORS.get(u, "#888888") for u in usernames]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} %", va="center", fontsize=11)

    ax.set_xlabel("Normalized Influence Weight (%)", fontsize=12)
    ax.set_xlim(0, 115)
    ax.set_title("Historical Influence Weights\n(based on avg |4h BTC return| after crypto tweets)",
                 fontsize=13)
    ax.axvline(100, linestyle="--", color="#AAAAAA", linewidth=1, alpha=0.7)
    fig.tight_layout()

    if save:
        _save(fig, "07_influence_weights.png")
    return fig


# ---------------------------------------------------------------------------
# 08  Influence over time (per person, quarterly)
# ---------------------------------------------------------------------------

def plot_influence_over_time(period_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Line chart per person of avg |4h return| over calendar quarters."""
    # Build a pivot: rows = sorted quarters, cols = persons
    period_df = period_df.copy()
    period_df["sort_key"] = period_df["year"] * 10 + period_df["quarter_num"]
    pivot = (
        period_df.pivot_table(
            index="sort_key",
            columns="username",
            values="avg_abs_return_pct",
            aggfunc="mean",
        )
        .sort_index()
    )
    # Map sort_key back to readable label
    key_to_label = {
        row["sort_key"]: row["quarter_str"]
        for _, row in period_df[["sort_key", "quarter_str"]].drop_duplicates().iterrows()
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    for username in pivot.columns:
        color = ACCOUNT_COLORS.get(username, "#888888")
        name  = ACCOUNTS.get(username, {}).get("name", username)
        series = pivot[username].dropna()
        ax.plot(range(len(series)), series.values, "o-", color=color,
                linewidth=1.8, markersize=4, label=name, alpha=0.85)

    # X-axis: every 4th quarter labelled
    xtick_positions = list(range(len(pivot)))
    xtick_labels    = [key_to_label.get(k, "") for k in pivot.index]
    step = max(1, len(xtick_labels) // 12)
    ax.set_xticks(xtick_positions[::step])
    ax.set_xticklabels(xtick_labels[::step], rotation=45, ha="right", fontsize=9)

    ax.axvline(
        xtick_positions[sum(1 for k in pivot.index if int(str(k)[:4]) < 2023) - 1],
        linestyle="--", color="steelblue", linewidth=1.5, alpha=0.7, label="Train/Test boundary"
    )

    ax.set_ylabel("Avg |4h BTC return| (%)", fontsize=12)
    ax.set_title("Who Was Most Influential — Per Quarter  (2018 – 2025)\n"
                 "Higher = tweets followed by larger price swings", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()

    if save:
        _save(fig, "08_influence_over_time.png")
    return fig


# ---------------------------------------------------------------------------
# 09  Backtest accuracy over time (rolling window)
# ---------------------------------------------------------------------------

def plot_backtest_accuracy_over_time(
    backtest_df: pd.DataFrame,
    window:      int  = 90,
    save:        bool = True,
) -> plt.Figure:
    """Rolling {window}-day accuracy vs 33.3 % random baseline."""
    df = backtest_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.sort_values("created_at").reset_index(drop=True)
    df.set_index("created_at", inplace=True)

    rolling_acc = (
        df["correct"]
        .resample("1D").mean()               # daily accuracy
        .rolling(window=window, min_periods=5)
        .mean() * 100
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_acc.index, rolling_acc.values, color="#457B9D", linewidth=1.5,
            label=f"{window}-day rolling accuracy")
    ax.axhline(33.3, linestyle="--", color="#AAAAAA", linewidth=1.5, label="Random baseline (33.3 %)")
    ax.fill_between(rolling_acc.index, 33.3, rolling_acc.values,
                    where=rolling_acc.values > 33.3, alpha=0.2, color="#2DC653",
                    label="Above baseline")
    ax.fill_between(rolling_acc.index, 33.3, rolling_acc.values,
                    where=rolling_acc.values < 33.3, alpha=0.2, color="#E63946",
                    label="Below baseline")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Backtest Accuracy Over Time  (test period: {TEST_START_DATE} → 2025)\n"
                 f"Rolling {window}-day window", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    if save:
        _save(fig, "09_backtest_accuracy_over_time.png")
    return fig


# ---------------------------------------------------------------------------
# 10  Year × Quarter accuracy heatmap
# ---------------------------------------------------------------------------

def plot_backtest_accuracy_heatmap(
    backtest_df: pd.DataFrame,
    save:        bool = True,
) -> plt.Figure:
    """Heatmap: rows = year, columns = quarter, values = accuracy %."""
    df = backtest_df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["year"]    = df["created_at"].dt.year
    df["quarter"] = df["created_at"].dt.quarter.apply(lambda q: f"Q{q}")

    pivot = (
        df.groupby(["year", "quarter"])["correct"]
        .mean()
        .unstack(fill_value=np.nan) * 100
    )
    # Ensure column order Q1..Q4
    for col in ["Q1", "Q2", "Q3", "Q4"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[["Q1", "Q2", "Q3", "Q4"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=33.3,
        vmin=0,
        vmax=70,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
    )
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Year",    fontsize=12)
    ax.set_title("Backtest Accuracy by Year and Quarter (%)\n"
                 "Random baseline = 33.3 %  (center / yellow)", fontsize=13)
    fig.tight_layout()

    if save:
        _save(fig, "10_backtest_heatmap_accuracy.png")
    return fig


# ---------------------------------------------------------------------------
# 11  Per-person accuracy
# ---------------------------------------------------------------------------

def plot_per_person_accuracy(
    backtest_df: pd.DataFrame,
    save:        bool = True,
) -> plt.Figure:
    """Bar chart: per-person accuracy + tweet count on test set."""
    stats = (
        backtest_df.groupby("username")["correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    stats["name"]     = stats["username"].apply(lambda u: ACCOUNTS.get(u, {}).get("name", u))
    stats["color"]    = stats["username"].apply(lambda u: ACCOUNT_COLORS.get(u, "#888888"))
    stats = stats.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(stats["name"], stats["mean"] * 100, color=stats["color"], alpha=0.85)
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row['mean']*100:.1f} %  (n={int(row['count'])})",
                va="center", fontsize=10)

    ax.axvline(33.3, linestyle="--", color="#AAAAAA", linewidth=1.5, label="Random baseline")
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 90)
    ax.set_title("Per-Person Prediction Accuracy\n(test set 2023–2025)", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save:
        _save(fig, "11_per_person_accuracy.png")
    return fig


# ---------------------------------------------------------------------------
# 12  Quantified return: predicted vs actual scatter
# ---------------------------------------------------------------------------

def plot_quantified_return_prediction(
    backtest_df: pd.DataFrame,
    save:        bool = True,
) -> plt.Figure:
    """Scatter: predicted expected return (x) vs actual 4h return (y)."""
    df = backtest_df.dropna(subset=["expected_return_pct", "actual_return_pct"]).copy()
    # Clip extreme outliers for visibility
    lo, hi = df["actual_return_pct"].quantile([0.01, 0.99])
    df = df[(df["actual_return_pct"] >= lo) & (df["actual_return_pct"] <= hi)]

    color_map = {1: "#2DC653", 0: "#E63946"}   # correct=green, wrong=red
    colors     = df["correct"].map(color_map)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df["expected_return_pct"], df["actual_return_pct"],
               c=colors, alpha=0.35, s=18, edgecolors="none")

    # Perfect-prediction diagonal
    lim = max(abs(df["expected_return_pct"].min()), abs(df["expected_return_pct"].max()),
              abs(df["actual_return_pct"].min()),    abs(df["actual_return_pct"].max())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
    ax.axhline(0, color="#AAAAAA", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="#AAAAAA", linewidth=0.8, alpha=0.6)

    legend_elements = [
        mpatches.Patch(color="#2DC653", label="Correct direction"),
        mpatches.Patch(color="#E63946", label="Wrong direction"),
        Line2D([0], [0], linestyle="--", color="black", label="Perfect prediction"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xlabel("Predicted expected 4h return (%)", fontsize=12)
    ax.set_ylabel("Actual 4h BTC return (%)",         fontsize=12)
    ax.set_title("Quantified Return Prediction vs Actual\n"
                 "E[r] = P(up)·μ_up + P(flat)·μ_flat + P(down)·μ_down", fontsize=13)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()

    if save:
        _save(fig, "12_quantified_return_scatter.png")
    return fig


# ---------------------------------------------------------------------------
# 13  Future price projection (fan chart)
# ---------------------------------------------------------------------------

def plot_future_projection(
    btc_df:              pd.DataFrame,
    weighted_up_prob:    float,
    expected_return_pct: float,
    conditional_returns: dict,
    save:                bool = True,
) -> plt.Figure:
    """
    Two panels:
      Left  : last 90 days of BTC actual closing price
      Right : 4-hour forward fan chart using model probabilities + historical
              return distributions
    """
    btc = btc_df.copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    btc = btc.sort_values("timestamp")

    # Last 90 days
    cutoff     = btc["timestamp"].max() - pd.Timedelta(days=90)
    recent_btc = btc[btc["timestamp"] >= cutoff]
    current_price = float(recent_btc["close"].iloc[-1])
    last_ts       = recent_btc["timestamp"].iloc[-1]

    # Build fan chart using historical return stats
    # Sample from historical distribution weighted by model probabilities
    np.random.seed(0)
    n_sim = 2000
    horizons_h = np.array([1, 2, 3, 4])    # hours out

    # μ and σ per class from conditional_returns
    class_stats = {
        "up":   {"mu": conditional_returns.get("up",   0.015), "sigma": 0.015},
        "flat": {"mu": conditional_returns.get("flat", 0.000), "sigma": 0.005},
        "down": {"mu": conditional_returns.get("down",-0.015), "sigma": 0.015},
    }
    class_probs = {
        "up":   weighted_up_prob,
        "flat": max(0, 1 - weighted_up_prob - max(0, 1 - weighted_up_prob - 0.2)),
        "down": max(0, 1 - weighted_up_prob),
    }
    # Re-normalise
    total = sum(class_probs.values())
    class_probs = {k: v / total for k, v in class_probs.items()}

    # Simulate 4-hour paths
    sim_end_prices = []
    for _ in range(n_sim):
        chosen_class = np.random.choice(
            list(class_stats.keys()),
            p=[class_probs["up"], class_probs["flat"], class_probs["down"]],
        )
        r = np.random.normal(class_stats[chosen_class]["mu"],
                             class_stats[chosen_class]["sigma"])
        sim_end_prices.append(current_price * (1 + r))

    sim_end_prices = np.array(sim_end_prices)
    p10  = np.percentile(sim_end_prices, 10)
    p25  = np.percentile(sim_end_prices, 25)
    p50  = np.percentile(sim_end_prices, 50)
    p75  = np.percentile(sim_end_prices, 75)
    p90  = np.percentile(sim_end_prices, 90)
    exp_price = current_price * (1 + expected_return_pct / 100)

    future_ts = last_ts + pd.Timedelta(hours=4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: last 90 days
    axes[0].plot(recent_btc["timestamp"], recent_btc["close"],
                 color="#1A1A2E", linewidth=1.5)
    axes[0].set_xlabel("Date", fontsize=11)
    axes[0].set_ylabel("BTC / USD", fontsize=11)
    axes[0].set_title("Last 90 Days — BTC Price", fontsize=12)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].axvline(last_ts, linestyle="--", color="gray", linewidth=1, alpha=0.6)

    # Right panel: fan chart
    xs = [last_ts, future_ts]
    axes[1].fill_between(xs, [current_price, p10], [current_price, p90],
                         alpha=0.12, color="steelblue", label="P10–P90")
    axes[1].fill_between(xs, [current_price, p25], [current_price, p75],
                         alpha=0.25, color="steelblue", label="P25–P75")
    axes[1].plot(xs, [current_price, p50],      color="steelblue",  linewidth=2,   label="Median")
    axes[1].plot(xs, [current_price, exp_price], color="#E63946",   linewidth=2,   linestyle="--",
                 label=f"Expected ({expected_return_pct:+.2f} %)")
    axes[1].plot([last_ts], [current_price], "ko", markersize=8)

    for price, label in [(p90, "P90"), (p10, "P10"), (exp_price, f"E = ${exp_price:,.0f}")]:
        axes[1].text(future_ts + pd.Timedelta(minutes=10), price,
                     label, va="center", fontsize=9)

    axes[1].set_xlim(last_ts - pd.Timedelta(hours=1), future_ts + pd.Timedelta(hours=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    axes[1].set_title(
        f"4-Hour Projection\n"
        f"P(up)={weighted_up_prob*100:.1f} %  |  E[r]={expected_return_pct:+.2f} %",
        fontsize=12,
    )
    axes[1].legend(fontsize=9, loc="upper left")

    fig.suptitle("Bitcoin Price: Recent History + Model Forecast", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        _save(fig, "13_future_price_projection.png")
    return fig


# ---------------------------------------------------------------------------
# Master caller
# ---------------------------------------------------------------------------

def plot_all(
    btc_df:              pd.DataFrame,
    tweets_df:           pd.DataFrame,
    labeled_df:          pd.DataFrame,
    backtest_df:         pd.DataFrame,
    weights:             dict,
    period_df:           pd.DataFrame,
    y_true_test:         list,
    y_pred_test:         list,
    test_report:         dict,
    weighted_up_prob:    float       = 0.5,
    expected_return_pct: float       = 0.0,
    conditional_returns: dict        = None,
    history_path:        str         = None,
) -> list[str]:
    """
    Generate all 13 plots.  Returns list of saved file paths.
    """
    conditional_returns = conditional_returns or {"up": 0.015, "flat": 0.0, "down": -0.015}
    history_path        = history_path        or os.path.join(MODEL_DIR, "training_history.json")

    print(f"\n{'='*55}")
    print("Generating all visualizations  →  plots/")
    print(f"{'='*55}")

    paths = []

    paths.append(plot_btc_price_history(btc_df))
    paths.append(plot_btc_with_tweet_events(btc_df, tweets_df))

    if os.path.exists(history_path):
        paths.append(plot_training_history(history_path))
    else:
        print("  Skipping plot 03: training_history.json not found")

    if len(y_true_test) > 0:
        paths.append(plot_confusion_matrix(y_true_test, y_pred_test, split_label="test"))
        paths.append(plot_classification_report_bars(test_report, split_label="test"))

    paths.append(plot_label_distribution(labeled_df))
    paths.append(plot_influence_weights(weights))

    if not period_df.empty:
        paths.append(plot_influence_over_time(period_df))

    if not backtest_df.empty:
        paths.append(plot_backtest_accuracy_over_time(backtest_df))
        paths.append(plot_backtest_accuracy_heatmap(backtest_df))
        paths.append(plot_per_person_accuracy(backtest_df))
        paths.append(plot_quantified_return_prediction(backtest_df))

    paths.append(plot_future_projection(
        btc_df, weighted_up_prob, expected_return_pct, conditional_returns
    ))

    print(f"\nAll done.  {len(paths)} plots saved to plots/")
    return [p for p in paths if isinstance(p, str)]
