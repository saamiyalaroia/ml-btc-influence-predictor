# Crypto Whisperers: Do the Internet's Loudest Voices Move Bitcoin?

**Saamiya Laroia · Tony Chen · Milena Harned · Alexander Liu**  
Applied Machine Learning · 2025

🌐 [View Project Website](https://saamiyalaroia.github.io/ml-btc-influence-predictor/)

---

## Abstract

In 2021, Elon Musk added '#bitcoin' to his Twitter bio. Bitcoin jumped 20% in hours. We live in an era where social media has given a handful of individuals the ability to shape market expectations in real time. Cryptocurrency, with no central bank and no earnings reports, is uniquely hypersensitive to sentiment and influence. We ask whether the public statements of five such figures contain a consistent, learnable signal about the direction of Bitcoin prices. Our project explores this question by fine-tuning BERT on labeled tweet data from January 2018 through December 2025, paired with a two-layer MLP classifier, to predict whether Bitcoin's price moves up, down, or stays flat in the 4 hours following a post. While prior work has focused on aggregate social media sentiment, we shift attention to individual voices, introducing a "historical influence weight" for each figure derived from their average absolute impact on Bitcoin returns in the hours following a post. Who is speaking, we argue, matters as much as what is being said.

---

## Repository Structure

```
crypto-whisperers/
│
├── src/
│   ├── main.py              # Entry point: runs full pipeline
│   ├── train.py             # BERT fine-tuning loop
│   ├── model.py             # BERT + MLP classifier definition
│   ├── config.py            # Hyperparameters and run settings
│   ├── dataset.py           # PyTorch Dataset class
│   ├── data_loading.py      # Loads and merges tweet + BTC price data
│   ├── labeling.py          # Assigns up/flat/down labels (±1% threshold)
│   ├── influence_weights.py # Computes per-speaker historical influence weights
│   ├── btc_prices.py        # Fetches and processes hourly BTC price data
│   ├── backtest.py          # Backtesting simulation on test predictions
│   └── visualize.py         # Generates all plots and figures
│
├── notebook/
│   └── finalnotebook.ipynb  # End-to-end walkthrough notebook
│
├── data/                    # Raw and processed tweet + price data (see note below)
│
├── models/
│   ├── best_model.pt        # Saved model weights (v5)
│   ├── tokenizer/           # Saved BERT tokenizer
│   ├── run_config.json      # Hyperparameters used in final run
│   ├── training_history.json
│   ├── split_report.json    # Train/val/test split statistics
│   └── test_report.json     # Per-class metrics on test set
│
├── results/
│   ├── backtest_metrics.json
│   ├── backtest_results.csv
│   ├── data_quality_report.json
│   ├── influence_by_period.csv   # Quarterly influence weights per speaker
│   └── runs/                     # Per-run logs
│
└── assets/                  # All generated figures (loss curves, influence charts, etc.)
```

---

## Setup

### Prerequisites

- Python 3.9
- Conda (recommended) or pip

### Install Dependencies

```bash
conda create -n crypto-whisperers python=3.9
conda activate crypto-whisperers
pip install -r requirements.txt
```

**Key packages:**
- `transformers` (HuggingFace) — BERT model and tokenizer
- `torch` — training loop
- `pandas`, `numpy` — data processing
- `scikit-learn` — evaluation metrics
- `matplotlib`, `seaborn` — visualization

> A full `requirements.txt` or `env.yaml` is included in the root directory.

---

## Data

### Tweets

Tweets were collected from five public figures across two windows:
- **Training:** January 2018 – December 2022
- **Testing:** January 2023 – December 2025

Due to Twitter/X API terms of service, raw tweet data is **not included** in this repository. To reproduce data collection, you will need a Twitter/X API key and should replicate the collection queries described in the paper.

### Bitcoin Prices

Hourly BTC price data was fetched from a public API and merged with tweet timestamps. See `btc_prices.py` for the collection logic.

### Labeling

Each tweet is assigned a class label based on the % change in BTC price over the 4 hours following the tweet:
- **Up:** price rises > 1%
- **Down:** price falls > 1%  
- **Flat:** price moves < 1% in either direction

Labels are computed in `labeling.py` using `pd.merge_asof` to ensure no lookahead leakage.

### Keyword Filter

A 29-term crypto keyword filter (e.g., `"bitcoin"`, `"btc"`, `"eth"`, `"crypto"`) is applied to reduce noise from non-crypto tweets. This does **not** alter the class distribution.

---

## Reproducing Results

### 1. Prepare Data

If you have raw tweet `.csv` files, place them in `data/raw/` and run:

```bash
python data_loading.py
python labeling.py
```

This will output processed, labeled data to `data/processed/`.

### 2. Train the Model

```bash
python train.py
```

This runs the v5 configuration: BERT-base fine-tuned with:
- Learning rate: `1e-5`
- Batch size: `16`
- Max epochs: `5` (early stopping with `patience=2` on validation Macro-F1)
- Class weighting: inverse frequency
- Random seed: `42`
- Split: chronological 70/10/20 (train/val/test)

Trained weights and logs will be saved to `models/`.

### 3. Evaluate

```bash
python main.py
```

Runs the full pipeline: loads the best checkpoint from `models/best_model.pt`, evaluates on the test set, computes influence weights, and writes results to `results/`.

### 4. Backtest

```bash
python backtest.py
```

Simulates a simple trading strategy based on model predictions. Outputs `results/backtest_metrics.json` and `results/backtest_results.csv`.

### 5. Visualize

```bash
python visualize.py
```

Generates all figures (loss curves, per-speaker accuracy, quarterly influence weights, BTC price history with train/test split) and saves them to `assets/plots/`.

---

## Model Architecture

```
Tweet Text
    ↓
BERT-base (fine-tuned)
    ↓
[CLS] Embedding (768-dim)
    ↓
2-Layer MLP: 768 → 256 → 3
    ↓
Softmax Scores
    ↑
Influence Weight (0–100%) — modulates prediction confidence per speaker
```

Influence weights are normalized relative to Michael Saylor (100%), whose posts are almost entirely Bitcoin-related and serve as the most reliable signal baseline.

---

## Results Summary

| Model | Macro-F1 | Notes |
|-------|----------|-------|
| v1 | 0.289 | Majority-class collapse |
| v3 | ~0.310 | Unfiltered test set |
| v4 | 0.334 | Corrected evaluation |
| **v5** | **0.329** | **Final model** |
| v6 | 0.318 | Underperformed v5 |

**Per-speaker test accuracy (2023–2025):**

| Speaker | Accuracy | n |
|---------|----------|---|
| Vitalik Buterin | 70.8% | 24 |
| Elon Musk | 67.7% | 768 |
| Changpeng Zhao | 66.1% | 121 |
| Donald Trump | 56.8% | 241 |
| Michael Saylor | 43.9% | 157 |

*Random baseline: 33%*

---

## Ethical Considerations

This model is intended for academic research only. The predictions it generates could theoretically be used to inform algorithmic trading or, in the worst case, to help influential figures anticipate the market impact of their own posts. We do not endorse such use. See the full ethics discussion in the paper.

---

## Citation

If you build on this work, please cite:

```
Laroia, S., Chen, T., Harned, M., & Liu, A. (2025). Crypto Whisperers: 
Do the Internet's Loudest Voices Move Bitcoin?
```

---

## Large Files

Some files exceed GitHub's size limit and are not included in this repository. This includes model weights (`best_model.pt`) and any large data files. You can download them here:

📁 [Google Drive — Large Files](https://drive.google.com/file/d/1CZacwoDmkMLDd0t79si4ISXC1fnZGgnd/view?usp=drive_link)

---

## License

See `LICENSE` for terms of use.
