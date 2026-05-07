# Class Project Template
Fill in this document at the start of the project and keep it up to date.

## 1) Project Overview

- **Title**: Crypto Whisperers: Do the Internet's Loudest Voices Move Bitcoin?
- **Team**: Saamiya Laroia (sdl2154) · Tony Chen (tc3238) · Milena Harned (mdh2192) · Alexander Liu (al4402)
- **Problem Statement**: Cryptocurrency markets lack the fundamental anchors of traditional assets, making Bitcoin prices uniquely sensitive to public sentiment. While prior research has used aggregate social media sentiment to predict price movements, it has relied on keyword-based approaches that miss contextual meaning and treats all voices as equivalent. This project investigates whether the tweets of five high-profile, market-moving individuals contain a consistent, learnable signal about Bitcoin's short-term price direction.
- **Hypothesis**: We hypothesize that fine-tuning BERT on tweets from specific influential figures, combined with a speaker-level historical influence weight, will produce better directional price predictions than a baseline model, because context-aware language encoding and speaker identity together capture signal that lexicon-based or aggregate approaches cannot.

## 2) Related Work (Short)

- Early crypto-Twitter sentiment models (Rhee et al., 2018; Prajapati, 2019; Zhu et al., 2019) predicted price direction with modest accuracy using gradient boosting, linear regression, and real-time platforms, but relied on keyword or lexicon-based sentiment scoring that cannot capture context, irony, or speaker identity.
- Li and Ma (2024) applied a lexicon-based approach (Hu and Liu, 2004) to five major cryptocurrencies from 2021 to 2022, finding that tweet volume correlated positively with price, while higher engagement correlated negatively, suggesting negative posts attract outsized attention.
- Counterintuitively, positive Bitcoin sentiment was associated with price decreases in Li and Ma's analysis, highlighting how the relationship between sentiment and price is far from straightforward.
- Khatri et al. (2024) identify persistent field-wide challenges including data quality, limited training data, and the complexity of the sentiment-price relationship, suggesting standard approaches leave meaningful signal on the table.
- Our approach addresses the lexicon limitation by using BERT for context-aware sentiment encoding and by focusing on posts from specific high-profile individuals rather than treating all Twitter activity as equivalent.

## 3) Data

- **Dataset(s)**: Tweets from five public figures — Vitalik Buterin, Changpeng Zhao, Elon Musk, Donald Trump, and Michael Saylor — collected via the Twitter/X API, paired with hourly Bitcoin price data from a public crypto exchange API. Training window: January 2018 – December 2022. Test window: January 2023 – December 2025. After applying a 29-term crypto keyword filter, the labeled dataset contains tweets assigned one of three classes: up (BTC +>1% in 4h), down (BTC ->1% in 4h), or flat (<1% move). The class distribution is heavily skewed toward flat (~71%).

- **How to access**: Raw tweet data cannot be redistributed due to Twitter/X API terms of service. Processed files (labeled_tweets.csv, btc_prices_1h.csv, test_split.csv) are available in the data/ directory of this repository. Large files that exceed GitHub's size limit are available via Google Drive: https://drive.google.com/file/d/1CZacwoDmkMLDd0t79si4ISXC1fnZGgnd/view?usp=drive_link

- **License/ethics**: Tweet data is subject to Twitter/X's Developer Agreement. We do not redistribute raw tweet text. All figures studied are public figures whose statements were made on a public platform. The model is intended for academic research only and should not be used for automated trading or market manipulation.

- **Train/val/test split**: Chronological 70/10/20 split. Training: January 2018 – December 2022. Validation: held out from the tail of the training window. Test: January 2023 – December 2025. Splits were constructed using pd.merge_asof with a strict time-based cutoff (random seed 42) to prevent lookahead leakage.

## 4) Baseline

- **Baseline model (v1)**: BERT-base-uncased with a 2-layer MLP head (768 → 256 → 3), no class weighting, no keyword filter, ±1% label threshold, lr = 2e-5, batch size 16, 3 epochs.
- **Baseline metrics**:
  - Accuracy: 76.5%, Macro-F1: 0.289
  - Per-class F1: down = 0.000, flat = 0.867, up = 0.000
  - The model just predicts "flat" for everything — the expected failure when classes are heavily imbalanced and nothing corrects for it.
- **Why this is a fair baseline**: Same architecture, same data, no modifications. Any improvement in the final model comes directly from the changes we make, not from a different setup.

## 5) Proposed Method

- **What we change** (v1 → v5, one addition per version):
  1. **Crypto keyword filter** — 29 keywords that reduce 53,880 tweets to 6,555 by removing posts with no Bitcoin-relevant content (Tesla earnings, political statements, etc.).
  2. **Class weighting** — the loss function is reweighted so the model is actually penalized for ignoring the up/down classes, not just rewarded for always predicting flat.
  3. **Filter alignment** — the same keyword filter is applied at evaluation time. Without this, the model is tested on a different distribution than it was trained on.
  4. **Lookahead-safe labeling** — fixed a bug where labels could accidentally use BTC price data from after the tweet was posted.
  5. **Chronological split** — train/val/test divided by date (70/10/20) rather than random shuffle, which would leak future prices into training.
  6. **Model selection by macro-F1** — best checkpoint chosen based on validation macro-F1, since we care about minority class performance, not just overall loss.
- **Why it should help**: The keyword filter removes most of the noise. The class weighting stops the model from ignoring the classes we actually care about. The rest ensures the evaluation reflects real-world conditions.
- **Ablations**:
  - v2: wider label threshold (±2% instead of ±1%)
  - v3: filter + class weights, but evaluated on the unfiltered test set
  - v4: adds the filter alignment fix
  - v6: prepends the author's username to the BERT input — tested and underperformed v5
  - v5.1: sweep over learning rate and batch size to check v5 settings are reasonable

## 6) Experiments

- **Metrics**: Macro-F1 is the main metric because accuracy is misleading here — 78% of tweets are labeled flat, so predicting flat every time gives 78% accuracy but is completely useless. Macro-F1 weights all three classes equally. We also report per-class F1 and per-speaker accuracy on the test set.
- **Compute budget**: Everything ran on a personal laptop (Apple M3 Pro). Each training run takes 10–25 minutes. All experiments combined took around 3 hours.
- **Experiment plan**:

  | Run | What we're testing |
  |---|---|
  | v1 | Baseline — what happens with no changes |
  | v2 | Whether a wider label threshold helps |
  | v3 | Effect of class weighting + keyword filter |
  | v4 | Effect of applying the filter at eval time too |
  | v5 | Full clean version with all fixes (final model) |
  | v6 | Whether telling BERT the speaker's name helps |
  | v5.1 (×5) | Whether our learning rate and batch size hold up |

- **Final result**: v5 macro-F1 = 0.329 vs v1 = 0.289, a 14% improvement. v6 scores 0.318 and is outperformed by v5.

## 7) Reproducibility

- **How to run training**:
  ```bash
  python3.9 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  python main.py train
  ```
- **How to run evaluation**:
  ```bash
  python main.py backtest     # evaluates on the test split
  python main.py visualize    # generates all plots
  ```
- **Run everything at once**:
  ```bash
  python main.py all
  ```
- **Where results are saved**:
  - `models/best_model.pt` — saved model weights (too large for GitHub, available via Google Drive)
  - `models/training_history.json` — loss and accuracy by epoch
  - `models/run_config.json` — exact settings used for the final run
  - `results/backtest_metrics.json` — all test metrics
  - `results/backtest_results.csv` — per-tweet predictions on the test set
  - `results/data_quality_report.json` — data ingestion summary
  - `results/runs/` — logs from every training run
  - `assets/` — all plots
