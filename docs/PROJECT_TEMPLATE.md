# Class Project Template

Fill in this document at the start of the project and keep it up to date.

# Class Project Template

Fill in this document at the start of the project and keep it up to date.
 
1) Project Overview
- **Title**: Crypto Whisperers: Do the Internet's Loudest Voices Move Bitcoin?
- **Team**: Saamiya Laroia (sdl2154) · Tony Chen (tc3238) · Milena Harned (mdh2192) · Alexander Liu (al4402)
- **Problem Statement**: Cryptocurrency markets lack the fundamental anchors of traditional assets, making Bitcoin prices uniquely sensitive to public sentiment. While prior research has used aggregate social media sentiment to predict price movements, it has relied on keyword-based approaches that miss contextual meaning and treats all voices as equivalent. This project investigates whether the tweets of five high-profile, market-moving individuals contain a consistent, learnable signal about Bitcoin's short-term price direction.
- **Hypothesis**: We hypothesize that fine-tuning BERT on tweets from specific influential figures, combined with a speaker-level historical influence weight, will produce better directional price predictions than a baseline model, because context-aware language encoding and speaker identity together capture signal that lexicon-based or aggregate approaches cannot.
2) Related Work (Short)
- Early crypto-Twitter sentiment models (Rhee et al., 2018; Prajapati, 2019; Zhu et al., 2019) predicted price direction with modest accuracy using gradient boosting, linear regression, and real-time platforms, but relied on keyword or lexicon-based sentiment scoring that cannot capture context, irony, or speaker identity.
- Li and Ma (2024) applied a lexicon-based approach (Hu and Liu, 2004) to five major cryptocurrencies from 2021 to 2022, finding that tweet volume correlated positively with price, while higher engagement correlated negatively, suggesting negative posts attract outsized attention.
- Counterintuitively, positive Bitcoin sentiment was associated with price decreases in Li and Ma's analysis, highlighting how the relationship between sentiment and price is far from straightforward.
- Khatri et al. (2024) identify persistent field-wide challenges including data quality, limited training data, and the complexity of the sentiment-price relationship, suggesting standard approaches leave meaningful signal on the table.
- Our approach addresses the lexicon limitation by using BERT for context-aware sentiment encoding and by focusing on posts from specific high-profile individuals rather than treating all Twitter activity as equivalent.

## 3) Data

- **Datasets**:
  - **Tweets — multi-source** (53,880 rows after cleaning, window 2023-03-24 → 2025-03-24):
    - Trump (Kaggle archive, X + Truth Social): 90,343 raw → 20,920 in-window
    - Musk (Kaggle archive): 55,099 raw → 30,820 in-window
    - CZ / Vitalik / Saylor (Apify scrapes, .xlsx with Chinese column headers from scraper locale): 703 / 275 / 1,645 raw rows respectively
  - **BTC hourly OHLCV** from Coinbase public REST (`api.exchange.coinbase.com/products/BTC-USD/candles`), 27,064 hourly candles. Binance was blocked from US IPs.
- **How to access**:
  - Kaggle CSVs available at the listed Kaggle pages (free with account).
  - Apify scrape was done via `apidojo/tweet-scraper-lite`; the resulting xlsx files are in `v5/data/mendeley/`.
  - BTC price fetcher: `v5/btc_prices.py` — public, no auth, just rerun.
  - All raw inputs are committed under `v5/data/mendeley/` (after the symlinks were replaced with real files).
- **License / ethics**:
  - Tweets are public posts retrieved from publicly viewable accounts; we did not access private accounts or DMs.
  - Apify scraping operates in a gray area against X ToS; we discuss this in the Ethics section of the blog post and in the data-quality report.
  - BTC price data is unauthenticated public market data with no licensing restriction.
- **Train/val/test split**: Strictly chronological 70 / 10 / 20 over the crypto-filtered subset (6,555 tweets):
  - Train: 4,588 tweets, 2023-03-24 → 2024-10-01
  - Val:     656 tweets, 2024-10-02 → 2024-12-02
  - Test:  1,311 tweets, 2024-12-02 → 2025-03-23
  Random splits are explicitly avoided to prevent temporal leakage between adjacent BTC return labels. Split boundaries persisted to `v5/models/split_report.json`.

## 4) Baseline

- **Baseline model (v1)**: BERT-base-uncased + 2-layer MLP (768 → 256 → 3), unweighted cross-entropy loss, no crypto-keyword filter, ±1 % label threshold, lr = 2e-5, batch 16, 3 epochs.
- **Baseline metrics** (on a date-based test set of 16,439 tweets):
  - Test accuracy: **76.5 %**
  - Macro-F1: **0.289**
  - Per-class F1: down = **0.000**, flat = **0.867**, up = **0.000**
  - Diagnosis: degenerate majority-class predictor — the canonical failure mode of unweighted CE on imbalanced data.
- **Why this is a fair baseline**: It is the exact same BERT + MLP architecture as our final model, trained on the same data window, with only the *interventions* (crypto filter, class weights, train/eval alignment, lookahead-safe labeling, chronological split, fixed seed) removed. Any improvement is therefore attributable to those interventions rather than to architecture, scale, or pretraining.

## 5) Proposed Method

- **What we change** (cumulatively, v1 → v5):
  1. **Crypto-keyword filter** on training data (29 keywords). Reduces 53,880 tweets to 6,555 by removing posts that have no detectable BTC content.
  2. **Inverse-frequency class weighting** in the cross-entropy loss, normalized so the average weight is 1.
  3. **Train/eval alignment**: the same crypto filter is applied at backtest time, enforced by a `FILTER_MODE` contract in `run_config.json` that backtest reads back and asserts.
  4. **Lookahead-safe labeling** via `pd.merge_asof(direction='backward')`, replacing the earlier `np.searchsorted` approach that could pick a candle whose close is one hour after the tweet.
  5. **Chronological 70/10/20 split** with explicit boundaries persisted to `split_report.json`.
  6. **Reproducibility infrastructure**: fixed seed 42 across Python/NumPy/Torch/MPS, `data_quality_report.json` audit trail, per-run `results/runs/<id>/` directory.
  7. **Model selection by val macro-F1** (not val_loss), with early stopping (patience = 2).
  8. **Author conditioning (v6, ablation)**: prepend `@username : ` to BERT input. **Tested and rejected** — see Experiments.
- **Why it should help**:
  - The crypto filter removes ~90 % of training tweets that are noise (Tesla / politics / memes), raising the effective signal-to-noise ratio.
  - Class weights prevent the model from collapsing to "always predict flat" — without them v1's down/up F1 are exactly 0.000.
  - Train/eval alignment removes a confounding factor where the model is asked at test time to predict tweets from a distribution it was never trained on.
  - Lookahead-safe labeling removes a small but principled source of label leakage that would inflate measured performance.
- **Ablations**:
  - v2: widen threshold ±1 % → ±2 % (test the choice of label boundary)
  - v3: keep crypto filter + class weights but evaluate on the unfiltered test set (test the importance of distribution alignment)
  - v4: add the alignment fix (test that the alignment really matters)
  - v6: add author prefix to BERT input (test whether explicit author conditioning helps)
  - v5.1: 5-cell hyperparameter sweep over lr ∈ {5e-6, 1e-5, 2e-5} × batch ∈ {8, 16, 32}

## 6) Experiments

- **Metrics**:
  - **Primary**: macro-F1 on the chronological test set.
  - **Secondary**: per-class precision / recall / F1, accuracy, lift over majority baseline, per-account accuracy, per-quarter accuracy, high-confidence subset accuracy (≥0.5, ≥0.7).
  - Why macro-F1 over accuracy: the class distribution is ~9 / 78 / 13 % (down / flat / up); a constant predictor scores 78 % accuracy and is reported as "good" by accuracy alone, but its macro-F1 is exactly F1_flat / 3 (≈ 0.289), which immediately surfaces the failure.
- **Compute budget**:
  - Single Apple M3 Pro laptop (MPS backend, no CUDA).
  - Per-run wall-clock: ~10–25 minutes (5 epochs, batch 16, ~4,500 train rows).
  - Total budget across all v1–v6 + v5.1 sweep: ~3 hours of training compute.
  - No cloud cost; everything runs locally.
- **Experiment plan** (each row is one fully reproducible run):

  | Run | Purpose | Expected outcome |
  |---|---|---|
  | v1 | Vanilla baseline — show the unweighted CE failure mode | High accuracy, zero minority-class F1 |
  | v2 | Threshold widening — test if ±2 % helps | Likely worse in this calm BTC window |
  | v3 | Crypto filter + class weights | First non-zero down/up F1 |
  | v4 | Add evaluation-side alignment | Improved lift, macro-F1 plateau ~0.33 |
  | v5 | Engineering refactor (seed, merge_asof, chrono split, contracts) | Reproduce v4's macro-F1 with full audit trail |
  | v6 | Author prefix in input | Test author-conditioning hypothesis |
  | v5.1 ×5 | LR/batch sensitivity sweep around v5 | Verify v5 is non-dominated |

- **Headline result**: v5 macro-F1 = **0.329** (vs v1 = 0.289, +14 % relative). v6 underperforms at 0.318. v5.1 sweep shows v5 is the only non-dominated point (any cell that beats v5 on macro-F1 loses on accuracy, and vice versa).

## 7) Reproducibility

- **How to run training**:
  ```bash
  cd v5
  python3.9 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  python main.py train      # ~12 minutes on M3 Pro MPS
  ```
- **How to run evaluation**:
  ```bash
  python main.py backtest    # walk-forward eval on the chronological test split
  python main.py visualize   # generate the 13 default plots
  python make_per_class_figure.py     # Figure 6 in the blog
  python make_per_speaker_figure.py   # Figure 14 in the blog
  ```
- **End-to-end (load → train → backtest → visualize)**:
  ```bash
  python main.py all
  ```
- **Hyperparameter ablation (v5.1)**:
  ```bash
  cd ../v5.1
  python run_ablation.py     # 5 cells, ~50 minutes total
  python make_ablation_figure.py
  ```
- **Where results are logged**:
  - `v5/data/labeled_tweets.csv` — labeled inputs after cleaning + filtering
  - `v5/models/best_model.pt` — selected checkpoint (1.3 GB)
  - `v5/models/training_history.json` — per-epoch loss / val_acc / val_macro_f1
  - `v5/models/run_config.json` — full config snapshot of the chosen run, used by backtest as a contract
  - `v5/results/data_quality_report.json` — per-source ingestion audit trail
  - `v5/results/backtest_metrics.json` — headline metrics + per-class + per-account + quarterly + high-confidence subsets
  - `v5/results/backtest_results.csv` — per-row test predictions with confidence and expected return
  - `v5/results/runs/<run_id>/` — full snapshot of every executed training run
  - `v5/plots/01–15_*.png` — all blog post figures
  - `v5.1/ablation_results/results.json` — hyperparameter sensitivity sweep results
  - `v5.1/ablation_results/ablation_comparison.png` — Figure 7 in the blog
  - `experiment_log.txt` — running per-version log of v1 through v6 + v5.1 sweep
  - `FINAL_REPORT.txt` — 3,034-word technical blog post in plain text
  - `v5/finalnotebook.ipynb` — read-only walkthrough that loads all artifacts and reproduces every figure in the blog post (outputs are pre-embedded; re-running it does not require GPU or retraining)
