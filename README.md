# Crypto Whisperers: Do the Internet's Loudest Voices Move Bitcoin?

## ₿ [View the Full Paper → saamiyalaroia.github.io/ml-btc-influence-predictor](https://saamiyalaroia.github.io/ml-btc-influence-predictor/)

---

**Saamiya Laroia · Tony Chen · Milena Harned · Alexander Liu**  
Columbia University · Applied Machine Learning · 2025

---

## Overview

This project investigates whether the public statements of five high-profile individuals contain a consistent, learnable signal about the direction of Bitcoin prices. We fine-tune **BERT** on labeled tweet data from January 2018 through December 2025, paired with a two-layer MLP classifier, to predict whether Bitcoin's price moves **up**, **down**, or stays **flat** in the 4 hours following a post.

We introduce a **historical influence weight** for each speaker derived from their average absolute impact on Bitcoin returns — arguing that *who is speaking matters as much as what is being said*.

---

## Speakers

| Figure | Followers |
|---|---|
| Vitalik Buterin | 6.2M |
| Changpeng Zhao | 11.2M |
| Elon Musk | 239.9M |
| Donald Trump | 111.5M |
| Michael Saylor | 5M |

---

## Results

| Metric | Score |
|---|---|
| Macro-F1 (v5) | 0.329 |
| Macro-F1 (v1 baseline) | 0.289 |
| Improvement | +14% |
| Test set size | 1,311 tweets |
| Prediction window | 4 hours |

---

## Repository Structure

```
assets/          ← figures and visualisations
data/            ← tweet and price data
docs/            ← GitHub Pages site (index.html)
src/             ← model training and evaluation code
script/          ← data collection scripts
```

---

## Keywords

`Bitcoin` `BERT` `NLP` `Sentiment Analysis` `Cryptocurrency` `Social Media Influence` `Price Prediction` `Historical Influence Weight`
