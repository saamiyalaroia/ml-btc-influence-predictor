# Crypto Whisperers: Do the Internet's Loudest Voices Move Bitcoin?

## [View the Full Paper → saamiyalaroia.github.io/ml-btc-influence-predictor](https://saamiyalaroia.github.io/ml-btc-influence-predictor/)

---

**Saamiya Laroia · Tony Chen · Milena Harned · Alexander Liu**  
Columbia University · Applied Machine Learning · 2025

---

## Abstract

In 2021, Elon Musk added ‘#bitcoin’ to his Twitter bio. Bitcoin jumped 20% in hours. We live in an era where social media has given a handful of individuals the ability to shape market expectations in real time. Cryptocurrency, with no central bank and no earnings reports, is uniquely hypersensitive to sentiment and influence. We ask whether the public statements of five such figures contain a consistent, learnable signal about the direction of Bitcoin prices. Our project explores this question by fine-tuning BERT on labeled tweet data from January 2018 through December 2025, paired with a two-layer MLP classifier, to predict whether Bitcoin’s price moves up, down, or stays flat in the 4 hours following a post. While prior work has focused on aggregate social media sentiment, we shift attention to individual voices, introducing a “historical influence weight” for each figure derived from their average absolute impact on Bitcoin returns in the hours following a post. Who is speaking, we argue, matters as much as what is being said.


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
