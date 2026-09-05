# Regime-Aware MoE Financial Forecaster

MSc Dissertation — Yash Rai, University of Bath
Supervisor: Prof. Mingzhi Dong

## Structure

```
dataset.py      Downloads OHLCV (yfinance), engineers features, fits 3-state HMM regime
                detector, returns PyTorch DataLoaders. (Stage 1 deliverable)
model.py        4-expert MoE: LSTM, Transformer, GRU, Linear baseline + regime-aware
                gating network. (Stage 2 deliverable)
train.py        Joint training loop. Loss = forecast + regime-consistency + diversity.
smoke_test.py   Synthetic end-to-end check — no internet/data needed. RUN THIS FIRST,
                especially on Hex, before burning shared GPU time on real data.

dataset.ipynb, model.ipynb, train.ipynb, smoke_test.ipynb
                Thin notebooks for cell-by-cell testing — each imports the
                corresponding .py module rather than duplicating its code.
```

## Why no ARIMA/GARCH expert inside the MoE

Classical ARIMA/GARCH require a non-differentiable per-window fit, which
breaks end-to-end joint training and is too slow across ~18k rolling
windows (5 tickers x ~15 years) on shared compute. Per the execution plan,
ARIMA is instead a **separate standalone baseline** (planned: `baselines.py`)
for the dissertation comparison table — not one of the 4 MoE experts.

## Current status vs. proposal

- [x] Stage 1 — OHLCV pipeline, feature engineering, 3-state HMM regime labelling
- [x] Stage 2 — 4-expert MoE (LSTM, Transformer, GRU, Linear) + joint training,
      regime-consistency + diversity losses
- [ ] Multimodal extension — FinBERT sentiment + FRED macro data (deferred;
      needs a news-data source decision — see open question below)
- [ ] Stage 3 — SHAP explainability layer
- [ ] Standalone baselines (ARIMA, single-LSTM, ensemble average) for
      comparison table

## Open question: sentiment data source

The proposal's Expert 4 (FinBERT-MLP) needs daily financial news text per
ticker. Options to evaluate before implementing:
- `yf.Ticker(ticker).news` — free, built into yfinance, but shallow history
- Kaggle financial news headline datasets — better history, manual download
- A paid news API — best coverage, cost/access tradeoff

## Running

```bash
pip install -r requirements.txt
python smoke_test.py      # synthetic check, no internet needed
python train.py --epochs 30
```
