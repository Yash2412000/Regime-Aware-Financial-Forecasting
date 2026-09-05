"""
baselines.py
Three standalone baselines for the dissertation comparison table, per the
Stage 2 execution plan: "ARIMA, single LSTM, ensemble average — needed for
dissertation comparison table."

1. Ensemble average (cheapest — no retraining)

2. Single LSTM
   Trains one LSTMExpert alone (same architecture as Expert 1 in the MoE)

3. ARIMA
   Classical univariate baseline, fit per ticker on the training split of
   log_return

Usage
    python baselines.py --checkpoint checkpoints/best_model.pt
    python baselines.py --skip-arima          # if statsmodels/time is tight
    python baselines.py --skip-lstm           # if a 30-epoch retrain is tight
"""

import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import (
    build_dataloaders, load_ticker, TICKERS, SEQ_LEN, PRED_HORIZON,
    FEATURE_COLS, N_REGIMES,
)
from model import LSTMExpert
from train import directional_accuracy, mape, smape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Shared metric reporting

def report(name: str, y_hat: np.ndarray, y: np.ndarray) -> dict:
    y_hat_t, y_t = torch.tensor(y_hat, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    row = {
        "model": name,
        "mae": (y_hat_t - y_t).abs().mean().item(),
        "rmse": ((y_hat_t - y_t) ** 2).mean().sqrt().item(),
        "dir_acc": directional_accuracy(y_hat_t, y_t),
        "mape": mape(y_hat_t, y_t),
        "smape": smape(y_hat_t, y_t),
    }
    print(f"  {name:14s}  MAE={row['mae']:.5f}  RMSE={row['rmse']:.5f}  "
          f"DA={row['dir_acc']:.3f}  MAPE={row['mape']:7.2f}%  SMAPE={row['smape']:6.2f}%")
    return row


# 1. Ensemble average (reuses the trained MoE checkpoint, no retraining)

def run_ensemble_average(checkpoint_path: str) -> dict:
    print("\n[1/3] Ensemble average (unweighted mean of the 4 trained experts)")
    loaders, scaler, hmm, hmm_scaler = build_dataloaders()

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    train_args = ckpt["args"]
    from model import MoEForecaster
    model = MoEForecaster(
        use_experts=tuple(getattr(train_args, "experts", "lstm,transformer,gru,linear").split(",")),
        lstm_hidden=train_args.lstm_hidden, lstm_layers=train_args.lstm_layers,
        d_model=train_args.d_model, nhead=train_args.nhead, tf_layers=train_args.tf_layers,
        dropout=train_args.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_y = [], []
    with torch.no_grad():
        for batch in loaders["train"]:
            x, y, _ = batch
            x = x.to(DEVICE)
            _, _, expert_preds = model(x)          
            avg_pred = expert_preds.mean(dim=1)     
            all_preds.append(avg_pred.cpu())
            all_y.append(y)

    y_hat = torch.cat(all_preds).numpy()
    y = torch.cat(all_y).numpy()
    return report("Ensemble-avg", y_hat, y)


#2. Single LSTM

def run_single_lstm(epochs: int = 30, lr: float = 3e-4) -> dict:
    print("\n[2/3] Single LSTM baseline (no gate, no other experts, forecast loss only)")
    loaders, scaler, hmm, hmm_scaler = build_dataloaders()

    model = LSTMExpert().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    huber = nn.HuberLoss(delta=0.01)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for batch in loaders["train"]:
            x, y, _ = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred, _ = model(x)
            loss = huber(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in loaders["val"]:
                x, y, _ = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred, _ = model(x)
                val_losses.append(huber(pred, y).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:03d}/{epochs}  [{time.time() - t0:.1f}s]  Val loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            x, y, _ = batch
            x = x.to(DEVICE)
            pred, _ = model(x)
            all_preds.append(pred.cpu())
            all_y.append(y)

    y_hat = torch.cat(all_preds).numpy()
    y = torch.cat(all_y).numpy()
    return report("Single-LSTM", y_hat, y)


#  3. ARIMA (per-ticker, walk-forward)

def run_arima(order=(2, 0, 2)) -> dict:
    print("\n[3/3] ARIMA baseline (per-ticker, walk-forward, 5-step-ahead)")
    from statsmodels.tsa.arima.model import ARIMA

    all_preds, all_y = [], []

    for ticker in TICKERS:
        print(f"  Fitting ARIMA{order} on {ticker}...")
        df, _ = load_ticker(ticker)
        returns = df["log_return"].values
        n = len(returns)
        t1 = int(n * 0.70)
        t2 = int(n * 0.85)

        train_series = returns[:t1]
        fitted = ARIMA(train_series, order=order).fit()

        preds = []
        history_model = fitted
        full_series = returns

        for t in range(t1, n - PRED_HORIZON):
            fc = history_model.get_forecast(steps=PRED_HORIZON)
            pred_5d_log_return = fc.predicted_mean.sum()  # sum of 1-step log returns ~= 5-day log return
            preds.append(pred_5d_log_return)
            history_model = history_model.append([full_series[t]], refit=False)

        targets = [np.sum(full_series[t + 1: t + 1 + PRED_HORIZON]) for t in range(t1, n - PRED_HORIZON)]

        test_start_offset = (t2 - t1)
        preds_test = preds[test_start_offset:]
        targets_test = targets[test_start_offset:]

        all_preds.extend(preds_test)
        all_y.extend(targets_test)

    y_hat = np.array(all_preds)
    y = np.array(all_y)
    return report("ARIMA", y_hat, y)


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--lstm-epochs", type=int, default=30)
    parser.add_argument("--arima-order", type=str, default="2,0,2")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-arima", action="store_true")
    parser.add_argument("--out", type=str, default="baseline_comparison.csv")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    rows = []

    if not args.skip_ensemble:
        rows.append(run_ensemble_average(args.checkpoint))
    if not args.skip_lstm:
        rows.append(run_single_lstm(epochs=args.lstm_epochs))
    if not args.skip_arima:
        order = tuple(int(x) for x in args.arima_order.split(","))
        rows.append(run_arima(order=order))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nComparison table saved to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()