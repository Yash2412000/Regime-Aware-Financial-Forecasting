"""
train.py
Joint training loop for the MoE financial forecaster.

Loss
    L = L_forecast + lambda1 * L_regime + lambda2 * L_diversity

L_forecast (Huber loss, robust to return outliers):
    L_forecast = HuberLoss(y_hat, y)

L_regime (Fisher-ratio regime-consistency loss):
    Groups samples by dominant HMM regime. Computes:

L_regime = within_variance / (between_variance + eps)

L_diversity (floor-based anti-collapse):
    Penalises expert average usage falling below 1/(2K). Zero when all
    experts are above the floor — does NOT pull toward exact uniformity.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_dataloaders, REGIME_IDX, N_REGIMES
from model import MoEForecaster

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Loss components

def loss_forecast(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.HuberLoss(delta=0.01)(y_hat, y)


def loss_regime_consistency(g_weights: torch.Tensor, x: torch.Tensor,
                             regime_idx: list[int] = REGIME_IDX,
                             eps: float = 1e-4) -> torch.Tensor:
    regime_post = x[:, -1, regime_idx]
    dominant    = regime_post.argmax(dim=-1)   # (B,) in {0,1,2}

    within_total, group_means, group_sizes = 0.0, [], []
    for r in range(N_REGIMES):
        mask = dominant == r
        n    = mask.sum()
        if n > 1:
            within_total += g_weights[mask].var(dim=0, unbiased=False).mean()
            group_means.append(g_weights[mask].mean(dim=0))
            group_sizes.append(n.float())

    if len(group_means) < 2:
        return torch.zeros((), device=g_weights.device)

    within     = within_total / len(group_means)
    means      = torch.stack(group_means)
    sizes      = torch.stack(group_sizes)
    weights    = sizes / sizes.sum()
    grand_mean = (means * weights.unsqueeze(-1)).sum(dim=0)
    between    = (weights * ((means - grand_mean) ** 2).sum(dim=-1)).sum()

    return within / (between + eps)


def loss_diversity(g_weights: torch.Tensor, floor: float = None) -> torch.Tensor:
    mean_usage = g_weights.mean(dim=0)
    k = g_weights.size(1)
    if floor is None:
        floor = 1.0 / (2 * k)
    return torch.relu(floor - mean_usage).pow(2).sum()


# Metrics

def directional_accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    correct = ((y_hat > 0) == (y > 0)).float()
    return correct.mean().item()


def mape(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> float:
    return (((y_hat - y).abs()) / (y.abs() + eps)).mean().item() * 100


def smape(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> float:
    return (2 * (y_hat - y).abs() / (y_hat.abs() + y.abs() + eps)).mean().item() * 100


# Train / eval loops

def run_epoch(model: MoEForecaster, loader, optimizer, lambda1: float, lambda2: float,
              training: bool) -> dict:

    model.train(training)

    total_loss = total_lf = total_lr = total_ld = 0.0
    all_yhat, all_y = [], []

    with torch.set_grad_enabled(training):

        for batch in loader:

            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_hat, g_weights, _ = model(x)

            lf = loss_forecast(y_hat, y)
            lr = loss_regime_consistency(g_weights, x)
            ld = loss_diversity(g_weights)

            loss = lf + lambda1 * lr + lambda2 * ld

            if training:
                optimizer.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            total_loss += loss.item()
            total_lf += lf.item()
            total_lr += lr.item()
            total_ld += ld.item()

            all_yhat.append(
                y_hat.detach().cpu()
            )

            all_y.append(
                y.detach().cpu()
            )


    n = len(loader)

    yhat = torch.cat(all_yhat)
    yt = torch.cat(all_y)

    mae = (yhat - yt).abs().mean().item()

    rmse = (
        (yhat - yt) ** 2
    ).mean().sqrt().item()

    da = directional_accuracy(
        yhat,
        yt
    )

    mp = mape(
        yhat,
        yt
    )

    smp = smape(
        yhat,
        yt
    )


    return {
        "loss": total_loss / n,
        "lf": total_lf / n,
        "lr": total_lr / n,
        "ld": total_ld / n,
        "mae": mae,
        "rmse": rmse,
        "dir_acc": da,
        "mape": mp,
        "smape": smp
    }

# Main

def train(args):
    print(f"\nDevice: {DEVICE}")
    print("Loading data...")
    loaders, scaler, hmm, hmm_scaler = build_dataloaders()

    model = MoEForecaster(
        use_experts=tuple(args.experts.split(",")),
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
        d_model=args.d_model, nhead=args.nhead, tf_layers=args.tf_layers,
        dropout=args.dropout,
    ).to(DEVICE)

    print(f"Experts: {model.expert_names}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_mae": [], "val_mae": [], "train_dir_acc": [], "val_dir_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], optimizer, args.lambda1, args.lambda2, training=True)
        va = run_epoch(model, loaders["val"],   optimizer, args.lambda1, args.lambda2, training=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs}  [{elapsed:.1f}s]  "
              f"Train loss={tr['loss']:.4f}  Val loss={va['loss']:.4f}  "
              f"Val MAE={va['mae']:.5f}  Val DA={va['dir_acc']:.3f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_mae"].append(tr["mae"])
        history["val_mae"].append(va["mae"])
        history["train_dir_acc"].append(tr["dir_acc"])
        history["val_dir_acc"].append(va["dir_acc"])

        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "scaler": scaler, "hmm": hmm, "hmm_scaler": hmm_scaler, "args": args},
                       "checkpoints/best_model.pt")
            print(f"  Saved best model (val loss={best_val_loss:.4f})")

    _save_loss_history(history, best_epoch=history["val_loss"].index(min(history["val_loss"])) + 1)


    ckpt = torch.load(
    "checkpoints/best_model.pt",
    map_location=DEVICE,
    weights_only=False
    )

    model.load_state_dict(ckpt["model_state"])

    model.eval()

    all_predictions = []
    all_actual = []
    all_tickers = []

    with torch.no_grad():
        for x, y, ticker in loaders["test"]:
            x = x.to(DEVICE)
            y_hat, _, _ = model(x)
            all_predictions.extend(y_hat.cpu().numpy())
            all_actual.extend(y.numpy())
            all_tickers.extend(ticker)

    import numpy as np
    os.makedirs("results", exist_ok=True)

    np.save(
    "results/predictions.npy",
    np.array(all_predictions)
    )

    np.save(
    "results/actual_returns.npy",
    np.array(all_actual)
    )

    np.save(
    "results/test_tickers.npy",
    np.array(all_tickers)
    )

    print("\nSaved:")
    print(" results/predictions.npy")
    print(" results/actual_returns.npy")
    print(" results/test_tickers.npy")

    te = run_epoch(
    model,
    loaders["test"],
    optimizer,
    args.lambda1,
    args.lambda2,
    training=False
    )

    print("\n-- Test Results --------------------------------------")
    print(f"  MAE              : {te['mae']:.5f}")
    print(f"  RMSE             : {te['rmse']:.5f}")
    print(f"  Directional Acc  : {te['dir_acc']:.3f}")
    print(f"  MAPE             : {te['mape']:.2f}%  (unstable near y=0 — see smape below)")
    print(f"  SMAPE            : {te['smape']:.2f}%")
    print(f"  Forecast Loss    : {te['lf']:.4f}")
    print(f"  Regime Loss      : {te['lr']:.4f}")
    print(f"  Diversity Loss   : {te['ld']:.4f}")

    return model, loaders, scaler


def _save_loss_history(history: dict, best_epoch: int, out_csv: str = "loss_history.csv",
                        out_png: str = "loss_curves.png"):
    """Writes per-epoch train/val loss + MAE + DA to CSV, and plots loss curves."""
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(history.keys()))
        for row in zip(*history.values()):
            writer.writerow(row)
    print(f"\nLoss history saved to {out_csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(history["epoch"], history["train_loss"], label="Train loss", linewidth=1.5)
    ax1.plot(history["epoch"], history["val_loss"],   label="Val loss",   linewidth=1.5)
    ax1.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label=f"Best checkpoint (ep {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Combined loss L")
    ax1.set_title("Training vs validation loss")
    ax1.legend()

    ax2.plot(history["epoch"], history["train_mae"], label="Train MAE", linewidth=1.5)
    ax2.plot(history["epoch"], history["val_mae"],   label="Val MAE",   linewidth=1.5)
    ax2.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.set_title("Training vs validation MAE")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Loss curves saved to {out_png}")


def parse_args():
    p = argparse.ArgumentParser(description="Train MoE financial forecaster")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--lambda1",     type=float, default=0.1, help="Weight for regime-consistency loss")
    p.add_argument("--lambda2",     type=float, default=0.05, help="Weight for diversity loss")
    p.add_argument("--experts",     type=str,   default="lstm,transformer,gru,linear",
                   help="Comma-separated list of experts to use")
    p.add_argument("--lstm_hidden", type=int,   default=128)
    p.add_argument("--lstm_layers", type=int,   default=2)
    p.add_argument("--d_model",     type=int,   default=128)
    p.add_argument("--nhead",       type=int,   default=4)
    p.add_argument("--tf_layers",   type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)