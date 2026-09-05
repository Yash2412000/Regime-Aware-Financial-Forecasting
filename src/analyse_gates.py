"""
Usage
    python analyse_gates.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import build_dataloaders, REGIME_IDX, FEATURE_COLS
from model import MoEForecaster

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGIME_NAMES = ["bull", "bear", "high-vol"]  # order matches REGIME_IDX (bull, bear, highvol)


def collect_gates_and_regimes(model, loader):
    """Run the test set through the model, collect gate weights + dominant regime per sample."""
    model.eval()

    all_gates = []
    all_regimes = []

    with torch.no_grad():

        for batch in loader:

            if len(batch) == 3:
                x, _, _ = batch
            else:
                x, _ = batch

            x = x.to(DEVICE)

            _, g_weights, _ = model(x)

            regime_post = x[:, -1, REGIME_IDX]

            regime_post = regime_post.clamp(min=0)

            regime_post = regime_post / (
                regime_post.sum(-1, keepdim=True) + 1e-9
            )

            dominant = regime_post.argmax(dim=-1)

            all_gates.append(
                g_weights.cpu().numpy()
            )

            all_regimes.append(
                dominant.cpu().numpy()
            )

    return (
        np.concatenate(all_gates),
        np.concatenate(all_regimes)
    )


def print_gate_by_regime_table(gates: np.ndarray, regimes: np.ndarray, expert_names: list[str]):
    print("\n" + "=" * 64)
    print("GATE WEIGHT BY DOMINANT REGIME  (test set)")
    print("=" * 64)
    header = f"{'Regime':<12}" + "".join(f"{name:>13}" for name in expert_names) + f"{'n_samples':>12}"
    print(header)
    print("-" * len(header))

    for r, rname in enumerate(REGIME_NAMES):
        mask = regimes == r
        n = mask.sum()
        if n == 0:
            print(f"{rname:<12}  (no test samples fell in this regime)")
            continue
        means = gates[mask].mean(axis=0)
        row = f"{rname:<12}" + "".join(f"{m:>13.3f}" for m in means) + f"{n:>12d}"
        print(row)

    print("-" * len(header))
    overall = gates.mean(axis=0)
    print(f"{'overall':<12}" + "".join(f"{m:>13.3f}" for m in overall))

    # Quick interpretation hint
    per_regime_means = np.array([
        gates[regimes == r].mean(axis=0) if (regimes == r).sum() > 0 else np.full(gates.shape[1], np.nan)
        for r in range(len(REGIME_NAMES))
    ])
    spread = np.nanstd(per_regime_means, axis=0).mean()
    print(f"\nMean cross-regime std of gate weights: {spread:.4f}")
    if spread < 0.02:
        print("-> WARNING: gate weights barely differ across regimes. The gate may not be")
        print("   meaningfully regime-aware yet — consider increasing lambda1 (regime-consistency")
        print("   loss weight) or checking that regime_post is reaching the gate correctly.")
    else:
        print("-> Gate weights visibly shift across regimes — supports the regime-aware claim.")


def plot_gates_over_time(gates: np.ndarray, regimes: np.ndarray, expert_names: list[str], out_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [4, 1]})

    for k, name in enumerate(expert_names):
        ax1.plot(gates[:, k], label=name, linewidth=1.2)
    ax1.set_ylabel("Gate weight")
    ax1.set_title("Gate weights over test set timeline")
    ax1.legend(loc="upper right", ncol=len(expert_names))

    regime_colors = {0: "#4CAF50", 1: "#F44336", 2: "#FF9800"}  # bull, bear, high-vol
    for r, color in regime_colors.items():
        mask = regimes == r
        ax2.fill_between(np.arange(len(regimes)), 0, 1, where=mask, color=color, alpha=0.6,
                          step="pre", label=REGIME_NAMES[r])
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("Test sample index (chronological)")
    ax2.legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--out", type=str, default="gate_analysis.png")
    args = parser.parse_args()

    print("Loading data (rebuilds the same train/val/test split as training)...")
    loaders, scaler, hmm, hmm_scaler = build_dataloaders()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    train_args = ckpt["args"]
    model = MoEForecaster(
        use_experts=tuple(getattr(train_args, "experts", "lstm,transformer,gru,linear").split(",")),
        lstm_hidden=train_args.lstm_hidden, lstm_layers=train_args.lstm_layers,
        d_model=train_args.d_model, nhead=train_args.nhead, tf_layers=train_args.tf_layers,
        dropout=train_args.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, experts={model.expert_names}")

    gates, regimes = collect_gates_and_regimes(model, loaders["test"])
    print_gate_by_regime_table(gates, regimes, model.expert_names)
    plot_gates_over_time(gates, regimes, model.expert_names, args.out)


if __name__ == "__main__":
    main()