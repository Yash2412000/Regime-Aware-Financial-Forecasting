"""
explain.py
Stage 3 SHAP explainability layer for the MoE regime-aware forecaster.

Produces two levels of explanation per the execution plan:

  1. Expert-level SHAP (GradientExplainer on each expert's prediction head)

  2. Gate-level SHAP (GradientExplainer on the gating network)

Usage
-----
    python explain.py --checkpoint checkpoints/best_model.pt
    python explain.py --checkpoint checkpoints/best_model.pt --n_bg 200 --n_explain 500
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False

from dataset import build_dataloaders, FEATURE_COLS, REGIME_IDX, N_REGIMES
from model import MoEForecaster

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGIME_NAMES = ["bull", "bear", "high-vol"]

# Human-readable feature names for plots
FEATURE_LABELS = [
    "Log return",
    "Volatility 10d",
    "Volatility 30d",
    "RSI-14",
    "MACD",
    "MACD signal",
    "Volume ratio",
    "BB width",
    "Regime: bull",
    "Regime: bear",
    "Regime: high-vol",
]


#Model wrappers

class ExpertWrapper(torch.nn.Module):
    """Wraps a single expert: input (B,T,F) → prediction (B,1)."""
    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred, _ = self.expert(x)
        return pred.unsqueeze(-1)


class GateWrapper(torch.nn.Module):
    """Wraps the full MoE: input (B,T,F) → gate weights (B,K)."""
    def __init__(self, moe: MoEForecaster):
        super().__init__()
        self.moe = moe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, g, _ = self.moe(x)
        return g


#Data collection


def collect_test_data(loader, device: torch.device, n_max: int = 2000):

    """
    Pull test-set sequences and dominant HMM regime labels.

    Returns
    -------
    X        : (N,T,F)
    regimes  : (N,)
    """

    all_x = []
    all_r = []

    count = 0


    for batch in loader:

        if len(batch) == 3:
            x, _, _ = batch
        else:
            x, _ = batch


        x = x.to(device)


        regime_post = x[:, -1, REGIME_IDX].clamp(min=0)

        regime_post = (
            regime_post /
            (regime_post.sum(-1, keepdim=True) + 1e-9)
        )


        dominant = (
            regime_post.argmax(dim=-1)
            .cpu()
            .numpy()
        )


        all_x.append(
            x.cpu()
        )

        all_r.append(
            dominant
        )


        count += x.size(0)


        if count >= n_max:
            break



    X = torch.cat(all_x)[:n_max]

    regimes = np.concatenate(all_r)[:n_max]


    return X, regimes


# SHAP computation
def compute_expert_shap(
    wrapper: ExpertWrapper,
    background: torch.Tensor,
    test_data: torch.Tensor,
) -> np.ndarray:

    """
    GradientExplainer SHAP for one expert.
    """

    wrapper.train()

    for param in wrapper.parameters():
        param.requires_grad = True


    background = background.clone().detach().requires_grad_(True)

    test_data = test_data.clone().detach().requires_grad_(True)


    explainer = shap.GradientExplainer(
        wrapper,
        background
    )


    vals = explainer.shap_values(
        test_data
    )


    arr = np.array(vals)


    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)


    return arr


def compute_gate_shap(
    gate_wrapper: GateWrapper,
    background:   torch.Tensor,
    test_data:    torch.Tensor,
    n_experts:    int,
) -> np.ndarray:
    """
    GradientExplainer SHAP for the gate outputs.

    Returns
    -------
    shap_vals : (K, N, T, F) — SHAP per expert output per sample per timestep per feature
    """
    explainer = shap.GradientExplainer(gate_wrapper, background)
    vals      = explainer.shap_values(test_data)   # (N, T, F, K)
    arr       = np.array(vals)                     # (N, T, F, K)
    # Rearrange to (K, N, T, F)
    return arr.transpose(3, 0, 1, 2)


# Plotting

def plot_expert_beeswarm_overall(
    shap_vals: np.ndarray,    # (N, T, F)
    test_data: np.ndarray,    # (N, T, F) raw feature values
    expert_name: str,
    out_dir: str,
):
    """Beeswarm of mean-over-time SHAP values for one expert (all regimes)."""
    # Average over time dimension → (N, F)
    sv_mean  = shap_vals.mean(axis=1)
    feat_mean = test_data.mean(axis=1)

    shap_exp = shap.Explanation(
        values     = sv_mean,
        base_values= np.zeros(len(sv_mean)),
        data       = feat_mean,
        feature_names = FEATURE_LABELS,
    )

    plt.figure(figsize=(9, 5))
    shap.plots.beeswarm(shap_exp, max_display=11, show=False)
    plt.title(f"SHAP — {expert_name.upper()} expert (all regimes, mean over 30-day window)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"shap_expert_{expert_name}_overall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_expert_by_regime(
    shap_vals: np.ndarray,   # (N, T, F)
    test_data: np.ndarray,   # (N, T, F)
    regimes:   np.ndarray,   # (N,)
    expert_name: str,
    out_dir: str,
):
    """Bar plot of mean |SHAP| per feature per regime for one expert."""
    n_features = shap_vals.shape[-1]
    sv_mean    = np.abs(shap_vals).mean(axis=1)   # (N, F) — mean |SHAP| over time

    regime_means = np.zeros((N_REGIMES, n_features))
    for r in range(N_REGIMES):
        mask = regimes == r
        if mask.sum() > 0:
            regime_means[r] = sv_mean[mask].mean(axis=0)

    fig, axes = plt.subplots(1, N_REGIMES, figsize=(14, 5), sharey=True)
    colors = ["#4CAF50", "#F44336", "#FF9800"]   # bull, bear, high-vol
    for r, (ax, rname, color) in enumerate(zip(axes, REGIME_NAMES, colors)):
        vals    = regime_means[r]
        order   = np.argsort(vals)
        ax.barh([FEATURE_LABELS[i] for i in order], vals[order], color=color, alpha=0.8)
        ax.set_title(f"{rname.capitalize()} regime")
        ax.set_xlabel("Mean |SHAP|")
    fig.suptitle(f"SHAP feature importance — {expert_name.upper()} expert by regime",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f"shap_expert_{expert_name}_by_regime.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_gate_shap(
    gate_shap:    np.ndarray,   # (K, N, T, F)
    expert_names: list[str],
    out_dir: str,
):
    """Heatmap: mean |gate SHAP| per feature per expert (rows=features, cols=experts)."""
    K, N, T, F = gate_shap.shape
    # Mean |SHAP| over samples and timesteps → (K, F)
    importance = np.abs(gate_shap).mean(axis=(1, 2))   # (K, F)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(importance.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(K))
    ax.set_xticklabels([n.upper() for n in expert_names], fontsize=11)
    ax.set_yticks(range(F))
    ax.set_yticklabels(FEATURE_LABELS, fontsize=9)
    ax.set_xlabel("Expert (gate output)", fontsize=11)
    ax.set_title("Gate SHAP — mean |SHAP| per feature per expert\n"
                 "(which features push the gate toward each expert?)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_gate_by_expert.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_gate_by_regime(
    gate_shap:    np.ndarray,   # (K, N, T, F)
    regimes:      np.ndarray,   # (N,)
    expert_names: list[str],
    out_dir: str,
):
    """Per-regime gate SHAP: which features drive routing to each expert?"""
    K, N, T, F = gate_shap.shape
    fig, axes = plt.subplots(1, N_REGIMES, figsize=(15, 6), sharey=True)
    colors = ["#4CAF50", "#F44336", "#FF9800"]

    for r, (ax, rname, color) in enumerate(zip(axes, REGIME_NAMES, colors)):
        mask = regimes == r
        if mask.sum() == 0:
            continue
        regime_gate = np.abs(gate_shap[:, mask]).mean(axis=(1, 2))  # (K, F)
        # Plot top expert's feature importance for this regime
        dominant_expert = regime_gate.mean(axis=-1).argmax()
        vals  = regime_gate[dominant_expert]
        order = np.argsort(vals)
        ax.barh([FEATURE_LABELS[i] for i in order], vals[order], color=color, alpha=0.8)
        ax.set_title(f"{rname.capitalize()}\n(dominant: {expert_names[dominant_expert].upper()})")
        ax.set_xlabel("Mean |SHAP| on gate")

    fig.suptitle("Gate SHAP by regime — features driving routing decision", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_gate_by_regime.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# Summary CSV

def save_summary_csv(
    expert_shap_dict: dict,   # {name: (N, T, F)}
    regimes: np.ndarray,
    out_dir: str,
):
    rows = []
    for expert_name, sv in expert_shap_dict.items():
        sv_mean = np.abs(sv).mean(axis=1)   # (N, F)
        for r, rname in enumerate(REGIME_NAMES):
            mask = regimes == r
            regime_sv = sv_mean[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(sv.shape[-1])
            for f, fname in enumerate(FEATURE_LABELS):
                rows.append({
                    "expert": expert_name,
                    "regime": rname,
                    "feature": fname,
                    "mean_abs_shap": regime_sv[f],
                })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "shap_summary.csv")
    df.to_csv(path, index=False)
    print(f"\nSHAP summary table saved to {path}")

    # Print top-3 features per expert per regime
    print("\n── Top 3 features per expert per regime ──")
    for expert_name in expert_shap_dict:
        for rname in REGIME_NAMES:
            top = (df[(df.expert == expert_name) & (df.regime == rname)]
                   .sort_values("mean_abs_shap", ascending=False)
                   .head(3))
            feats = ", ".join(f"{r.feature} ({r.mean_abs_shap:.4f})" for _, r in top.iterrows())
            print(f"  {expert_name:12s} | {rname:8s} | {feats}")


# Main

def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for MoE forecaster")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--n_bg",        type=int, default=100,
                        help="Background samples for GradientExplainer")
    parser.add_argument("--n_explain",   type=int, default=300,
                        help="Test samples to explain (more = slower but more stable)")
    parser.add_argument("--out_dir",     type=str, default="shap_outputs")
    parser.add_argument("--skip_gate",   action="store_true",
                        help="Skip gate SHAP (faster for quick expert-only run)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Background samples: {args.n_bg} | Explain samples: {args.n_explain}")

    #Load data
    print("\nLoading data...")
    loaders, feat_scaler, hmm_model, hmm_scaler = build_dataloaders()

    print(f"\nCollecting {args.n_explain} test samples...")
    X_test, regimes = collect_test_data(loaders["test"], DEVICE, n_max=args.n_explain)
    print(f"  Regime distribution — "
          f"bull: {(regimes==0).sum()}  bear: {(regimes==1).sum()}  "
          f"high-vol: {(regimes==2).sum()}")

    # Load model
    print("\nLoading checkpoint...")
    ckpt       = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    train_args = ckpt["args"]
    model = MoEForecaster(
        use_experts  = tuple(getattr(train_args, "experts", "lstm,transformer,gru,linear").split(",")),
        lstm_hidden  = train_args.lstm_hidden,
        lstm_layers  = train_args.lstm_layers,
        d_model      = train_args.d_model,
        nhead        = train_args.nhead,
        tf_layers    = train_args.tf_layers,
        dropout      = train_args.dropout,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, experts={model.expert_names}")

    # Background samples from training set
    print(f"\nBuilding background set ({args.n_bg} train samples)...")
    bg_list = []
    for batch in loaders["train"]:
        if len(batch) == 3:
            x, _, _ = batch
        else:
             x, _ = batch
        bg_list.append(x)
        if sum(len(b) for b in bg_list) >= args.n_bg:
            break
    background = torch.cat(bg_list)[:args.n_bg].to(DEVICE)
    X_explain  = X_test[:args.n_explain].to(DEVICE)
    X_np       = X_test[:args.n_explain].numpy()

    # Expert SHAP
    expert_shap_dict = {}

    for expert_name, expert in zip(model.expert_names, model.experts):
        print(f"\n── Expert SHAP: {expert_name.upper()} ──")
        wrapper   = ExpertWrapper(expert).to(DEVICE)
        wrapper.train()
        for param in wrapper.parameters():
            param.requires_grad = True

        print(f"  Computing GradientExplainer SHAP...")
        sv = compute_expert_shap(wrapper, background, X_explain)  # (N, T, F)
        expert_shap_dict[expert_name] = sv

        print(f"  Plotting beeswarm (overall)...")
        plot_expert_beeswarm_overall(sv, X_np, expert_name, args.out_dir)

        print(f"  Plotting by regime...")
        plot_expert_by_regime(sv, X_np, regimes, expert_name, args.out_dir)

    # Gate SHAP
    if not args.skip_gate:
        print(f"\n── Gate SHAP ──")
        gate_wrapper = GateWrapper(model).to(DEVICE)
        gate_wrapper.train()

        for param in gate_wrapper.parameters():
            param.requires_grad = True

        print(f"  Computing GradientExplainer SHAP on gate outputs...")
        gate_sv = compute_gate_shap(
            gate_wrapper, background, X_explain, model.n_experts
        )   # (K, N, T, F)

        print(f"  Plotting gate SHAP heatmap...")
        plot_gate_shap(gate_sv, model.expert_names, args.out_dir)

        print(f"  Plotting gate SHAP by regime...")
        plot_gate_by_regime(gate_sv, regimes, model.expert_names, args.out_dir)

    # Summary CSV
    save_summary_csv(expert_shap_dict, regimes, args.out_dir)

    print(f"\n✓ SHAP analysis complete. Outputs in: {args.out_dir}/")
    print("  Files produced:")
    for f in sorted(os.listdir(args.out_dir)):
        print(f"    {f}")


if __name__ == "__main__":
    main()
