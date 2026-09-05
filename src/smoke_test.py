import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FinancialSequenceDataset, FEATURE_COLS, SEQ_LEN, REGIME_IDX
from model import MoEForecaster
from train import run_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_synthetic_loaders(n_train=500, n_val=100, n_test=100,
                            seq_len=SEQ_LEN, n_features=len(FEATURE_COLS),
                            batch_size=32):
    def make_split(n):
        X = np.random.randn(n + seq_len, n_features).astype(np.float32)
        regime_raw = np.random.dirichlet(alpha=[1, 1, 1], size=n + seq_len)
        X[:, REGIME_IDX] = regime_raw
        y = np.random.randn(n + seq_len).astype(np.float32) * 0.01
        return FinancialSequenceDataset(X, y)

    return {
        "train": DataLoader(make_split(n_train), batch_size=batch_size, shuffle=False),
        "val":   DataLoader(make_split(n_val),   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(make_split(n_test),  batch_size=batch_size, shuffle=False),
    }


def main():
    print(f"Device: {DEVICE}")
    print("Building synthetic loaders...")
    loaders = make_synthetic_loaders()

    model = MoEForecaster().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print(f"Experts: {model.expert_names}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("\nRunning 3 synthetic epochs...\n")

    for epoch in range(1, 4):
        tr = run_epoch(model, loaders["train"], optimizer, 0.1, 0.05, training=True)
        va = run_epoch(model, loaders["val"],   optimizer, 0.1, 0.05, training=False)
        print(f"  Epoch {epoch}  Train loss={tr['loss']:.4f}  "
              f"Val loss={va['loss']:.4f}  Val DA={va['dir_acc']:.3f}")

    te = run_epoch(model, loaders["test"], optimizer, 0.1, 0.05, training=False)
    print(f"\nTest MAE={te['mae']:.5f}  RMSE={te['rmse']:.5f}  DA={te['dir_acc']:.3f}")

    print("\nGate weight distribution (mean over test batches):")
    model.eval()
    all_gates = []
    with torch.no_grad():
        for x, _ in loaders["test"]:
            _, g, _ = model(x.to(DEVICE))
            all_gates.append(g.cpu())
    gates = torch.cat(all_gates).numpy()
    for i, name in enumerate(model.expert_names):
        print(f"  {name:12s} mean weight : {gates[:, i].mean():.3f} +/- {gates[:, i].std():.3f}")

    print("\nSmoke test passed — all components working correctly.")


if __name__ == "__main__":
    main()
