"""
dataset.py
Downloads OHLCV data for S&P 500 sector ETFs, engineers features, fits a
3-state HMM to label market regimes, and returns PyTorch DataLoader objects
for train / val / test splits.

Usage
    from dataset import build_dataloaders, FEATURE_COLS, N_REGIMES
    loaders, scaler, hmm, hmm_scaler = build_dataloaders()
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Config 
TICKERS  =  [
    # 🇺🇸 United States
    "SPY",       
    "QQQ",       
    "AAPL",
    "MSFT",

    # 🇬🇧 United Kingdom
    "^FTSE",     
    "HSBA.L",    
    "AZN.L",     

    # 🇮🇳 India
    "^NSEI",
    "TCS.NS",
    "INFY.NS",
]
START_DATE   = "2018-01-01"
END_DATE     = "2025-12-31"
SEQ_LEN      = 30
PRED_HORIZON = 5
N_REGIMES    = 3
BATCH_SIZE   = 64
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15

FEATURE_COLS = [
    "log_return",
    "volatility_10d",
    "volatility_30d",
    "rsi_14",
    "macd",
    "macd_signal",
    "volume_ratio",
    "bb_width",
    "regime_bull",
    "regime_bear",
    "regime_highvol",
]

REGIME_IDX = [
    FEATURE_COLS.index("regime_bull"),
    FEATURE_COLS.index("regime_bear"),
    FEATURE_COLS.index("regime_highvol"),
]

# HMM observable columns — kept separate for clarity
HMM_COLS = ["log_return", "volatility_10d"]


# Feature engineering

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    vol   = df["Volume"]
    df = df.copy()

    df["log_return"]     = np.log(close / close.shift(1))
    df["volatility_10d"] = df["log_return"].rolling(10).std()
    df["volatility_30d"] = df["log_return"].rolling(30).std()
    df["rsi_14"]         = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["volume_ratio"] = vol / vol.rolling(20).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_width"] = (2 * std20) / (sma20 + 1e-9)

    df["target"] = np.log(close.shift(-PRED_HORIZON) / close)
    return df


# HMM fitting (train-only)
def fit_hmm(train_df: pd.DataFrame) -> tuple[GaussianHMM, StandardScaler]:
    obs_raw = train_df[HMM_COLS].dropna().values  # (T_train, 2)

    
    hmm_scaler = StandardScaler()
    obs_scaled = hmm_scaler.fit_transform(obs_raw)

    
    km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10)
    km.fit(obs_scaled)

    hmm_model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="diag",   
        n_iter=500,               
        tol=1e-3,
        init_params="t",          
        params="stmc",            
        random_state=42,
        verbose=False,            
    )

  
    hmm_model.startprob_ = np.ones(N_REGIMES) / N_REGIMES
    hmm_model.means_     = km.cluster_centers_
    hmm_model.covars_    = np.tile(
        np.var(obs_scaled, axis=0).clip(min=1e-3),
        (N_REGIMES, 1),
    )

    hmm_model.fit(obs_scaled)

    return hmm_model, hmm_scaler


def apply_hmm(hmm_model: GaussianHMM, hmm_scaler: StandardScaler,
              df: pd.DataFrame) -> np.ndarray:
    obs_raw    = df[HMM_COLS].dropna().values
    obs_scaled = hmm_scaler.transform(obs_raw)
    posteriors = hmm_model.predict_proba(obs_scaled)   # (T, N_REGIMES)

    # Order states by mean log_return in original space (inverse-transform means)
    means_original = hmm_scaler.inverse_transform(hmm_model.means_)[:, 0]
    order          = np.argsort(means_original)        # ascending: bear, high-vol, bull
    return posteriors[:, order]


def _attach_regime_posteriors(df: pd.DataFrame, posteriors: np.ndarray) -> pd.DataFrame:
    df       = df.copy()
    valid_idx = df[HMM_COLS].dropna().index
    regime_df = pd.DataFrame(
        posteriors,
        index=valid_idx,
        columns=["regime_bear", "regime_highvol", "regime_bull"],
    )
    return df.join(regime_df)


# Download + feature engineering (no HMM)

def load_ticker(ticker: str) -> tuple[pd.DataFrame, None]:
    raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = _engineer_features(raw)
    df.dropna(subset=HMM_COLS, inplace=True)
    return df, None


#PyTorch Dataset

class FinancialSequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        tickers: np.ndarray
    ):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.tickers = tickers

    def __len__(self):
        return len(self.X) - SEQ_LEN

    def __getitem__(self, idx):
        return (
            self.X[idx: idx + SEQ_LEN],
            self.y[idx + SEQ_LEN],
            self.tickers[idx + SEQ_LEN]
        )


#Main builder

def build_dataloaders(
    tickers: list[str] = TICKERS,
    batch_size: int = BATCH_SIZE,
) -> tuple[dict, StandardScaler, GaussianHMM, StandardScaler]:
    
    # Step 1: download and engineer features for all tickers
    all_frames = []
    for ticker in tickers:
        print(f"  Downloading {ticker}...")
        df, _ = load_ticker(ticker)
        df["ticker"] = ticker
        if ticker in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]:
            df["country"] = "US"
        elif ticker in ["^FTSE", "SHEL.L", "HSBA.L", "AZN.L"]:
            df["country"] = "UK"
        elif ticker in ["^NSEI", "RELIANCE.NS", "TCS.NS", "INFY.NS"]:
            df["country"] = "India"
        else:
            df["country"] = "Other"
        all_frames.append(df)

    data = pd.concat(all_frames).sort_index()

    # Step 2: chronological split (BEFORE HMM)
    n_raw = len(data)
    t1    = int(n_raw * TRAIN_FRAC)
    t2    = int(n_raw * (TRAIN_FRAC + VAL_FRAC))

    train_raw = data.iloc[:t1]
    
    # Step 3: fit HMM on training data only 
    print("\nFitting HMM on training split only...")
    hmm_model, hmm_scaler = fit_hmm(train_raw)
    print(f"  HMM converged. States ordered: bear / high-vol / bull")
    print(f"  Means (original scale):\n"
          f"    {hmm_scaler.inverse_transform(hmm_model.means_)}")

    # Step 4: apply HMM to full dataset 
    posteriors_all = apply_hmm(hmm_model, hmm_scaler, data)
    data = _attach_regime_posteriors(data, posteriors_all)
    data.dropna(inplace=True)

    # Step 5: extract features/targets and re-split
    features = data[FEATURE_COLS].values
    targets  = data["target"].values
    ticker_values = data["ticker"].values
    n        = len(features)
    t1       = int(n * TRAIN_FRAC)
    t2       = int(n * (TRAIN_FRAC + VAL_FRAC))

    X_train, y_train, ticker_train = (
    features[:t1],
    targets[:t1],
    ticker_values[:t1]
    )

    X_val, y_val, ticker_val = (
    features[t1:t2],
    targets[t1:t2],
    ticker_values[t1:t2]
    )

    X_test, y_test, ticker_test = (
    features[t2:],
    targets[t2:],
    ticker_values[t2:]
    )

    # Step 6: scale features on train only
    feat_scaler = StandardScaler()
    X_train     = feat_scaler.fit_transform(X_train)
    X_val       = feat_scaler.transform(X_val)
    X_test      = feat_scaler.transform(X_test)

    # Step 7: DataLoaders
    train_ds = FinancialSequenceDataset(
    X_train, y_train, ticker_train
    )

    val_ds = FinancialSequenceDataset(
    X_val, y_val, ticker_val
    )

    test_ds = FinancialSequenceDataset(
    X_test, y_test, ticker_test
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    }

    print(
        f"\nDataset summary:"
        f"\n  Features : {len(FEATURE_COLS)}"
        f"\n  Train    : {len(train_ds):,} sequences"
        f"\n  Val      : {len(val_ds):,} sequences"
        f"\n  Test     : {len(test_ds):,} sequences"
    )

    return loaders, feat_scaler, hmm_model, hmm_scaler


if __name__ == "__main__":
    loaders, feat_scaler, hmm_model, hmm_scaler = build_dataloaders()
    x, y, ticker = next(iter(loaders["train"]))
    print(f"\nBatch shapes — X: {x.shape}, y: {y.shape}")
    print(f"Ticker sample: {ticker[:5]}")
    print("HMM means (bear / high-vol / bull, original scale):")
    print(hmm_scaler.inverse_transform(hmm_model.means_))
