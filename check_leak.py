import pandas as pd
import numpy as np
import os

data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"
print("Loading data...")
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))
print("Done loading")

start = "2005-01-03"
end = "2019-12-31"

rets = returns.loc[start:end]
univ = universe.loc[start:end].astype(bool)

# Mask returns with universe
rets = rets.where(univ, np.nan)

feature_names = features.columns.levels[0].tolist() if isinstance(features.columns, pd.MultiIndex) else []
if not feature_names:
    feature_names = [
        "relative_strength_index", "williams_r", "volatility_20", "volatility_60",
        "trend_1_3", "trend_5_20", "trend_20_60", "average_true_range", "macd", "trix",
        "commodity_channel_index", "chande_momentum_oscillator", "ichimoku",
        "know_sure_thing", "ultimate_oscillator", "aroon", "stochastic_oscillator",
        "on_balance_volume", "ease_of_movement", "chaikin_money_flow",
        "accumulation_distribution_index", "volume"
    ]

# We are predicting rets at time t, using features at time t-1.
# So we shift features by 1.
for name in feature_names:
    feat = features[name].loc[start:end]
    feat = feat.where(univ, np.nan)
    
    # Check correlation of feature_t with return_t
    corr_t = feat.corrwith(rets, axis=1).mean()
    
    # Check correlation of feature_{t-1} with return_t (What we are supposed to predict)
    feat_shifted = feat.shift(1)
    corr_t_minus_1 = feat_shifted.corrwith(rets, axis=1).mean()
    
    # Check correlation of feature_{t-2} with return_t
    feat_shifted_2 = feat.shift(2)
    corr_t_minus_2 = feat_shifted_2.corrwith(rets, axis=1).mean()
    
    print(f"Feature: {name:35}")
    print(f"  Corr (t)  : {corr_t:.5f}")
    print(f"  Corr (t-1): {corr_t_minus_1:.5f}")
    print(f"  Corr (t-2): {corr_t_minus_2:.5f}")
    print("-" * 50)
