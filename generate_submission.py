import pandas as pd
import numpy as np
import os

print("Loading data...")
data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

# ======= STRATEGY CONFIGURATION =======
MEAN_REVERSION_FACTORS = [
    "accumulation_distribution_index",
    "ichimoku",
    "trend_5_20",
    "trend_1_3",
    "chaikin_money_flow",
    "average_true_range",
]
MEAN_REVERSION_WEIGHT = 1.0

VOLUME_FACTORS = [
    "volume",
    "on_balance_volume",
    "ease_of_movement",
]
VOLUME_WEIGHT = 0.7

EWMA_SPAN = 5
# =======================================

def generate_full_submission():
    start_date = universe.index[0].strftime('%Y-%m-%d')
    end_date = universe.index[-1].strftime('%Y-%m-%d')
    
    universe_boolean = universe.astype(bool)
    
    portfolio = 0
    
    # Mean reversion factors: rank descending
    print("Processing mean reversion factors...")
    for name in MEAN_REVERSION_FACTORS:
        sig = features[name].shift(1)
        sig = sig.where(universe_boolean, np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=False)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sig = sig.fillna(0)
        portfolio += sig * MEAN_REVERSION_WEIGHT
    
    # Volume/flow factors: rank ascending
    print("Processing volume factors...")
    for name in VOLUME_FACTORS:
        sig = features[name].shift(1)
        sig = sig.where(universe_boolean, np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=True)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sig = sig.fillna(0)
        portfolio += sig * VOLUME_WEIGHT
    
    # Universe mask
    portfolio = portfolio.where(universe_boolean, np.nan)
    
    # EWMA smoothing
    print("Smoothing portfolio...")
    portfolio = portfolio.ewm(span=EWMA_SPAN, min_periods=1, ignore_na=False).mean()
    
    # Re-mask and demean
    portfolio = portfolio.where(universe_boolean, np.nan)
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    # Scale to unit capital (0.5 long, 0.5 short)
    sum_pos = portfolio[portfolio > 0].sum(axis=1)
    sum_neg = portfolio[portfolio < 0].sum(axis=1)
    
    df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
    df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5
    
    portfolio_scaled = df_pos.fillna(0) + df_neg.fillna(0)
    portfolio_scaled = portfolio_scaled.fillna(0)
    
    portfolio_scaled = portfolio_scaled.reindex(index=universe.index, columns=universe.columns, fill_value=0)
    portfolio_scaled.to_csv(os.path.join(data_dir, "submission.csv"))
    print(f"Submission saved! Shape: {portfolio_scaled.shape}")
    print(f"Date range: {portfolio_scaled.index[0]} to {portfolio_scaled.index[-1]}")

if __name__ == "__main__":
    generate_full_submission()
