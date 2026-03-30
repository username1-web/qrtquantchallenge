import pandas as pd
import numpy as np
import os
import utils

data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))
returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))

def generate_ensemble_portfolio(start_date, end_date):
    universe_boolean = universe.loc[start_date:end_date].astype(bool)
    features_ = features.loc[:end_date]
    
    # We invert the strong negative factors so we have positive weights
    best_negative_factors = [
        "accumulation_distribution_index",
        "ichimoku",
        "ultimate_oscillator",
        "stochastic_oscillator",
        "williams_r",
        "trend_5_20"
    ]
    
    portfolio = 0
    
    # Equal weight for best factors
    for name in best_negative_factors:
        sig = features_[name].shift(1).loc[start_date:end_date]
        sig = sig.where(universe_boolean, np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=False)
        sig = sig.sub(sig.mean(axis=1), axis=0) # Demean
        sig = sig.fillna(0)
        
        portfolio += sig

    portfolio = portfolio.where(universe_boolean, np.nan)
    
    # Smooth signal to reduce turnover!
    portfolio = portfolio.ewm(span=3, min_periods=1, ignore_na=False).mean()
    
    portfolio = portfolio.where(universe_boolean, np.nan)
    
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    sum_pos = portfolio[portfolio > 0].sum(axis=1)
    sum_neg = portfolio[portfolio < 0].sum(axis=1)
    
    df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
    df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5
    
    portfolio_scaled = df_pos.fillna(0) + df_neg.fillna(0)
    return portfolio_scaled.fillna(0)

if __name__ == "__main__":
    start = "2005-01-03"
    end = "2019-12-31"
    port = generate_ensemble_portfolio(start, end)
    net_sharpe, _ = utils.backtest_portfolio(port, returns.loc[start:end], universe.loc[start:end], False, True)
    print(f"Smoothed Ensemble Net Sharpe: {net_sharpe}")
