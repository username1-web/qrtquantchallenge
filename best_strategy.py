"""
Best Strategy: Multi-Factor Mean Reversion + Volume Ensemble
=============================================================
6 mean reversion factors (ranked descending, equal-weighted):
  - accumulation_distribution_index
  - ichimoku
  - trend_5_20
  - trend_1_3
  - chaikin_money_flow 
  - average_true_range

3 positive-IC volume/flow factors (ranked ascending, weight=0.7):
  - volume
  - on_balance_volume
  - ease_of_movement

EWMA span=5 for turnover control. No ML model.
"""
import pandas as pd
import numpy as np
import os
import utils

data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"

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


def generate_portfolio_vectorized(
    entire_features: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
):
    universe_boolean = universe.loc[:end_date].astype(bool)
    
    portfolio = 0
    
    # Mean reversion factors: rank descending (short high-ranked)
    for name in MEAN_REVERSION_FACTORS:
        sig = entire_features[name].shift(1).loc[start_date:end_date]
        sig = sig.where(universe_boolean.loc[start_date:end_date], np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=False)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sig = sig.fillna(0)
        portfolio += sig * MEAN_REVERSION_WEIGHT
    
    # Volume/flow factors: rank ascending (buy high volume stocks)
    for name in VOLUME_FACTORS:
        sig = entire_features[name].shift(1).loc[start_date:end_date]
        sig = sig.where(universe_boolean.loc[start_date:end_date], np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=True)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sig = sig.fillna(0)
        portfolio += sig * VOLUME_WEIGHT
    
    # Universe mask
    portfolio = portfolio.where(universe_boolean.loc[start_date:end_date], np.nan)
    
    # EWMA smoothing
    portfolio = portfolio.ewm(span=EWMA_SPAN, min_periods=1, ignore_na=False).mean()
    
    # Re-mask and demean
    portfolio = portfolio.where(universe_boolean.loc[start_date:end_date], np.nan)
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    # Scale to unit capital (0.5 long, 0.5 short)
    sum_pos = portfolio[portfolio > 0].sum(axis=1)
    sum_neg = portfolio[portfolio < 0].sum(axis=1)
    
    df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
    df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5
    
    portfolio_scaled = df_pos.fillna(0) + df_neg.fillna(0)
    return portfolio_scaled.fillna(0).loc[start_date:end_date]


def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict[str, float]:
    """
    Iterative version: called once per trading day.
    features: MultiIndex DataFrame with past features only.
    today_universe: Series with 0/1 for tradable stocks.
    """
    if features.shape[0] == 0:
        return (today_universe * 1).replace(0, np.nan).dropna().fillna(0).to_dict()
    
    lookback = min(features.shape[0], 15)
    recent_features = features.iloc[-lookback:]
    
    ensemble = 0
    
    # Mean reversion factors
    for name in MEAN_REVERSION_FACTORS:
        sig = recent_features.xs(name, axis=1, level=0)
        sig = sig.rank(axis=1, pct=True, ascending=False)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        ensemble += sig.fillna(0) * MEAN_REVERSION_WEIGHT
    
    # Volume/flow factors
    for name in VOLUME_FACTORS:
        sig = recent_features.xs(name, axis=1, level=0)
        sig = sig.rank(axis=1, pct=True, ascending=True)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        ensemble += sig.fillna(0) * VOLUME_WEIGHT
    
    # EWMA smoothing
    smoothed = ensemble.ewm(span=EWMA_SPAN, min_periods=1, ignore_na=False).mean()
    alpha = smoothed.iloc[-1]
    
    # Universe masking
    alpha = utils.get_universe_adjusted_series(alpha, today_universe)
    alpha = alpha.sub(alpha.mean())
    
    # Scale to 0.5 long / 0.5 short
    alpha = utils.scale_to_book_long_short(alpha)
    alpha = alpha.replace(0, np.nan).dropna()
    
    return alpha.to_dict()


if __name__ == "__main__":
    print("Loading data...")
    features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
    returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
    universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))
    
    start = "2005-01-03"
    end = "2019-12-31"
    
    print("Generating vectorized portfolio...")
    port = generate_portfolio_vectorized(features, universe, start, end)
    
    print("\n--- Full Period (2005-2019) ---")
    net_sharpe, _ = utils.backtest_portfolio(
        port, returns.loc[start:end], universe.loc[start:end], False, True
    )
    print(f"Net Sharpe: {net_sharpe}")
    
    # Sub-period robustness
    for period_name, s, e in [
        ("2005-2009", "2005-01-03", "2009-12-31"),
        ("2010-2014", "2010-01-04", "2014-12-31"),
        ("2015-2019", "2015-01-02", "2019-12-31"),
    ]:
        try:
            idx = port.loc[s:e].index.intersection(returns.loc[s:e].index)
            if len(idx) > 50:
                print(f"\n--- {period_name} ---")
                utils.backtest_portfolio(
                    port.loc[idx], returns.loc[idx], universe.loc[idx], False, True
                )
        except Exception as ex:
            print(f"  {period_name}: {ex}")
    
    # Match implementations
    print("\n--- Matching implementations ---")
    utils.match_implementations(get_weights, port, features, universe, returns)
