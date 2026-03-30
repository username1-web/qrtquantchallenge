import json
import os

notebook_path = '/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi/main.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

new_vectorized = """def generate_portfolio_vectorized(
    entire_features: pd.DataFrame,
    universe: pd.DataFrame,
    start_date: str,
    end_date: str,
):
    import datetime
    
    try:
        start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        cutoff_date = datetime.datetime.strptime('2005-01-01', '%Y-%m-%d')
    except ValueError:
        raise ValueError("start_date and end_date must be strings in 'YYYY-MM-DD' format.")

    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date.")

    if start_dt < cutoff_date:
        raise ValueError("start_date must be later than '2005-01-01'.")

    trading_days = universe.index[(universe.index >= start_dt) & (universe.index <= end_dt)]
    if len(trading_days) == 0:
        raise ValueError("No Trading Days in the specified dates")
    
    universe_boolean = universe.loc[:end_date].astype(bool)
    
    # Configuration
    MEAN_REVERSION_FACTORS = [
        "accumulation_distribution_index", "ichimoku", 
        "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"
    ]
    MEAN_REVERSION_WEIGHT = 1.0
    
    VOLUME_FACTORS = ["volume", "on_balance_volume", "ease_of_movement"]
    VOLUME_WEIGHT = 0.7
    EWMA_SPAN = 5
    
    portfolio = 0
    
    # Mean reversion factors: rank descending
    for name in MEAN_REVERSION_FACTORS:
        sig = entire_features[name].shift(1).loc[start_date:end_date]
        sig = sig.where(universe_boolean.loc[start_date:end_date], np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=False)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sig = sig.fillna(0)
        portfolio += sig * MEAN_REVERSION_WEIGHT
    
    # Volume/flow factors: rank ascending
    for name in VOLUME_FACTORS:
        sig = entire_features[name].shift(1).loc[start_date:end_date]
        sig = sig.where(universe_boolean.loc[start_date:end_date], np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=True)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sig = sig.fillna(0)
        portfolio += sig * VOLUME_WEIGHT
    
    portfolio = portfolio.where(universe_boolean.loc[start_date:end_date], np.nan)
    portfolio = portfolio.ewm(span=EWMA_SPAN, min_periods=1, ignore_na=False).mean()
    portfolio = portfolio.where(universe_boolean.loc[start_date:end_date], np.nan)
    
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    sum_pos = portfolio[portfolio > 0].sum(axis=1)
    sum_neg = portfolio[portfolio < 0].sum(axis=1)
    
    df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
    df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5
    
    portfolio_scaled = df_pos.fillna(0) + df_neg.fillna(0)
    return portfolio_scaled.fillna(0).loc[start_date:end_date]
"""

new_get_weights = """def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict[str, float]:
    \"\"\"
    Calculate stock weights for the portfolio on the current trading day.
    \"\"\"
    if features.shape[0] == 0:
        return (today_universe * 1).replace(0, np.nan).dropna().fillna(0).to_dict()

    lookback = min(features.shape[0], 15)
    recent_features = features.iloc[-lookback:]
    
    MEAN_REVERSION_FACTORS = [
        "accumulation_distribution_index", "ichimoku", 
        "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"
    ]
    MEAN_REVERSION_WEIGHT = 1.0
    
    VOLUME_FACTORS = ["volume", "on_balance_volume", "ease_of_movement"]
    VOLUME_WEIGHT = 0.7
    EWMA_SPAN = 5
    
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
    
    smoothed = ensemble.ewm(span=EWMA_SPAN, min_periods=1, ignore_na=False).mean()
    alpha = smoothed.iloc[-1]
    
    import utils
    alpha = utils.get_universe_adjusted_series(alpha, today_universe)
    alpha = alpha.sub(alpha.mean(), fill_value=0)
    
    alpha = utils.scale_to_book_long_short(alpha)
    alpha = alpha.replace(0, np.nan).dropna()

    return alpha.to_dict()
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def generate_portfolio_vectorized(" in source:
            cell['source'] = [line + "\n" for line in new_vectorized.split("\n")][:-1]
        elif "def get_weights(" in source:
            cell['source'] = [line + "\n" for line in new_get_weights.split("\n")][:-1]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
