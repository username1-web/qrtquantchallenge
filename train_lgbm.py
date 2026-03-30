import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor

data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

start = "2005-01-03"
end = "2019-12-31"

rets = returns.loc[start:end]
univ_bool = universe.loc[start:end].astype(bool)

rets = rets.where(univ_bool, np.nan)
rets_demeaned = rets.sub(rets.mean(axis=1), axis=0)

feature_names = features.columns.levels[0].tolist() if isinstance(features.columns, pd.MultiIndex) else [
    "relative_strength_index", "williams_r", "volatility_20", "volatility_60",
    "trend_1_3", "trend_5_20", "trend_20_60", "average_true_range", "macd", "trix",
    "commodity_channel_index", "chande_momentum_oscillator", "ichimoku",
    "know_sure_thing", "ultimate_oscillator", "aroon", "stochastic_oscillator",
    "on_balance_volume", "ease_of_movement", "chaikin_money_flow",
    "accumulation_distribution_index", "volume"
]

X_list = []
for name in feature_names:
    feat = features[name].shift(1).loc[start:end]
    feat = feat.where(univ_bool, np.nan)
    feat_ranked = feat.rank(axis=1, pct=True, method='first')
    feat_demeaned = feat_ranked.sub(feat_ranked.mean(axis=1), axis=0)
    X_list.append(feat_demeaned.values.flatten())

X = np.column_stack(X_list)
Y = rets_demeaned.values.flatten()

valid = ~np.isnan(Y) & ~np.isnan(X).any(axis=1)
X_valid = X[valid]
Y_valid = Y[valid]

np.random.seed(42)
sample_size = min(1000000, len(X_valid))
indices = np.random.choice(len(X_valid), sample_size, replace=False)
X_sample = X_valid[indices]
Y_sample = Y_valid[indices]

model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_sample, Y_sample)

Y_pred_full = np.zeros_like(Y)
Y_pred_full[valid] = model.predict(X_valid)
Y_pred_matrix = Y_pred_full.reshape(rets.shape)

portfolio = pd.DataFrame(Y_pred_matrix, index=rets.index, columns=rets.columns)
portfolio = portfolio.where(univ_bool, np.nan)

# Rank cross sectionally to avoid concentration and max weight violation
portfolio = portfolio.rank(axis=1, pct=True, method='first')
portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)

portfolio = portfolio.ewm(span=5, min_periods=1, ignore_na=False).mean()
portfolio = portfolio.where(univ_bool, np.nan)
portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)

sum_pos = portfolio[portfolio > 0].sum(axis=1)
sum_neg = portfolio[portfolio < 0].sum(axis=1)

df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5

portfolio_scaled = df_pos.fillna(0) + df_neg.fillna(0)
portfolio_scaled = portfolio_scaled.fillna(0)

import utils
net_sharpe, _ = utils.backtest_portfolio(portfolio_scaled, returns.loc[start:end], universe.loc[start:end], False, True)
print(f"ML Model + Rank + EWMA Sharpe: {net_sharpe}")
