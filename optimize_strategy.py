import pandas as pd
import numpy as np
import os
import utils
import itertools

data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"
print("Loading data...")
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

start = "2005-01-03"
end = "2019-12-31"

univ_bool = universe.loc[start:end].astype(bool)
rets = returns.loc[start:end]

# Precompute rank versions of all relevant features to save time
print("Precomputing functional representations...")
feature_data = {}

# Negative correlation signals
neg_features = {
    "accumulation_distribution_index": -2.303,
    "ichimoku": -1.822,
    "ultimate_oscillator": -1.783,
    "stochastic_oscillator": -1.742,
    "williams_r": -1.742,
    "trend_5_20": -1.693,
    "trend_1_3": -1.53,
    "chaikin_money_flow": -1.529,
    "average_true_range": -1.456,
    "trend_20_60": -1.406
}

# Positive correlation signals
pos_features = {
    "volume": 0.781,
    "volatility_60": 0.311,
    "volatility_20": 0.29
}

def get_signal(name, ascending):
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=ascending)
    # Demean
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    return sig

for name in neg_features:
    feature_data[name] = get_signal(name, ascending=False)
for name in pos_features:
    feature_data[name] = get_signal(name, ascending=True)

print("Running optimization...")
best_sharpe = 0
best_params = {}

for num_neg in [6, 8, 10]:
    for use_pos in [False, True]:
        for weight_scheme in ["equal", "sharpe"]:
            for span in [3, 4, 5, 6, 7]:
                
                selected_neg = list(neg_features.keys())[:num_neg]
                
                portfolio = 0
                for name in selected_neg:
                    w = abs(neg_features[name]) if weight_scheme == "sharpe" else 1.0
                    portfolio += feature_data[name] * w
                    
                if use_pos:
                    for name in pos_features:
                        w = pos_features[name] if weight_scheme == "sharpe" else (0.5 if weight_scheme=="equal" else 1.0)
                        portfolio += feature_data[name] * w
                        
                # Smooth
                p_smoothed = portfolio.where(univ_bool, np.nan).ewm(span=span, min_periods=1, ignore_na=False).mean()
                p_smoothed = p_smoothed.where(univ_bool, np.nan)
                
                # Scale
                p_smoothed = p_smoothed.sub(p_smoothed.mean(axis=1), axis=0)
                sum_pos = p_smoothed[p_smoothed > 0].sum(axis=1)
                sum_neg = p_smoothed[p_smoothed < 0].sum(axis=1)
                
                df_pos = p_smoothed[p_smoothed > 0].div(sum_pos, axis=0) * 0.5
                df_neg = p_smoothed[p_smoothed < 0].div(sum_neg.abs(), axis=0) * 0.5
                
                final_port = df_pos.fillna(0) + df_neg.fillna(0)
                final_port = final_port.fillna(0)
                
                # Eval
                net_sharpe, _ = utils.backtest_portfolio(final_port, rets, univ_bool, False, False)
                print(f"Neg: {num_neg}, Pos: {use_pos}, W: {weight_scheme}, Span: {span} => Sharpe: {net_sharpe}")
                
                if net_sharpe > best_sharpe:
                    best_sharpe = net_sharpe
                    best_params = {"neg": num_neg, "pos": use_pos, "weight": weight_scheme, "span": span}

print(f"\nBest Sharpe: {best_sharpe}")
print(f"Best Params: {best_params}")
