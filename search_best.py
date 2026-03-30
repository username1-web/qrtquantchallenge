"""
Systematic Factor Search: Find the optimal factor combination.
Tests various factor counts, weighting schemes, and EWMA spans.
"""
import pandas as pd
import numpy as np
import os
import utils
from itertools import combinations

data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"
print("Loading data...")
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

start = "2005-01-03"
end = "2019-12-31"
univ_bool = universe.loc[start:end].astype(bool)
rets = returns.loc[start:end]

# Pre-compute all factor signals
print("Precomputing signals...")

# All factors sorted by absolute lag-1 IC  
ALL_FACTORS = {
    # Negative IC (mean reversion) - rank descending 
    "ultimate_oscillator":        {"ic": -0.00927, "asc": False},
    "trend_20_60":                {"ic": -0.00923, "asc": False},
    "williams_r":                 {"ic": -0.00872, "asc": False},
    "stochastic_oscillator":      {"ic": -0.00872, "asc": False},
    "relative_strength_index":    {"ic": -0.00807, "asc": False},
    "trend_5_20":                 {"ic": -0.00763, "asc": False},
    "chaikin_money_flow":         {"ic": -0.00695, "asc": False},
    "chande_momentum_oscillator": {"ic": -0.00618, "asc": False},
    "ichimoku":                   {"ic": -0.00404, "asc": False},
    "average_true_range":         {"ic": -0.00392, "asc": False},
    "commodity_channel_index":    {"ic": -0.00388, "asc": False},
    "aroon":                      {"ic": -0.00336, "asc": False},
    "accumulation_distribution_index": {"ic": -0.00274, "asc": False},
    "macd":                       {"ic": -0.00282, "asc": False},
    "trix":                       {"ic": -0.00259, "asc": False},
    "know_sure_thing":            {"ic": -0.00204, "asc": False},
    "trend_1_3":                  {"ic": -0.00270, "asc": False},
    # Positive IC - rank ascending
    "volume":                     {"ic": 0.00384, "asc": True},
    "volatility_60":              {"ic": 0.00301, "asc": True},
    "on_balance_volume":          {"ic": 0.00248, "asc": True},
    "volatility_20":              {"ic": 0.00223, "asc": True},
    "ease_of_movement":           {"ic": 0.00049, "asc": True},
}

signals = {}
for name, info in ALL_FACTORS.items():
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=info["asc"])
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    signals[name] = sig

def test_config(factor_list, weights, span):
    portfolio = sum(signals[f] * w for f, w in zip(factor_list, weights))
    
    portfolio = portfolio.where(univ_bool, np.nan)
    portfolio = portfolio.ewm(span=span, min_periods=1, ignore_na=False).mean()
    portfolio = portfolio.where(univ_bool, np.nan)
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    sum_pos = portfolio[portfolio > 0].sum(axis=1)
    sum_neg = portfolio[portfolio < 0].sum(axis=1)
    
    df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
    df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5
    
    final = df_pos.fillna(0) + df_neg.fillna(0)
    final = final.fillna(0)
    
    net_sharpe, _ = utils.backtest_portfolio(final, rets, univ_bool, False, False)
    return net_sharpe

# --- Test Approach 1: Top N negative factors only, equal weight ---
print("\n=== TOP N NEGATIVE FACTORS, EQUAL WEIGHT ===")
neg_sorted = sorted(
    [(k, v) for k, v in ALL_FACTORS.items() if v["ic"] < 0], 
    key=lambda x: x[1]["ic"]
)

for n in [4, 5, 6, 7, 8, 9, 10, 12, 14, 16]:
    facs = [f for f, _ in neg_sorted[:n]]
    wts = [1.0] * n
    for span in [3, 4, 5]:
        s = test_config(facs, wts, span)
        print(f"  Top {n:2d} neg, equal, span={span} => {s}")

# --- Test Approach 2: Top N negative factors, IC-weighted ---
print("\n=== TOP N NEGATIVE FACTORS, IC-WEIGHTED ===")
for n in [4, 5, 6, 7, 8, 9, 10, 12, 14, 16]:
    facs = [f for f, _ in neg_sorted[:n]]
    wts = [abs(ALL_FACTORS[f]["ic"]) for f in facs]
    for span in [3, 4, 5]:
        s = test_config(facs, wts, span)
        print(f"  Top {n:2d} neg, IC-wt, span={span} => {s}")

# --- Test Approach 3: Top N neg + positive factors ---
print("\n=== TOP N NEG + ALL POS, EQUAL WEIGHT ===")
pos_sorted = sorted(
    [(k, v) for k, v in ALL_FACTORS.items() if v["ic"] > 0], 
    key=lambda x: -x[1]["ic"]
)
pos_facs = [f for f, _ in pos_sorted]

for n in [6, 8, 10, 12]:
    neg_facs = [f for f, _ in neg_sorted[:n]]
    all_facs = neg_facs + pos_facs
    wts = [1.0] * len(all_facs)
    for span in [3, 4, 5]:
        s = test_config(all_facs, wts, span)
        print(f"  Top {n:2d} neg + all pos, equal, span={span} => {s}")

# --- Test Approach 4: Top N neg + pos, IC-weighted ---
print("\n=== TOP N NEG + ALL POS, IC-WEIGHTED ===")
for n in [6, 8, 10, 12]:
    neg_facs = [f for f, _ in neg_sorted[:n]]
    all_facs = neg_facs + pos_facs
    wts = [abs(ALL_FACTORS[f]["ic"]) for f in all_facs]
    for span in [3, 4, 5]:
        s = test_config(all_facs, wts, span)
        print(f"  Top {n:2d} neg + all pos, IC-wt, span={span} => {s}")

# --- Test Approach 5: Only strongest 3-4 pos factors (no weak ones) ---
print("\n=== TOP N NEG + TOP 3 POS, IC-WEIGHTED ===")
top_pos = pos_facs[:3]
for n in [6, 8, 10, 12]:
    neg_facs = [f for f, _ in neg_sorted[:n]]
    all_facs = neg_facs + top_pos
    wts = [abs(ALL_FACTORS[f]["ic"]) for f in all_facs]
    for span in [3, 4, 5]:
        s = test_config(all_facs, wts, span)
        print(f"  Top {n:2d} neg + top3 pos, IC-wt, span={span} => {s}")

# --- Test Approach 6: ONLY the original 6 factors from verify_strategy.py ---
print("\n=== ORIGINAL 6 FACTORS (from verify_strategy.py) ===")
orig_6 = ["accumulation_distribution_index", "ichimoku", "ultimate_oscillator",
           "stochastic_oscillator", "williams_r", "trend_5_20"]
for span in [2, 3, 4, 5]:
    s = test_config(orig_6, [1.0]*6, span)
    print(f"  Original 6, equal, span={span} => {s}")

# IC-weighted version
wts6 = [abs(ALL_FACTORS[f]["ic"]) for f in orig_6]
for span in [2, 3, 4, 5]:
    s = test_config(orig_6, wts6, span)
    print(f"  Original 6, IC-wt, span={span} => {s}")

print("\nDone!")
