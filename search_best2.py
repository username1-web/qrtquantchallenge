"""
Replicate the optimizer's exact configs AND try more combinations.
The previous optimizer used factors ordered by individual factor sharpe:
  accumulation_distribution_index, ichimoku, ultimate_oscillator,
  stochastic_oscillator, williams_r, trend_5_20,
  trend_1_3, chaikin_money_flow, average_true_range, trend_20_60
"""
import pandas as pd
import numpy as np
import os
import utils

data_dir = "/Users/tantheta/Downloads/qrt-quant-challenge-2026-iit-delhi"
print("Loading data...")
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

start = "2005-01-03"
end = "2019-12-31"
univ_bool = universe.loc[start:end].astype(bool)
rets = returns.loc[start:end]

# The exact factor ordering from optimize_strategy.py (by individual sharpe)
NEG_FACTORS_BY_SHARPE = [
    "accumulation_distribution_index",  # sharpe: -2.303
    "ichimoku",                         # sharpe: -1.822
    "ultimate_oscillator",              # sharpe: -1.783
    "stochastic_oscillator",            # sharpe: -1.742
    "williams_r",                       # sharpe: -1.742
    "trend_5_20",                       # sharpe: -1.693
    "trend_1_3",                        # sharpe: -1.53
    "chaikin_money_flow",               # sharpe: -1.529
    "average_true_range",               # sharpe: -1.456
    "trend_20_60",                      # sharpe: -1.406
]

NEG_SHARPE_VALS = {
    "accumulation_distribution_index": -2.303,
    "ichimoku": -1.822,
    "ultimate_oscillator": -1.783,
    "stochastic_oscillator": -1.742,
    "williams_r": -1.742,
    "trend_5_20": -1.693,
    "trend_1_3": -1.53,
    "chaikin_money_flow": -1.529,
    "average_true_range": -1.456,
    "trend_20_60": -1.406,
}

POS_FACTORS = ["volume", "volatility_60", "volatility_20"]
POS_SHARPE_VALS = {"volume": 0.781, "volatility_60": 0.311, "volatility_20": 0.29}

# Precompute signals
print("Precomputing signals...")
neg_signals = {}
for name in NEG_FACTORS_BY_SHARPE:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=False)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    neg_signals[name] = sig

pos_signals = {}
for name in POS_FACTORS:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=True)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    pos_signals[name] = sig

# Also try: more factors from the "non-selected" group
extra_neg = ["chande_momentum_oscillator", "relative_strength_index", 
             "commodity_channel_index", "aroon", "macd", "trix",
             "know_sure_thing", "on_balance_volume"]
extra_signals = {}
for name in extra_neg:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=False)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    extra_signals[name] = sig

# Also positive extras
extra_pos = ["on_balance_volume", "ease_of_movement"]
for name in extra_pos:
    if name not in pos_signals:
        sig = features[name].shift(1).loc[start:end]
        sig = sig.where(univ_bool, np.nan)
        sig = sig.rank(axis=1, pct=True, ascending=True)
        sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
        pos_signals[name] = sig


def build_and_test(neg_facs, pos_facs, neg_wts, pos_wts, span):
    portfolio = 0
    for f, w in zip(neg_facs, neg_wts):
        portfolio += neg_signals[f] * w
    for f, w in zip(pos_facs, pos_wts):
        portfolio += pos_signals[f] * w
    
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


print("\n=== REPLICATE OPTIMIZER'S BEST: 10 neg + pos, sharpe-wt, span=3 ===")
neg_facs = NEG_FACTORS_BY_SHARPE[:10]
neg_wts = [abs(NEG_SHARPE_VALS[f]) for f in neg_facs]
pos_facs = POS_FACTORS
pos_wts = [POS_SHARPE_VALS[f] for f in pos_facs]
s = build_and_test(neg_facs, pos_facs, neg_wts, pos_wts, 3)
print(f"  => {s}")

print("\n=== TESTING EXHAUSTIVE SWEEP ===")
best_sharpe = 0
best_config = ""

for n in [6, 7, 8, 9, 10]:
    for use_pos in [False, True]:
        for wt_scheme in ["equal", "sharpe"]:
            for span in [2, 3, 4]:
                nf = NEG_FACTORS_BY_SHARPE[:n]
                if wt_scheme == "equal":
                    nw = [1.0] * n
                else:
                    nw = [abs(NEG_SHARPE_VALS[f]) for f in nf]
                
                if use_pos:
                    pf = POS_FACTORS
                    if wt_scheme == "equal":
                        pw = [0.5] * len(pf)
                    else:
                        pw = [POS_SHARPE_VALS[f] for f in pf]
                else:
                    pf, pw = [], []
                
                s = build_and_test(nf, pf, nw, pw, span)
                label = f"Neg={n}, Pos={use_pos}, W={wt_scheme}, Span={span}"
                print(f"  {label} => {s}")
                if s > best_sharpe:
                    best_sharpe = s
                    best_config = label

print(f"\nBest: {best_sharpe} ({best_config})")

# --- test adding extra factors to top configs ---
print("\n=== ADDING EXTRA FACTORS TO BEST BASE ===")
# Start from 10 neg, sharpe-wt, span=3 as base
base_neg = NEG_FACTORS_BY_SHARPE[:10]
base_nw = [abs(NEG_SHARPE_VALS[f]) for f in base_neg]

for extra_name in extra_neg:
    if extra_name not in base_neg:
        test_neg = base_neg + [extra_name]
        test_nw = base_nw + [1.0]
        # Make this extra factor also a neg signal
        neg_signals[extra_name] = extra_signals[extra_name]
        
        for span in [2, 3]:
            s = build_and_test(test_neg, POS_FACTORS, test_nw, 
                              [POS_SHARPE_VALS[f] for f in POS_FACTORS], span)
            print(f"  +{extra_name}, span={span} => {s}")

print("\nDone!")
