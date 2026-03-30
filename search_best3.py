"""
Deep dive around the best config: 9 neg factors, no pos, equal weight.
Also try small variations and different spans.
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

NEG_FACTORS_BY_SHARPE = [
    "accumulation_distribution_index",
    "ichimoku",
    "ultimate_oscillator",
    "stochastic_oscillator",
    "williams_r",
    "trend_5_20",
    "trend_1_3",
    "chaikin_money_flow",
    "average_true_range",
    "trend_20_60",
]

neg_sharpe = {
    "accumulation_distribution_index": 2.303,
    "ichimoku": 1.822,
    "ultimate_oscillator": 1.783,
    "stochastic_oscillator": 1.742,
    "williams_r": 1.742,
    "trend_5_20": 1.693,
    "trend_1_3": 1.53,
    "chaikin_money_flow": 1.529,
    "average_true_range": 1.456,
    "trend_20_60": 1.406,
}

# ALL neg factors we might consider
all_neg = [
    "accumulation_distribution_index", "ichimoku", "ultimate_oscillator",
    "stochastic_oscillator", "williams_r", "trend_5_20",
    "trend_1_3", "chaikin_money_flow", "average_true_range", "trend_20_60",
    "chande_momentum_oscillator", "relative_strength_index",
    "commodity_channel_index", "aroon", "macd", "trix", "know_sure_thing",
]

print("Precomputing signals...")
signals = {}
for name in all_neg:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=False)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    signals[name] = sig

# Also precompute positive signals
pos_names = ["volume", "volatility_60", "volatility_20", "on_balance_volume", "ease_of_movement"]
pos_signals = {}
for name in pos_names:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=True)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    pos_signals[name] = sig

def test(factor_signals, weights, span):
    portfolio = sum(s * w for s, w in zip(factor_signals, weights))
    portfolio = portfolio.where(univ_bool, np.nan)
    portfolio = portfolio.ewm(span=span, min_periods=1, ignore_na=False).mean()
    portfolio = portfolio.where(univ_bool, np.nan)
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    sum_pos = portfolio[portfolio > 0].sum(axis=1)
    sum_neg = portfolio[portfolio < 0].sum(axis=1)
    df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
    df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5
    final = df_pos.fillna(0) + df_neg.fillna(0)
    
    net_sharpe, pnl = utils.backtest_portfolio(final.fillna(0), rets, univ_bool, False, False)
    return net_sharpe

# ===== Fine-tune around best: 9 neg factors =====
best_9 = NEG_FACTORS_BY_SHARPE[:9]  
print("\n=== BEST 9 FACTOR DETAILED SPAN SWEEP ===")
for span in [2, 3, 4, 5, 6, 7, 8, 10]:
    s = test([signals[f] for f in best_9], [1.0]*9, span)
    print(f"  9 neg, equal, span={span} => {s}")

# ===== Try dropping each one from the 9 =====
print("\n=== DROP-ONE ANALYSIS (from 9) ===")
for i, dropped in enumerate(best_9):
    facs = [f for j, f in enumerate(best_9) if j != i]
    sigs = [signals[f] for f in facs]
    s = test(sigs, [1.0]*8, 4)
    print(f"  Drop {dropped:40s} => {s}")

# ===== Try adding each extra factor to the 9 =====
print("\n=== ADD-ONE ANALYSIS (to 9) ===")
for extra in all_neg:
    if extra not in best_9:
        facs = best_9 + [extra]
        sigs = [signals[f] for f in facs]
        for span in [3, 4, 5]:
            s = test(sigs, [1.0]*10, span)
            print(f"  +{extra:35s}, span={span} => {s}")

# ===== Try adding positive factors to the 9 =====
print("\n=== ADD POS TO 9 NEG ===")
base_sigs = [signals[f] for f in best_9]
for pname in pos_names:
    for pw in [0.3, 0.5, 0.7, 1.0]:
        sigs = base_sigs + [pos_signals[pname]]
        wts = [1.0]*9 + [pw]
        for span in [3, 4]:
            s = test(sigs, wts, span)
            print(f"  9neg + {pname}(w={pw}), span={span} => {s}")

# ===== Try weighting the 9 differently =====
print("\n=== WEIGHTED 9 FACTORS ===")
for span in [3, 4, 5]:
    # Sharpe-weighted
    wts = [neg_sharpe[f] for f in best_9]
    s = test([signals[f] for f in best_9], wts, span)
    print(f"  9 neg, sharpe-wt, span={span} => {s}")
    
    # Sqrt-sharpe weighted
    wts = [np.sqrt(neg_sharpe[f]) for f in best_9]
    s = test([signals[f] for f in best_9], wts, span)
    print(f"  9 neg, sqrt-sharpe-wt, span={span} => {s}")

print("\nDone!")
