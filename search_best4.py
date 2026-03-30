"""
Final optimization: Explore the best paths from drop-one analysis.
Key insights:
- Dropping stochastic_oscillator OR williams_r from 9 => 2.064
- Adding volume to 9 => 2.02
- Combining both?
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

# All candidate factors
all_neg_names = [
    "accumulation_distribution_index", "ichimoku", "ultimate_oscillator",
    "stochastic_oscillator", "williams_r", "trend_5_20",
    "trend_1_3", "chaikin_money_flow", "average_true_range", "trend_20_60",
    "chande_momentum_oscillator", "relative_strength_index",
    "commodity_channel_index", "aroon", "macd", "trix", "know_sure_thing",
]
pos_names = ["volume", "on_balance_volume", "ease_of_movement", "volatility_60", "volatility_20"]

print("Precomputing signals...")
neg_sigs = {}
for name in all_neg_names:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=False)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    neg_sigs[name] = sig

pos_sigs = {}
for name in pos_names:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=True)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    pos_sigs[name] = sig


def test(factor_sigs, weights, span, verbose=False):
    portfolio = sum(s * w for s, w in zip(factor_sigs, weights))
    portfolio = portfolio.where(univ_bool, np.nan)
    portfolio = portfolio.ewm(span=span, min_periods=1, ignore_na=False).mean()
    portfolio = portfolio.where(univ_bool, np.nan)
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    sum_pos = portfolio[portfolio > 0].sum(axis=1)
    sum_neg = portfolio[portfolio < 0].sum(axis=1)
    df_pos = portfolio[portfolio > 0].div(sum_pos, axis=0) * 0.5
    df_neg = portfolio[portfolio < 0].div(sum_neg.abs(), axis=0) * 0.5
    final = df_pos.fillna(0) + df_neg.fillna(0)
    
    net_sharpe, pnl = utils.backtest_portfolio(final.fillna(0), rets, univ_bool, False, verbose)
    return net_sharpe

# ===== Path 1: Drop stochastic_oscillator (williams_r is redundant with it) =====
print("\n=== DROP STOCHASTIC+WILLIAMS FROM 9 (they're identical IC) ===")
base_7 = ["accumulation_distribution_index", "ichimoku", "ultimate_oscillator",
           "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"]

for span in [2, 3, 4, 5, 6]:
    s = test([neg_sigs[f] for f in base_7], [1.0]*7, span)
    print(f"  7 neg (no stoch/will), span={span} => {s}")

# Drop one from 9 (keep williams_r, drop stochastic)
base_8_no_stoch = ["accumulation_distribution_index", "ichimoku", "ultimate_oscillator",
                    "williams_r", "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"]
print("\n=== 8 NEG (drop stochastic only) ===")
for span in [2, 3, 4, 5, 6]:
    s = test([neg_sigs[f] for f in base_8_no_stoch], [1.0]*8, span)
    print(f"  8 neg (no stoch), span={span} => {s}")

# Drop one from 9 (keep stochastic, drop williams_r)
base_8_no_will = ["accumulation_distribution_index", "ichimoku", "ultimate_oscillator",
                   "stochastic_oscillator", "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"]
print("\n=== 8 NEG (drop williams only) ===")
for span in [2, 3, 4, 5, 6]:
    s = test([neg_sigs[f] for f in base_8_no_will], [1.0]*8, span)
    print(f"  8 neg (no williams), span={span} => {s}")

# Drop ultimate_oscillator too (it was 2.056)
base_7_no_ult_stoch = ["accumulation_distribution_index", "ichimoku",
                        "williams_r", "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"]
print("\n=== 7 NEG (drop stoch+ult) ===")
for span in [2, 3, 4, 5, 6]:
    s = test([neg_sigs[f] for f in base_7_no_ult_stoch], [1.0]*7, span)
    print(f"  7 neg (no stoch, no ult), span={span} => {s}")

# ===== Path 2: Best 8 + volume/OBV =====
print("\n=== 8 NEG (no stoch) + VOLUME ===")
for vw in [0.3, 0.5, 0.7, 1.0, 1.5]:
    sigs = [neg_sigs[f] for f in base_8_no_stoch] + [pos_sigs["volume"]]
    wts = [1.0]*8 + [vw]
    for span in [3, 4, 5]:
        s = test(sigs, wts, span)
        print(f"  8neg + vol(w={vw}), span={span} => {s}")

print("\n=== 8 NEG (no stoch) + OBV ===")
for vw in [0.3, 0.5, 0.7, 1.0]:
    sigs = [neg_sigs[f] for f in base_8_no_stoch] + [pos_sigs["on_balance_volume"]]
    wts = [1.0]*8 + [vw]
    for span in [3, 4, 5]:
        s = test(sigs, wts, span)
        print(f"  8neg + obv(w={vw}), span={span} => {s}")

print("\n=== 8 NEG (no stoch) + VOLUME + OBV ===")
for vw in [0.5, 0.7, 1.0]:
    sigs = [neg_sigs[f] for f in base_8_no_stoch] + [pos_sigs["volume"], pos_sigs["on_balance_volume"]]
    wts = [1.0]*8 + [vw, vw]
    for span in [3, 4, 5]:
        s = test(sigs, wts, span)
        print(f"  8neg + vol+obv(w={vw}), span={span} => {s}")

print("\n=== 8 NEG (no stoch) + VOLUME + EOM ===")
for vw in [0.5, 0.7, 1.0]:
    sigs = [neg_sigs[f] for f in base_8_no_stoch] + [pos_sigs["volume"], pos_sigs["ease_of_movement"]]
    wts = [1.0]*8 + [vw, vw]
    for span in [3, 4, 5]:
        s = test(sigs, wts, span)
        print(f"  8neg + vol+eom(w={vw}), span={span} => {s}")

# ===== Path 3: 7 neg (no stoch, no ult) + volume combos =====
print("\n=== 7 NEG (no stoch, no ult) + VOLUME ===")
for vw in [0.5, 0.7, 1.0, 1.5]:
    sigs = [neg_sigs[f] for f in base_7_no_ult_stoch] + [pos_sigs["volume"]]
    wts = [1.0]*7 + [vw]
    for span in [3, 4, 5]:
        s = test(sigs, wts, span)
        print(f"  7neg + vol(w={vw}), span={span} => {s}")

# ====== FINAL: print the best candidate with full detail ======
print("\n\n===== FINAL CANDIDATE DETAILED OUTPUT =====")
# Best candidate: 8 neg (no stoch), equal, span=4
s = test([neg_sigs[f] for f in base_8_no_stoch], [1.0]*8, 4, verbose=True)
print(f"  8 neg (no stoch), equal, span=4 => NET SHARPE: {s}")

print("\nDone!")
