"""
Final deep dive on the best path: 7 neg (no stoch, no ult) + volume combinations.
Also explore adding OBV, EOM, and other combos.
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

# Best 7 neg factors (no stochastic, no williams_r, no ultimate_oscillator)
BEST_7 = ["accumulation_distribution_index", "ichimoku",
           "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range",
           "williams_r"]  # Note: keep williams_r, drop stoch+ult  

# Actually from the results, "7 neg (no stoch, no ult)" means dropping stoch AND ultimate from the 9:
# Original 9 = [ADI, ichimoku, ultimate_osc, stoch_osc, williams_r, trend_5_20, trend_1_3, chaikin_money_flow, avg_true_range]
# No stoch, no ult = [ADI, ichimoku, williams_r, trend_5_20, trend_1_3, chaikin_money_flow, avg_true_range]
BEST_7_V2 = ["accumulation_distribution_index", "ichimoku",
              "williams_r", "trend_5_20", "trend_1_3", 
              "chaikin_money_flow", "average_true_range"]

# Also "7 neg (no stoch/will)" was 2.227 which is:
# [ADI, ichimoku, ultimate_osc, trend_5_20, trend_1_3, chaikin_money_flow, avg_true_range]
BEST_7_V1 = ["accumulation_distribution_index", "ichimoku", "ultimate_oscillator",
              "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"]

pos_names = ["volume", "on_balance_volume", "ease_of_movement", "volatility_60", "volatility_20"]

print("Precomputing signals...")
all_names = list(set(BEST_7_V1 + BEST_7_V2 + pos_names))
sigs = {}
for name in all_names:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    # Determine direction
    if name in pos_names:
        sig = sig.rank(axis=1, pct=True, ascending=True)
    else:
        sig = sig.rank(axis=1, pct=True, ascending=False)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    sigs[name] = sig


def test(fac_names, weights, span, verbose=False):
    portfolio = sum(sigs[f] * w for f, w in zip(fac_names, weights))
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


# ===== V2: 7 neg (no stoch, no ult) + volume fine-tune =====
print("\n=== V2 (no stoch, no ult) + volume fine-tune ===")
for vw in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
    facs = BEST_7_V2 + ["volume"]
    wts = [1.0]*7 + [vw]
    for span in [4, 5, 6, 7]:
        s = test(facs, wts, span)
        print(f"  V2 + vol(w={vw}), span={span} => {s}")

# ===== V2 + volume + OBV =====
print("\n=== V2 + volume + OBV ===")
for vw in [0.7, 1.0, 1.3]:
    for obv_w in [0.3, 0.5, 0.7, 1.0]:
        facs = BEST_7_V2 + ["volume", "on_balance_volume"]
        wts = [1.0]*7 + [vw, obv_w]
        for span in [4, 5, 6]:
            s = test(facs, wts, span)
            print(f"  V2 + vol({vw}) + obv({obv_w}), span={span} => {s}")

# ===== V2 + volume + EOM =====            
print("\n=== V2 + volume + EOM ===")
for vw in [0.7, 1.0, 1.3]:
    for eom_w in [0.3, 0.5, 0.7, 1.0]:
        facs = BEST_7_V2 + ["volume", "ease_of_movement"]
        wts = [1.0]*7 + [vw, eom_w]
        for span in [4, 5, 6]:
            s = test(facs, wts, span)
            print(f"  V2 + vol({vw}) + eom({eom_w}), span={span} => {s}")

# ===== V2 + volume + OBV + EOM =====
print("\n=== V2 + volume + OBV + EOM ===")
for vw in [0.7, 1.0]:
    facs = BEST_7_V2 + ["volume", "on_balance_volume", "ease_of_movement"]
    wts = [1.0]*7 + [vw, vw, vw]
    for span in [4, 5, 6]:
        s = test(facs, wts, span)
        print(f"  V2 + vol+obv+eom(w={vw}), span={span} => {s}")

# ===== V1 comparison (no stoch/will but with ult) + volume =====
print("\n=== V1 (with ult, no stoch/will) + volume fine-tune ===")
for vw in [0.7, 1.0, 1.3]:
    facs = BEST_7_V1 + ["volume"]
    wts = [1.0]*7 + [vw]
    for span in [4, 5, 6]:
        s = test(facs, wts, span)
        print(f"  V1 + vol(w={vw}), span={span} => {s}")

# ===== Drop-one from V2 to see if 6 factors is even better =====
print("\n=== DROP-ONE FROM V2 ===")
for i, dropped in enumerate(BEST_7_V2):
    facs = [f for j, f in enumerate(BEST_7_V2) if j != i]
    for span in [4, 5, 6]:
        s = test(facs, [1.0]*6, span)
        print(f"  Drop {dropped:40s}, span={span} => {s}")

# Print best candidate with detail
print("\n\n===== DETAILED: V2 + vol(1.0), span=5 =====")
facs = BEST_7_V2 + ["volume"]
wts = [1.0]*7 + [1.0]
s = test(facs, wts, 5, verbose=True)
print(f"  => NET SHARPE: {s}")

print("\nDone!")
