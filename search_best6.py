"""
Final fine-tune around the best: 7 neg (V2) + volume + OBV + EOM.
Also drop-one from V2 was telling: dropping williams_r gives 2.42+ and 
dropping chaikin_money_flow also helps. Let's explore.
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

BEST_7_V2 = ["accumulation_distribution_index", "ichimoku",
              "williams_r", "trend_5_20", "trend_1_3", 
              "chaikin_money_flow", "average_true_range"]

pos_names = ["volume", "on_balance_volume", "ease_of_movement", "volatility_60", "volatility_20"]

print("Precomputing signals...")
neg_names = BEST_7_V2 + ["ultimate_oscillator", "stochastic_oscillator", "trend_20_60",
                          "chande_momentum_oscillator", "relative_strength_index",
                          "commodity_channel_index", "aroon"]
all_names = list(set(neg_names + pos_names))
sigs = {}
for name in all_names:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
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
    
    sp = portfolio[portfolio > 0].sum(axis=1)
    sn = portfolio[portfolio < 0].sum(axis=1)
    dp = portfolio[portfolio > 0].div(sp, axis=0) * 0.5
    dn = portfolio[portfolio < 0].div(sn.abs(), axis=0) * 0.5
    final = dp.fillna(0) + dn.fillna(0)
    
    net_sharpe, pnl = utils.backtest_portfolio(final.fillna(0), rets, univ_bool, False, verbose)
    return net_sharpe


# ===== Fine-tune weights for V2 + vol + obv + eom =====
print("\n=== V2 + vol + obv + eom: weight fine-tune ===")
for w in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
    facs = BEST_7_V2 + ["volume", "on_balance_volume", "ease_of_movement"]
    wts = [1.0]*7 + [w, w, w]
    for span in [3, 4, 5]:
        s = test(facs, wts, span)
        print(f"  w={w}, span={span} => {s}")

# ===== Try different pos factor weights =====
print("\n=== V2 + vol/obv/eom with different individual weights ===")
for vw in [0.7, 1.0, 1.3]:
    for ow in [0.5, 0.7, 1.0]:
        for ew in [0.7, 1.0, 1.3]:
            facs = BEST_7_V2 + ["volume", "on_balance_volume", "ease_of_movement"]
            wts = [1.0]*7 + [vw, ow, ew]
            s = test(facs, wts, 4)
            print(f"  vol={vw}, obv={ow}, eom={ew}, span=4 => {s}")

# ===== 6 neg (drop williams_r from V2) + vol + obv + eom =====
print("\n=== 6 NEG (V2 minus williams_r) + vol + obv + eom ===")
V2_NO_WILL = [f for f in BEST_7_V2 if f != "williams_r"]
for w in [0.5, 0.7, 1.0, 1.3]:
    facs = V2_NO_WILL + ["volume", "on_balance_volume", "ease_of_movement"]
    wts = [1.0]*6 + [w, w, w]
    for span in [4, 5, 6]:
        s = test(facs, wts, span)
        print(f"  6neg + pos(w={w}), span={span} => {s}")

# ===== 5 neg (drop williams_r AND chaikin_money_flow) + vol + obv + eom =====
print("\n=== 5 NEG (drop will+cmf) + vol + obv + eom ===")
V2_NO_WILL_CMF = [f for f in BEST_7_V2 if f not in ["williams_r", "chaikin_money_flow"]]
for w in [0.7, 1.0, 1.3]:
    facs = V2_NO_WILL_CMF + ["volume", "on_balance_volume", "ease_of_movement"]
    wts = [1.0]*5 + [w, w, w]
    for span in [4, 5, 6, 7]:
        s = test(facs, wts, span)
        print(f"  5neg + pos(w={w}), span={span} => {s}")

# ===== Add volatility factors too? =====
print("\n=== V2 + ALL 5 POS FACTORS ===")
for w in [0.5, 0.7, 1.0]:
    facs = BEST_7_V2 + pos_names
    wts = [1.0]*7 + [w]*5
    for span in [4, 5, 6]:
        s = test(facs, wts, span)
        print(f"  V2 + all5pos(w={w}), span={span} => {s}")

# FINAL best candidate
print("\n\n===== BEST CANDIDATE DETAILED =====")
facs = BEST_7_V2 + ["volume", "on_balance_volume", "ease_of_movement"]
wts = [1.0]*7 + [1.0, 1.0, 1.0]
s = test(facs, wts, 4, verbose=True)
print(f"  V2 + vol+obv+eom(1.0), span=4 => NET SHARPE: {s}")

print("\nDone!")
