"""
Final: fine-tune around 6neg + vol/obv/eom and check robustness.
6 neg = [ADI, ichimoku, trend_5_20, trend_1_3, chaikin_money_flow, average_true_range]
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

NEG_6 = ["accumulation_distribution_index", "ichimoku",
         "trend_5_20", "trend_1_3", "chaikin_money_flow", "average_true_range"]
POS_3 = ["volume", "on_balance_volume", "ease_of_movement"]

print("Precomputing signals...")
sigs = {}
for name in NEG_6:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=False)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    sigs[name] = sig
for name in POS_3:
    sig = features[name].shift(1).loc[start:end]
    sig = sig.where(univ_bool, np.nan)
    sig = sig.rank(axis=1, pct=True, ascending=True)
    sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
    sigs[name] = sig


def build_portfolio(fac_names, weights, span, date_range=None):
    if date_range:
        s, e = date_range
        ub = universe.loc[s:e].astype(bool)
    else:
        ub = univ_bool
    
    portfolio = 0
    for f, w in zip(fac_names, weights):
        if date_range:
            sig = features[f].shift(1).loc[s:e]
            sig = sig.where(ub, np.nan)
            if f in POS_3:
                sig = sig.rank(axis=1, pct=True, ascending=True)
            else:
                sig = sig.rank(axis=1, pct=True, ascending=False)
            sig = sig.sub(sig.mean(axis=1), axis=0).fillna(0)
            portfolio += sig * w
        else:
            portfolio += sigs[f] * w
    
    portfolio = portfolio.where(ub, np.nan)
    portfolio = portfolio.ewm(span=span, min_periods=1, ignore_na=False).mean()
    portfolio = portfolio.where(ub, np.nan)
    portfolio = portfolio.sub(portfolio.mean(axis=1), axis=0)
    
    sp = portfolio[portfolio > 0].sum(axis=1)
    sn = portfolio[portfolio < 0].sum(axis=1)
    dp = portfolio[portfolio > 0].div(sp, axis=0) * 0.5
    dn = portfolio[portfolio < 0].div(sn.abs(), axis=0) * 0.5
    return (dp.fillna(0) + dn.fillna(0)).fillna(0)


def test(fac_names, weights, span, verbose=False):
    final = build_portfolio(fac_names, weights, span)
    net_sharpe, _ = utils.backtest_portfolio(final, rets, univ_bool, False, verbose)
    return net_sharpe


# ===== Fine-tune pos weights individually =====
print("\n=== FINE-TUNE: 6neg + vol/obv/eom individual weights ===")
best_s = 0
best_cfg = ""
for vw in [0.4, 0.5, 0.6, 0.7, 0.8]:
    for ow in [0.3, 0.5, 0.7, 0.9]:
        for ew in [0.3, 0.5, 0.7, 0.9]:
            facs = NEG_6 + POS_3
            wts = [1.0]*6 + [vw, ow, ew]
            for span in [4, 5, 6]:
                s = test(facs, wts, span)
                if s > best_s:
                    best_s = s
                    best_cfg = f"vol={vw}, obv={ow}, eom={ew}, span={span}"
                    print(f"  NEW BEST: {s} ({best_cfg})")

print(f"\nOverall Best: {best_s} ({best_cfg})")

# ===== Sub-period robustness =====
print("\n\n=== SUB-PERIOD ROBUSTNESS ===")
# Use best config: w=0.7, span=5
facs = NEG_6 + POS_3
wts = [1.0]*6 + [0.7, 0.7, 0.7]
span = 5

final = build_portfolio(facs, wts, span)
for period_name, s, e in [
    ("FULL 2005-2019", "2005-01-03", "2019-12-31"),
    ("2005-2009", "2005-01-03", "2009-12-31"),
    ("2010-2014", "2010-01-04", "2014-12-31"),
    ("2015-2019", "2015-01-02", "2019-12-31"),
    ("2005-2012", "2005-01-03", "2012-12-31"),
    ("2013-2019", "2013-01-02", "2019-12-31"),
]:
    try:
        idx = final.loc[s:e].index.intersection(rets.loc[s:e].index)
        if len(idx) > 50:
            print(f"\n--- {period_name} ---")
            utils.backtest_portfolio(final.loc[idx], rets.loc[idx], universe.loc[idx], False, True)
    except Exception as ex:
        print(f"  {period_name}: {ex}")

print("\n\nDone!")
