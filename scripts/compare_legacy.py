"""
Старый и исправленный run_strategies_real_odds.py на одних и тех же исходах.

Старая версия берётся из git (коммит до исправлений), исходы — первые 4000
симуляций из simulation-current/bet_outcomes_real_odds.npz (ROI 7%).

  python scripts/compare_legacy.py   # -> reports/legacy_before_after.json
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUR = os.path.join(ROOT, 'simulation-current')
OLD_REV = '1a34258'

CASES = [
    ('dynamic_percentage 1.5%', 'dynamic_percentage_strategy_with_real_odds', {'bet_size_pct': 1.5}, False),
    ('dynamic_percentage 1.5% + вариация', 'dynamic_percentage_strategy_with_real_odds', {'bet_size_pct': 1.5}, True),
    ('dynamic_percentage 5% + вариация', 'dynamic_percentage_strategy_with_real_odds', {'bet_size_pct': 5.0}, True),
    ('kelly_criterion risk=2', 'kelly_criterion_strategy_with_real_odds', {'risk': 2.0, 'kelly_fraction': 1.0}, False),
    ('fixed_fraction 2%', 'fixed_fraction_strategy_with_real_odds', {'fixed_percent': 2.0}, False),
    ('linear_roi (по умолчанию)', 'linear_roi_strategy_with_real_odds', {}, False),
    ('proportional_kelly', 'proportional_kelly_strategy_with_real_odds', {}, False),
    ('dynamic_kelly', 'dynamic_kelly_strategy_with_real_odds', {}, False),
    ('target_based 3%→1%', 'target_based_strategy_with_real_odds', {}, False),
    ('volatility_adjusted', 'volatility_adjusted_strategy_with_real_odds', {}, False),
]


def summarize(bank, bets):
    before = bank[:, :-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(before > 0, bets / before * 100, np.nan)
    profit = (bank[:, -1] / 1000 - 1) * 100
    peaks = np.maximum.accumulate(np.maximum(bank, 1e-9), axis=1)
    dd = ((peaks - bank) / peaks).max(axis=1) * 100
    return {'mean_profit': float(profit.mean()), 'median_profit': float(np.median(profit)),
            'min_bank': float(bank.min()), 'p_negative_bank': float((bank.min(axis=1) < 0).mean() * 100),
            'p_dd50': float((dd >= 50).mean() * 100),
            'stake_pct_live_median': float(np.nanmedian(pct[:, 500])),
            'stake_pct_live_spread': float(np.nanpercentile(pct[:, 500], 95) - np.nanpercentile(pct[:, 500], 5))}


def main():
    sys.path.insert(0, CUR)
    os.chdir(CUR)
    src = subprocess.run(['git', 'show', f'{OLD_REV}:simulation-current/run_strategies_real_odds.py'],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location('rs_old', f.name)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    import run_strategies_real_odds as new

    data = np.load('bet_outcomes_real_odds.npz')
    outcomes, odds = data['outcomes'][:4000], data['odds']
    rows = []
    for label, fn, kw, var in CASES:
        res = {}
        for tag, mod in (('old', old), ('new', new)):
            with contextlib.redirect_stdout(io.StringIO()):
                bank, bets, *_ = getattr(mod, fn)(outcomes, odds, apply_variation=var, **kw)
            res[tag] = summarize(bank, bets)
        rows.append({'case': label, **res})
        o, n = res['old'], res['new']
        print(f"{label:36s} было: медиана {o['median_profit']:8.1f}%, мин. банк {o['min_bank']:11.1f} | "
              f"стало: медиана {n['median_profit']:8.1f}%, мин. банк {n['min_bank']:7.1f}")
    out = os.path.join(ROOT, 'reports', 'legacy_before_after.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print('->', out)


if __name__ == '__main__':
    main()
