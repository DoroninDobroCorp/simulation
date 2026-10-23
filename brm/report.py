"""
Исследования на корректном движке -> reports/results.json и reports/RESULTS.md.

  python -m brm.report            # полный прогон (~15–20 минут)
  python -m brm.report --quick    # быстрый прогон для проверки (~1 минута)

1. Горизонт: сколько ставок нужно, чтобы ROI 2.5% перестал быть лотереей.
2. Сравнение стратегий при РАВНОМ риске (калибровка размера под профиль).
3. Доля Келли и ошибка модели EV: что будет, если модель завышает перевес.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace

import numpy as np

from . import strategies as S
from .data import REPO_ROOT, load_real_bets
from .engine import SimConfig, make_market, run
from .metrics import summarize
from .optimize import RISK_PROFILES, leaderboard

OUT_DIR = os.path.join(REPO_ROOT, 'reports')

KEEP = ('median_profit_pct', 'mean_profit_pct', 'p5_profit_pct', 'p95_profit_pct', 'p_loss_pct',
        'median_max_dd_pct', 'p_dd50_pct', 'p_dd80_pct', 'avg_stake_pct', 'roi_turnover_pct',
        'median_growth_per_bet_pct')


def slim(m: dict) -> dict:
    return {k: round(float(m[k]), 3) for k in KEEP if k in m}


def unit_return_std(pool, roi):
    """Стандартное отклонение дохода единичной ставки при p = (1+roi)/кэф."""
    o = pool.odds
    p = np.clip((1 + roi) / o, 0, 1)
    second = np.mean(p * (o - 1) ** 2 + (1 - p))
    return math.sqrt(second - roi ** 2)


def horizon_study(pool, cfg, roi):
    sd = unit_return_std(pool, roi)
    rows = []
    for n in (250, 500, 1000, 2000, 5000, 10000):
        analytic = 0.5 * math.erfc(roi * math.sqrt(n) / sd / math.sqrt(2))
        # флэт 0.1% начального банка: на любой дистанции до банкротства далеко, P(минус) = P(ROI < 0)
        m = summarize(run(S.FlatStake(0.1), make_market(pool, roi=roi), replace(cfg, n_bets=n)))
        rows.append({'n_bets': n, 'p_roi_negative_analytic_pct': round(analytic * 100, 2),
                     'p_loss_flat_mc_pct': round(m['p_loss_pct'], 2),
                     'roi_sd_pp': round(sd / math.sqrt(n) * 100, 2)})
        print(f"  {n:6d} ставок: P(ROI<0) = {analytic * 100:5.1f}% (аналитика), {m['p_loss_pct']:5.1f}% (MC), "
              f"σ ROI = {sd / math.sqrt(n) * 100:.2f} п.п.")
    return {'unit_return_sd': round(sd, 4), 'rows': rows}


def risk_matched(pool, cfg, roi, profiles):
    markets = {
        'flat': make_market(pool, roi=roi, ev_model='flat'),
        'scaled_calibrated': make_market(pool, roi=roi, ev_model='scaled', calibrated=True),
    }
    out = {}
    for mname, market in markets.items():
        out[mname] = {'description': market.description, 'profiles': {}}
        for prof in profiles:
            print(f"\n[{mname}] профиль {prof}: {RISK_PROFILES[prof][0]} <= {RISK_PROFILES[prof][1]}%")
            rows = leaderboard(market, cfg, prof)
            out[mname]['profiles'][prof] = [dict(slim(r), key=r['key'], scale=round(r['scale'], 4),
                                                 scale_max=round(r['scale_max'], 4), label=r['label'])
                                            for r in rows]
    return out


def kelly_model_error(pool, cfg, roi):
    mults = (0.1, 0.25, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    cases = {
        'calibrated': make_market(pool, roi=roi, ev_model='scaled', calibrated=True),
        'overstated': make_market(pool, roi=roi, ev_model='scaled', calibrated=False),
        'overstated_noise': make_market(pool, roi=roi, ev_model='scaled', calibrated=False, noise=0.05),
    }
    out = {}
    for name, market in cases.items():
        rows = []
        for k in mults:
            m = summarize(run(S.Kelly(k), market, replace(cfg, cap=1.0)))
            rows.append(dict(slim(m), mult=k))
        best = max(rows, key=lambda r: r['median_profit_pct'])
        print(f"  {name:17s}: лучший множитель Келли по медиане = {best['mult']}, "
              f"медиана {best['median_profit_pct']:.1f}%, P(DD≥50) {best['p_dd50_pct']:.1f}%")
        out[name] = {'description': market.description, 'rows': rows}
    return out


def write_markdown(res, path):
    L = []
    a = L.append
    s = res['settings']
    a('# Результаты исследований (корректный движок brm)\n')
    a(f"Сгенерировано `python -m brm.report`: {s['n_sims']} симуляций × {s['n_bets']} ставок, "
      f"кэфы бутстрепом из {s['pool_size']} реальных ставок (средний {s['avg_odds']:.2f}), "
      f"истинный средний ROI {s['roi'] * 100:.1f}%, кап ставки {s['cap'] * 100:.0f}% банка (в разделе 3 — без капа), seed {s['seed']}.\n")

    h = res['horizon']
    a('## 1. Горизонт: ROI 2.5% — это много шума\n')
    a(f"Флэт-ставка 0.1% начального банка, {min(s['n_sims'] * 4, 20000)} симуляций на каждую дистанцию.\n")
    a(f"Стандартное отклонение дохода одной единичной ставки ≈ {h['unit_return_sd']:.2f}, поэтому "
      'фактический ROI на коротких дистанциях пляшет сильнее, чем сам перевес. '
      'Никакой банкролл-менеджмент это не исправит: при флэт-ставке вероятность закончить в минусе — это просто P(ROI < 0).\n')
    a('| Ставок | σ ROI, п.п. | P(ROI < 0), аналитика | P(минус), Monte Carlo |')
    a('|---:|---:|---:|---:|')
    for r in h['rows']:
        a(f"| {r['n_bets']} | {r['roi_sd_pp']:.1f} | {r['p_roi_negative_analytic_pct']:.1f}% | {r['p_loss_flat_mc_pct']:.1f}% |")
    a('')

    a('## 2. Сравнение стратегий при равном риске\n')
    a('Размер ставки каждой стратегии умножен на коэффициент `scale`: сначала бисекцией найдена граница, где '
      'стратегия упирается в ограничение профиля, затем внутри допустимой области выбран размер с лучшей медианой '
      '(если лимит не держит, оптимум ищется по сетке от границы вниз и уточняется золотым сечением). '
      'Все прогоны на одних и тех же ставках и исходах. У стратегий с порогами (защита от просадки, динамический %) '
      'медиана между соседними размерами дрожит на несколько десятых п.п., поэтому разницу меньше ~1 п.п. считайте ничьей. '
      'Сортировка по медианной прибыли — по среднему сортировать нельзя: его делают несколько сверхудачных симуляций.\n')
    names = {'flat': 'Все ставки с одинаковым EV (как в simulation-current)',
             'scaled_calibrated': 'EV каждой ставки из данных, отмасштабирован к среднему ROI (стратегия знает истинный EV)'}
    for mname, block in res['risk_matched'].items():
        for prof, rows in block['profiles'].items():
            metric, limit = RISK_PROFILES[prof]
            a(f"### {names[mname]} — профиль `{prof}` ({metric.replace('p_dd', 'P(DD≥').replace('_pct', '%)')} ≤ {limit}%)\n")
            a('| # | Стратегия | scale (граница) | Медиана | Среднее | P5 | P(минус) | Медиана DD | P(DD≥50) | P(DD≥80) | Ср. ставка |')
            a('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for i, r in enumerate(rows, 1):
                a(f"| {i} | `{r['key']}` — {r['label']} | {r['scale']:.2f} ({r['scale_max']:.2f}) | {r['median_profit_pct']:+.1f}% | "
                  f"{r['mean_profit_pct']:+.1f}% | {r['p5_profit_pct']:+.1f}% | {r['p_loss_pct']:.1f}% | "
                  f"{r['median_max_dd_pct']:.1f}% | {r['p_dd50_pct']:.1f}% | {r['p_dd80_pct']:.1f}% | {r['avg_stake_pct']:.2f}% |")
            a('')

    a('## 3. Доля Келли и ошибка модели EV\n')
    a('Ставка = множитель × EV / (кэф − 1), здесь без капа 10% (cap = 100%), чтобы было видно и перебор. Три случая: стратегия знает истинный EV; '
      'модель завышает EV (истинный = k × оценка, как если бы средний ROI модели 5.1% был на деле 2.5%); '
      'то же плюс шум модели σ = 5% на каждую ставку.\n')
    cases = list(res['kelly_model_error'])
    a('| Множитель | ' + ' | '.join(f'{c}: медиана / P(DD≥50)' for c in cases) + ' |')
    a('|---:|' + '---:|' * len(cases))
    for i, mult in enumerate(r['mult'] for r in res['kelly_model_error'][cases[0]]['rows']):
        cells = []
        for c in cases:
            r = res['kelly_model_error'][c]['rows'][i]
            cells.append(f"{r['median_profit_pct']:+.1f}% / {r['p_dd50_pct']:.1f}%")
        a(f"| {mult} | " + ' | '.join(cells) + ' |')
    a('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--roi', type=float, default=0.025)
    ap.add_argument('--sims', type=int, default=None)
    args = ap.parse_args()
    n_sims = args.sims or (600 if args.quick else 2500)
    cfg = SimConfig(n_sims=n_sims, n_bets=1000, cap=0.10, seed=2026, quantile_every=1000)
    pool = load_real_bets()
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    res = {'settings': {'n_sims': n_sims, 'n_bets': cfg.n_bets, 'roi': args.roi, 'cap': cfg.cap,
                        'seed': cfg.seed, 'pool_size': len(pool), 'avg_odds': float(pool.odds.mean())}}
    print('1. Горизонт')
    res['horizon'] = horizon_study(pool, replace(cfg, n_sims=min(n_sims * 4, 20000)), args.roi)
    print('\n2. Сравнение при равном риске')
    res['risk_matched'] = risk_matched(pool, cfg, args.roi, ('conservative', 'cautious', 'balanced', 'risky', 'extreme'))
    print('\n3. Келли и ошибка модели')
    res['kelly_model_error'] = kelly_model_error(pool, cfg, args.roi)
    res['settings']['seconds'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_markdown(res, os.path.join(OUT_DIR, 'RESULTS.md'))
    print(f"\nГотово за {res['settings']['seconds']} с: reports/RESULTS.md, reports/results.json")


if __name__ == '__main__':
    main()
