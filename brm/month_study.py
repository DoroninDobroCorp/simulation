"""
Оптимальный размер ставки на месяц под конкретные условия.

Условия по умолчанию: ROI 5% у каждой ставки, кэф от 1.5 до 4 (средний 2.75,
разброс как у реальных ставок), 1000 ставок = месяц, не больше 10% банка на ставку.

Задачи (максимум медианной прибыли месяца при ограничении на просадку от пика):
  aggressive    — просадка ≥ 50% не чаще чем в 1 месяце из 10;
  moderate      — просадка ≥ 30% не чаще чем в 1 месяце из 10;
  conservative  — большой банк: просадка ≥ 20% не чаще чем в 1 месяце из 20
                  и месяц хуже −10% не чаще чем 1 из 20.
Отдельно — потолок: доля Келли с максимальной медианой без ограничений.

Для каждой стратегии строится кривая «размер -> рост/риск» на общих случайных
числах, на ней ищется лучший размер под задачу, затем найденные настройки
перепроверяются на независимом seed с большим числом симуляций.

  python -m brm.month_study            # ~10 минут -> reports/MONTH_ROI5.md, month_roi5.json
  python -m brm.month_study --quick    # быстрый прогон
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
from .data import REPO_ROOT, beta_odds_pool, real_odds_in_range
from .engine import Market, SimConfig, make_market, run
from .metrics import summarize
from .optimize import calibrate

OUT_DIR = os.path.join(REPO_ROOT, 'reports')

ODDS_MIN, ODDS_MAX, ODDS_MEAN, ROI = 1.5, 4.0, 2.75, 0.05
# концентрация Beta подобрана под разброс реальных кэфов в этом диапазоне (σ ≈ 0.65)
CONCENTRATION = 3.0

# key -> (название, build(scale), scale_lo, scale_hi, что означает scale)
FAMILIES = {
    'kelly': ('Келли', lambda s: S.Kelly(s), 0.05, 3.0, 'доля от полного Келли'),
    'fixed': ('Фиксированный % банка', lambda s: S.FixedFraction(s), 0.1, 10.0, '% текущего банка'),
    'flat': ('Флэт от начального банка', lambda s: S.FlatStake(s), 0.1, 10.0, '% начального банка'),
    'log_formula': ('Лог-формула (backend)', lambda s: S.LogFormula(risk=s), 1.5, 60.0, 'risk (меньше = агрессивнее)'),
    # пол = 1 - допустимая глубина просадки задачи, подставляется в solve()
    'floor_kelly': ('Келли с полом просадки', None, 0.1, 2.0, 'доля Келли от подушки над полом'),
    'kelly_guard': ('Келли с урезанием на просадке', lambda s: S.DrawdownGuard(S.Kelly(s), ((0.10, 0.5), (0.20, 0.25))),
                    0.05, 4.0, 'доля Келли до просадки 10%'),
    'dynamic_percentage': ('Динамический %', lambda s: S.DynamicPercentage(s), 0.1, 8.0, 'базовый %'),
    'anti_martingale': ('Анти-мартингейл', lambda s: S.AntiMartingale(base_percent=s), 0.1, 6.0, 'базовый %'),
}

TASKS = {
    'aggressive': ('Агрессивная', ('p_dd50_pct', 10.0), 'просадка на 50% от пика не чаще 1 месяца из 10'),
    'moderate': ('Умеренная', ('p_dd30_pct', 10.0), 'просадка на 30% от пика не чаще 1 месяца из 10'),
    'conservative': ('Консервативная, большой банк', (('p_dd20_pct', 5.0), ('p_profit_below_m10_pct', 5.0)),
                     'просадка на 20% от пика не чаще 1 месяца из 20 и месяц хуже −10% не чаще 1 из 20'),
}
FRONTIER_KEYS = ('kelly', 'fixed', 'flat', 'log_formula')
# при почти равной медиане рекомендуем стратегию выше по списку: она проще и обоснованнее
PREFERENCE = ('kelly', 'log_formula', 'floor_kelly', 'fixed', 'flat', 'kelly_guard', 'dynamic_percentage', 'anti_martingale')
TIE_PP = 1.5

KEEP = ('median_profit_pct', 'mean_profit_pct', 'p_profit_below_m10_pct', 'p1_profit_pct', 'p5_profit_pct', 'p25_profit_pct', 'p75_profit_pct',
        'p95_profit_pct', 'p_loss_pct', 'median_max_dd_pct', 'p95_max_dd_pct', 'p_dd10_pct', 'p_dd15_pct', 'p_dd20_pct',
        'p_dd25_pct', 'p_dd30_pct', 'p_dd50_pct', 'p_dd80_pct', 'avg_stake_pct', 'roi_turnover_pct',
        'median_growth_per_bet_pct', 'p_ruin_pct')
STAKE_ODDS = (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0)


def slim(m):
    return {k: round(float(m[k]), 3) for k in KEEP if k in m}


def is_inverse(key):
    return key == 'log_formula'      # у лог-формулы ставка обратна параметру risk


def rules_of(task):
    c = TASKS[task][1]
    return list(c) if isinstance(c[0], tuple) else [c]


def floor_for(task):
    metric = rules_of(task)[0][0]
    return 1.0 - int(metric[len('p_dd'):-len('_pct')]) / 100.0


def build_for(key, scale, task=None):
    """Сборка стратегии; для лог-формулы scale = 1/risk, чтобы рост scale всегда означал большую ставку."""
    if key == 'floor_kelly':
        return S.FloorKelly(scale, floor_for(task) if task else 0.5)
    _, build, *_ = FAMILIES[key]
    return build(1.0 / scale) if is_inverse(key) else build(scale)


def scale_range(key):
    _, _, lo, hi, _ = FAMILIES[key]
    return (1.0 / hi, 1.0 / lo) if is_inverse(key) else (lo, hi)


def param_value(key, scale):
    return 1.0 / scale if is_inverse(key) else scale


def evaluate(key, scale, market, cfg, task=None):
    return summarize(run(build_for(key, scale, task), market, cfg))


def frontier(key, market, cfg, points):
    lo, hi = scale_range(key)
    grid = np.geomspace(lo, hi, points)
    return [dict(slim(evaluate(key, s, market, cfg)), scale=float(s), param=float(param_value(key, s))) for s in grid]


def best_growth(key, market, cfg, front):
    """Максимум медианы: точка сетки + золотое сечение между соседями."""
    i = int(np.argmax([r['median_profit_pct'] for r in front]))
    a = front[max(i - 1, 0)]['scale']
    b = front[min(i + 1, len(front) - 1)]['scale']
    g = (math.sqrt(5) - 1) / 2
    cache = {}

    def f(s):
        if s not in cache:
            cache[s] = summarize(run(build_for(key, s), market, cfg))
        return cache[s]['median_profit_pct']

    x1, x2 = b - g * (b - a), a + g * (b - a)
    for _ in range(7):
        if f(x1) >= f(x2):
            b, x2 = x2, x1
            x1 = b - g * (b - a)
        else:
            a, x1 = x1, x2
            x2 = a + g * (b - a)
    best = max(cache, key=lambda s: cache[s]['median_profit_pct'])
    return best, cache[best]


def solve(key, task, market, cfg):
    lo, hi = scale_range(key)
    scale, m = calibrate(lambda s: build_for(key, s, task), market, cfg, rules_of(task), None, scale_hi=hi,
                         refine=(0.3, 0.45, 0.6, 0.75, 0.9))
    return scale, m


def stake_table(key, scale, task=None, bank0=1000.0, cap=0.10):
    """Ставка в % банка при разных кэфах (банк на пике, без просадки)."""
    strat = build_for(key, scale, task)
    n = len(STAKE_ODDS)
    strat.reset(n, bank0)
    from types import SimpleNamespace
    bank = np.full(n, bank0)
    c = SimpleNamespace(odds=np.array(STAKE_ODDS), edge=np.full(n, ROI), bank=bank, peak=bank, base=bank,
                        bank0=bank0, step=0, n_bets=1000, n_sims=n)
    f = np.clip(np.broadcast_to(np.asarray(strat.fraction(c), dtype=float), (n,)), 0, cap)
    return {str(o): round(float(v) * 100, 3) for o, v in zip(STAKE_ODDS, f)}


def overstated_market(odds, est, true):
    return Market(odds, np.full_like(odds, est), np.full_like(odds, true), 0.0,
                  f'оценка EV {est:.1%}, на деле {true:.1%}')


def robustness(key, scale, task, pool, cfg_eval):
    out = {}
    odds = pool.odds

    def go(mk, cfg):
        return slim(summarize(run(build_for(key, scale, task), mk, cfg)))

    for true in (0.05, 0.035, 0.025, 0.0):
        out[f'true_roi_{true:.3f}'] = go(overstated_market(odds, ROI, true), cfg_eval)
    base_mk = make_market(pool, roi=ROI, ev_model='flat')
    out['backend_rules'] = go(base_mk, replace(cfg_eval, bank0=10000.0, recalc_every=(30, 70), round_to=5.0, min_stake=10.0))
    out['real_odds'] = go(make_market(real_odds_in_range(ODDS_MIN, ODDS_MAX, ROI), roi=ROI), cfg_eval)
    out['horizon'] = {}
    for months in (3, 6, 12):
        out['horizon'][months] = go(base_mk, replace(cfg_eval, n_sims=min(cfg_eval.n_sims, 6000), n_bets=1000 * months))
    return out


def conservative_tolerance(market, cfg, keys=('kelly', 'floor_kelly', 'fixed', 'flat')):
    """Как меняется консервативный размер при разной допустимой просадке (P ≤ 5%)."""
    rows = []
    for level in (10, 15, 20, 25, 30):
        metric = f'p_dd{level}_pct'
        row = {'dd_level': level}
        for key in keys:
            lo, hi = scale_range(key)
            if key == 'floor_kelly':
                build = lambda s, fl=1 - level / 100: S.FloorKelly(s, fl)  # noqa: E731
            else:
                build = lambda s, k=key: build_for(k, s)  # noqa: E731
            scale, m = calibrate(build, market, cfg, metric, 5.0, scale_hi=hi, refine=(0.45, 0.6, 0.75, 0.9))
            row[key] = None if m is None else dict(slim(m), scale=scale, param=param_value(key, scale))
        rows.append(row)
        print(f"  просадка ≥ {level}% не чаще 5%: " + ', '.join(
            f"{k}={row[k]['param']:.3f} ({row[k]['median_profit_pct']:+.1f}%)" for k in keys if row[k]))
    return rows


def pick(rows):
    """Лучшая медиана; если другие отстают меньше чем на TIE_PP — стратегия выше в PREFERENCE."""
    best = max(r['eval']['median_profit_pct'] for r in rows)
    near = [r for r in rows if r['eval']['median_profit_pct'] >= best - TIE_PP]
    return min(near, key=lambda r: PREFERENCE.index(r['key']))


def pp(x, d=1):
    return f'{x:+.{d}f}%'.replace('-', '−')


def pc(x, d=1):
    return f'{x:.{d}f}%'


def write_markdown(res, path):
    s = res['settings']
    fam = {k: v[0] for k, v in FAMILIES.items()}
    unit = {k: v[4] for k, v in FAMILIES.items()}
    L = []
    a = L.append

    def row_metrics(m):
        return (f"{pp(m['median_profit_pct'])} | {pp(m['mean_profit_pct'], 0)} | {pp(m['p5_profit_pct'])} | {pp(m['p95_profit_pct'], 0)} | "
                f"{pc(m['p_loss_pct'])} | {pc(m.get('p_profit_below_m10_pct', float('nan')))} | {pc(m['median_max_dd_pct'])} | {pc(m['p_dd20_pct'])} | {pc(m['p_dd30_pct'])} | "
                f"{pc(m['p_dd50_pct'])} | {pc(m['p_dd80_pct'])}")
    head = ('Медиана месяца | Среднее | Плохой месяц (5%) | Хороший (95%) | Месяц в минусе | Месяц хуже −10% | Медиана макс. просадки | '
            'Просадка ≥ 20% | ≥ 30% | ≥ 50% | ≥ 80%')
    ncol = 11

    a('# Месяц ставок: ROI 5%, кэф 1.5–4\n')
    a(f"**Условия.** У каждой ставки истинный ROI {s['roi'] * 100:.0f}%. Кэф от {s['odds_min']} до {s['odds_max']}, средний "
      f"{s['avg_odds']:.2f}, разброс σ {s['sd_odds']:.2f} — как у реальных ставок в этом диапазоне. {s['n_bets']} ставок в месяц, "
      f"ставка от текущего банка, не больше {s['cap'] * 100:.0f}% банка. Размер подбирался на {s['n_sims_search']} симуляциях на точку, "
      f"все цифры ниже — перепроверка на независимых {s['n_sims_eval']} симуляциях. Прибыль и просадка — за месяц, от банка на его начало.\n")
    a('**Задачи.** Везде максимизируется медианная прибыль месяца, ограничение — на просадку от пика внутри месяца:\n')
    for t, (label, _, text) in TASKS.items():
        a(f'- **{label}** — {text}.')
    a('')
    a('Медиана, а не среднее: среднее делают несколько сверхудачных месяцев, и оно растёт с размером ставки далеко за разумные пределы.\n')

    a('## Рекомендации\n')
    a('| Задача | Стратегия | Параметр | Ставка, % банка, при кэфе 1.5 / 2 / 2.75 / 3.5 / 4 | ' + head + ' |')
    a('|---|---|---|---|' + '---:|' * ncol)
    for task, r in res['recommended'].items():
        st = r['stakes']
        stakes = ' / '.join(f"{st[k]:.2f}" for k in ('1.5', '2.0', '2.75', '3.5', '4.0'))
        a(f"| {TASKS[task][0]} | {fam[r['key']]} | {r['param']:.3g} ({unit[r['key']]}) | {stakes} | {row_metrics(r['eval'])} |")
    c = res['ceiling']
    st = c['stakes']
    stakes = ' / '.join(f"{st[k]:.2f}" for k in ('1.5', '2.0', '2.75', '3.5', '4.0'))
    a(f"| Потолок, без ограничений | Келли | {c['scale']:.3g} (доля от полного Келли) | {stakes} | {row_metrics(c['eval'])} |")
    a('')
    for task, r in res['recommended'].items():
        eq = [x for x in res['tasks'][task] if x['key'] in ('log_formula', 'floor_kelly', 'fixed') and x['key'] != r['key']]
        parts = [f"{fam[x['key']]} {x['param']:.3g} ({unit[x['key']]}) — медиана {pp(x['eval']['median_profit_pct'])}" for x in eq]
        a(f"- **{TASKS[task][0]}.** Альтернативы на том же риске: " + '; '.join(parts) + '.')
    a('')

    for task, (label, _, text) in TASKS.items():
        a(f'## {label}: все стратегии\n')
        a(f'Ограничение: {text}. Отсортировано по медиане.\n')
        a('| # | Стратегия | Параметр | Ср. ставка | ' + head + ' |')
        a('|---:|---|---|---:|' + '---:|' * ncol)
        rows = sorted(res['tasks'][task], key=lambda r: r['eval']['median_profit_pct'], reverse=True)
        for i, r in enumerate(rows, 1):
            a(f"| {i} | {fam[r['key']]} | {r['param']:.3g} {unit[r['key']]} | {pc(r['eval']['avg_stake_pct'], 2)} | {row_metrics(r['eval'])} |")
        a('')

    a('## Консервативная: сколько ставить при разной допустимой просадке\n')
    a('Ограничение: просадка не глубже X% от пика в 95% месяцев. В ячейке — параметр и медиана месяца '
      '(на поисковых симуляциях, без перепроверки).\n')
    keys = [k for k in res['tolerance'][0] if k != 'dd_level']
    a('| Допустимая просадка | ' + ' | '.join(f"{fam[k]} ({unit[k]})" for k in keys) + ' |')
    a('|---:|' + '---|' * len(keys))
    for row in res['tolerance']:
        cells = ['—' if row[k] is None else f"{row[k]['param']:.3g} → {pp(row[k]['median_profit_pct'])}" for k in keys]
        a(f"| {row['dd_level']}% | " + ' | '.join(cells) + ' |')
    a('')

    a('## Устойчивость\n')
    a('Стратегия считает размер ставки по ROI 5%, а на деле ROI может оказаться меньше. Ещё два сценария: правила backend '
      '(снимок банка раз в 30–70 ставок, округление до 5, минимум 10 при банке 10 000) и реальные кэфы 1.5–4 вместо модели.\n')
    names = {'true_roi_0.050': 'ROI 5% (как задумано)', 'true_roi_0.035': 'ROI на деле 3.5%', 'true_roi_0.025': 'ROI на деле 2.5%',
             'true_roi_0.000': 'ROI на деле 0%', 'backend_rules': 'правила backend', 'real_odds': 'реальные кэфы 1.5–4'}
    a('| Стратегия | Сценарий | ' + head + ' |')
    a('|---|---|' + '---:|' * ncol)
    for task, r in res['recommended'].items():
        for variant in [r] + r.get('alternatives', []):
            title = f"{TASKS[task][0]}: {fam[variant['key']]} {variant['param']:.3g}"
            for sk, sname in names.items():
                a(f"| {title} | {sname} | {row_metrics(variant['robustness'][sk])} |")
                title = ''
    a('')

    a('## Несколько месяцев подряд\n')
    a('Банк не выводится, ставки продолжаются от текущего банка. Это теоретический потолок: на практике ставка быстро упрётся '
      'в лимиты букмекеров, поэтому смотрите в первую очередь на риск (просадка, шанс минуса), а не на проценты роста.\n')
    a('| Стратегия | Месяцев | Медиана | Плохой исход (5%) | Хороший (95%) | В минусе | Медиана макс. просадки | Просадка ≥ 50% |')
    a('|---|---:|---:|---:|---:|---:|---:|---:|')
    for task, r in res['recommended'].items():
        m = r['eval']
        a(f"| {TASKS[task][0]}: {fam[r['key']]} {r['param']:.3g} | 1 | {pp(m['median_profit_pct'])} | {pp(m['p5_profit_pct'])} | "
          f"{pp(m['p95_profit_pct'], 0)} | {pc(m['p_loss_pct'])} | {pc(m['median_max_dd_pct'])} | {pc(m['p_dd50_pct'])} |")
        for months, m in r['robustness']['horizon'].items():
            a(f"| | {months} | {pp(m['median_profit_pct'], 0)} | {pp(m['p5_profit_pct'], 0)} | {pp(m['p95_profit_pct'], 0)} | "
              f"{pc(m['p_loss_pct'])} | {pc(m['median_max_dd_pct'])} | {pc(m['p_dd50_pct'])} |")
    a('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    n_search = 800 if args.quick else 4000
    n_eval = 2000 if args.quick else 20000
    points = 10 if args.quick else 16
    t0 = time.time()

    pool = beta_odds_pool(ODDS_MIN, ODDS_MAX, ODDS_MEAN, ROI, CONCENTRATION)
    market = make_market(pool, roi=ROI, ev_model='flat')
    cfg = SimConfig(n_sims=n_search, n_bets=1000, cap=0.10, seed=101, quantile_every=1000)
    cfg_eval = replace(cfg, n_sims=n_eval, seed=202)
    cfg_rob = replace(cfg_eval, n_sims=min(n_eval, 10000))
    res = {'settings': {'roi': ROI, 'odds_min': ODDS_MIN, 'odds_max': ODDS_MAX, 'avg_odds': float(pool.odds.mean()),
                        'sd_odds': float(pool.odds.std()), 'n_bets': cfg.n_bets, 'cap': cfg.cap,
                        'n_sims_search': n_search, 'n_sims_eval': n_eval}}

    print('Кривые «размер -> рост/риск»')
    res['frontier'] = {}
    for key in FRONTIER_KEYS:
        res['frontier'][key] = frontier(key, market, cfg, points)
        print(f'  {FAMILIES[key][0]}: {len(res["frontier"][key])} точек')
    scale, _ = best_growth('kelly', market, cfg, res['frontier']['kelly'])
    res['ceiling'] = {'scale': scale, 'eval': slim(evaluate('kelly', scale, market, cfg_eval)),
                      'stakes': stake_table('kelly', scale)}
    print(f"Потолок: Келли {scale:.3f}, медиана {res['ceiling']['eval']['median_profit_pct']:+.1f}%")

    res['tasks'] = {}
    for task, (label, _, _) in TASKS.items():
        print(f'\n{label}: ' + ', '.join(f'{k} <= {v}%' for k, v in rules_of(task)))
        metric = rules_of(task)[0][0]
        rows = []
        for key in FAMILIES:
            scale, m = solve(key, task, market, cfg)
            if m is None:
                print(f'  {key:20s}: не укладывается в ограничение')
                continue
            ev = slim(evaluate(key, scale, market, cfg_eval, task))
            rows.append({'key': key, 'scale': scale, 'param': param_value(key, scale), 'search': slim(m), 'eval': ev})
            print(f"  {key:20s} параметр {param_value(key, scale):7.3f}  медиана {ev['median_profit_pct']:+7.1f}%  "
                  f"{metric} {ev[metric]:5.1f}%  в минусе {ev['p_loss_pct']:4.1f}%  ставка {ev['avg_stake_pct']:.2f}%")
        res['tasks'][task] = rows

    res['recommended'] = {}
    for task, (label, _, _) in TASKS.items():
        rows = [r for r in res['tasks'][task] if all(r['eval'][k] <= v + 0.5 for k, v in rules_of(task))] or res['tasks'][task]
        rec = dict(pick(rows))
        rec['stakes'] = stake_table(rec['key'], rec['scale'], task)
        print(f"\n{label}: рекомендую {FAMILIES[rec['key']][0]} {rec['param']:.3f}; устойчивость…")
        rec['robustness'] = robustness(rec['key'], rec['scale'], task, pool, cfg_rob)
        rec['alternatives'] = []
        for alt in res['tasks'][task]:
            if alt['key'] in ('floor_kelly', 'kelly') and alt['key'] != rec['key']:
                alt = dict(alt, stakes=stake_table(alt['key'], alt['scale'], task),
                           robustness=robustness(alt['key'], alt['scale'], task, pool, cfg_rob))
                rec['alternatives'].append(alt)
        res['recommended'][task] = rec

    print('\nКонсервативная при разной допустимой просадке')
    res['tolerance'] = conservative_tolerance(market, cfg)
    res['settings']['seconds'] = round(time.time() - t0, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'month_roi5.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    write_markdown(res, os.path.join(OUT_DIR, 'MONTH_ROI5.md'))
    print(f"\nГотово за {res['settings']['seconds']} с: reports/MONTH_ROI5.md")


if __name__ == '__main__':
    main()
