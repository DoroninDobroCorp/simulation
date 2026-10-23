"""
Оптимальный размер ставки на месяц под конкретные условия.

Условия по умолчанию: ROI 5% у каждой ставки, кэф от 1.5 до 4 (средний 2.75,
разброс как у реальных ставок), 1000 ставок = месяц, не больше 10% банка на ставку.

Задачи (максимум медианной прибыли месяца при ограничении на просадку от пика; основной рецепт —
дробный Келли, рядом — стратегия с максимальной медианой):
  aggressive    — просадка ≥ 50% не чаще чем в 1 месяце из 10;
  moderate      — просадка ≥ 30% не чаще чем в 1 месяце из 10;
  conservative  — большой банк: просадка ≥ 20% не чаще чем в 1 месяце из 20
                  и месяц хуже −10% не чаще чем 1 из 20.
Отдельно — потолок: доля Келли с максимальной медианой без ограничений.

Для каждой стратегии лучший размер под задачу ищется бисекцией к границе лимита и
перебором внутри допустимой области (brm.optimize.calibrate; у Келли с полом ещё и
подбор пола), на общих случайных числах. Найденные настройки перепроверяются на
независимом seed с большим числом симуляций. Для четырёх основных стратегий
дополнительно строятся кривые «размер -> рост/риск».

  python -m brm.month_study            # ~15–20 минут -> reports/MONTH_ROI5.md, month_roi5.json
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
from .optimize import calibrate, calibrate_floor_kelly

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
    # две ручки: пол и доля Келли от подушки; обе подбирает calibrate_floor_kelly в solve()
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
# Основной рецепт в каждой задаче — дробный Келли: по результату он почти совпадает с лог-формулой backend
# при подходящем risk и у него меньше месяцев в минусе, чем у стратегии с максимальной медианой, в том числе
# если ROI завышен. Стратегия с максимальной медианой показывается рядом как альтернатива со своей ценой.
PRIMARY = 'kelly'

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


def build_for(key, scale, floor=None):
    """Сборка стратегии; для лог-формулы scale = 1/risk, чтобы рост scale всегда означал большую ставку."""
    if key == 'floor_kelly':
        return S.FloorKelly(scale, 0.5 if floor is None else floor)
    _, build, *_ = FAMILIES[key]
    return build(1.0 / scale) if is_inverse(key) else build(scale)


def scale_range(key):
    _, _, lo, hi, _ = FAMILIES[key]
    return (1.0 / hi, 1.0 / lo) if is_inverse(key) else (lo, hi)


def param_value(key, scale):
    return 1.0 / scale if is_inverse(key) else scale


def evaluate(key, scale, market, cfg, floor=None):
    return summarize(run(build_for(key, scale, floor), market, cfg))


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


def solve(key, rules, market, cfg):
    """Лучший размер под ограничения; у Келли с полом подбираются и пол, и доля. -> (scale, метрики, пол)."""
    lo, hi = scale_range(key)
    refine = (0.3, 0.45, 0.6, 0.75, 0.9)
    if key == 'floor_kelly':
        scale, m = calibrate_floor_kelly(market, cfg, rules, None, mult_hi=hi, refine=refine)
        return scale, m, (None if m is None else m['floor'])
    scale, m = calibrate(lambda s: build_for(key, s), market, cfg, rules, None, scale_hi=hi, refine=refine)
    return scale, m, None


def stake_table(key, scale, floor=None, bank0=1000.0, cap=0.10):
    """Ставка в % банка при разных кэфах (банк на пике, без просадки)."""
    strat = build_for(key, scale, floor)
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


def robustness(row, pool, cfg_eval, cfg_horizon):
    key, scale, floor = row['key'], row['scale'], row.get('floor')
    out = {}
    odds = pool.odds

    def go(mk, cfg):
        return slim(summarize(run(build_for(key, scale, floor), mk, cfg)))

    out['true_roi_0.050'] = row['eval']          # это и есть основная перепроверка
    for true in (0.035, 0.025, 0.0):
        out[f'true_roi_{true:.3f}'] = go(overstated_market(odds, ROI, true), cfg_eval)
    base_mk = make_market(pool, roi=ROI, ev_model='flat')
    out['backend_rules'] = go(base_mk, replace(cfg_eval, bank0=10000.0, recalc_every=(30, 70), round_to=5.0, min_stake=10.0))
    out['real_odds'] = go(make_market(real_odds_in_range(ODDS_MIN, ODDS_MAX, ROI), roi=ROI), cfg_eval)
    out['horizon'] = {}
    for months in (3, 6, 12):
        out['horizon'][months] = go(base_mk, replace(cfg_horizon, n_bets=1000 * months))
    return out


def conservative_tolerance(market, cfg, keys=('kelly', 'floor_kelly', 'fixed', 'flat')):
    """Как меняется консервативный размер при разной допустимой просадке (P ≤ 5%)."""
    rows = []
    for level in (10, 15, 20, 25, 30):
        row = {'dd_level': level}
        for key in keys:
            scale, m, floor = solve(key, [(f'p_dd{level}_pct', 5.0)], market, cfg)
            row[key] = None if m is None else dict(slim(m), scale=scale, param=param_value(key, scale), floor=floor)
        rows.append(row)
        print(f"  просадка ≥ {level}% не чаще 5%: " + ', '.join(
            f"{k}={row[k]['param']:.3f} ({row[k]['median_profit_pct']:+.1f}%)" for k in keys if row[k]))
    return rows


def feasible_rows(task, rows):
    """Строки, которые выполняют ограничения задачи и на независимой перепроверке."""
    return [r for r in rows if all(r['eval'][k] <= v for k, v in rules_of(task))]


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

    def ptxt(r):
        if r['key'] == 'floor_kelly':
            return f"{r['param']:.3g} Келли от подушки над полом {r['floor']:.0%} пика"
        return f"{r['param']:.3g} ({unit[r['key']]})"

    def row_metrics(m):
        return (f"{pp(m['median_profit_pct'])} | {pp(m['mean_profit_pct'], 0)} | {pp(m['p5_profit_pct'])} | {pp(m['p95_profit_pct'], 0)} | "
                f"{pc(m['p_loss_pct'])} | {pc(m.get('p_profit_below_m10_pct', float('nan')))} | {pc(m['median_max_dd_pct'])} | "
                f"{pc(m['p_dd20_pct'])} | {pc(m['p_dd30_pct'])} | {pc(m['p_dd50_pct'])} | {pc(m['p_dd80_pct'])}")
    head = ('Медиана месяца | Среднее | Плохой месяц (5%) | Хороший (95%) | Месяц в минусе | Месяц хуже −10% | Медиана макс. просадки | '
            'Просадка ≥ 20% | ≥ 30% | ≥ 50% | ≥ 80%')
    ncol = 11

    def stakes_txt(st):
        return ' / '.join(f"{st[k]:.2f}" for k in ('1.5', '2.0', '2.75', '3.5', '4.0'))

    a('# Месяц ставок: ROI 5%, кэф 1.5–4\n')
    a(f"**Условия.** У каждой ставки истинный ROI {s['roi'] * 100:.0f}%. Кэф от {s['odds_min']} до {s['odds_max']} из Beta-распределения, "
      f"средний {s['avg_odds']:.2f}, разброс σ {s['sd_odds']:.2f} (у реальных ставок в этом диапазоне σ 0.65 при среднем 2.63). "
      f"{s['n_bets']} ставок в месяц, ставка от текущего банка, не больше {s['cap'] * 100:.0f}% банка. "
      f"Размер подбирался на {s['n_sims_search']} симуляциях на точку с запасом {100 - s['search_margin'] * 100:.0f}% от лимита; "
      f"если на перепроверке лимит всё же нарушался, размер уменьшался шагами по 3%. "
      f"Основные таблицы, кривые и устойчивость — перепроверка на независимых {s['n_sims_eval']} симуляциях, "
      f"несколько месяцев подряд — {s['n_sims_horizon']} симуляций. Прибыль и просадка — за месяц, от банка на его начало.\n")
    a('**Задачи.** Везде максимизируется медианная прибыль месяца, ограничение — на просадку от пика внутри месяца:\n')
    for t, (label, _, text) in TASKS.items():
        a(f'- **{label}** — {text}.')
    a('')
    a('Медиана, а не среднее: среднее делают несколько сверхудачных месяцев, и оно растёт с размером ставки далеко за разумные пределы.\n')
    a('**Как выбран рецепт.** Основной рецепт в каждой задаче — дробный Келли. По результату он почти совпадает с лог-формулой '
      'backend при подходящем risk (он указан ниже), хотя ставки по кэфам распределяются по-разному. У него меньше месяцев '
      'в минусе, чем у стратегии с максимальной медианой, в том числе если ROI окажется ниже 5%; в агрессивной и умеренной '
      'задачах при меньшем ROI у него ещё и заметно выше медиана. Рядом — стратегия с максимальной медианой при том же '
      'ограничении и её цена.\n')

    a('## Рекомендации\n')
    a('Ставка указана при банке на пике (у Келли с полом она меньше, когда банк ниже пика).\n')
    a('| Задача | Вариант | Стратегия | Параметр | Ставка, % банка, при кэфе 1.5 / 2 / 2.75 / 3.5 / 4 | ' + head + ' |')
    a('|---|---|---|---|---|' + '---:|' * ncol)
    for task, r in res['recommended'].items():
        a(f"| {TASKS[task][0]} | основной | {fam[r['key']]} | {ptxt(r)} | {stakes_txt(r['stakes'])} | {row_metrics(r['eval'])} |")
        for alt in r['alternatives']:
            a(f"| | максимум медианы | {fam[alt['key']]} | {ptxt(alt)} | {stakes_txt(alt['stakes'])} | {row_metrics(alt['eval'])} |")
    c = res['ceiling']
    a(f"| Потолок, без ограничений | | Келли | {c['scale']:.3g} (доля от полного Келли) | {stakes_txt(c['stakes'])} | {row_metrics(c['eval'])} |")
    a('')
    for task, r in res['recommended'].items():
        log = next((x for x in res['tasks'][task] if x['key'] == 'log_formula'), None)
        fixed = next((x for x in res['tasks'][task] if x['key'] == 'fixed'), None)
        parts = []
        if log:
            parts.append(f"в лог-формуле backend это risk ≈ {log['param']:.0f} (медиана {pp(log['eval']['median_profit_pct'])})")
        if fixed:
            parts.append(f"фиксированный процент на том же ограничении — {fixed['param']:.2f}% банка, медиана {pp(fixed['eval']['median_profit_pct'])}")
        for alt in r['alternatives']:
            m0, m1 = r['eval'], alt['eval']
            r0, r1 = r['robustness']['true_roi_0.025'], alt['robustness']['true_roi_0.025']
            text = (f"{fam[alt['key']]} даёт медиану на {m1['median_profit_pct'] - m0['median_profit_pct']:.1f} п.п. выше, "
                    f"но месяцев в минусе {pc(m1['p_loss_pct'])} против {pc(m0['p_loss_pct'])}, а при ROI 2.5% медиана "
                    f"{pp(r1['median_profit_pct'])} против {pp(r0['median_profit_pct'])}")
            b0, b1 = r['robustness']['backend_rules'], alt['robustness']['backend_rules']
            broken = [(k, v) for k, v in rules_of(task) if b1[k] > v]
            if broken:
                k, v = broken[0]
                label = k.replace('p_dd', 'просадка ≥ ').replace('_pct', '%').replace('p_profit_below_m10', 'месяц хуже −10%')
                text += (f"; при правилах backend (снимок банка раз в 30–70 ставок) её ограничение не выполняется: "
                         f"{label} в {pc(b1[k])} месяцев при лимите {v:g}% (у основного {pc(b0[k])})")
            parts.append(text)
        text = '; '.join(parts)
        a(f"- **{TASKS[task][0]}.** " + text[:1].upper() + text[1:] + '.')
    a('')

    for task, (label, _, text) in TASKS.items():
        a(f'## {label}: все стратегии\n')
        a(f'Ограничение: {text}. Отсортировано по медиане.\n')
        a('| # | Стратегия | Параметр | Ср. ставка | ' + head + ' |')
        a('|---:|---|---|---:|' + '---:|' * ncol)
        rows = sorted(res['tasks'][task], key=lambda r: r['eval']['median_profit_pct'], reverse=True)
        for i, r in enumerate(rows, 1):
            a(f"| {i} | {fam[r['key']]} | {ptxt(r)} | {pc(r['eval']['avg_stake_pct'], 2)} | {row_metrics(r['eval'])} |")
        a('')

    a('## Консервативная: сколько ставить при разной допустимой просадке\n')
    a('Ограничение: просадка не глубже X% от пика в 95% месяцев. В ячейке — параметр и медиана месяца '
      '(на поисковых симуляциях, без перепроверки).\n')
    keys = [k for k in res['tolerance'][0] if k != 'dd_level']
    a('| Допустимая просадка | ' + ' | '.join(f"{fam[k]} ({unit[k]})" for k in keys) + ' |')
    a('|---:|' + '---|' * len(keys))
    for row in res['tolerance']:
        cells = []
        for k in keys:
            x = row[k]
            if x is None:
                cells.append('—')
            elif k == 'floor_kelly':
                cells.append(f"пол {x['floor']:.0%}, доля {x['param']:.3g} → {pp(x['median_profit_pct'])}")
            else:
                cells.append(f"{x['param']:.3g} → {pp(x['median_profit_pct'])}")
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
        for variant in [r] + r['alternatives']:
            title = f"{TASKS[task][0]}: {fam[variant['key']]} {ptxt(variant)}"
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
        for variant in [r] + r['alternatives']:
            m = variant['eval']
            a(f"| {TASKS[task][0]}: {fam[variant['key']]} {ptxt(variant)} | 1 | {pp(m['median_profit_pct'])} | {pp(m['p5_profit_pct'])} | "
              f"{pp(m['p95_profit_pct'], 0)} | {pc(m['p_loss_pct'])} | {pc(m['median_max_dd_pct'])} | {pc(m['p_dd50_pct'])} |")
            for months, m in variant['robustness']['horizon'].items():
                a(f"| | {months} | {pp(m['median_profit_pct'], 0)} | {pp(m['p5_profit_pct'], 0)} | {pp(m['p95_profit_pct'], 0)} | "
                  f"{pc(m['p_loss_pct'])} | {pc(m['median_max_dd_pct'])} | {pc(m['p_dd50_pct'])} |")
    a('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


SEARCH_MARGIN = 0.95   # подбор идёт к 95% лимита, чтобы на перепроверке лимит выполнялся, а не был на грани


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
    cfg_horizon = replace(cfg_eval, n_sims=min(n_eval, 6000))
    res = {'settings': {'roi': ROI, 'odds_min': ODDS_MIN, 'odds_max': ODDS_MAX, 'avg_odds': float(pool.odds.mean()),
                        'sd_odds': float(pool.odds.std()), 'n_bets': cfg.n_bets, 'cap': cfg.cap,
                        'n_sims_search': n_search, 'n_sims_eval': n_eval, 'n_sims_horizon': cfg_horizon.n_sims,
                        'search_margin': SEARCH_MARGIN, 'primary': PRIMARY}}

    # кривые и потолок считаются на той же выборке, что и итоговые цифры, чтобы точки рекомендаций лежали на кривых
    print('Кривые «размер -> рост/риск»')
    res['frontier'] = {}
    for key in FRONTIER_KEYS:
        res['frontier'][key] = frontier(key, market, cfg_eval, points)
        print(f'  {FAMILIES[key][0]}: {len(res["frontier"][key])} точек')
    scale, m = best_growth('kelly', market, cfg_eval, res['frontier']['kelly'])
    res['ceiling'] = {'scale': scale, 'eval': slim(m), 'stakes': stake_table('kelly', scale)}
    print(f"Потолок: Келли {scale:.3f}, медиана {res['ceiling']['eval']['median_profit_pct']:+.1f}%")

    res['tasks'] = {}
    for task, (label, _, _) in TASKS.items():
        rules = rules_of(task)
        print(f'\n{label}: ' + ', '.join(f'{k} <= {v}%' for k, v in rules))
        search_rules = [(k, v * SEARCH_MARGIN) for k, v in rules]
        metric = rules[0][0]
        rows = []
        for key in FAMILIES:
            scale, m, floor = solve(key, search_rules, market, cfg)
            if m is None:
                print(f'  {key:20s}: не укладывается в ограничение')
                continue
            ev = slim(evaluate(key, scale, market, cfg_eval, floor))
            # если на независимой перепроверке лимит всё же нарушен (шум поиска), уменьшаем размер шагами по 3%
            shrink = 0
            while not all(ev[k] <= v for k, v in rules) and shrink < 40:
                scale *= 0.97
                shrink += 1
                ev = slim(evaluate(key, scale, market, cfg_eval, floor))
            row = {'key': key, 'scale': scale, 'param': param_value(key, scale), 'search': slim(m), 'eval': ev,
                   'shrink_steps': shrink}
            if floor is not None:
                row['floor'] = floor
            rows.append(row)
            print(f"  {key:20s} параметр {param_value(key, scale):7.3f}{'' if floor is None else f' пол {floor:.2f}'}  "
                  f"медиана {ev['median_profit_pct']:+7.1f}%  {metric} {ev[metric]:5.1f}%  в минусе {ev['p_loss_pct']:4.1f}%  "
                  f"ставка {ev['avg_stake_pct']:.2f}%")
        res['tasks'][task] = rows

    res['recommended'] = {}
    for task, (label, _, _) in TASKS.items():
        rows = feasible_rows(task, res['tasks'][task])
        if not rows:
            raise RuntimeError(f'{label}: ни одна стратегия не выполнила ограничения на перепроверке')
        primary = next((r for r in rows if r['key'] == PRIMARY), None)
        if primary is None:
            raise RuntimeError(f'{label}: основной рецепт ({PRIMARY}) не выполнил ограничения на перепроверке')
        best = max(rows, key=lambda r: r['eval']['median_profit_pct'])
        rec = dict(primary, stakes=stake_table(primary['key'], primary['scale'], primary.get('floor')))
        print(f"\n{label}: основной {FAMILIES[rec['key']][0]} {rec['param']:.3f}, максимум медианы — {FAMILIES[best['key']][0]}; устойчивость…")
        rec['robustness'] = robustness(rec, pool, cfg_eval, cfg_horizon)
        rec['alternatives'] = []
        if best['key'] != primary['key']:
            alt = dict(best, stakes=stake_table(best['key'], best['scale'], best.get('floor')))
            alt['robustness'] = robustness(alt, pool, cfg_eval, cfg_horizon)
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
