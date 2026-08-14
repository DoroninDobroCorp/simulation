"""
Отбор ТОП-безопасных стратегий по жёсткому критерию:
  ни одна из 10000 симуляций не опускается ниже 34% от пика (max DD <= 66%),
  при железном правиле "ставка <= 10% текущего банка" (динамический БМ).

Для отобранного топа:
  - полный прогон 10000 траекторий (с сохранением истории),
  - строгая перепроверка корректности,
  - графики: 2000 траекторий (спагетти) + перцентильный веер,
  - сводка "сколько стратегий пробивают порог -X%".

Результаты: top_data/ (PNG + top_summary.json) для дашборда.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sim_engine

ROI = 0.025
CAP = 0.10
NUM_SIMS = 10000
NUM_BETS = 10000
SEED = 42
FLOOR_DD = -66.0  # банк не ниже 34% от пика
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top_data')

# Отобранные и проверенные конфигурации (см. скан). Доходные И безопасные.
# Параметры подобраны и проверены на устойчивость по 4 сидам (42,7,123,2024):
# все три держат критерий "никогда ниже 34% от пика" С ЗАПАСОМ на всех сидах.
TOP = [
    {'id': 'log_roi_bp0.45',    'key': 'log_roi',    'params': {'base_percent': 0.45}, 'label': 'Logarithmic ROI (base 0.45%)'},
    {'id': 'linear_roi_bp0.35', 'key': 'linear_roi', 'params': {'base_percent': 0.35}, 'label': 'Linear ROI (base 0.35%)'},
    {'id': 'hybrid_bp1.0',      'key': 'hybrid',     'params': {'base_percent': 1.0},  'label': 'Hybrid ROI/odds (base 1.0%)'},
]

DD_THRESHOLDS = [20, 34, 50, 66, 80, 100]


def metrics_from_bankroll(br):
    peaks = np.maximum.accumulate(br, axis=1)
    dd = (br - peaks) / peaks * 100.0
    maxdd = dd.min(axis=1)
    final = br[:, -1]
    profit = (final - sim_engine.INITIAL_BANKROLL) / sim_engine.INITIAL_BANKROLL * 100.0
    return dd, maxdd, final, profit


def recheck(br, bet_history, odds, outcomes_seed_ok=True):
    """Строгие инварианты корректности. Возвращает dict с результатами проверок."""
    checks = {}
    # 1. Банк никогда не отрицателен
    checks['no_negative_bank'] = bool(br.min() >= -1e-6)
    # 2. Каждая ставка <= 10% банка перед ней (+эпсилон)
    before = br[:, :NUM_BETS]
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(before > 0, bet_history / before * 100, 0.0)
    checks['bet_le_10pct'] = bool(np.nanmax(pct) <= 10.0 + 1e-6)
    checks['max_bet_pct'] = float(np.nanmax(pct))
    # 3. Просадка не глубже -100% (физический предел)
    _, maxdd, final, _ = metrics_from_bankroll(br)
    checks['dd_ge_minus100'] = bool(maxdd.min() >= -100.0 - 1e-6)
    # 4. Жёсткий критерий: ни одна симуляция не ниже 34% от пика
    checks['worst_dd'] = float(maxdd.min())
    checks['passes_floor_-66'] = bool(maxdd.min() >= FLOOR_DD)
    # 5. Инвариант баланса: финал == старт + sum(выигрыши) - sum(проигрыши) на одной симуляции
    #    (проверяем на симуляции 0; пересчитываем независимо)
    s0 = 0
    bank = sim_engine.INITIAL_BANKROLL
    for i in range(NUM_BETS):
        bet = bet_history[s0, i]
        bank += bet * (odds[i] - 1) if outcomes_glob[s0, i] else -bet
    checks['balance_invariant_ok'] = bool(abs(bank - br[s0, -1]) < 1e-3)
    checks['balance_recomputed'] = float(bank)
    checks['balance_stored'] = float(br[s0, -1])
    return checks


def spaghetti_and_fan(br, label, out_png, n_lines=2000):
    """Рисует n_lines случайных траекторий + перцентильный веер."""
    num_sims, T = br.shape
    x = np.arange(T)
    # прореживаем по времени для рендера
    step = max(1, T // 400)
    xs = x[::step]
    sub = br[:, ::step]

    rng = np.random.default_rng(0)
    idx = rng.choice(num_sims, size=min(n_lines, num_sims), replace=False)

    pcts = {p: np.percentile(sub, p, axis=0) for p in [1, 5, 25, 50, 75, 95, 99]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Спагетти
    for j in idx:
        ax1.plot(xs, sub[j], color='#4c9aff', alpha=0.02, linewidth=0.5)
    ax1.plot(xs, pcts[50], color='#ffd43b', linewidth=2.2, label='Медиана')
    ax1.plot(xs, pcts[5], color='#ff6b6b', linewidth=1.8, label='P5 (худшие 5%)')
    ax1.plot(xs, pcts[95], color='#51cf66', linewidth=1.8, label='P95 (лучшие 5%)')
    ax1.axhline(sim_engine.INITIAL_BANKROLL, color='#888', linestyle='--', linewidth=0.8)
    ax1.set_yscale('log')
    ax1.set_title(f'{label}\n{n_lines} случайных траекторий (лог-шкала)')
    ax1.set_xlabel('Номер ставки'); ax1.set_ylabel('Банкролл (лог)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Веер перцентилей
    ax2.fill_between(xs, pcts[1], pcts[99], color='#4c9aff', alpha=0.12, label='P1-P99')
    ax2.fill_between(xs, pcts[5], pcts[95], color='#4c9aff', alpha=0.20, label='P5-P95')
    ax2.fill_between(xs, pcts[25], pcts[75], color='#4c9aff', alpha=0.35, label='P25-P75')
    ax2.plot(xs, pcts[50], color='#ffd43b', linewidth=2.2, label='Медиана')
    ax2.axhline(sim_engine.INITIAL_BANKROLL, color='#888', linestyle='--', linewidth=0.8)
    ax2.set_yscale('log')
    ax2.set_title('Перцентильный веер роста банкролла')
    ax2.set_xlabel('Номер ставки'); ax2.set_ylabel('Банкролл (лог)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor='white')
    plt.close(fig)

    # лёгкие данные для интерактивного дашборда
    return {
        'x': [int(v) for v in xs],
        'p1': [round(float(v), 2) for v in pcts[1]],
        'p5': [round(float(v), 2) for v in pcts[5]],
        'p25': [round(float(v), 2) for v in pcts[25]],
        'p50': [round(float(v), 2) for v in pcts[50]],
        'p75': [round(float(v), 2) for v in pcts[75]],
        'p95': [round(float(v), 2) for v in pcts[95]],
        'p99': [round(float(v), 2) for v in pcts[99]],
    }


def dd_distribution(maxdd):
    """Гистограмма макс-просадок (для дашборда)."""
    counts, edges = np.histogram(maxdd, bins=np.linspace(maxdd.min(), 0, 41))
    centers = (edges[:-1] + edges[1:]) / 2
    return {'x': [round(float(c), 1) for c in centers], 'y': [int(c) for c in counts]}


outcomes_glob = None  # заполняется в main для recheck


def main():
    global outcomes_glob
    os.makedirs(OUT, exist_ok=True)

    # один датасет для всех (одинаковые исходы -> честное сравнение)
    sim_engine.set_roi(ROI)
    outcomes_glob, odds = sim_engine.build_dataset(NUM_SIMS, NUM_BETS, ROI, seed=SEED)
    actual_roi = sim_engine.actual_roi_from_outcomes(outcomes_glob, odds)
    print(f"ROI оборота данных: {actual_roi:.3f}% (цель {ROI*100:.2f}%)")

    top_out = {
        'roi_pct': ROI * 100,
        'actual_roi_turnover_pct': actual_roi,
        'cap_pct': CAP * 100,
        'num_sims': NUM_SIMS, 'num_bets': NUM_BETS,
        'floor_dd_pct': FLOOR_DD,
        'strategies': [],
        'threshold_summary': None,
    }

    # Для сводки порогов нужно прогнать ВСЕ стратегии (берём из готового capped-кэша)
    cap_summary = json.load(open(os.path.join(sim_engine.CACHE_DIR + '_capped', 'summary.json')))

    robustness_seeds = [42, 7, 123, 2024]

    for cfg in TOP:
        print(f"\n=== {cfg['label']} ===")
        br, bet_history, odds = sim_engine.run_single(
            cfg['key'], variation=False, num_sims=NUM_SIMS, num_bets=NUM_BETS,
            roi=ROI, seed=SEED, cap=CAP, override_params=cfg['params'])

        # Проверка устойчивости на нескольких сидах (не overfit к одному набору исходов)
        robustness = []
        for sd in robustness_seeds:
            b2, _, _ = sim_engine.run_single(cfg['key'], False, NUM_SIMS, NUM_BETS,
                                             roi=ROI, seed=sd, cap=CAP, override_params=cfg['params'])
            p2 = np.maximum.accumulate(b2, axis=1)
            w2 = float(((b2 - p2) / p2 * 100).min(axis=1).min())
            robustness.append({'seed': sd, 'worst_dd_pct': round(w2, 1), 'safe': bool(w2 >= FLOOR_DD)})

        dd, maxdd, final, profit = metrics_from_bankroll(br)
        checks = recheck(br, bet_history, odds)
        print("  Проверки корректности:")
        for k in ['no_negative_bank', 'bet_le_10pct', 'dd_ge_minus100',
                  'passes_floor_-66', 'balance_invariant_ok']:
            print(f"    {k}: {checks[k]}")
        print(f"    worst_dd={checks['worst_dd']:.2f}%  max_bet={checks['max_bet_pct']:.3f}%")

        png = os.path.join(OUT, f"{cfg['id']}.png")
        fan = spaghetti_and_fan(br, cfg['label'], png)
        print(f"  График: {png}")

        row = {
            'id': cfg['id'], 'label': cfg['label'],
            'params': cfg['params'],
            'median_profit_pct': float(np.median(profit)),
            'avg_profit_pct': float(np.mean(profit)),
            'p5_profit_pct': float(np.percentile(profit, 5)),
            'p1_profit_pct': float(np.percentile(profit, 1)),
            'worst_dd_pct': float(maxdd.min()),
            'median_maxdd_pct': float(np.median(maxdd)),
            'p5_maxdd_pct': float(np.percentile(maxdd, 5)),  # 5й перцентиль самых глубоких
            'bankrupt_pct': float((final < 1.0).mean() * 100),
            'avg_bet_pct': checks['max_bet_pct'],  # покажем и средн. ниже
            'checks': checks,
            'robustness': robustness,
            'fan': fan,
            'maxdd_hist': dd_distribution(maxdd),
            'png': os.path.basename(png),
        }
        # средняя ставка
        before = br[:, :NUM_BETS]
        pct = np.where(before > 0, bet_history / before * 100, 0.0)
        row['avg_bet_pct'] = float(pct[pct > 0].mean())
        top_out['strategies'].append(row)

    # Сводка порогов: сколько стратегий (из всех 50 capped) НЕ пробивают -X%
    summary = []
    rows = cap_summary['strategies']
    for thr in DD_THRESHOLDS:
        safe = sum(1 for r in rows if r['worst_dd_pct'] >= -thr)
        summary.append({'threshold_pct': thr, 'safe_count': safe, 'total': len(rows)})
    top_out['threshold_summary'] = summary

    with open(os.path.join(OUT, 'top_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(top_out, f, ensure_ascii=False, indent=2)

    print("\n=== СВОДКА ПО ПОРОГАМ (capped, 50 прогонов) ===")
    for s in summary:
        print(f"  Не падают ниже {100-s['threshold_pct']}% от пика "
              f"(DD<={s['threshold_pct']}%): {s['safe_count']}/{s['total']}")
    print(f"\nГотово. Данные: {OUT}/top_summary.json")


if __name__ == '__main__':
    main()
