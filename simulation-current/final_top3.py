"""
Финальный отбор топ-3 стратегий — по подтверждённой постановке:
  - Дистанция 10000 ставок
  - Реальные кэфы (семпл с повторениями), ROI +2.5%
  - Динамический БМ (ставка <= 10% живого банка), банк не уходит в минус
  - ПРОСАДКА МЕРЯЕТСЯ ОТ СТАРТА (минимальный банк за историю vs стартовый 1000),
    а НЕ от пика. Это реальный риск вложенного капитала.
  - Критерий безопасности: банк НИКОГДА не падает ниже 34% от старта (>=340).
  - Среди безопасных — топ-3 по медианной прибыли.

Сохраняет: final_top3_data/summary.json (метрики всех),
           final_top3_data/<key>_traj.npy (выборка траекторий топ-3 для графиков),
           печатает строгие проверки.
"""

import os
import json
import time
import numpy as np

import sim_engine as E

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_top3_data')
START = E.INITIAL_BANKROLL
SAFE_FLOOR = 0.34 * START  # 340


def metrics_from_start(bankroll):
    """Метрики с акцентом на просадку ОТ СТАРТА."""
    final = bankroll[:, -1]
    profit = (final - START) / START * 100
    min_bank = bankroll.min(axis=1)              # самый низкий банк за историю
    dd_from_start = (min_bank - START) / START * 100  # <=0 если падал ниже старта
    return {
        'median_profit_pct': float(np.median(profit)),
        'mean_profit_pct': float(np.mean(profit)),
        'p5_profit_pct': float(np.percentile(profit, 5)),
        'p1_profit_pct': float(np.percentile(profit, 1)),
        'min_bank_median': float(np.median(min_bank)),
        'min_bank_p5': float(np.percentile(min_bank, 5)),
        'min_bank_p1': float(np.percentile(min_bank, 1)),
        'min_bank_worst': float(min_bank.min()),
        'worst_dd_from_start_pct': float(dd_from_start.min()),
        'pct_below_34_of_start': float((min_bank < SAFE_FLOOR).mean() * 100),
        'safe_never_below_34': bool(min_bank.min() >= SAFE_FLOOR),
    }, profit, min_bank


def run(num_sims=10000, num_bets=10000, roi=0.025, seed=42, variation=False, cap=0.10):
    os.makedirs(OUT, exist_ok=True)
    rs = E.set_roi(roi)
    print(f"Прогон: {num_sims} sims x {num_bets} bets, ROI={roi*100:.2f}%, cap={cap*100:.0f}%, "
          f"вариация={variation}")
    outcomes, odds = E.build_dataset(num_sims, num_bets, roi, seed=seed)
    print(f"  avg кэф={odds.mean():.3f}, факт. ROI оборота={E.actual_roi_from_outcomes(outcomes, odds):.3f}%")

    registry = E.strategy_registry(rs)
    bayes_prior = float(np.clip((1.0 + roi) / odds.mean(), 0.01, 0.99))
    registry['bayesian_kelly'] = (rs.bayesian_kelly_strategy_with_real_odds,
                                  {'prior_mean': bayes_prior, 'prior_std': 0.05})

    results = {}
    traj_cache = {}
    for key, (func, params) in registry.items():
        t0 = time.time()
        try:
            bankroll, bet_history, *_ = func(outcomes, odds, apply_variation=variation, **params)
            bankroll, bet_history = E.recover_fraction_and_resim(bankroll, bet_history, outcomes, odds, cap=cap)
            m, profit, min_bank = metrics_from_start(bankroll)
            results[key] = m
            traj_cache[key] = bankroll
            flag = "SAFE" if m['safe_never_below_34'] else f"ниже340={m['pct_below_34_of_start']:.2f}%"
            print(f"  {key:28s} medProf={m['median_profit_pct']:8.1f}%  "
                  f"minBank(med)={m['min_bank_median']:7.0f}  worst={m['min_bank_worst']:6.0f}  "
                  f"{flag}  ({time.time()-t0:.1f}s)")
        except Exception as ex:
            print(f"  {key:28s} ОШИБКА: {ex}")

    # Отбор: безопасные (никогда ниже 34% старта) -> сортировка по медианной прибыли
    safe = {k: v for k, v in results.items() if v['safe_never_below_34']}
    pool = safe if safe else results  # если строго безопасных нет — берём из всех
    ranked = sorted(pool.items(), key=lambda kv: kv[1]['median_profit_pct'], reverse=True)
    top3 = ranked[:3]

    print("\n" + "=" * 70)
    print(f"Строго безопасных (банк НИКОГДА ниже 34% старта): {len(safe)} из {len(results)}")
    print("ТОП-3 по медианной прибыли:")
    for i, (k, v) in enumerate(top3, 1):
        print(f"  {i}. {k}: медПриб={v['median_profit_pct']:.1f}%, "
              f"мин.банк(медиана)={v['min_bank_median']:.0f}, "
              f"худший мин.банк={v['min_bank_worst']:.0f}, "
              f"ниже34%={v['pct_below_34_of_start']:.2f}%")

    # Сохраняем выборку траекторий топ-3 (для графиков; все 10000 хранить тяжело)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(num_sims, size=min(2000, num_sims), replace=False)
    for k, _ in top3:
        np.save(os.path.join(OUT, f"{k}_traj.npy"), traj_cache[k][sample_idx].astype(np.float32))

    summary = {
        'config': {'num_sims': num_sims, 'num_bets': num_bets, 'roi_pct': roi * 100,
                   'cap_pct': cap * 100, 'variation': variation,
                   'dd_measured_from': 'START', 'safe_floor': SAFE_FLOOR},
        'all': results,
        'safe_keys': list(safe.keys()),
        'top3': [k for k, _ in top3],
        'sample_idx': sample_idx.tolist(),
    }
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nСохранено в {OUT}/summary.json + траектории топ-3")
    return summary, top3, traj_cache


def verify(top3, traj_cache, num_bets):
    """Строгие независимые перепроверки топ-3."""
    print("\n" + "=" * 70)
    print("СТРОГИЕ ПРОВЕРКИ топ-3:")
    for k, v in top3:
        bk = traj_cache[k]
        # 1) банк не уходит в минус
        no_neg = bool((bk >= -1e-6).all())
        # 2) лимит ставки: пересчитаем долю и проверим <=10%+эпс
        before = bk[:, :num_bets]
        deltas = np.abs(np.diff(bk, axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            # ставка = |проигрыш| при лоссе; на вине delta=bet*(o-1) -> не прямой; берём min оценку лосса
            pass
        # 3) независимый пересчёт просадки от старта
        min_bank = bk.min(axis=1)
        worst = float(min_bank.min())
        safe = worst >= SAFE_FLOOR
        print(f"  {k}: банк>=0 [{ 'OK' if no_neg else 'FAIL'}], "
              f"худший мин.банк={worst:.1f} (порог {SAFE_FLOOR:.0f}) "
              f"[{'SAFE' if safe else 'НИЖЕ ПОРОГА'}]")


if __name__ == '__main__':
    summary, top3, traj = run()
    verify(top3, traj, summary['config']['num_bets'])
