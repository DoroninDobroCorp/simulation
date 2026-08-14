"""
Оркестратор прогона всех стратегий при ROI=2.5%.

Сначала SMOKE-прогон (маленький набор) для проверки корректности на СТАРТЕ:
  - фактический ROI оборота ≈ целевой
  - метрики/распределения считаются, кэш пишется
Затем (если smoke ok) — ПОЛНЫЙ прогон 10000 x 10000, оба режима.

Использование:
  python3 run_all_2.5.py smoke   # только проверка
  python3 run_all_2.5.py full    # полный прогон
  python3 run_all_2.5.py         # smoke, затем full
"""

import sys
import numpy as np
import sim_engine

ROI = 0.025


def smoke():
    print("=" * 70)
    print("SMOKE-ПРОВЕРКА (500 sims x 1000 bets) — корректность на старте")
    print("=" * 70)
    summary = sim_engine.run_all(num_sims=500, num_bets=1000, roi=ROI,
                                 cache_dir=sim_engine.CACHE_DIR + '_smoke')

    actual = summary['actual_roi_turnover_pct']
    target = summary['roi_pct']
    ok = abs(actual - target) < 0.5  # допуск 0.5 п.п. на шум маленькой выборки
    print("\n--- SANITY CHECK ---")
    print(f"Целевой ROI: {target:.3f}%, фактический ROI оборота: {actual:.3f}% -> "
          f"{'OK' if ok else 'РАССОГЛАСОВАНИЕ!'}")
    n = len(summary['strategies'])
    print(f"Стратегий x режимов посчитано: {n} (ожидалось {25*2})")
    # Проверка, что метрики не вырождены
    profits = [s['avg_profit_pct'] for s in summary['strategies']]
    print(f"Диапазон avg_profit: {min(profits):.1f}% .. {max(profits):.1f}%")
    assert ok, "Фактический ROI не совпал с целевым — прогон некорректен!"
    assert n == 25 * 2, "Не все стратегии/режимы посчитаны!"
    print("SMOKE OK\n")
    return True


def full():
    print("=" * 70)
    print("ПОЛНЫЙ ПРОГОН (10000 sims x 10000 bets), оба режима, ROI=2.5%")
    print("=" * 70)
    summary = sim_engine.run_all(num_sims=10000, num_bets=10000, roi=ROI,
                                 cache_dir=sim_engine.CACHE_DIR)
    actual = summary['actual_roi_turnover_pct']
    print("\n--- ИТОГ ---")
    print(f"Фактический ROI оборота: {actual:.3f}% (целевой {summary['roi_pct']:.2f}%)")
    print(f"Результаты в: {sim_engine.CACHE_DIR}/")
    print("Запусти дашборд:  python3 dashboard.py")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode == 'smoke':
        smoke()
    elif mode == 'full':
        full()
    else:
        smoke()
        full()
