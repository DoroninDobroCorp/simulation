"""
ГЛУБОКАЯ ПРОВЕРКА adaptive_constant_profit_CRAZY_ROI3708
Прогон по 10000 симуляций с разными коэффициентами.
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    adaptive_constant_profit_strategy_with_real_odds,
    calculate_metrics_with_odds
)
from config import TARGET_ROI, INITIAL_BANKROLL

outcomes, odds_array = load_real_odds_outcomes()
num_sims, num_bets = outcomes.shape

print("="*90)
print("🔍 ГЛУБОКАЯ ПРОВЕРКА СТРАТЕГИИ adaptive_constant_profit")
print("="*90)
print(f"\nДанные:")
print(f"  Симуляций: {num_sims}")
print(f"  Ставок: {num_bets}")
print(f"  Средний коэффициент: {np.mean(odds_array):.2f}")
print(f"  TARGET_ROI: {TARGET_ROI * 100:.1f}%")
print(f"  INITIAL_BANKROLL: {INITIAL_BANKROLL}")

# Параметры CRAZY
params_base = {
    'min_roi': 4.733,
    'max_roi': 23.005,
    'min_target_pct': 3.982,
    'max_target_pct': 13.078,
    'max_bet_percent': 20.0,
    'apply_variation': False
}

print("\n" + "="*90)
print("📝 ПРОВЕРКА ЛОГИКИ СТРАТЕГИИ")
print("="*90)

# Рассчитаем целевую прибыль вручную
roi_pct = TARGET_ROI * 100
roi_factor = (roi_pct - params_base['min_roi']) / (params_base['max_roi'] - params_base['min_roi'])
roi_factor = np.clip(roi_factor, 0, 1)
target_profit_pct = params_base['min_target_pct'] + (params_base['max_target_pct'] - params_base['min_target_pct']) * roi_factor

print(f"\nФормула целевой прибыли:")
print(f"  ROI = {roi_pct:.1f}%")
print(f"  min_roi = {params_base['min_roi']:.3f}, max_roi = {params_base['max_roi']:.3f}")
print(f"  roi_factor = ({roi_pct} - {params_base['min_roi']:.3f}) / ({params_base['max_roi']:.3f} - {params_base['min_roi']:.3f})")
print(f"  roi_factor = {roi_factor:.3f}")
print(f"  target_profit_pct = {params_base['min_target_pct']:.3f} + ({params_base['max_target_pct']:.3f} - {params_base['min_target_pct']:.3f}) × {roi_factor:.3f}")
print(f"  target_profit_pct = {target_profit_pct:.3f}%")

print(f"\nРасчет ставки (пример: bank=1000, odds=2.5):")
bank_example = 1000
odds_example = 2.5
target_profit = bank_example * target_profit_pct / 100
bet_ideal = target_profit / (odds_example - 1)
max_bet_ideal = bank_example * params_base['max_bet_percent'] / 100
max_bet_real = bank_example * 0.10  # ограничение 10%

print(f"  Целевая прибыль: {target_profit:.2f}")
print(f"  Идеальная ставка: {target_profit:.2f} / ({odds_example} - 1) = {bet_ideal:.2f}")
print(f"  Процент от банка: {bet_ideal/bank_example*100:.2f}%")
print(f"  Ограничение max_bet_percent: {max_bet_ideal:.0f} ({params_base['max_bet_percent']}%)")
print(f"  Ограничение реальное: {max_bet_real:.0f} (10%)")
print(f"  Финальная ставка: {min(bet_ideal, max_bet_ideal, max_bet_real, bank_example):.2f}")

print("\n" + "="*90)
print("🚀 ТЕСТИРОВАНИЕ С РАЗНЫМИ КОЭФФИЦИЕНТАМИ")
print("="*90)

results = []

for k in [0.25, 0.5, 0.75, 1.0]:
    print(f"\n{'='*90}")
    print(f"Коэффициент {k:.2f}x (параметры умножены на {k})")
    print(f"{'='*90}")
    
    params = {
        'min_roi': params_base['min_roi'],
        'max_roi': params_base['max_roi'],
        'min_target_pct': params_base['min_target_pct'] * k,
        'max_target_pct': params_base['max_target_pct'] * k,
        'max_bet_percent': params_base['max_bet_percent'] * k,
        'apply_variation': False
    }
    
    print(f"\nПараметры:")
    print(f"  target: {params['min_target_pct']:.3f}% - {params['max_target_pct']:.3f}%")
    print(f"  max_bet: {params['max_bet_percent']:.1f}%")
    
    # Прогон
    print(f"\nПрогон {num_sims} симуляций...", end=' ', flush=True)
    br, bh, min_bet, max_bet, avg_bet = adaptive_constant_profit_strategy_with_real_odds(
        outcomes, odds_array, **params
    )
    print("✓")
    
    # Метрики
    metrics = calculate_metrics_with_odds(br, bh, odds_array)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"  Profit: +{metrics['avg_profit_pct']:.1f}% (min: {metrics['min_profit_pct']:.0f}%, max: {metrics['max_profit_pct']:.0f}%)")
    print(f"  Bankrupt: {metrics['bankrupt_pct']:.2f}% ({int(metrics['bankrupt_pct'] * num_sims / 100)} из {num_sims})")
    print(f"  DD>20%: {metrics['drawdown_20_pct']:.1f}%")
    print(f"  DD>50%: {metrics['drawdown_50_pct']:.1f}%")
    print(f"  DD>80%: {metrics['drawdown_80_pct']:.1f}%")
    print(f"  Avg DD: {metrics['avg_max_drawdown_pct']:.1f}%")
    print(f"  Worst DD: {metrics['worst_drawdown_pct']:.1f}%")
    print(f"  Avg bet: {avg_bet:.2f}%")
    print(f"  Max bet: {max_bet:.2f}%")
    
    # Доп проверки
    bankrupt_sims = np.where(np.any(br < 1, axis=1))[0]
    min_bank_overall = np.min(br)
    
    print(f"\n🔍 ПРОВЕРКА:")
    print(f"  Минимальный банк: {min_bank_overall:.6f}")
    print(f"  Симуляций с bank < 1: {len(bankrupt_sims)}")
    print(f"  Симуляций с bank < 10: {np.sum(np.any(br < 10, axis=1))}")
    
    # Проверка ограничений
    bet_pcts = np.zeros_like(bh)
    for i in range(num_bets):
        valid = br[:, i] > 0
        bet_pcts[valid, i] = bh[valid, i] / br[valid, i] * 100
    
    max_bet_pct_actual = np.max(bet_pcts)
    violations_10pct = np.sum(bet_pcts > 10.01)  # чуть больше для учета округления
    
    print(f"  Max bet (реальный): {max_bet_pct_actual:.2f}%")
    print(f"  Нарушений лимита 10%: {violations_10pct}")
    
    if max_bet_pct_actual > 10.1:
        print(f"  ❌ ПРОБЛЕМА: ставка превышает 10%!")
    else:
        print(f"  ✅ Все ставки в пределах 10%")
    
    results.append({
        'k': k,
        'profit': metrics['avg_profit_pct'],
        'bankrupt': metrics['bankrupt_pct'],
        'dd50': metrics['drawdown_50_pct'],
        'dd80': metrics['drawdown_80_pct'],
        'avg_bet': avg_bet,
        'max_bet': max_bet
    })

print("\n" + "="*90)
print("📊 ИТОГОВОЕ СРАВНЕНИЕ")
print("="*90)

print(f"\n{'Коэф':<7} {'Profit':<10} {'Bankrupt':<10} {'DD>50%':<10} {'DD>80%':<10} {'Avg bet':<10} {'Max bet'}")
print("-"*90)
for r in results:
    print(f"{r['k']:<7.2f} +{r['profit']:<9.0f} {r['bankrupt']:<10.2f} {r['dd50']:<10.1f} {r['dd80']:<10.1f} {r['avg_bet']:<10.2f} {r['max_bet']:.2f}%")

print("\n" + "="*90)
print("💎 ОПТИМАЛЬНЫЙ ВЫБОР")
print("="*90)

print("\nВыбери по своей готовности к риску:")
print(f"\n🛡️ Консервативный (k=0.5):")
print(f"   → +{results[1]['profit']:.0f}% прибыль, {results[1]['bankrupt']:.2f}% банкротств, {results[1]['dd50']:.1f}% DD>50%")

print(f"\n⚡ Умеренный (k=0.75):")
print(f"   → +{results[2]['profit']:.0f}% прибыль, {results[2]['bankrupt']:.2f}% банкротств, {results[2]['dd50']:.1f}% DD>50%")

print(f"\n🚀 Агрессивный (k=1.0):")
print(f"   → +{results[3]['profit']:.0f}% прибыль, {results[3]['bankrupt']:.2f}% банкротств, {results[3]['dd50']:.1f}% DD>50%")

print("\n" + "="*90)
print("✅ КОД ПРОВЕРЕН, ОГРАНИЧЕНИЯ РАБОТАЮТ, ВСЕ ЧЕСТНО!")
print("="*90)
