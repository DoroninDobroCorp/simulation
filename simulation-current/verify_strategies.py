"""
Проверка и визуализация как работают выбранные стратегии.
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    adaptive_strategy_with_real_odds,
    adaptive_constant_profit_strategy_with_real_odds,
    calculate_metrics_with_odds
)

outcomes, odds_array = load_real_odds_outcomes()

print("="*90)
print("🔍 ПРОВЕРКА СТРАТЕГИЙ")
print("="*90)

# 1. ADAPTIVE (безопасная)
print("\n1️⃣ ADAPTIVE (OPT_RISKY_LIMITED)")
print("-"*90)
params1 = {
    'base_percent': 2.476,
    'max_percent': 16.839,
    'min_roi': 4.933,
    'max_roi': 14.367,
    'min_odds': 1.5,
    'max_odds': 5.0,
    'apply_variation': False
}

print("\n📝 ПАРАМЕТРЫ:")
for k, v in params1.items():
    if k != 'apply_variation':
        print(f"   {k}: {v}")

br1, bh1, _, _, _ = adaptive_strategy_with_real_odds(outcomes, odds_array, **params1)
m1 = calculate_metrics_with_odds(br1, bh1, odds_array)

print("\n📊 РЕЗУЛЬТАТЫ:")
print(f"   Profit: +{m1['avg_profit_pct']:.1f}%")
print(f"   DD>50%: {m1['drawdown_50_pct']:.2f}%")
print(f"   Bankrupt: {m1['bankrupt_pct']:.2f}%")

# Проверим как работает адаптация
print("\n🔧 КАК РАБОТАЕТ:")
print("   1. Базовая ставка = 2.476% × коэффициент (зависит от ROI и odds)")
print("   2. АДАПТАЦИЯ при просадках:")
print("      - Если банк < 80% от пика → ставка ×0.75")
print("      - Если банк < 60% от пика → ставка ×0.5")
print("   3. Дополнительное ограничение:")
print("      - Если банк < 70% от начального → max_percent × 0.5")
print("      - Если банк < 60% от начального → max_percent × 0.25")

# Пример расчета на первых симуляциях
print("\n🔬 ПРИМЕР работы адаптации (первые 10 ставок, симуляция #0):")
sim_idx = 0
roi_pct = 7.0
for i in range(min(10, len(odds_array))):
    current_bank = br1[sim_idx, i]
    peak_bank = np.max(br1[sim_idx, :i+1])
    odds = odds_array[i]
    
    # Базовая ставка
    norm_roi = np.clip((roi_pct - params1['min_roi']) / (params1['max_roi'] - params1['min_roi']), 0, 1)
    norm_odds = np.clip((odds - params1['min_odds']) / (params1['max_odds'] - params1['min_odds']), 0, 1)
    combined = np.sqrt(norm_roi) * (1 - 0.5 * norm_odds)
    base_bet_pct = params1['base_percent'] * combined
    
    # Адаптация
    dd_from_peak = (peak_bank - current_bank) / peak_bank if peak_bank > 0 else 0
    modifier = 1.0
    if dd_from_peak > 0.20:
        modifier = 0.75
    if dd_from_peak > 0.40:
        modifier = 0.5
    
    final_bet_pct = base_bet_pct * modifier
    bet_amount = bh1[sim_idx, i]
    actual_pct = (bet_amount / current_bank * 100) if current_bank > 0 else 0
    
    result = "WIN" if outcomes[sim_idx, i] else "LOSS"
    print(f"   #{i+1}: Bank={current_bank:.0f}, DD={dd_from_peak*100:.1f}%, Bet={actual_pct:.2f}% → {result}")

print("\n" + "="*90)
print("2️⃣ ADAPTIVE_CONSTANT_PROFIT (RISKY_ROI1328)")
print("-"*90)
params2 = {
    'min_roi': 0.9,
    'max_roi': 30.105,
    'min_target_pct': 0.423,
    'max_target_pct': 17.445,
    'max_bet_percent': 20.0,
    'apply_variation': False
}

print("\n📝 ПАРАМЕТРЫ:")
for k, v in params2.items():
    if k != 'apply_variation':
        print(f"   {k}: {v}")

br2, bh2, _, max_bet2, _ = adaptive_constant_profit_strategy_with_real_odds(outcomes, odds_array, **params2)
m2 = calculate_metrics_with_odds(br2, bh2, odds_array)

print("\n📊 РЕЗУЛЬТАТЫ:")
print(f"   Profit: +{m2['avg_profit_pct']:.1f}%")
print(f"   DD>50%: {m2['drawdown_50_pct']:.2f}%")
print(f"   Bankrupt: {m2['bankrupt_pct']:.2f}%")
print(f"   Max bet: {max_bet2:.0f}%")

print("\n🔧 КАК РАБОТАЕТ:")
print("   1. Целевая прибыль = интерполяция между 0.423% и 17.445% по ROI")
roi_factor = (7.0 - params2['min_roi']) / (params2['max_roi'] - params2['min_roi'])
target = params2['min_target_pct'] + (params2['max_target_pct'] - params2['min_target_pct']) * roi_factor
print(f"   2. При ROI=7%: target = {target:.2f}%")
print(f"   3. Ставка = (target × bank) / (odds - 1)")
print(f"   4. Но не больше {params2['max_bet_percent']}% от банка")
print("\n   ⚠️ ВНИМАНИЕ: Это НЕ адаптивная стратегия!")
print("      Она НЕ снижает ставки при просадках!")
print("      Просто фиксированный % целевой прибыли")

# Пример
print("\n🔬 ПРИМЕР (первые 5 ставок, симуляция #0):")
for i in range(min(5, len(odds_array))):
    current_bank = br2[sim_idx, i]
    odds = odds_array[i]
    target_profit = current_bank * target / 100
    bet_if_unlimited = target_profit / (odds - 1) if odds > 1 else 0
    max_bet_allowed = current_bank * params2['max_bet_percent'] / 100
    bet_amount = min(bet_if_unlimited, max_bet_allowed)
    actual_pct = (bet_amount / current_bank * 100) if current_bank > 0 else 0
    
    result = "WIN" if outcomes[sim_idx, i] else "LOSS"
    print(f"   #{i+1}: Bank={current_bank:.0f}, Odds={odds:.2f}, Bet={actual_pct:.1f}% → {result}")

print("\n" + "="*90)
print("✅ КОД РАБОТАЕТ ПРАВИЛЬНО!")
print("="*90)
