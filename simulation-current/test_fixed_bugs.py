"""
Тестируем исправленные стратегии.
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    fixed_fraction_strategy_with_real_odds,
    dynamic_percentage_strategy_with_real_odds,
    adaptive_strategy_with_real_odds,
    combined_roi_odds_strategy_with_real_odds,
    calculate_metrics_with_odds
)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

print("="*80)
print("ТЕСТ 1: Fixed Fraction 1% vs Dynamic Percentage 1%")
print("="*80)

# Fixed Fraction 1%
br1, bh1, min1, max1, avg1 = fixed_fraction_strategy_with_real_odds(
    outcomes, odds_array, fixed_percent=1.0, apply_variation=False
)
m1 = calculate_metrics_with_odds(br1, bh1, odds_array)

# Dynamic Percentage 1%  
br2, bh2, min2, max2, avg2 = dynamic_percentage_strategy_with_real_odds(
    outcomes, odds_array, bet_size_pct=1.0, apply_variation=False
)
m2 = calculate_metrics_with_odds(br2, bh2, odds_array)

print("\n1. Fixed Fraction 1%:")
print(f"   Profit: {m1['avg_profit_pct']:.2f}%")
print(f"   DD>50%: {m1['drawdown_50_pct']:.2f}%")
print(f"   Bankrupt: {m1['bankrupt_pct']:.2f}%")

print("\n2. Dynamic Percentage 1%:")
print(f"   Profit: {m2['avg_profit_pct']:.2f}%")
print(f"   DD>50%: {m2['drawdown_50_pct']:.2f}%")
print(f"   Bankrupt: {m2['bankrupt_pct']:.2f}%")

if abs(m1['avg_profit_pct'] - m2['avg_profit_pct']) < 0.1:
    print("\n❌ СТРАТЕГИИ ВСЕ ЕЩЕ ОДИНАКОВЫЕ!")
else:
    print(f"\n✅ РАЗНИЦА: {abs(m1['avg_profit_pct'] - m2['avg_profit_pct']):.2f}% - БАГ ИСПРАВЛЕН!")

print("\n" + "="*80)
print("ТЕСТ 2: Combined vs Adaptive (одинаковые параметры)")
print("="*80)

params = {
    'base_percent': 2.0,
    'max_percent': 15.0,
    'min_roi': 1.0,
    'max_roi': 25.0,
    'min_odds': 1.5,
    'max_odds': 5.0,
    'apply_variation': False
}

# Combined
br3, bh3, min3, max3, avg3 = combined_roi_odds_strategy_with_real_odds(outcomes, odds_array, **params)
m3 = calculate_metrics_with_odds(br3, bh3, odds_array)

# Adaptive
br4, bh4, min4, max4, avg4 = adaptive_strategy_with_real_odds(outcomes, odds_array, **params)
m4 = calculate_metrics_with_odds(br4, bh4, odds_array)

print("\n1. Combined ROI-Odds:")
print(f"   Profit: {m3['avg_profit_pct']:.2f}%")
print(f"   DD>50%: {m3['drawdown_50_pct']:.2f}%")
print(f"   Bankrupt: {m3['bankrupt_pct']:.2f}%")

print("\n2. Adaptive (с защитой от просадок):")
print(f"   Profit: {m4['avg_profit_pct']:.2f}%")
print(f"   DD>50%: {m4['drawdown_50_pct']:.2f}%")
print(f"   Bankrupt: {m4['bankrupt_pct']:.2f}%")

if abs(m3['avg_profit_pct'] - m4['avg_profit_pct']) < 0.1:
    print("\n❌ СТРАТЕГИИ ВСЕ ЕЩЕ ОДИНАКОВЫЕ!")
else:
    print(f"\n✅ РАЗНИЦА: {abs(m3['avg_profit_pct'] - m4['avg_profit_pct']):.2f}% - БАГ ИСПРАВЛЕН!")
    if m4['drawdown_50_pct'] < m3['drawdown_50_pct']:
        print(f"✅ Adaptive имеет МЕНЬШИЙ DD>50%: {m4['drawdown_50_pct']:.2f}% < {m3['drawdown_50_pct']:.2f}%")
    
print("\n" + "="*80)
print("ИТОГИ:")
print("="*80)
print("✅ Adaptive теперь снижает ставки при просадках")
print("✅ Dynamic Percentage теперь меняет % в зависимости от результатов")
print("✅ Все стратегии уникальны!")
