"""
ФИНАЛЬНЫЙ ТЕСТ ТОП-3 СТРАТЕГИЙ на 10,000 симуляций
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from realistic_simulation import adaptive_constant_profit_realistic, calculate_metrics_realistic

# Импортируем функции теста
import sys
sys.path.insert(0, '/Users/vladimirdoronin/VovkaNowEngineer/simulation_new')
from test_all_strategies_realistic import test_linear_roi_realistic, test_exponential_roi_realistic, test_anti_martingale_realistic


outcomes, odds_array = load_real_odds_outcomes()

print("="*100)
print("🏆 ФИНАЛЬНЫЙ ТЕСТ ТОП-3 СТРАТЕГИЙ (10,000 симуляций)")
print("="*100)
print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")

results = []

print("\n" + "="*100)
print("1️⃣ LINEAR ROI (base=5.0%)")
print("="*100)
print("Тестируем...", end=' ', flush=True)
br1 = test_linear_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=5.0, max_percent=10.0)
m1 = calculate_metrics_realistic(br1, np.zeros_like(outcomes, dtype=float), odds_array)
print("✓")
print(f"\n📊 Результаты:")
print(f"  💰 Profit: +{m1['avg_profit_pct']:.0f}%")
print(f"  ⚠️  Bankrupt: {m1['bankrupt_pct']:.2f}% ({int(m1['bankrupt_pct']*100):.0f} из 10000)")
print(f"  ⚠️  DD>50%: {m1['drawdown_50_pct']:.1f}%, DD>80%: {m1['drawdown_80_pct']:.1f}%")
print(f"  📊 Worst DD: {m1['worst_drawdown_pct']:.1f}%")

print(f"\n💶 НА 1,000 ЕВРО:")
print(f"  Прибыль: +{1000 * m1['avg_profit_pct'] / 100:,.0f} евро")
results.append(('linear_roi base=5.0%', m1))

print("\n" + "="*100)
print("2️⃣ ADAPTIVE CONSTANT PROFIT (k=2.0)")
print("="*100)
print("Тестируем...", end=' ', flush=True)
br2, _, _, _, _ = adaptive_constant_profit_realistic(
    outcomes, odds_array,
    min_roi=4.733, max_roi=23.005,
    min_target_pct=3.982 * 2.0,
    max_target_pct=13.078 * 2.0,
    max_bet_percent=20.0 * 2.0,
    apply_variation=False,
    recalc_min=30, recalc_max=70
)
m2 = calculate_metrics_realistic(br2, np.zeros_like(outcomes, dtype=float), odds_array)
print("✓")
print(f"\n📊 Результаты:")
print(f"  💰 Profit: +{m2['avg_profit_pct']:.0f}%")
print(f"  ⚠️  Bankrupt: {m2['bankrupt_pct']:.2f}% ({int(m2['bankrupt_pct']*100):.0f} из 10000)")
print(f"  ⚠️  DD>50%: {m2['drawdown_50_pct']:.1f}%, DD>80%: {m2['drawdown_80_pct']:.1f}%")
print(f"  📊 Worst DD: {m2['worst_drawdown_pct']:.1f}%")

print(f"\n💶 НА 1,000 ЕВРО:")
print(f"  Прибыль: +{1000 * m2['avg_profit_pct'] / 100:,.0f} евро")
results.append(('adaptive_constant_profit k=2.0', m2))

print("\n" + "="*100)
print("3️⃣ ADAPTIVE CONSTANT PROFIT (k=1.5) - 0% СЛИВОВ")
print("="*100)
print("Тестируем...", end=' ', flush=True)
br3, _, _, _, _ = adaptive_constant_profit_realistic(
    outcomes, odds_array,
    min_roi=4.733, max_roi=23.005,
    min_target_pct=3.982 * 1.5,
    max_target_pct=13.078 * 1.5,
    max_bet_percent=20.0 * 1.5,
    apply_variation=False,
    recalc_min=30, recalc_max=70
)
m3 = calculate_metrics_realistic(br3, np.zeros_like(outcomes, dtype=float), odds_array)
print("✓")
print(f"\n📊 Результаты:")
print(f"  💰 Profit: +{m3['avg_profit_pct']:.0f}%")
print(f"  ⚠️  Bankrupt: {m3['bankrupt_pct']:.2f}% ({int(m3['bankrupt_pct']*100):.0f} из 10000)")
print(f"  ⚠️  DD>50%: {m3['drawdown_50_pct']:.1f}%, DD>80%: {m3['drawdown_80_pct']:.1f}%")
print(f"  📊 Worst DD: {m3['worst_drawdown_pct']:.1f}%")

print(f"\n💶 НА 1,000 ЕВРО:")
print(f"  Прибыль: +{1000 * m3['avg_profit_pct'] / 100:,.0f} евро")
results.append(('adaptive_constant_profit k=1.5', m3))

print("\n" + "="*100)
print("4️⃣ EXPONENTIAL ROI (base=3.0%, exponent=2.0)")
print("="*100)
print("Тестируем...", end=' ', flush=True)
br4 = test_exponential_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=3.0, exponent=2.0, max_percent=10.0)
m4 = calculate_metrics_realistic(br4, np.zeros_like(outcomes, dtype=float), odds_array)
print("✓")
print(f"\n📊 Результаты:")
print(f"  💰 Profit: +{m4['avg_profit_pct']:.0f}%")
print(f"  ⚠️  Bankrupt: {m4['bankrupt_pct']:.2f}% ({int(m4['bankrupt_pct']*100):.0f} из 10000)")
print(f"  ⚠️  DD>50%: {m4['drawdown_50_pct']:.1f}%, DD>80%: {m4['drawdown_80_pct']:.1f}%")
print(f"  📊 Worst DD: {m4['worst_drawdown_pct']:.1f}%")

print(f"\n💶 НА 1,000 ЕВРО:")
print(f"  Прибыль: +{1000 * m4['avg_profit_pct'] / 100:,.0f} евро")
results.append(('exponential_roi base=3%,e=2', m4))

print("\n" + "="*100)
print("5️⃣ ANTI-MARTINGALE (base=3.0%, multiplier=1.5) - 0% СЛИВОВ")
print("="*100)
print("Тестируем...", end=' ', flush=True)
br5 = test_anti_martingale_realistic(outcomes, odds_array, base_percent=3.0, multiplier=1.5, max_percent=10.0)
m5 = calculate_metrics_realistic(br5, np.zeros_like(outcomes, dtype=float), odds_array)
print("✓")
print(f"\n📊 Результаты:")
print(f"  💰 Profit: +{m5['avg_profit_pct']:.0f}%")
print(f"  ⚠️  Bankrupt: {m5['bankrupt_pct']:.2f}% ({int(m5['bankrupt_pct']*100):.0f} из 10000)")
print(f"  ⚠️  DD>50%: {m5['drawdown_50_pct']:.1f}%, DD>80%: {m5['drawdown_80_pct']:.1f}%")
print(f"  📊 Worst DD: {m5['worst_drawdown_pct']:.1f}%")

print(f"\n💶 НА 1,000 ЕВРО:")
print(f"  Прибыль: +{1000 * m5['avg_profit_pct'] / 100:,.0f} евро")
results.append(('anti_martingale base=3%,m=1.5', m5))

print("\n" + "="*100)
print("💎 ИТОГОВОЕ СРАВНЕНИЕ")
print("="*100)

print(f"\n{'Стратегия':<40} {'Profit':<12} {'Bankrupt':<12} {'DD>50%':<12} {'DD>80%'}")
print("-"*100)
for name, m in results:
    print(f"{name:<40} +{m['avg_profit_pct']:<11.0f} {m['bankrupt_pct']:<12.2f} {m['drawdown_50_pct']:<12.1f} {m['drawdown_80_pct']:.1f}%")

# Лучший с 0% банкротств
zero_bankrupt = [(name, m) for name, m in results if m['bankrupt_pct'] == 0]
if zero_bankrupt:
    best_zero = max(zero_bankrupt, key=lambda x: x[1]['avg_profit_pct'])
    print(f"\n🛡️ ЛУЧШИЙ С 0% БАНКРОТСТВ:")
    print(f"  {best_zero[0]}")
    print(f"  → +{best_zero[1]['avg_profit_pct']:.0f}% (+{1000 * best_zero[1]['avg_profit_pct'] / 100:,.0f} евро на 1000)")

# Максимальная прибыль
best_profit = max(results, key=lambda x: x[1]['avg_profit_pct'])
print(f"\n🚀 МАКСИМАЛЬНАЯ ПРИБЫЛЬ:")
print(f"  {best_profit[0]}")
print(f"  → +{best_profit[1]['avg_profit_pct']:.0f}% (+{1000 * best_profit[1]['avg_profit_pct'] / 100:,.0f} евро на 1000)")
print(f"  → Bankrupt: {best_profit[1]['bankrupt_pct']:.2f}%")

print("\n" + "="*100)
print("✅ ФИНАЛЬНЫЙ ТЕСТ ЗАВЕРШЕН!")
print("="*100)
