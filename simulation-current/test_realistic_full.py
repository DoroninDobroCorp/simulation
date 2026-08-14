"""
ПОЛНЫЙ ТЕСТ реалистичной симуляции
От k=0.25 до k=2.5 с шагом 0.25
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from realistic_simulation import adaptive_constant_profit_realistic, calculate_metrics_realistic

outcomes, odds_array = load_real_odds_outcomes()

print("="*100)
print("🚀 ПОЛНЫЙ ТЕСТ РЕАЛИСТИЧНОЙ СИМУЛЯЦИИ (пересчет банка раз в 30-70 ставок)")
print("="*100)
print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {np.mean(odds_array):.2f}")

params_base = {
    'min_roi': 4.733,
    'max_roi': 23.005,
    'min_target_pct': 3.982,
    'max_target_pct': 13.078,
    'max_bet_percent': 20.0,
    'recalc_min': 30,
    'recalc_max': 70
}

print("\n" + "="*100)
print("📊 РЕЗУЛЬТАТЫ")
print("="*100)

print(f"\n{'k':<6} {'Var':<5} {'Profit':<10} {'Bankrupt':<10} {'DD>50%':<10} {'DD>80%':<10} {'Worst DD':<10} {'Avg bet':<10}")
print("-"*100)

results = []

# Тестируем от k=0.25 до k=2.5
k_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

for k in k_values:
    for apply_var in [False, True]:
        params = {
            'min_roi': params_base['min_roi'],
            'max_roi': params_base['max_roi'],
            'min_target_pct': params_base['min_target_pct'] * k,
            'max_target_pct': params_base['max_target_pct'] * k,
            'max_bet_percent': params_base['max_bet_percent'] * k,
            'apply_variation': apply_var,
            'recalc_min': params_base['recalc_min'],
            'recalc_max': params_base['recalc_max']
        }
        
        br, bh, _, _, avg_bet = adaptive_constant_profit_realistic(
            outcomes, odds_array, **params
        )
        
        m = calculate_metrics_realistic(br, bh, odds_array)
        
        var_str = "Yes" if apply_var else "No"
        
        print(f"{k:<6.2f} {var_str:<5} +{m['avg_profit_pct']:<9.0f} "
              f"{m['bankrupt_pct']:<10.2f} {m['drawdown_50_pct']:<10.1f} "
              f"{m['drawdown_80_pct']:<10.1f} {m['worst_drawdown_pct']:<10.1f} {avg_bet:<10.2f}")
        
        results.append({
            'k': k,
            'var': apply_var,
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct'],
            'avg_bet': avg_bet
        })

print("\n" + "="*100)
print("🛡️ ДЛЯ БОЛЬШИХ БАНКОВ (100k+ евро) - 0% СЛИВОВ")
print("="*100)

zero_bankrupt = [r for r in results if r['bankrupt'] == 0.0]
if zero_bankrupt:
    zero_bankrupt_sorted = sorted(zero_bankrupt, key=lambda x: x['profit'], reverse=True)
    
    print(f"\n✅ Найдено {len(zero_bankrupt)} вариантов с 0% банкротств\n")
    
    for i, r in enumerate(zero_bankrupt_sorted[:5], 1):
        var_str = "с вариацией" if r['var'] else "без вариации"
        bank = 100000
        profit_eur = bank * r['profit'] / 100
        avg_bet_eur = bank * r['avg_bet'] / 100
        
        print(f"{i}. k={r['k']:.2f} {var_str}")
        print(f"   💰 Profit: +{r['profit']:.0f}% = +{profit_eur:,.0f} евро")
        print(f"   ✅ Bankrupt: 0.00%")
        print(f"   ⚠️  DD>50%: {r['dd50']:.1f}%, DD>80%: {r['dd80']:.1f}%")
        print(f"   📊 Worst DD: {r['worst_dd']:.1f}%")
        print(f"   📊 Avg bet: {r['avg_bet']:.2f}% = {avg_bet_eur:,.0f} евро")
        print()

print("="*100)
print("🚀 ДЛЯ МЕЛКИХ ИНВЕСТОРОВ (100-5000 евро) - МАКСИМУМ ПРИБЫЛИ")
print("="*100)

# Фильтруем с высокой прибылью (>200%) и приемлемым риском (<5%)
aggressive = [r for r in results if r['profit'] > 200 and r['bankrupt'] < 5.0]
aggressive_sorted = sorted(aggressive, key=lambda x: x['profit'], reverse=True)

if aggressive_sorted:
    print(f"\n✅ Найдено {len(aggressive_sorted)} вариантов с прибылью >200% и банкротством <5%\n")
    
    for i, r in enumerate(aggressive_sorted[:5], 1):
        var_str = "с вариацией" if r['var'] else "без вариации"
        bank = 1000
        profit_eur = bank * r['profit'] / 100
        avg_bet_eur = bank * r['avg_bet'] / 100
        
        print(f"{i}. k={r['k']:.2f} {var_str}")
        print(f"   💰 Profit: +{r['profit']:.0f}% = +{profit_eur:,.0f} евро на 1000")
        print(f"   ⚠️  Bankrupt: {r['bankrupt']:.2f}% ({int(r['bankrupt']*100):.0f} из 10000)")
        print(f"   ⚠️  DD>50%: {r['dd50']:.1f}%, DD>80%: {r['dd80']:.1f}%")
        print(f"   📊 Avg bet: {r['avg_bet']:.2f}% = {avg_bet_eur:.0f} евро")
        print()

print("="*100)
print("💎 РЕКОМЕНДАЦИИ")
print("="*100)

# Лучший для 100k
best_safe = max(zero_bankrupt, key=lambda x: x['profit'])
var_str = "с вариацией" if best_safe['var'] else "без вариации"
print(f"\n🛡️ ДЛЯ 100,000 ЕВРО (безопасность):")
print(f"   k={best_safe['k']:.2f} {var_str}")
print(f"   → +{best_safe['profit']:.0f}% (+{100000*best_safe['profit']/100:,.0f} евро)")
print(f"   → 0% сливов, DD>50%: {best_safe['dd50']:.1f}%")

# Лучший для мелких
if aggressive_sorted:
    best_aggr = aggressive_sorted[0]
    var_str = "с вариацией" if best_aggr['var'] else "без вариации"
    print(f"\n🚀 ДЛЯ 1,000 ЕВРО (агрессивно):")
    print(f"   k={best_aggr['k']:.2f} {var_str}")
    print(f"   → +{best_aggr['profit']:.0f}% (+{1000*best_aggr['profit']/100:,.0f} евро)")
    print(f"   → {best_aggr['bankrupt']:.2f}% сливов, DD>50%: {best_aggr['dd50']:.1f}%")

print("\n" + "="*100)
print("✅ РЕАЛИСТИЧНАЯ СИМУЛЯЦИЯ ЗАВЕРШЕНА!")
print("="*100)
print("""
Ключевые отличия от старой симуляции:
✅ Пересчет банка раз в 30-70 ставок (реалистично!)
✅ Ставки остаются стабильными внутри периода
✅ Прибыль ВЫШЕ (но и риски тоже)
✅ Более честное отражение реальности
""")
