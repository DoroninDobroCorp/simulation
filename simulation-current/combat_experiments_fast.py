"""
БЫСТРЫЕ БОЕВЫЕ ЭКСПЕРИМЕНТЫ
Только самые перспективные параметры для скорости
"""

import numpy as np
import pandas as pd
from generate_real_odds_simulations import load_real_odds_outcomes
from realistic_simulation import adaptive_constant_profit_realistic, calculate_metrics_realistic
from test_all_strategies_realistic import (
    test_linear_roi_realistic, 
    test_exponential_roi_realistic,
    test_anti_martingale_realistic
)

print("="*100)
print("⚔️ БЫСТРЫЕ БОЕВЫЕ ЭКСПЕРИМЕНТЫ")
print("="*100)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Данные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")

results = []

# =============================================================================
# 1. ADAPTIVE CONSTANT PROFIT - КЛЮЧЕВЫЕ ЗНАЧЕНИЯ
# =============================================================================
print("\n1. ADAPTIVE CONSTANT PROFIT...")

# Широкий диапазон для поиска всех 3 категорий
k_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

params_base = {
    'min_roi': 4.733,
    'max_roi': 23.005,
    'min_target_pct': 3.982,
    'max_target_pct': 13.078,
    'max_bet_percent': 20.0,
}

for k in k_values:
    print(f"  k={k:.2f}...", end=' ', flush=True)
    
    br, _, _, _, avg_bet = adaptive_constant_profit_realistic(
        outcomes, odds_array,
        min_roi=params_base['min_roi'],
        max_roi=params_base['max_roi'],
        min_target_pct=params_base['min_target_pct'] * k,
        max_target_pct=params_base['max_target_pct'] * k,
        max_bet_percent=params_base['max_bet_percent'] * k,
        apply_variation=True,
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'adaptive_constant_profit',
        'params': f'k={k:.2f}_var',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    })
    
    print(f"+{m['avg_profit_pct']:.0f}%, B:{m['bankrupt_pct']:.2f}%")

# =============================================================================
# 2. ANTI-MARTINGALE - КЛЮЧЕВЫЕ КОМБИНАЦИИ
# =============================================================================
print("\n2. ANTI-MARTINGALE...")

combinations = [
    (2.0, 1.3), (2.0, 1.5), (2.0, 2.0),
    (3.0, 1.3), (3.0, 1.5), (3.0, 2.0),
    (4.0, 1.3), (4.0, 1.5), (4.0, 2.0),
    (5.0, 1.5), (5.0, 2.0), (5.0, 2.5),
    (6.0, 1.5), (6.0, 2.0),
    (7.0, 1.5), (7.0, 2.0),
    (8.0, 2.0), (10.0, 2.0)
]

for base, mult in combinations:
    print(f"  base={base}% m={mult}...", end=' ', flush=True)
    
    br = test_anti_martingale_realistic(outcomes, odds_array, base_percent=base, multiplier=mult, max_percent=10.0)
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'anti_martingale',
        'params': f'base={base}%_m={mult}',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    })
    
    print(f"+{m['avg_profit_pct']:.0f}%, B:{m['bankrupt_pct']:.2f}%")

# =============================================================================
# 3. LINEAR ROI
# =============================================================================
print("\n3. LINEAR ROI...")

for base in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
    print(f"  base={base}%...", end=' ', flush=True)
    
    br = test_linear_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=base, max_percent=10.0)
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'linear_roi',
        'params': f'base={base}%',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    })
    
    print(f"+{m['avg_profit_pct']:.0f}%, B:{m['bankrupt_pct']:.2f}%")

# =============================================================================
# 4. EXPONENTIAL ROI - КЛЮЧЕВЫЕ КОМБИНАЦИИ
# =============================================================================
print("\n4. EXPONENTIAL ROI...")

exp_combinations = [
    (2.0, 1.5), (2.0, 2.0), (2.0, 2.5),
    (3.0, 1.5), (3.0, 2.0), (3.0, 2.5),
    (4.0, 2.0), (4.0, 2.5),
    (5.0, 2.0), (5.0, 2.5), (5.0, 3.0),
    (6.0, 2.5), (7.0, 2.5)
]

for base, exp in exp_combinations:
    print(f"  base={base}% e={exp}...", end=' ', flush=True)
    
    br = test_exponential_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=base, exponent=exp, max_percent=10.0)
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'exponential_roi',
        'params': f'base={base}%_e={exp}',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    })
    
    print(f"+{m['avg_profit_pct']:.0f}%, B:{m['bankrupt_pct']:.2f}%")

# =============================================================================
# АНАЛИЗ
# =============================================================================
print("\n" + "="*100)
print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*100)

df = pd.DataFrame(results).sort_values('profit', ascending=False)
df.to_csv('combat_results.csv', index=False)

print(f"\nВсего: {len(df)} вариантов")

# 1. 0% СЛИВОВ
zero = df[df['bankrupt'] == 0.0].sort_values('profit', ascending=False)
print(f"\n🛡️ 0% БАНКРОТСТВ: {len(zero)} вариантов")
if len(zero) > 0:
    for i, (_, r) in enumerate(zero.head(3).iterrows(), 1):
        print(f"  {i}. {r['strategy']:<30} {r['params']:<20} +{r['profit']:.0f}%")

# 2. ДО 5%
agg = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)].sort_values('profit', ascending=False)
print(f"\n⚡ АГРЕССИВНАЯ (до 5%): {len(agg)} вариантов")
if len(agg) > 0:
    for i, (_, r) in enumerate(agg.head(3).iterrows(), 1):
        print(f"  {i}. {r['strategy']:<30} {r['params']:<20} +{r['profit']:.0f}%, B:{r['bankrupt']:.2f}%")

# 3. 5-25%
mega = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)].sort_values('profit', ascending=False)
print(f"\n🔥 МЕГА АГРЕССИВНАЯ (5-25%): {len(mega)} вариантов")
if len(mega) > 0:
    for i, (_, r) in enumerate(mega.head(3).iterrows(), 1):
        print(f"  {i}. {r['strategy']:<30} {r['params']:<20} +{r['profit']:.0f}%, B:{r['bankrupt']:.2f}%")

# ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ
print("\n" + "="*100)
print("💎 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ (БОЕВЫЕ УСЛОВИЯ)")
print("="*100)

if len(zero) > 0:
    best = zero.iloc[0]
    print(f"\n🥇 БЕЗ СЛИВОВ:")
    print(f"   {best['strategy']} ({best['params']})")
    print(f"   +{best['profit']:.0f}% | 0% bankrupt | DD>50: {best['dd50']:.1f}% | Worst: {best['worst_dd']:.1f}%")

if len(agg) > 0:
    best = agg.iloc[0]
    print(f"\n🥈 АГРЕССИВНАЯ:")
    print(f"   {best['strategy']} ({best['params']})")
    print(f"   +{best['profit']:.0f}% | {best['bankrupt']:.2f}% bankrupt | DD>50: {best['dd50']:.1f}%")

if len(mega) > 0:
    best = mega.iloc[0]
    print(f"\n🥉 МЕГА АГРЕССИВНАЯ:")
    print(f"   {best['strategy']} ({best['params']})")
    print(f"   +{best['profit']:.0f}% | {best['bankrupt']:.2f}% bankrupt | DD>50: {best['dd50']:.1f}%")

print("\n✅ Результаты в combat_results.csv")
print("="*100)
