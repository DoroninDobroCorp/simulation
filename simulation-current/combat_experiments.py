"""
БОЕВЫЕ ЭКСПЕРИМЕНТЫ - РЕАЛЬНЫЕ УСЛОВИЯ
- Пересчет банка раз в 30-70 ставок (рандом)
- Вариация ставок 35-115% (рандом)

Ищем ТРИ стратегии:
1. 0% банкротств (вообще никогда не сливает)
2. Агрессивная (до 5% банкротств)
3. МЕГА агрессивная (5-25% банкротств)
"""

import numpy as np
import pandas as pd
from generate_real_odds_simulations import load_real_odds_outcomes
from realistic_simulation import adaptive_constant_profit_realistic, calculate_metrics_realistic
from test_all_strategies_realistic import (
    test_linear_roi_realistic, 
    test_exponential_roi_realistic,
    test_anti_martingale_realistic,
    test_sqrt_roi_realistic,
    test_fixed_fraction_realistic
)

print("="*100)
print("⚔️ БОЕВЫЕ ЭКСПЕРИМЕНТЫ - РЕАЛЬНЫЕ УСЛОВИЯ")
print("="*100)
print("\nУсловия:")
print("  ✅ Пересчет банка: раз в 30-70 ставок (рандом)")
print("  ✅ Вариация ставок: 35-115% от расчетной (рандом)")
print("  ✅ Ограничение: max 10% от текущего банка")

# Загружаем данные
outcomes, odds_array = load_real_odds_outcomes()
print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")

results = []

# =============================================================================
# 1. ADAPTIVE CONSTANT PROFIT - РАСШИРЕННЫЙ ДИАПАЗОН
# =============================================================================
print("\n" + "="*100)
print("1. ADAPTIVE CONSTANT PROFIT (с вариацией 35-115%)")
print("="*100)

# От очень консервативных до экстремально агрессивных
k_values = [
    # Консервативные (ищем 0% сливов)
    0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 0.35, 0.4, 0.45, 0.5,
    # Умеренные
    0.6, 0.7, 0.8, 0.9, 1.0,
    # Агрессивные (ищем до 5%)
    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    # МЕГА агрессивные (ищем 5-25%)
    2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0
]

params_base = {
    'min_roi': 4.733,
    'max_roi': 23.005,
    'min_target_pct': 3.982,
    'max_target_pct': 13.078,
    'max_bet_percent': 20.0,
}

for k in k_values:
    print(f"  Testing k={k:.2f}...", end=' ', flush=True)
    
    br, _, _, _, avg_bet = adaptive_constant_profit_realistic(
        outcomes, odds_array,
        min_roi=params_base['min_roi'],
        max_roi=params_base['max_roi'],
        min_target_pct=params_base['min_target_pct'] * k,
        max_target_pct=params_base['max_target_pct'] * k,
        max_bet_percent=params_base['max_bet_percent'] * k,
        apply_variation=True,  # БОЕВЫЕ УСЛОВИЯ!
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'adaptive_constant_profit',
        'params': f'k={k:.2f}_var',
        'k': k,
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct'],
        'avg_bet': avg_bet
    })
    
    print(f"Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 2. ANTI-MARTINGALE - РАСШИРЕННЫЙ
# =============================================================================
print("\n" + "="*100)
print("2. ANTI-MARTINGALE")
print("="*100)

base_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]
multiplier_values = [1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]

for base in base_values:
    for mult in multiplier_values:
        print(f"  Testing base={base}% mult={mult}...", end=' ', flush=True)
        
        br = test_anti_martingale_realistic(
            outcomes, odds_array,
            base_percent=base,
            multiplier=mult,
            max_percent=10.0
        )
        
        m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
        
        results.append({
            'strategy': 'anti_martingale',
            'params': f'base={base}%_m={mult}',
            'k': None,
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct'],
            'avg_bet': None
        })
        
        print(f"Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 3. LINEAR ROI - РАСШИРЕННЫЙ
# =============================================================================
print("\n" + "="*100)
print("3. LINEAR ROI")
print("="*100)

base_values_linear = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]

for base in base_values_linear:
    print(f"  Testing base={base}%...", end=' ', flush=True)
    
    br = test_linear_roi_realistic(
        outcomes, odds_array,
        base_roi=5.0,
        base_percent=base,
        max_percent=10.0
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'linear_roi',
        'params': f'base={base}%',
        'k': None,
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct'],
        'avg_bet': None
    })
    
    print(f"Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 4. EXPONENTIAL ROI - РАСШИРЕННЫЙ
# =============================================================================
print("\n" + "="*100)
print("4. EXPONENTIAL ROI")
print("="*100)

base_values_exp = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
exponent_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]

for base in base_values_exp:
    for exp in exponent_values:
        print(f"  Testing base={base}% exp={exp}...", end=' ', flush=True)
        
        br = test_exponential_roi_realistic(
            outcomes, odds_array,
            base_roi=5.0,
            base_percent=base,
            exponent=exp,
            max_percent=10.0
        )
        
        m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
        
        results.append({
            'strategy': 'exponential_roi',
            'params': f'base={base}%_e={exp}',
            'k': None,
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct'],
            'avg_bet': None
        })
        
        print(f"Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 5. FIXED FRACTION - РАСШИРЕННЫЙ
# =============================================================================
print("\n" + "="*100)
print("5. FIXED FRACTION")
print("="*100)

for pct in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]:
    print(f"  Testing {pct}%...", end=' ', flush=True)
    
    br = test_fixed_fraction_realistic(
        outcomes, odds_array,
        fixed_percent=pct
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'fixed_fraction',
        'params': f'{pct}%',
        'k': None,
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct'],
        'avg_bet': None
    })
    
    print(f"Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# СОХРАНЯЕМ И АНАЛИЗИРУЕМ
# =============================================================================
print("\n" + "="*100)
print("💾 АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*100)

df = pd.DataFrame(results)
df = df.sort_values('profit', ascending=False)
df.to_csv('combat_experiments_results.csv', index=False)

print(f"\n✅ Всего протестировано: {len(results)} вариантов")
print(f"✅ Сохранено в: combat_experiments_results.csv")

# =============================================================================
# ПОИСК ЛУЧШИХ ДЛЯ КАЖДОЙ КАТЕГОРИИ
# =============================================================================
print("\n" + "="*100)
print("🎯 ПОИСК ЛУЧШИХ СТРАТЕГИЙ")
print("="*100)

# 1. 0% БАНКРОТСТВ
zero_bankrupt = df[df['bankrupt'] == 0.0].sort_values('profit', ascending=False)
print(f"\n🛡️ 0% БАНКРОТСТВ: {len(zero_bankrupt)} вариантов")
if len(zero_bankrupt) > 0:
    print(f"\n   ТОП-5:")
    for i, (_, row) in enumerate(zero_bankrupt.head(5).iterrows(), 1):
        print(f"   {i}. {row['strategy']:<30} {row['params']:<20} → +{row['profit']:.0f}%, DD>50: {row['dd50']:.1f}%")

# 2. АГРЕССИВНАЯ (до 5%)
aggressive = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)].sort_values('profit', ascending=False)
print(f"\n⚡ АГРЕССИВНАЯ (до 5% банкротств): {len(aggressive)} вариантов")
if len(aggressive) > 0:
    print(f"\n   ТОП-5:")
    for i, (_, row) in enumerate(aggressive.head(5).iterrows(), 1):
        print(f"   {i}. {row['strategy']:<30} {row['params']:<20} → +{row['profit']:.0f}%, B: {row['bankrupt']:.2f}%, DD>50: {row['dd50']:.1f}%")

# 3. МЕГА АГРЕССИВНАЯ (5-25%)
mega = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)].sort_values('profit', ascending=False)
print(f"\n🔥 МЕГА АГРЕССИВНАЯ (5-25% банкротств): {len(mega)} вариантов")
if len(mega) > 0:
    print(f"\n   ТОП-5:")
    for i, (_, row) in enumerate(mega.head(5).iterrows(), 1):
        print(f"   {i}. {row['strategy']:<30} {row['params']:<20} → +{row['profit']:.0f}%, B: {row['bankrupt']:.2f}%, DD>50: {row['dd50']:.1f}%")

# =============================================================================
# ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ
# =============================================================================
print("\n" + "="*100)
print("💎 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
print("="*100)

if len(zero_bankrupt) > 0:
    best_zero = zero_bankrupt.iloc[0]
    print(f"\n🥇 ЛУЧШАЯ БЕЗ СЛИВОВ:")
    print(f"   {best_zero['strategy']} ({best_zero['params']})")
    print(f"   → +{best_zero['profit']:.0f}% прибыль")
    print(f"   → 0% банкротств")
    print(f"   → DD>50%: {best_zero['dd50']:.1f}%, DD>80%: {best_zero['dd80']:.1f}%")
    print(f"   → Worst DD: {best_zero['worst_dd']:.1f}%")

if len(aggressive) > 0:
    best_agg = aggressive.iloc[0]
    print(f"\n🥈 ЛУЧШАЯ АГРЕССИВНАЯ:")
    print(f"   {best_agg['strategy']} ({best_agg['params']})")
    print(f"   → +{best_agg['profit']:.0f}% прибыль")
    print(f"   → {best_agg['bankrupt']:.2f}% банкротств")
    print(f"   → DD>50%: {best_agg['dd50']:.1f}%, DD>80%: {best_agg['dd80']:.1f}%")

if len(mega) > 0:
    best_mega = mega.iloc[0]
    print(f"\n🥉 ЛУЧШАЯ МЕГА АГРЕССИВНАЯ:")
    print(f"   {best_mega['strategy']} ({best_mega['params']})")
    print(f"   → +{best_mega['profit']:.0f}% прибыль")
    print(f"   → {best_mega['bankrupt']:.2f}% банкротств")
    print(f"   → DD>50%: {best_mega['dd50']:.1f}%, DD>80%: {best_mega['dd80']:.1f}%")

print("\n" + "="*100)
print("✅ БОЕВЫЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
print("="*100)
