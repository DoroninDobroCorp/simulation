"""
МАССОВЫЕ ЭКСПЕРИМЕНТЫ
Тестируем ВСЕ возможные комбинации параметров для разных стратегий.
Записываем все результаты в CSV, потом анализируем.
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
print("🔬 МАССОВЫЕ ЭКСПЕРИМЕНТЫ - ПОИСК ИДЕАЛЬНЫХ СТРАТЕГИЙ")
print("="*100)

# Загружаем данные
outcomes, odds_array = load_real_odds_outcomes()
print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")

results = []

# =============================================================================
# 1. ADAPTIVE CONSTANT PROFIT - ТОНКАЯ НАСТРОЙКА
# =============================================================================
print("\n" + "="*100)
print("1. ADAPTIVE CONSTANT PROFIT - тестируем k от 0.1 до 3.0")
print("="*100)

k_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
            0.6, 0.7, 0.75, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
            2.2, 2.4, 2.6, 2.8, 3.0]

params_base_acp = {
    'min_roi': 4.733,
    'max_roi': 23.005,
    'min_target_pct': 3.982,
    'max_target_pct': 13.078,
    'max_bet_percent': 20.0,
}

for k in k_values:
    for var in [False, True]:
        print(f"Testing adaptive_constant_profit k={k:.2f} var={var}...", end=' ', flush=True)
        
        br, _, _, _, avg_bet = adaptive_constant_profit_realistic(
            outcomes, odds_array,
            min_roi=params_base_acp['min_roi'],
            max_roi=params_base_acp['max_roi'],
            min_target_pct=params_base_acp['min_target_pct'] * k,
            max_target_pct=params_base_acp['max_target_pct'] * k,
            max_bet_percent=params_base_acp['max_bet_percent'] * k,
            apply_variation=var,
            recalc_min=30, recalc_max=70
        )
        
        m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
        
        results.append({
            'strategy': 'adaptive_constant_profit',
            'params': f'k={k:.2f}',
            'variation': 'Yes' if var else 'No',
            'k': k,
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct'],
            'avg_bet': avg_bet,
            'min_profit': m['min_profit_pct'],
            'max_profit': m['max_profit_pct']
        })
        
        print(f"✓ Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 2. ANTI-MARTINGALE - РАЗНЫЕ КОМБИНАЦИИ
# =============================================================================
print("\n" + "="*100)
print("2. ANTI-MARTINGALE - тестируем разные base и multiplier")
print("="*100)

base_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
multiplier_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5]

for base in base_values:
    for mult in multiplier_values:
        print(f"Testing anti_martingale base={base}% mult={mult}...", end=' ', flush=True)
        
        br = test_anti_martingale_realistic(
            outcomes, odds_array,
            base_percent=base,
            multiplier=mult,
            max_percent=10.0
        )
        
        m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
        
        results.append({
            'strategy': 'anti_martingale',
            'params': f'base={base}%,m={mult}',
            'variation': 'No',
            'k': None,
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct'],
            'avg_bet': None,
            'min_profit': m['min_profit_pct'],
            'max_profit': m['max_profit_pct']
        })
        
        print(f"✓ Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 3. LINEAR ROI - РАЗНЫЕ BASE
# =============================================================================
print("\n" + "="*100)
print("3. LINEAR ROI - тестируем разные base")
print("="*100)

base_values_linear = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

for base in base_values_linear:
    print(f"Testing linear_roi base={base}%...", end=' ', flush=True)
    
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
        'variation': 'No',
        'k': None,
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct'],
        'avg_bet': None,
        'min_profit': m['min_profit_pct'],
        'max_profit': m['max_profit_pct']
    })
    
    print(f"✓ Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 4. EXPONENTIAL ROI - РАЗНЫЕ BASE И EXPONENT
# =============================================================================
print("\n" + "="*100)
print("4. EXPONENTIAL ROI - тестируем разные base и exponent")
print("="*100)

base_values_exp = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
exponent_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]

for base in base_values_exp:
    for exp in exponent_values:
        print(f"Testing exponential_roi base={base}% exp={exp}...", end=' ', flush=True)
        
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
            'params': f'base={base}%,e={exp}',
            'variation': 'No',
            'k': None,
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct'],
            'avg_bet': None,
            'min_profit': m['min_profit_pct'],
            'max_profit': m['max_profit_pct']
        })
        
        print(f"✓ Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 5. SQRT ROI - РАЗНЫЕ BASE
# =============================================================================
print("\n" + "="*100)
print("5. SQRT ROI - тестируем разные base")
print("="*100)

for base in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    print(f"Testing sqrt_roi base={base}%...", end=' ', flush=True)
    
    br = test_sqrt_roi_realistic(
        outcomes, odds_array,
        base_roi=5.0,
        base_percent=base,
        max_percent=10.0
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'sqrt_roi',
        'params': f'base={base}%',
        'variation': 'No',
        'k': None,
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct'],
        'avg_bet': None,
        'min_profit': m['min_profit_pct'],
        'max_profit': m['max_profit_pct']
    })
    
    print(f"✓ Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# 6. FIXED FRACTION - РАЗНЫЕ ПРОЦЕНТЫ
# =============================================================================
print("\n" + "="*100)
print("6. FIXED FRACTION - тестируем разные проценты")
print("="*100)

for pct in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
    print(f"Testing fixed_fraction {pct}%...", end=' ', flush=True)
    
    br = test_fixed_fraction_realistic(
        outcomes, odds_array,
        fixed_percent=pct
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes, dtype=float), odds_array)
    
    results.append({
        'strategy': 'fixed_fraction',
        'params': f'{pct}%',
        'variation': 'No',
        'k': None,
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct'],
        'avg_bet': None,
        'min_profit': m['min_profit_pct'],
        'max_profit': m['max_profit_pct']
    })
    
    print(f"✓ Profit: +{m['avg_profit_pct']:.0f}%, Bankrupt: {m['bankrupt_pct']:.2f}%")

# =============================================================================
# СОХРАНЯЕМ ВСЕ РЕЗУЛЬТАТЫ
# =============================================================================
print("\n" + "="*100)
print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*100)

df = pd.DataFrame(results)
df = df.sort_values('profit', ascending=False)
df.to_csv('massive_experiments_results.csv', index=False)

print(f"\n✅ Всего протестировано: {len(results)} комбинаций")
print(f"✅ Сохранено в: massive_experiments_results.csv")

# =============================================================================
# СТАТИСТИКА
# =============================================================================
print("\n" + "="*100)
print("📊 СТАТИСТИКА")
print("="*100)

print(f"\nВсего протестировано: {len(df)} вариантов")
print(f"По стратегиям:")
for strategy in df['strategy'].unique():
    count = len(df[df['strategy'] == strategy])
    print(f"  - {strategy}: {count} вариантов")

# 0% сливов
zero_bankrupt = df[df['bankrupt'] == 0.0]
print(f"\n✅ С 0% банкротств: {len(zero_bankrupt)} вариантов")
print(f"   Макс прибыль: +{zero_bankrupt['profit'].max():.0f}%")
print(f"   Мин прибыль: +{zero_bankrupt['profit'].min():.0f}%")

# Очень низкий риск (<0.1%)
low_risk = df[df['bankrupt'] < 0.1]
print(f"\n⚡ С <0.1% банкротств: {len(low_risk)} вариантов")
print(f"   Макс прибыль: +{low_risk['profit'].max():.0f}%")

# Агрессивная (до 5%)
aggressive = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)]
print(f"\n⚡ АГРЕССИВНАЯ (до 5% банкротств): {len(aggressive)} вариантов")
if len(aggressive) > 0:
    print(f"   Макс прибыль: +{aggressive['profit'].max():.0f}%")
    best = aggressive.loc[aggressive['profit'].idxmax()]
    print(f"   Лучший: {best['strategy']} {best['params']} → +{best['profit']:.0f}%, bankrupt {best['bankrupt']:.2f}%")

# Мега агрессивная (5-25%)
mega_aggressive = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)]
print(f"\n🔥 МЕГА АГРЕССИВНАЯ (5-25% банкротств): {len(mega_aggressive)} вариантов")
if len(mega_aggressive) > 0:
    print(f"   Макс прибыль: +{mega_aggressive['profit'].max():.0f}%")
    best = mega_aggressive.loc[mega_aggressive['profit'].idxmax()]
    print(f"   Лучший: {best['strategy']} {best['params']} → +{best['profit']:.0f}%, bankrupt {best['bankrupt']:.2f}%")

# Экстрим (>25%)
extreme_risk = df[df['bankrupt'] > 25.0]
print(f"\n🚨 ЭКСТРИМ (>25% банкротств): {len(extreme_risk)} вариантов")
if len(extreme_risk) > 0:
    print(f"   Макс прибыль: +{extreme_risk['profit'].max():.0f}%")

print("\n" + "="*100)
print("🏆 ТОП-10 ПО ПРИБЫЛИ (ВСЕ)")
print("="*100)

print(f"\n{'#':<4} {'Стратегия':<25} {'Params':<25} {'Profit':<10} {'Bankrupt':<10} {'DD>50%'}")
print("-"*100)

for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
    print(f"{i:<4} {row['strategy']:<25} {row['params']:<25} +{row['profit']:<9.0f} {row['bankrupt']:<10.2f} {row['dd50']:.1f}%")

print("\n" + "="*100)
print("🛡️ ТОП-10 ПО ПРИБЫЛИ (0% БАНКРОТСТВ)")
print("="*100)

print(f"\n{'#':<4} {'Стратегия':<25} {'Params':<25} {'Profit':<10} {'DD>50%':<10} {'DD>80%'}")
print("-"*100)

for i, (_, row) in enumerate(zero_bankrupt.head(10).iterrows(), 1):
    print(f"{i:<4} {row['strategy']:<25} {row['params']:<25} +{row['profit']:<9.0f} {row['dd50']:<10.1f} {row['dd80']:.1f}%")

print("\n" + "="*100)
print("✅ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
print("="*100)
print("\nВсе данные сохранены в massive_experiments_results.csv")
print("Теперь можно анализировать и выбирать лучшие для каждой категории!")
