"""
Генерация агрессивных стратегий с ЖЕСТКИМ ограничением max_bet <= 10%.
Цель: profit 200-300%, bankrupt < 20%
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import *

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

# Создадим много вариаций агрессивных стратегий
strategies = []

# ADAPTIVE с разными параметрами + жесткое ограничение max_bet
for base_pct in [3.0, 3.5, 4.0, 4.5, 5.0]:
    for max_pct in [8.0, 9.0, 10.0]:
        for min_roi in [2.0, 2.5, 3.0]:
            strategies.append(('adaptive', adaptive_strategy_with_real_odds, {
                'base_percent': base_pct,
                'max_percent': max_pct,
                'min_roi': min_roi,
                'max_roi': min_roi + 20.0,
                'min_odds': 1.3,
                'max_odds': 6.0,
                'apply_variation': False
            }))

# LINEAR_ROI_ODDS с агрессивными параметрами
for base_roi in [3.0, 3.5, 4.0, 4.5, 5.0]:
    for base_pct in [2.0, 2.5, 3.0]:
        for max_pct in [8.0, 9.0, 10.0]:
            strategies.append(('linear_roi_odds', linear_roi_odds_strategy_with_real_odds, {
                'base_roi': base_roi,
                'base_percent': base_pct,
                'max_percent': max_pct,
                'odds_penalty_factor': 0.5,
                'min_odds': 1.3,
                'max_odds': 6.0,
                'apply_variation': False
            }))

# CONSTANT_PROFIT с высокими целями
for target in [3.0, 3.5, 4.0, 4.5, 5.0]:
    for max_pct in [8.0, 9.0, 10.0]:
        strategies.append(('constant_profit', constant_profit_strategy_with_real_odds, {
            'target_profit_pct': target,
            'max_percent': max_pct,
            'apply_variation': False
        }))

# LOG_ROI с агрессивными параметрами
for base_roi in [6.0, 7.0, 8.0]:
    for base_pct in [2.5, 3.0, 3.5]:
        for max_pct in [8.0, 9.0, 10.0]:
            strategies.append(('log_roi', log_roi_strategy_with_real_odds, {
                'base_roi': base_roi,
                'base_percent': base_pct,
                'max_percent': max_pct,
                'apply_variation': False
            }))

print("="*90)
print("ГЕНЕРАЦИЯ АГРЕССИВНЫХ СТРАТЕГИЙ С ОГРАНИЧЕНИЕМ max_bet <= 10%")
print("="*90)
print(f"Будет протестировано: {len(strategies)} наборов параметров\n")

count = 0
added = 0
results = []

for strategy_name, strategy_func, params in strategies:
    count += 1
    
    if count % 20 == 0:
        print(f"[{count}/{len(strategies)}] Обработано, найдено подходящих: {added}")
    
    try:
        br, bh, min_bet, max_bet, avg_bet = strategy_func(outcomes, odds_array, **params)
        metrics = calculate_metrics_with_odds(br, bh, odds_array)
        
        # ЖЕСТКИЕ фильтры
        if max_bet > 10:
            continue
        if metrics['avg_profit_pct'] < 100:
            continue
        if metrics['bankrupt_pct'] > 20:
            continue
        
        # Генерируем уникальное имя
        params_str = "_".join([f"{v:.1f}" for v in list(params.values())[:3]])
        name = f"{strategy_name}_AGG_LIMITED_{params_str}"
        
        result = {
            'strategy_name': name,
            'base_strategy': strategy_name,
            'strategy_params': params,
            'with_variation': "No",
            'description': f"CUSTOM AGGRESSIVE LIMITED: {strategy_name}",
            'avg_bet_pct': avg_bet,
            'min_bet_pct': min_bet,
            'max_bet_pct': max_bet,
            **metrics
        }
        
        save_results_to_csv(result, filename='results_aggressive_limited_maxbet10.csv')
        added += 1
        
        results.append({
            'name': name,
            'profit': metrics['avg_profit_pct'],
            'bankrupt': metrics['bankrupt_pct'],
            'dd50': metrics['drawdown_50_pct'],
            'max_bet': max_bet
        })
        
    except Exception as e:
        pass

print(f"\n{'='*90}")
print(f"✅ Найдено стратегий: {added}")
print(f"{'='*90}")

if len(results) > 0:
    # Сортируем по прибыли
    results_sorted = sorted(results, key=lambda x: x['profit'], reverse=True)
    
    print(f"\n🏆 ТОП-10 по прибыли:")
    print(f"{'Стратегия':<50} {'Profit':<9} {'Bankrupt':<9} {'Max bet'}")
    print("-"*90)
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i:2}. {r['name'][:48]:<48} +{r['profit']:>5.0f}%  {r['bankrupt']:>6.1f}%  {r['max_bet']:>6.1f}%")
    
    # Лучшая по соотношению прибыль/риск
    best_ratio = max(results_sorted, key=lambda x: x['profit'] / (x['bankrupt'] + 1))
    print(f"\n💎 ЛУЧШЕЕ СООТНОШЕНИЕ profit/risk:")
    print(f"   {best_ratio['name']}")
    print(f"   Profit: +{best_ratio['profit']:.0f}%")
    print(f"   Bankrupt: {best_ratio['bankrupt']:.1f}%")
    print(f"   DD>50%: {best_ratio['dd50']:.1f}%")
    print(f"   Max bet: {best_ratio['max_bet']:.1f}%")
    
    print(f"\nФайл: results_aggressive_limited_maxbet10.csv")
else:
    print("\n❌ Не найдено стратегий удовлетворяющих критериям!")
    print("\nВозможные причины:")
    print("  1. При ROI=7% сложно получить 200%+ прибыль с max_bet<=10%")
    print("  2. Нужны более агрессивные параметры")
    print("  3. Или выше ROI в исходных данных")
