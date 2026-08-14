"""
ТОЧНЫЕ параметры из отчета - те что отмечены !
Тестируем их напрямую.
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    linear_roi_odds_strategy_with_real_odds,
    adaptive_constant_profit_strategy_with_real_odds,
    adaptive_strategy_with_real_odds,
    dynamic_kelly_strategy_with_real_odds,
    linear_scaled_strategy_with_real_odds,
    linear_roi_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

# ТОЧНЫЕ стратегии из отчета с ! 
exact_strategies = [
    # ===== CONSERVATIVE =====
    ('linear_roi_odds', 'CONSERVATIVE_ROI111', linear_roi_odds_strategy_with_real_odds, {
        'base_roi': 9.263, 'base_percent': 1.669, 'max_percent': 8.464,
        'odds_penalty_factor': 0.7, 'min_odds': 1.41, 'max_odds': 3.65
    }),
    
    # ===== CAUTIOUS =====
    ('linear_roi_odds', 'CAUTIOUS_ROI209', linear_roi_odds_strategy_with_real_odds, {
        'base_roi': 11.003, 'base_percent': 2.84, 'max_percent': 6.789,
        'odds_penalty_factor': 0.7, 'min_odds': 1.552, 'max_odds': 4.034
    }),
    
    # ===== BALANCED =====
    ('linear_roi_odds', 'BALANCED_ROI282', linear_roi_odds_strategy_with_real_odds, {
        'base_roi': 5.649, 'base_percent': 1.533, 'max_percent': 15.089,
        'odds_penalty_factor': 0.7, 'min_odds': 1.686, 'max_odds': 4.627
    }),
    
    # ===== RISKY - Adaptive (ROI: 2602%!) =====
    ('adaptive', 'RISKY_ROI2602', adaptive_strategy_with_real_odds, {
        'base_percent': 2.476, 'max_percent': 16.839,
        'min_roi': 4.933, 'max_roi': 14.367
    }),
    
    # ===== CRAZY - Adaptive (ROI: 4931%!) =====
    ('adaptive', 'CRAZY_ROI4931', adaptive_strategy_with_real_odds, {
        'base_percent': 1.14, 'max_percent': 29.865,
        'min_roi': 3.266, 'max_roi': 34.689
    }),
    
    # ===== EXTREME - Adaptive (ROI: 16173%!) =====
    ('adaptive', 'EXTREME_ROI16173', adaptive_strategy_with_real_odds, {
        'base_percent': 2.777, 'max_percent': 19.534,
        'min_roi': 2.732, 'max_roi': 25.191
    }),
    
    # ===== EXTREME - Linear Scaled (ROI: 11034%!) =====
    ('linear_scaled', 'EXTREME_ROI11034', linear_scaled_strategy_with_real_odds, {
        'min_roi': 4.925, 'max_roi': 21.285,
        'min_percent': 3.526, 'max_percent': 12.475
    }),
    
    # ===== RISKY - Dynamic Kelly (ROI: 1414%) =====
    ('dynamic_kelly', 'RISKY_ROI1414', dynamic_kelly_strategy_with_real_odds, {
        'risk': 4.97, 'min_fraction': 0.426, 'max_fraction': 1.162,
        'min_roi': 2.608, 'max_roi': 22.442
    }),
    
    # ===== RISKY - Adaptive Constant Profit (ROI: 1328%) =====
    ('adaptive_constant_profit', 'RISKY_ROI1328', adaptive_constant_profit_strategy_with_real_odds, {
        'min_roi': 0.9, 'max_roi': 30.105,
        'min_target_pct': 0.423, 'max_target_pct': 17.445,
        'max_bet_percent': 20.0
    }),
    
    # ===== CRAZY - Adaptive Constant Profit (ROI: 3708%) =====
    ('adaptive_constant_profit', 'CRAZY_ROI3708', adaptive_constant_profit_strategy_with_real_odds, {
        'min_roi': 4.733, 'max_roi': 23.005,
        'min_target_pct': 3.982, 'max_target_pct': 13.078,
        'max_bet_percent': 20.0
    }),
    
    # ===== CRAZY - Dynamic Kelly (ROI: 3669%) =====
    ('dynamic_kelly', 'CRAZY_ROI3669', dynamic_kelly_strategy_with_real_odds, {
        'risk': 4.143, 'min_fraction': 0.33, 'max_fraction': 1.39,
        'min_roi': 3.727, 'max_roi': 13.513
    }),
    
    # ===== EXTREME - Dynamic Kelly (ROI: 9191%) =====
    ('dynamic_kelly', 'EXTREME_ROI9191', dynamic_kelly_strategy_with_real_odds, {
        'risk': 1.531, 'min_fraction': 0.168, 'max_fraction': 0.456,
        'min_roi': 1.396, 'max_roi': 13.922
    }),
]

print("="*80)
print("ТЕСТИРОВАНИЕ ТОЧНЫХ СТРАТЕГИЙ ИЗ ОТЧЕТА (отмеченные !)")
print("="*80)
print(f"Будет протестировано: {len(exact_strategies)} × 2 = {len(exact_strategies)*2} вариантов\n")

count = 0
added = 0
results = []

for strategy_name, profile, strategy_func, params in exact_strategies:
    print(f"\n{'='*80}")
    print(f"{profile} - {strategy_name}")
    print(f"{'='*80}")
    
    for apply_var in [False, True]:
        count += 1
        var_str = "Yes" if apply_var else "No"
        var_suffix = "_var" if apply_var else ""
        
        name = f"{strategy_name}_{profile}{var_suffix}"
        
        print(f"[{count}/{len(exact_strategies)*2}] {name[:60]:<60}", end=' ', flush=True)
        
        try:
            bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                outcomes, odds_array, **params, apply_variation=apply_var
            )
            
            metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
            
            # Просто сохраняем ВСЕ результаты без фильтров
            result = {
                'strategy_name': name,
                'base_strategy': strategy_name,
                'strategy_params': params,
                'with_variation': var_str,
                'description': f"{profile}: {strategy_name}",
                'avg_bet_pct': avg_bet,
                'min_bet_pct': min_bet,
                'max_bet_pct': max_bet,
                **metrics
            }
            
            # Сохраняем в разные файлы в зависимости от bankrupt
            if metrics['bankrupt_pct'] <= 10:
                save_results_to_csv(result, filename='results_aggressive_bankrupt10.csv')
                print(f"✅ +{metrics['avg_profit_pct']:.0f}% B:{metrics['bankrupt_pct']:.1f}% DD50:{metrics['drawdown_50_pct']:.1f}% → bankrupt10")
            elif metrics['bankrupt_pct'] <= 20:
                save_results_to_csv(result, filename='results_aggressive_bankrupt20.csv')
                print(f"✅ +{metrics['avg_profit_pct']:.0f}% B:{metrics['bankrupt_pct']:.1f}% DD50:{metrics['drawdown_50_pct']:.1f}% → bankrupt20")
            else:
                save_results_to_csv(result, filename='results_aggressive_bankrupt30plus.csv')
                print(f"⚠️ +{metrics['avg_profit_pct']:.0f}% B:{metrics['bankrupt_pct']:.1f}% DD50:{metrics['drawdown_50_pct']:.1f}% → bankrupt30+")
            
            added += 1
            results.append(result)
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")

print("\n" + "="*80)
print("📊 ИТОГИ")
print("="*80)
print(f"✅ Протестировано: {added}/{len(exact_strategies)*2}")

if results:
    print("\n🏆 ТОП-10 по прибыли:")
    sorted_results = sorted(results, key=lambda x: x['avg_profit_pct'], reverse=True)[:10]
    for i, r in enumerate(sorted_results, 1):
        print(f"{i:2}. {r['strategy_name'][:50]:<50} +{r['avg_profit_pct']:>6.0f}% B:{r['bankrupt_pct']:>4.1f}%")
    
    print("\n📂 Файлы созданы:")
    print("  - results_aggressive_bankrupt10.csv (0-10% сливов)")
    print("  - results_aggressive_bankrupt20.csv (10-20% сливов)")
    print("  - results_aggressive_bankrupt30plus.csv (>20% сливов)")
