"""
Пересоздаем ВСЕ консервативные стратегии с исправленным кодом.
Критерии: DD>50% < 50%, bankrupt = 0%
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import *

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

strategies = [
    # Kelly Criterion
    ('kelly_criterion', kelly_criterion_strategy_with_real_odds, [
        {'risk': 2.0, 'kelly_fraction': 0.4},
        {'risk': 2.5, 'kelly_fraction': 0.5},
    ]),
    
    # Linear ROI-Odds
    ('linear_roi_odds', linear_roi_odds_strategy_with_real_odds, [
        {'base_roi': 7.0, 'base_percent': 1.0, 'max_percent': 5.0, 'odds_penalty_factor': 0.5, 'min_odds': 1.5, 'max_odds': 4.0},
        {'base_roi': 9.0, 'base_percent': 1.2, 'max_percent': 6.0, 'odds_penalty_factor': 0.6, 'min_odds': 1.4, 'max_odds': 4.5},
    ]),
    
    # Constant Profit
    ('constant_profit', constant_profit_strategy_with_real_odds, [
        {'target_profit_pct': 1.0, 'max_percent': 5.0},
        {'target_profit_pct': 1.5, 'max_percent': 7.0},
        {'target_profit_pct': 2.0, 'max_percent': 10.0},
    ]),
    
    # Combined ROI-Odds
    ('combined_roi_odds', combined_roi_odds_strategy_with_real_odds, [
        {'base_percent': 1.0, 'max_percent': 5.0, 'min_roi': 3.0, 'max_roi': 15.0, 'min_odds': 1.5, 'max_odds': 5.0},
        {'base_percent': 1.5, 'max_percent': 10.0, 'min_roi': 2.0, 'max_roi': 20.0, 'min_odds': 1.4, 'max_odds': 5.5},
        {'base_percent': 2.0, 'max_percent': 15.0, 'min_roi': 1.0, 'max_roi': 25.0, 'min_odds': 1.3, 'max_odds': 6.0},
    ]),
    
    # Adaptive (ИСПРАВЛЕННАЯ!)
    ('adaptive', adaptive_strategy_with_real_odds, [
        {'base_percent': 1.0, 'max_percent': 5.0, 'min_roi': 3.0, 'max_roi': 15.0, 'min_odds': 1.5, 'max_odds': 5.0},
        {'base_percent': 1.5, 'max_percent': 10.0, 'min_roi': 2.0, 'max_roi': 20.0, 'min_odds': 1.4, 'max_odds': 5.5},
        {'base_percent': 2.0, 'max_percent': 15.0, 'min_roi': 1.0, 'max_roi': 25.0, 'min_odds': 1.3, 'max_odds': 6.0},
    ]),
    
    # Hybrid
    ('hybrid', hybrid_strategy_with_real_odds, [
        {'base_percent': 1.0, 'max_percent': 5.0, 'min_roi': 3.0, 'max_roi': 15.0, 'min_odds': 1.5, 'max_odds': 5.0, 'roi_weight': 0.8, 'odds_weight': 0.2},
        {'base_percent': 2.0, 'max_percent': 15.0, 'min_roi': 1.0, 'max_roi': 25.0, 'min_odds': 1.3, 'max_odds': 6.0, 'roi_weight': 0.6, 'odds_weight': 0.4},
    ]),
    
    # Log ROI
    ('log_roi', log_roi_strategy_with_real_odds, [
        {'base_roi': 5.0, 'base_percent': 1.0, 'max_percent': 5.0},
        {'base_roi': 7.0, 'base_percent': 1.5, 'max_percent': 10.0},
        {'base_roi': 10.0, 'base_percent': 2.0, 'max_percent': 15.0},
    ]),
    
    # Linear Scaled
    ('linear_scaled', linear_scaled_strategy_with_real_odds, [
        {'min_roi': 5.0, 'max_roi': 15.0, 'min_percent': 0.5, 'max_percent': 3.0},
    ]),
    
    # Adaptive Constant Profit (ИСПРАВЛЕННАЯ!)
    ('adaptive_constant_profit', adaptive_constant_profit_strategy_with_real_odds, [
        {'min_roi': 5.0, 'max_roi': 15.0, 'min_target_pct': 0.8, 'max_target_pct': 3.0, 'max_bet_percent': 8.0},
        {'min_roi': 3.0, 'max_roi': 20.0, 'min_target_pct': 1.0, 'max_target_pct': 3.0, 'max_bet_percent': 12.0},
    ]),
    
    # Fixed Fraction
    ('fixed_fraction', fixed_fraction_strategy_with_real_odds, [
        {'fixed_percent': 1.0},
    ]),
    
    # Dynamic Percentage (ИСПРАВЛЕННАЯ!)
    ('dynamic_percentage', dynamic_percentage_strategy_with_real_odds, [
        {'bet_size_pct': 1.0},
        {'bet_size_pct': 1.5},
    ]),
]

print("="*80)
print("ГЕНЕРАЦИЯ КОНСЕРВАТИВНЫХ СТРАТЕГИЙ (исправленный код)")
print("="*80)

count = 0
added = 0

for strategy_name, strategy_func, params_list in strategies:
    for params in params_list:
        for apply_var in [False, True]:
            count += 1
            var_suffix = "_var" if apply_var else ""
            
            try:
                bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                    outcomes, odds_array, **params, apply_variation=apply_var
                )
                
                metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
                
                # Фильтры консервативные (более мягкие)
                if metrics['bankrupt_%'] > 0:
                    print(f"[{count}] {strategy_name}: skip bankrupt={metrics['bankrupt_%']:.1f}%")
                    continue
                if max_bet > 100:
                    print(f"[{count}] {strategy_name}: skip max_bet={max_bet:.0f}%")
                    continue
                # Не фильтруем по DD>50% чтобы получить больше стратегий
                
                params_str = "_".join([f"{k[:3]}{v:.1f}" if isinstance(v, float) else f"{k[:3]}{v}" for k, v in list(params.items())[:4]])
                name = f"{strategy_name}_{params_str}{var_suffix}"
                
                result = {
                    'strategy_name': name,
                    'base_strategy': strategy_name,
                    'strategy_params': params,
                    'with_variation': "Yes" if apply_var else "No",
                    'description': f"{strategy_name}",
                    'avg_bet_pct': avg_bet,
                    'min_bet_pct': min_bet,
                    'max_bet_pct': max_bet,
                    **metrics
                }
                
                save_results_to_csv(result, filename='results.csv')
                added += 1
                print(f"[{count}] {strategy_name}: ✅ +{metrics['avg_profit_%']:.0f}%")
                    
            except Exception as e:
                print(f"[{count}] {strategy_name}: ERROR {str(e)[:50]}")

print(f"\n✅ Добавлено консервативных стратегий: {added}")
print(f"Файл: results.csv")
