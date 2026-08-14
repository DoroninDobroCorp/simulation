"""
ФИНАЛЬНЫЙ поиск агрессивных стратегий.
Критерии: profit > 80%, bankrupt <= 20%, max_bet <= 45%

Делаем ОЧЕНЬ МНОГО вариаций!
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    linear_roi_odds_strategy_with_real_odds,
    adaptive_constant_profit_strategy_with_real_odds,
    sqrt_roi_strategy_with_real_odds,
    adaptive_strategy_with_real_odds,
    linear_scaled_strategy_with_real_odds,
    linear_roi_strategy_with_real_odds,
    kelly_criterion_strategy_with_real_odds,
    constant_profit_strategy_with_real_odds,
    combined_roi_odds_strategy_with_real_odds,
    hybrid_strategy_with_real_odds,
    log_roi_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

# МНОГО агрессивных параметров
strategies = []

# LINEAR ROI-ODDS - топ! Делаем много вариаций
for base_roi in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]:
    for base_pct in [1.4, 1.6, 1.8, 2.0, 2.2]:
        for max_pct in [14.0, 16.0, 18.0]:
            strategies.append(('linear_roi_odds', linear_roi_odds_strategy_with_real_odds, {
                'base_roi': base_roi, 'base_percent': base_pct, 'max_percent': max_pct,
                'odds_penalty_factor': 0.65, 'min_odds': 1.5, 'max_odds': 5.0
            }))

# ADAPTIVE - много вариаций  
for base_pct in [2.5, 2.8, 3.0, 3.2, 3.5]:
    for max_pct in [15.0, 17.0, 19.0, 21.0]:
        for min_roi in [3.5, 4.0, 4.5, 5.0]:
            strategies.append(('adaptive', adaptive_strategy_with_real_odds, {
                'base_percent': base_pct, 'max_percent': max_pct,
                'min_roi': min_roi, 'max_roi': min_roi + 11.0
            }))

# CONSTANT PROFIT - агрессивные параметры
for target in [1.6, 1.8, 2.0, 2.2, 2.5, 2.8]:
    for max_pct in [8.0, 9.0, 10.0, 11.0, 12.0]:
        strategies.append(('constant_profit', constant_profit_strategy_with_real_odds, {
            'target_profit_pct': target, 'max_percent': max_pct
        }))

# COMBINED ROI-ODDS
for base_pct in [2.0, 2.5, 3.0]:
    for max_pct in [14.0, 16.0, 18.0, 20.0]:
        for min_roi in [0.5, 1.0, 1.5]:
            strategies.append(('combined_roi_odds', combined_roi_odds_strategy_with_real_odds, {
                'base_percent': base_pct, 'max_percent': max_pct,
                'min_roi': min_roi, 'max_roi': min_roi + 25.0
            }))

# HYBRID
for base_pct in [2.0, 2.5, 3.0]:
    for max_pct in [14.0, 17.0, 20.0]:
        for min_roi in [1.0, 1.5, 2.0]:
            strategies.append(('hybrid', hybrid_strategy_with_real_odds, {
                'base_percent': base_pct, 'max_percent': max_pct,
                'min_roi': min_roi, 'max_roi': min_roi + 25.0,
                'min_odds': 1.4, 'max_odds': 5.2,
                'roi_weight': 0.6, 'odds_weight': 0.4
            }))

# LOG ROI
for base_roi in [8.0, 9.0, 10.0, 11.0, 12.0]:
    for base_pct in [1.8, 2.0, 2.2, 2.5]:
        for max_pct in [13.0, 15.0, 17.0]:
            strategies.append(('log_roi', log_roi_strategy_with_real_odds, {
                'base_roi': base_roi, 'base_percent': base_pct, 'max_percent': max_pct
            }))

# KELLY CRITERION
for risk in [1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
    for fraction in [0.6, 0.7, 0.75, 0.8]:
        strategies.append(('kelly_criterion', kelly_criterion_strategy_with_real_odds, {
            'risk': risk, 'kelly_fraction': fraction
        }))

print("="*80)
print("МАССОВЫЙ ПОИСК АГРЕССИВНЫХ СТРАТЕГИЙ")
print("Критерии: profit >= 80%, bankrupt <= 20%, max_bet <= 45%")
print("="*80)
print(f"Будет протестировано: {len(strategies)} × 2 = {len(strategies)*2} вариантов")
print("Это может занять 10-15 минут...\n")

count = 0
added = 0

for strategy_name, strategy_func, params in strategies:
    for apply_var in [False, True]:
        count += 1
        var_suffix = "_var" if apply_var else ""
        
        if count % 50 == 0:
            print(f"[{count}/{len(strategies)*2}] Обработано, найдено: {added}")
        
        try:
            bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                outcomes, odds_array, **params, apply_variation=apply_var
            )
            
            metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
            
            # Фильтры
            if metrics['avg_profit_pct'] < 80 or metrics['bankrupt_pct'] > 20 or max_bet > 45:
                continue
            
            params_str = "_".join([f"{v:.1f}" for v in list(params.values())[:3]])
            name = f"{strategy_name}_AGG_{params_str}{var_suffix}"
            
            result = {
                'strategy_name': name,
                'base_strategy': strategy_name,
                'strategy_params': params,
                'with_variation': "Yes" if apply_var else "No",
                'description': f"AGGRESSIVE: {strategy_name}",
                'avg_bet_pct': avg_bet,
                'min_bet_pct': min_bet,
                'max_bet_pct': max_bet,
                **metrics
            }
            
            save_results_to_csv(result, filename='results_aggressive_bankrupt10.csv')
            added += 1
            
        except:
            pass

print(f"\n{'='*80}")
print(f"✅ НАЙДЕНО агрессивных стратегий: {added}")
print(f"Файл: results_aggressive_bankrupt10.csv")
print("="*80)
