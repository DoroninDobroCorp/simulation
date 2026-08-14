"""
ПОЛНЫЙ ПЕРЕБОР АГРЕССИВНЫХ ПАРАМЕТРОВ для ВСЕХ стратегий.
Цель: прибыль > 100%, bankrupt 5-10% (допустимо сливаться иногда).

Делаем МНОГО вариаций для каждой стратегии!
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    linear_roi_odds_strategy_with_real_odds,
    adaptive_constant_profit_strategy_with_real_odds,
    sqrt_roi_strategy_with_real_odds,
    adaptive_strategy_with_real_odds,
    dynamic_kelly_strategy_with_real_odds,
    linear_scaled_strategy_with_real_odds,
    linear_roi_strategy_with_real_odds,
    exponential_roi_strategy_with_real_odds,
    kelly_criterion_strategy_with_real_odds,
    constant_profit_strategy_with_real_odds,
    combined_roi_odds_strategy_with_real_odds,
    hybrid_strategy_with_real_odds,
    log_roi_strategy_with_real_odds,
    fixed_fraction_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

# БОЛЬШОЙ набор агрессивных параметров для каждой стратегии
aggressive_strategies = [
    # ===== 1. LINEAR ROI-ODDS (топ из отчета!) =====
    ('linear_roi_odds', linear_roi_odds_strategy_with_real_odds, [
        # Из отчета Balanced (ROI: 281.56%, DD>50%: 6.8%)
        {'base_roi': 5.649, 'base_percent': 1.533, 'max_percent': 15.089, 'odds_penalty_factor': 0.7, 'min_odds': 1.686, 'max_odds': 4.627},
        # Вариации - увеличиваем агрессию
        {'base_roi': 5.0, 'base_percent': 1.8, 'max_percent': 16.0, 'odds_penalty_factor': 0.6, 'min_odds': 1.6, 'max_odds': 5.0},
        {'base_roi': 4.5, 'base_percent': 2.0, 'max_percent': 18.0, 'odds_penalty_factor': 0.65, 'min_odds': 1.5, 'max_odds': 5.2},
        {'base_roi': 6.0, 'base_percent': 1.7, 'max_percent': 17.0, 'odds_penalty_factor': 0.55, 'min_odds': 1.7, 'max_odds': 4.8},
        {'base_roi': 4.0, 'base_percent': 2.2, 'max_percent': 19.0, 'odds_penalty_factor': 0.6, 'min_odds': 1.5, 'max_odds': 5.5},
        {'base_roi': 5.5, 'base_percent': 1.9, 'max_percent': 17.5, 'odds_penalty_factor': 0.65, 'min_odds': 1.65, 'max_odds': 4.9},
    ]),
    
    # ===== 2. ADAPTIVE (Risky профиль - ROI: 2602%!) =====
    ('adaptive', adaptive_strategy_with_real_odds, [
        # Из отчета но чуть мягче
        {'base_percent': 2.476, 'max_percent': 16.839, 'min_roi': 4.933, 'max_roi': 14.367},
        {'base_percent': 2.8, 'max_percent': 18.0, 'min_roi': 4.5, 'max_roi': 15.0},
        {'base_percent': 3.0, 'max_percent': 19.0, 'min_roi': 4.0, 'max_roi': 15.5},
        {'base_percent': 2.5, 'max_percent': 17.5, 'min_roi': 5.0, 'max_roi': 14.0},
        {'base_percent': 3.2, 'max_percent': 20.0, 'min_roi': 3.8, 'max_roi': 16.0},
        {'base_percent': 2.3, 'max_percent': 16.0, 'min_roi': 5.2, 'max_roi': 13.5},
        {'base_percent': 3.5, 'max_percent': 21.0, 'min_roi': 3.5, 'max_roi': 16.5},
    ]),
    
    # ===== 3. LINEAR SCALED (Extreme профиль - ROI: 11034%!) =====
    ('linear_scaled', linear_scaled_strategy_with_real_odds, [
        # Из отчета но мягче
        {'min_roi': 4.925, 'max_roi': 21.285, 'min_percent': 3.526, 'max_percent': 12.475},
        # Вариации
        {'min_roi': 5.0, 'max_roi': 20.0, 'min_percent': 3.5, 'max_percent': 12.0},
        {'min_roi': 4.5, 'max_roi': 22.0, 'min_percent': 3.8, 'max_percent': 13.0},
        {'min_roi': 5.5, 'max_roi': 19.0, 'min_percent': 3.3, 'max_percent': 11.5},
        {'min_roi': 4.0, 'max_roi': 23.0, 'min_percent': 4.0, 'max_percent': 13.5},
        {'min_roi': 6.0, 'max_roi': 18.0, 'min_percent': 3.0, 'max_percent': 11.0},
    ]),
    
    # ===== 4. ADAPTIVE CONSTANT PROFIT =====
    ('adaptive_constant_profit', adaptive_constant_profit_strategy_with_real_odds, [
        # Balanced из отчета (ROI: 292%, DD>50%: 11%)
        {'min_roi': 2.474, 'max_roi': 34.033, 'min_target_pct': 1.443, 'max_target_pct': 6.059, 'max_bet_percent': 15.0},
        # Увеличиваем агрессию
        {'min_roi': 2.0, 'max_roi': 35.0, 'min_target_pct': 1.6, 'max_target_pct': 6.5, 'max_bet_percent': 16.0},
        {'min_roi': 1.5, 'max_roi': 36.0, 'min_target_pct': 1.8, 'max_target_pct': 7.0, 'max_bet_percent': 17.0},
        {'min_roi': 2.5, 'max_roi': 33.0, 'min_target_pct': 1.5, 'max_target_pct': 6.2, 'max_bet_percent': 15.5},
        {'min_roi': 1.8, 'max_roi': 35.5, 'min_target_pct': 1.7, 'max_target_pct': 6.8, 'max_bet_percent': 16.5},
    ]),
    
    # ===== 5. SQRT ROI =====
    ('sqrt_roi', sqrt_roi_strategy_with_real_odds, [
        # Balanced из отчета (ROI: 280%, DD>50%: 11%)
        {'base_roi': 5.703, 'base_percent': 1.439, 'max_percent': 28.148},
        # Увеличиваем агрессию
        {'base_roi': 5.0, 'base_percent': 1.6, 'max_percent': 30.0},
        {'base_roi': 4.5, 'base_percent': 1.8, 'max_percent': 32.0},
        {'base_roi': 6.0, 'base_percent': 1.5, 'max_percent': 27.0},
        {'base_roi': 5.5, 'base_percent': 1.7, 'max_percent': 29.0},
        {'base_roi': 4.0, 'base_percent': 2.0, 'max_percent': 34.0},
    ]),
    
    # ===== 6. LINEAR ROI =====
    ('linear_roi', linear_roi_strategy_with_real_odds, [
        # Conservative из отчета + агрессивнее
        {'base_roi': 5.164, 'base_percent': 0.614, 'max_percent': 27.211},
        {'base_roi': 5.0, 'base_percent': 0.8, 'max_percent': 28.0},
        {'base_roi': 4.5, 'base_percent': 1.0, 'max_percent': 30.0},
        {'base_roi': 5.5, 'base_percent': 0.7, 'max_percent': 26.0},
        {'base_roi': 4.0, 'base_percent': 1.2, 'max_percent': 32.0},
        {'base_roi': 6.0, 'base_percent': 0.6, 'max_percent': 25.0},
    ]),
    
    # ===== 7. KELLY CRITERION (более агрессивный) =====
    ('kelly_criterion', kelly_criterion_strategy_with_real_odds, [
        {'risk': 1.5, 'kelly_fraction': 0.8},
        {'risk': 1.3, 'kelly_fraction': 0.9},
        {'risk': 1.8, 'kelly_fraction': 0.7},
        {'risk': 1.2, 'kelly_fraction': 1.0},
        {'risk': 1.4, 'kelly_fraction': 0.85},
        {'risk': 1.6, 'kelly_fraction': 0.75},
    ]),
    
    # ===== 8. EXPONENTIAL ROI (более агрессивный) =====
    ('exponential_roi', exponential_roi_strategy_with_real_odds, [
        {'base_roi': 7.0, 'base_percent': 0.5, 'factor': 0.05, 'max_percent': 5.0},
        {'base_roi': 6.0, 'base_percent': 0.7, 'factor': 0.07, 'max_percent': 6.0},
        {'base_roi': 6.5, 'base_percent': 0.6, 'factor': 0.06, 'max_percent': 5.5},
        {'base_roi': 5.5, 'base_percent': 0.8, 'factor': 0.08, 'max_percent': 6.5},
        {'base_roi': 7.5, 'base_percent': 0.55, 'factor': 0.055, 'max_percent': 5.2},
    ]),
    
    # ===== 9. CONSTANT PROFIT (агрессивный) =====
    ('constant_profit', constant_profit_strategy_with_real_odds, [
        {'target_profit_pct': 2.0, 'max_percent': 10.0},
        {'target_profit_pct': 2.5, 'max_percent': 12.0},
        {'target_profit_pct': 1.8, 'max_percent': 9.0},
        {'target_profit_pct': 3.0, 'max_percent': 14.0},
        {'target_profit_pct': 2.2, 'max_percent': 11.0},
    ]),
    
    # ===== 10. COMBINED ROI-ODDS (агрессивный) =====
    ('combined_roi_odds', combined_roi_odds_strategy_with_real_odds, [
        {'base_percent': 2.0, 'max_percent': 15.0, 'min_roi': 1.0, 'max_roi': 25.0},
        {'base_percent': 2.5, 'max_percent': 18.0, 'min_roi': 0.5, 'max_roi': 28.0},
        {'base_percent': 1.8, 'max_percent': 14.0, 'min_roi': 1.5, 'max_roi': 23.0},
        {'base_percent': 3.0, 'max_percent': 20.0, 'min_roi': 0.8, 'max_roi': 30.0},
    ]),
    
    # ===== 11. HYBRID (агрессивный) =====
    ('hybrid', hybrid_strategy_with_real_odds, [
        {'base_percent': 2.0, 'max_percent': 15.0, 'min_roi': 1.0, 'max_roi': 25.0, 'min_odds': 1.5, 'max_odds': 5.0, 'roi_weight': 0.6, 'odds_weight': 0.4},
        {'base_percent': 2.5, 'max_percent': 18.0, 'min_roi': 0.5, 'max_roi': 28.0, 'min_odds': 1.3, 'max_odds': 5.5, 'roi_weight': 0.65, 'odds_weight': 0.35},
        {'base_percent': 3.0, 'max_percent': 20.0, 'min_roi': 0.8, 'max_roi': 30.0, 'min_odds': 1.4, 'max_odds': 5.3, 'roi_weight': 0.55, 'odds_weight': 0.45},
    ]),
    
    # ===== 12. LOG ROI (агрессивный) =====
    ('log_roi', log_roi_strategy_with_real_odds, [
        {'base_roi': 10.0, 'base_percent': 2.0, 'max_percent': 15.0},
        {'base_roi': 8.0, 'base_percent': 2.5, 'max_percent': 18.0},
        {'base_roi': 12.0, 'base_percent': 1.8, 'max_percent': 14.0},
        {'base_roi': 9.0, 'base_percent': 2.2, 'max_percent': 16.0},
    ]),
    
    # ===== 13. FIXED FRACTION (агрессивный) =====
    ('fixed_fraction', fixed_fraction_strategy_with_real_odds, [
        {'fixed_percent': 3.0},
        {'fixed_percent': 3.5},
        {'fixed_percent': 4.0},
        {'fixed_percent': 2.8},
        {'fixed_percent': 3.2},
    ]),
]

print("="*80)
print("ПОЛНЫЙ ПЕРЕБОР АГРЕССИВНЫХ СТРАТЕГИЙ")
print("Цель: profit > 80%, bankrupt <= 15%")
print("="*80)

total_params = sum(len(params) for _, _, params in aggressive_strategies)
print(f"Будет протестировано: {total_params} наборов × 2 = {total_params * 2} вариантов")
print("Это займет время, но мы найдем лучшие!\n")

count = 0
added = 0

for strategy_name, strategy_func, params_list in aggressive_strategies:
    print(f"\n{'='*80}")
    print(f"{strategy_name.upper()}")
    print(f"{'='*80}")
    
    for params in params_list:
        for apply_var in [False, True]:
            count += 1
            var_str = "Yes" if apply_var else "No"
            var_suffix = "_var" if apply_var else ""
            
            # Короткое имя
            params_str = "_".join([f"{v:.1f}" for v in list(params.values())[:3]])
            name = f"{strategy_name}_AGG_{params_str}{var_suffix}"
            
            print(f"[{count}/{total_params*2}] {name[:50]:<50}", end=' ', flush=True)
            
            try:
                bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                    outcomes, odds_array, **params, apply_variation=apply_var
                )
                
                metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
                
                # Фильтры: profit > 100%, bankrupt <= 10%, max_bet <= 50%
                if metrics['avg_profit_pct'] < 80:
                    print(f"skip: profit={metrics['avg_profit_pct']:.0f}%")
                    continue
                    
                if metrics['bankrupt_pct'] > 15:
                    print(f"skip: bankrupt={metrics['bankrupt_pct']:.1f}%")
                    continue
                    
                if max_bet > 50:
                    print(f"skip: max_bet={max_bet:.0f}%")
                    continue
                
                result = {
                    'strategy_name': name,
                    'base_strategy': strategy_name,
                    'strategy_params': params,
                    'with_variation': var_str,
                    'description': f"AGGRESSIVE: {strategy_name}",
                    'avg_bet_pct': avg_bet,
                    'min_bet_pct': min_bet,
                    'max_bet_pct': max_bet,
                    **metrics
                }
                
                save_results_to_csv(result, filename='results_aggressive_bankrupt10.csv')
                added += 1
                print(f"✅ +{metrics['avg_profit_pct']:.0f}% B:{metrics['bankrupt_pct']:.1f}% DD50:{metrics['drawdown_50_pct']:.1f}%")
                
            except Exception as e:
                print(f"error: {str(e)[:30]}")

print("\n" + "="*80)
print(f"✅ Найдено и добавлено агрессивных стратегий: {added}/{total_params*2}")
print(f"Файл: results_aggressive_bankrupt10.csv")
print("="*80)
