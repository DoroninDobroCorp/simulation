"""
Генерация АГРЕССИВНЫХ стратегий из отчета об оптимизации.
Цель: прибыль > 100%, bankrupt <= 10%, DD>50% допустим до 50%.

Берем параметры из Cautious, Balanced, Risky профилей + делаем много вариаций.
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
    calculate_metrics_with_odds,
    save_results_to_csv
)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

# Агрессивные параметры из отчета + вариации
aggressive_strategies = [
    # ===== LINEAR ROI-ODDS - топ из отчета =====
    # Cautious профиль (ROI: 208.90%, DD>50%: 2%)
    ('linear_roi_odds', linear_roi_odds_strategy_with_real_odds, [
        {'base_roi': 11.003, 'base_percent': 2.84, 'max_percent': 6.789, 'odds_penalty_factor': 0.7, 'min_odds': 1.552, 'max_odds': 4.034},
        {'base_roi': 10.0, 'base_percent': 2.5, 'max_percent': 7.0, 'odds_penalty_factor': 0.7, 'min_odds': 1.5, 'max_odds': 4.0},
        {'base_roi': 12.0, 'base_percent': 3.0, 'max_percent': 6.5, 'odds_penalty_factor': 0.75, 'min_odds': 1.6, 'max_odds': 4.2},
    ]),
    
    # Balanced профиль (ROI: 281.56%, DD>50%: 6.8%)
    ('linear_roi_odds', linear_roi_odds_strategy_with_real_odds, [
        {'base_roi': 5.649, 'base_percent': 1.533, 'max_percent': 15.089, 'odds_penalty_factor': 0.7, 'min_odds': 1.686, 'max_odds': 4.627},
        {'base_roi': 6.0, 'base_percent': 1.6, 'max_percent': 14.0, 'odds_penalty_factor': 0.65, 'min_odds': 1.7, 'max_odds': 4.5},
        {'base_roi': 5.0, 'base_percent': 1.4, 'max_percent': 16.0, 'odds_penalty_factor': 0.75, 'min_odds': 1.6, 'max_odds': 4.8},
        {'base_roi': 7.0, 'base_percent': 1.8, 'max_percent': 13.0, 'odds_penalty_factor': 0.7, 'min_odds': 1.75, 'max_odds': 4.3},
    ]),
    
    # ===== ADAPTIVE CONSTANT PROFIT =====
    # Cautious (ROI: 218.30%, DD>50%: 3.2%)
    ('adaptive_constant_profit', adaptive_constant_profit_strategy_with_real_odds, [
        {'min_roi': 0.117, 'max_roi': 31.685, 'min_target_pct': 0.203, 'max_target_pct': 7.41, 'max_bet_percent': 15.0},
        {'min_roi': 0.5, 'max_roi': 30.0, 'min_target_pct': 0.3, 'max_target_pct': 7.0, 'max_bet_percent': 14.0},
        {'min_roi': 1.0, 'max_roi': 32.0, 'min_target_pct': 0.25, 'max_target_pct': 7.5, 'max_bet_percent': 16.0},
    ]),
    
    # Balanced (ROI: 292.31%, DD>50%: 11%)
    ('adaptive_constant_profit', adaptive_constant_profit_strategy_with_real_odds, [
        {'min_roi': 2.474, 'max_roi': 34.033, 'min_target_pct': 1.443, 'max_target_pct': 6.059, 'max_bet_percent': 15.0},
        {'min_roi': 2.0, 'max_roi': 33.0, 'min_target_pct': 1.5, 'max_target_pct': 6.0, 'max_bet_percent': 14.0},
        {'min_roi': 3.0, 'max_roi': 35.0, 'min_target_pct': 1.4, 'max_target_pct': 6.5, 'max_bet_percent': 16.0},
        {'min_roi': 2.5, 'max_roi': 34.5, 'min_target_pct': 1.6, 'max_target_pct': 5.8, 'max_bet_percent': 15.5},
    ]),
    
    # ===== SQRT ROI =====
    # Balanced (ROI: 280.23%, DD>50%: 11%)
    ('sqrt_roi', sqrt_roi_strategy_with_real_odds, [
        {'base_roi': 5.703, 'base_percent': 1.439, 'max_percent': 28.148},
        {'base_roi': 6.0, 'base_percent': 1.5, 'max_percent': 25.0},
        {'base_roi': 5.5, 'base_percent': 1.4, 'max_percent': 30.0},
        {'base_roi': 6.5, 'base_percent': 1.6, 'max_percent': 23.0},
    ]),
    
    # ===== ADAPTIVE =====
    # Risky (ROI: 2602.50%, DD>50%: 65.4%, bankrupt: 2.6%) - но смягчим параметры
    ('adaptive', adaptive_strategy_with_real_odds, [
        {'base_percent': 2.476, 'max_percent': 16.839, 'min_roi': 4.933, 'max_roi': 14.367},
        {'base_percent': 2.2, 'max_percent': 15.0, 'min_roi': 5.0, 'max_roi': 15.0},
        {'base_percent': 2.7, 'max_percent': 18.0, 'min_roi': 4.5, 'max_roi': 14.0},
        {'base_percent': 2.0, 'max_percent': 14.0, 'min_roi': 5.5, 'max_roi': 15.5},
        {'base_percent': 2.8, 'max_percent': 17.0, 'min_roi': 4.8, 'max_roi': 13.5},
    ]),
    
    # ===== DYNAMIC KELLY =====
    # Cautious (ROI: 211.58%, DD>50%: 3.4%)
    ('dynamic_kelly', dynamic_kelly_strategy_with_real_odds, [
        {'risk': 4.264, 'min_fraction': 0.166, 'max_fraction': 0.552, 'min_roi': 5.369, 'max_roi': 24.12},
        {'risk': 4.0, 'min_fraction': 0.15, 'max_fraction': 0.55, 'min_roi': 5.0, 'max_roi': 25.0},
        {'risk': 4.5, 'min_fraction': 0.18, 'max_fraction': 0.50, 'min_roi': 5.5, 'max_roi': 23.0},
    ]),
    
    # ===== LINEAR SCALED =====
    # Extreme профиль - смягчим
    ('linear_scaled', linear_scaled_strategy_with_real_odds, [
        {'min_roi': 4.925, 'max_roi': 21.285, 'min_percent': 3.526, 'max_percent': 12.475},
        {'min_roi': 5.0, 'max_roi': 20.0, 'min_percent': 3.5, 'max_percent': 12.0},
        {'min_roi': 4.5, 'max_roi': 22.0, 'min_percent': 3.7, 'max_percent': 13.0},
        {'min_roi': 5.5, 'max_roi': 19.0, 'min_percent': 3.3, 'max_percent': 11.5},
    ]),
    
    # ===== LINEAR ROI =====
    # Дополнительные вариации
    ('linear_roi', linear_roi_strategy_with_real_odds, [
        {'base_roi': 5.164, 'base_percent': 0.614, 'max_percent': 27.211},
        {'base_roi': 5.0, 'base_percent': 0.7, 'max_percent': 25.0},
        {'base_roi': 5.5, 'base_percent': 0.8, 'max_percent': 28.0},
        {'base_roi': 4.8, 'base_percent': 0.65, 'max_percent': 26.0},
    ]),
    
    # ===== EXPONENTIAL ROI =====
    # Более агрессивные параметры
    ('exponential_roi', exponential_roi_strategy_with_real_odds, [
        {'base_roi': 7.0, 'base_percent': 0.5, 'factor': 0.05, 'max_percent': 5.0},
        {'base_roi': 6.5, 'base_percent': 0.6, 'factor': 0.06, 'max_percent': 6.0},
        {'base_roi': 7.5, 'base_percent': 0.55, 'factor': 0.055, 'max_percent': 5.5},
    ]),
]

print("="*80)
print("ГЕНЕРАЦИЯ АГРЕССИВНЫХ СТРАТЕГИЙ (profit > 100%, bankrupt <= 10%)")
print("="*80)

total_params = sum(len(params) for _, _, params in aggressive_strategies)
print(f"Будет протестировано: {total_params} наборов × 2 = {total_params * 2} вариантов\n")

count = 0
added = 0
skipped_profit = 0
skipped_bankrupt = 0
skipped_bets = 0

for strategy_name, strategy_func, params_list in aggressive_strategies:
    print(f"\n{'='*80}")
    print(f"СТРАТЕГИЯ: {strategy_name.upper()}")
    print(f"{'='*80}")
    
    for params in params_list:
        for apply_var in [False, True]:
            count += 1
            var_str = "Yes" if apply_var else "No"
            var_suffix = "_var" if apply_var else ""
            
            # Генерируем короткое имя
            params_str = "_".join([f"{v:.1f}" for v in list(params.values())[:3]])
            name = f"{strategy_name}_AGG_{params_str}{var_suffix}"
            
            print(f"[{count}/{total_params*2}] {name[:55]:<55}", end=' ', flush=True)
            
            try:
                bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                    outcomes, odds_array, **params, apply_variation=apply_var
                )
                
                metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
                
                # Фильтры для агрессивных стратегий
                if metrics['avg_profit_pct'] < 100:
                    print(f"⚠️ profit={metrics['avg_profit_pct']:.0f}% < 100%")
                    skipped_profit += 1
                    continue
                    
                if metrics['bankrupt_pct'] > 10:
                    print(f"⚠️ bankrupt={metrics['bankrupt_pct']:.1f}% > 10%")
                    skipped_bankrupt += 1
                    continue
                    
                if max_bet > 50:
                    print(f"⚠️ max_bet={max_bet:.0f}% > 50%")
                    skipped_bets += 1
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
                print(f"✅ +{metrics['avg_profit_pct']:.0f}% DD50:{metrics['drawdown_50_pct']:.1f}% B:{metrics['bankrupt_pct']:.1f}%")
                
            except Exception as e:
                print(f"❌ {str(e)[:40]}")

print("\n" + "="*80)
print("📊 ИТОГИ ГЕНЕРАЦИИ")
print("="*80)
print(f"✅ Добавлено агрессивных стратегий: {added}")
print(f"⚠️ Пропущено (profit < 100%): {skipped_profit}")
print(f"⚠️ Пропущено (bankrupt > 10%): {skipped_bankrupt}")
print(f"⚠️ Пропущено (max_bet > 50%): {skipped_bets}")
print(f"\nФайл: results_aggressive_bankrupt10.csv")
