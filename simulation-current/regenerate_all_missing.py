"""
Перегенерация всех недостающих/проблемных стратегий с адекватными параметрами.
Цель: прибыль 20-200%, DD>50% от 0% до 40%, без безумных ставок.
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    kelly_criterion_strategy_with_real_odds,
    linear_roi_strategy_with_real_odds,
    sqrt_roi_strategy_with_real_odds,
    exponential_roi_strategy_with_real_odds,
    constant_profit_strategy_with_real_odds,
    linear_scaled_strategy_with_real_odds,
    linear_roi_odds_strategy_with_real_odds,
    adaptive_constant_profit_strategy_with_real_odds,
    fixed_fraction_strategy_with_real_odds,
    anti_martingale_strategy_with_real_odds,
    proportional_kelly_strategy_with_real_odds,
    volatility_adjusted_strategy_with_real_odds,
    target_based_strategy_with_real_odds,
    sharpe_optimized_strategy_with_real_odds,
    multi_objective_strategy_with_real_odds,
    bayesian_kelly_strategy_with_real_odds,
    streak_aware_strategy_with_real_odds,
    portfolio_theory_strategy_with_real_odds,
    ml_adaptive_strategy_with_real_odds,
    dynamic_percentage_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок\n")

# Параметры с адекватными значениями (избегаем экстремальных)
strategies_to_generate = [
    # Kelly - более агрессивные параметры
    ('kelly_criterion', kelly_criterion_strategy_with_real_odds, [
        {'risk': 2.0, 'kelly_fraction': 0.4},  # Умеренная
        {'risk': 1.8, 'kelly_fraction': 0.6},  # Агрессивная
        {'risk': 1.5, 'kelly_fraction': 0.8},  # Очень агрессивная
    ]),
    
    # Linear ROI - средние параметры
    ('linear_roi', linear_roi_strategy_with_real_odds, [
        {'base_roi': 5.0, 'base_percent': 1.5, 'max_percent': 6.0},
        {'base_roi': 5.0, 'base_percent': 2.0, 'max_percent': 8.0},
        {'base_roi': 5.0, 'base_percent': 2.5, 'max_percent': 10.0},
    ]),
    
    # Sqrt ROI - дополнительные параметры
    ('sqrt_roi', sqrt_roi_strategy_with_real_odds, [
        {'base_roi': 5.0, 'base_percent': 1.5, 'max_percent': 8.0},
        {'base_roi': 7.0, 'base_percent': 2.0, 'max_percent': 12.0},
    ]),
    
    # Exponential ROI - более мягкие факторы
    ('exponential_roi', exponential_roi_strategy_with_real_odds, [
        {'base_roi': 5.0, 'base_percent': 1.0, 'factor': 0.08, 'max_percent': 8.0},
        {'base_roi': 7.0, 'base_percent': 1.5, 'factor': 0.12, 'max_percent': 12.0},
    ]),
    
    # Constant Profit - умеренные цели
    ('constant_profit', constant_profit_strategy_with_real_odds, [
        {'target_profit_pct': 1.5, 'max_percent': 7.0},
        {'target_profit_pct': 2.0, 'max_percent': 9.0},
        {'target_profit_pct': 2.5, 'max_percent': 11.0},
    ]),
    
    # Linear Scaled - средние диапазоны
    ('linear_scaled', linear_scaled_strategy_with_real_odds, [
        {'min_roi': 5.0, 'max_roi': 15.0, 'min_percent': 1.0, 'max_percent': 5.0},
        {'min_roi': 3.0, 'max_roi': 18.0, 'min_percent': 1.5, 'max_percent': 8.0},
    ]),
    
    # Linear ROI-Odds - умеренные penalty
    ('linear_roi_odds', linear_roi_odds_strategy_with_real_odds, [
        {'base_roi': 5.0, 'base_percent': 1.5, 'max_percent': 8.0, 'odds_penalty_factor': 0.5, 'min_odds': 1.5, 'max_odds': 5.0},
        {'base_roi': 7.0, 'base_percent': 2.0, 'max_percent': 12.0, 'odds_penalty_factor': 0.6, 'min_odds': 1.5, 'max_odds': 5.0},
    ]),
    
    # Adaptive Constant Profit - мягкие цели
    ('adaptive_constant_profit', adaptive_constant_profit_strategy_with_real_odds, [
        {'min_roi': 5.0, 'max_roi': 15.0, 'min_target_pct': 0.8, 'max_target_pct': 2.0, 'max_bet_percent': 10.0},
    ]),
    
    # Fixed Fraction - умеренные проценты
    ('fixed_fraction', fixed_fraction_strategy_with_real_odds, [
        {'fixed_percent': 1.5},
        {'fixed_percent': 3.0},
    ]),
    
    # Anti-Martingale - консервативные multiplier
    ('anti_martingale', anti_martingale_strategy_with_real_odds, [
        {'base_percent': 1.5, 'multiplier': 1.4, 'max_percent': 7.0, 'max_streak': 4},
        {'base_percent': 2.0, 'multiplier': 1.5, 'max_percent': 10.0, 'max_streak': 4},
        {'base_percent': 2.5, 'multiplier': 1.6, 'max_percent': 12.0, 'max_streak': 5},
    ]),
    
    # Proportional Kelly - более высокий confidence
    ('proportional_kelly', proportional_kelly_strategy_with_real_odds, [
        {'risk': 2.0, 'confidence': 0.8, 'max_percent': 8.0},
        {'risk': 1.8, 'confidence': 0.9, 'max_percent': 10.0},
        {'risk': 1.5, 'confidence': 1.0, 'max_percent': 12.0},
    ]),
    
    # Volatility-Adjusted - адекватные base
    ('volatility_adjusted', volatility_adjusted_strategy_with_real_odds, [
        {'base_percent': 2.5, 'lookback': 40, 'volatility_factor': 0.8},
        {'base_percent': 3.0, 'lookback': 60, 'volatility_factor': 1.0},
        {'base_percent': 3.5, 'lookback': 80, 'volatility_factor': 1.2},
    ]),
    
    # Target-Based - умеренные проценты
    ('target_based', target_based_strategy_with_real_odds, [
        {'target_bankroll_percent': 150.0, 'aggressive_pct': 2.5, 'conservative_pct': 0.8},
        {'target_bankroll_percent': 180.0, 'aggressive_pct': 3.0, 'conservative_pct': 1.0},
        {'target_bankroll_percent': 220.0, 'aggressive_pct': 3.5, 'conservative_pct': 1.2},
    ]),
    
    # Sharpe Optimized - меньше base
    ('sharpe_optimized', sharpe_optimized_strategy_with_real_odds, [
        {'base_percent': 2.5, 'lookback': 80, 'risk_free_rate': 0.0},
        {'base_percent': 3.0, 'lookback': 120, 'risk_free_rate': 0.0},
        {'base_percent': 3.5, 'lookback': 160, 'risk_free_rate': 0.0},
    ]),
    
    # Multi-Objective - разные веса
    ('multi_objective', multi_objective_strategy_with_real_odds, [
        {'base_percent': 2.5, 'w_profit': 0.6, 'w_drawdown': 0.25, 'w_volatility': 0.15, 'lookback': 60},
        {'base_percent': 3.0, 'w_profit': 0.5, 'w_drawdown': 0.35, 'w_volatility': 0.15, 'lookback': 60},
        {'base_percent': 3.5, 'w_profit': 0.4, 'w_drawdown': 0.45, 'w_volatility': 0.15, 'lookback': 60},
    ]),
    
    # Bayesian Kelly - более высокий prior_mean
    ('bayesian_kelly', bayesian_kelly_strategy_with_real_odds, [
        {'prior_mean': 0.40, 'prior_std': 0.04, 'risk_factor': 1.8, 'max_percent': 8.0},
        {'prior_mean': 0.42, 'prior_std': 0.05, 'risk_factor': 1.5, 'max_percent': 10.0},
        {'prior_mean': 0.45, 'prior_std': 0.06, 'risk_factor': 1.3, 'max_percent': 12.0},
    ]),
    
    # Streak Aware - умеренные multipliers
    ('streak_aware', streak_aware_strategy_with_real_odds, [
        {'base_percent': 2.5, 'win_streak_multiplier': 1.15, 'loss_streak_divider': 1.25, 'max_multiplier': 2.5},
        {'base_percent': 3.0, 'win_streak_multiplier': 1.2, 'loss_streak_divider': 1.3, 'max_multiplier': 3.0},
        {'base_percent': 3.5, 'win_streak_multiplier': 1.25, 'loss_streak_divider': 1.35, 'max_multiplier': 3.5},
    ]),
    
    # Portfolio Theory - умеренные base
    ('portfolio_theory', portfolio_theory_strategy_with_real_odds, [
        {'base_percent': 2.5, 'rebalance_frequency': 120},
        {'base_percent': 3.0, 'rebalance_frequency': 160},
        {'base_percent': 3.5, 'rebalance_frequency': 200},
    ]),
    
    # ML Adaptive - меньше learning_rate
    ('ml_adaptive', ml_adaptive_strategy_with_real_odds, [
        {'base_percent': 2.5, 'learning_rate': 0.02, 'lookback': 40},
        {'base_percent': 3.0, 'learning_rate': 0.04, 'lookback': 60},
        {'base_percent': 3.5, 'learning_rate': 0.06, 'lookback': 80},
    ]),
    
    # Dynamic % - умеренные
    ('dynamic_percentage', dynamic_percentage_strategy_with_real_odds, [
        {'bet_size_pct': 2.5},
        {'bet_size_pct': 3.5},
    ]),
]

count = 0
total = sum(len(params) * 2 for _, _, params in strategies_to_generate)

print(f"Будет сгенерировано: {total} вариантов\n")
print("="*70)

for strategy_name, strategy_func, params_list in strategies_to_generate:
    print(f"\n{strategy_name.upper()}")
    print("-"*70)
    
    for params in params_list:
        for apply_var in [False, True]:
            count += 1
            var_str = "Yes" if apply_var else "No"
            
            # Генерируем краткое имя
            params_str = "_".join([f"{k[:3]}{v}" for k, v in list(params.items())[:3]])
            name = f"{strategy_name}_{params_str}{'_var' if apply_var else ''}"
            
            print(f"[{count}/{total}] {name[:60]}... ", end='', flush=True)
            
            try:
                bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                    outcomes, odds_array, **params, apply_variation=apply_var
                )
                
                metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
                
                # Проверяем на адекватность
                if max_bet > 50 or metrics['avg_profit_pct'] < 5:
                    print(f"⚠️ Skip (max_bet={max_bet:.1f}%, profit={metrics['avg_profit_pct']:.1f}%)")
                    continue
                
                result = {
                    'strategy_name': name,
                    'base_strategy': strategy_name,
                    'strategy_params': params,
                    'with_variation': var_str,
                    'description': f"{strategy_name} {params_str}",
                    'avg_bet_pct': avg_bet,
                    'min_bet_pct': min_bet,
                    'max_bet_pct': max_bet,
                    **metrics
                }
                
                save_results_to_csv(result)
                print(f"✅ profit={metrics['avg_profit_pct']:.0f}%, DD50={metrics['drawdown_50_pct']:.1f}%")
                
            except Exception as e:
                print(f"❌ Error: {e}")

print("\n" + "="*70)
print(f"✅ Генерация завершена!")
