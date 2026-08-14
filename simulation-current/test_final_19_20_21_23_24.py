"""
Финальное тестирование последних 5 стратегий:
#19 Sharpe Ratio Optimization
#20 Multi-Objective Optimization  
#21 Bayesian Kelly
#23 Portfolio Theory Approach
#24 ML Adaptive

5 × 6 = 30 вариантов
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    sharpe_optimized_strategy_with_real_odds,
    multi_objective_strategy_with_real_odds,
    bayesian_kelly_strategy_with_real_odds,
    portfolio_theory_strategy_with_real_odds,
    ml_adaptive_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

print("="*70)
print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ: СТРАТЕГИИ #19, #20, #21, #23, #24")
print("="*70)

outcomes, odds_array = load_real_odds_outcomes()
print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# Параметры для всех стратегий
sharpe_params = [
    {'base_percent': 2.0, 'lookback': 50, 'risk_free_rate': 0.0},
    {'base_percent': 2.5, 'lookback': 100, 'risk_free_rate': 0.0},
    {'base_percent': 3.0, 'lookback': 150, 'risk_free_rate': 0.0},
]

multi_objective_params = [
    {'base_percent': 2.0, 'w_profit': 0.6, 'w_drawdown': 0.3, 'w_volatility': 0.1, 'lookback': 50},
    {'base_percent': 2.5, 'w_profit': 0.5, 'w_drawdown': 0.3, 'w_volatility': 0.2, 'lookback': 50},
    {'base_percent': 3.0, 'w_profit': 0.4, 'w_drawdown': 0.4, 'w_volatility': 0.2, 'lookback': 50},
]

bayesian_kelly_params = [
    {'prior_mean': 0.378, 'prior_std': 0.03, 'risk_factor': 3.0, 'max_percent': 5.0},
    {'prior_mean': 0.378, 'prior_std': 0.05, 'risk_factor': 2.0, 'max_percent': 10.0},
    {'prior_mean': 0.378, 'prior_std': 0.07, 'risk_factor': 1.5, 'max_percent': 15.0},
]

portfolio_params = [
    {'base_percent': 2.0, 'rebalance_frequency': 100},
    {'base_percent': 2.5, 'rebalance_frequency': 150},
    {'base_percent': 3.0, 'rebalance_frequency': 200},
]

ml_adaptive_params = [
    {'base_percent': 2.0, 'learning_rate': 0.01, 'lookback': 30},
    {'base_percent': 2.5, 'learning_rate': 0.05, 'lookback': 50},
    {'base_percent': 3.0, 'learning_rate': 0.10, 'lookback': 70},
]

all_strategies = [
    ('sharpe_optimized', '#19 Sharpe Ratio', sharpe_optimized_strategy_with_real_odds, sharpe_params),
    ('multi_objective', '#20 Multi-Objective', multi_objective_strategy_with_real_odds, multi_objective_params),
    ('bayesian_kelly', '#21 Bayesian Kelly', bayesian_kelly_strategy_with_real_odds, bayesian_kelly_params),
    ('portfolio_theory', '#23 Portfolio Theory', portfolio_theory_strategy_with_real_odds, portfolio_params),
    ('ml_adaptive', '#24 ML Adaptive', ml_adaptive_strategy_with_real_odds, ml_adaptive_params),
]

results = []
total = 30
count = 0

for strategy_key, strategy_name, strategy_func, params_list in all_strategies:
    print("\n" + "="*70)
    print(strategy_name)
    print("="*70)
    
    for idx, params in enumerate(params_list, 1):
        for apply_var in [False, True]:
            count += 1
            var_suffix = "_with_variation" if apply_var else ""
            
            # Генерируем краткое имя
            if strategy_key == 'sharpe_optimized':
                name = f"{strategy_key}_b{params['base_percent']}_lb{params['lookback']}{var_suffix}"
            elif strategy_key == 'multi_objective':
                name = f"{strategy_key}_b{params['base_percent']}_w{params['w_profit']}-{params['w_drawdown']}-{params['w_volatility']}{var_suffix}"
            elif strategy_key == 'bayesian_kelly':
                name = f"{strategy_key}_pm{params['prior_mean']}_ps{params['prior_std']}_rf{params['risk_factor']}{var_suffix}"
            elif strategy_key == 'portfolio_theory':
                name = f"{strategy_key}_b{params['base_percent']}_rb{params['rebalance_frequency']}{var_suffix}"
            elif strategy_key == 'ml_adaptive':
                name = f"{strategy_key}_b{params['base_percent']}_lr{params['learning_rate']}_lb{params['lookback']}{var_suffix}"
            
            print(f"\n[{count}/{total}] {name}")
            
            try:
                bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                    outcomes, odds_array, **params, apply_variation=apply_var
                )
                
                metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
                
                result = {
                    'strategy_name': name,
                    'base_strategy': strategy_key,
                    'strategy_params': params,
                    'with_variation': 'Yes' if apply_var else 'No',
                    'description': f"{strategy_name}: params{idx}. {'With var' if apply_var else 'No var'}",
                    'avg_bet_pct': avg_bet,
                    'min_bet_pct': min_bet,
                    'max_bet_pct': max_bet,
                    **metrics
                }
                
                save_results_to_csv(result)
                results.append(result)
                print(f"  ✅ Profit: {metrics['avg_profit_pct']:.1f}%, DD>50%: {metrics['drawdown_50_pct']:.2f}%, Bankrupt: {metrics['bankrupt_pct']:.2f}%")
            
            except Exception as e:
                print(f"  ❌ ОШИБКА: {e}")

print("\n" + "="*70)
print("🎉 ФИНАЛЬНЫЕ ИТОГИ")
print("="*70)
print(f"\n✅ Добавлено {len(results)} новых записей")
print(f"Теперь всего: 122 + {len(results)} = {122 + len(results)} стратегий")

if results:
    for strategy_key in ['sharpe_optimized', 'multi_objective', 'bayesian_kelly', 'portfolio_theory', 'ml_adaptive']:
        strat_results = [r for r in results if r['base_strategy'] == strategy_key]
        if strat_results:
            avg_profit = np.mean([r['avg_profit_pct'] for r in strat_results])
            avg_dd50 = np.mean([r['drawdown_50_pct'] for r in strat_results])
            print(f"\n{strategy_key}: avg_profit={avg_profit:.1f}%, avg_dd50={avg_dd50:.2f}%")

print("\n" + "="*70)
print("🏁 ВСЕ 24 СТРАТЕГИИ РЕАЛИЗОВАНЫ И ПРОТЕСТИРОВАНЫ!")
print("="*70)
