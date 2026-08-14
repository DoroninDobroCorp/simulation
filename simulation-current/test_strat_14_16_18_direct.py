"""
Прямое тестирование стратегий #14, #16, #18.
Вызываем функции напрямую, без run_strategy_with_real_odds.
"""

import numpy as np
import pandas as pd
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    fixed_fraction_strategy_with_real_odds,
    proportional_kelly_strategy_with_real_odds,
    target_based_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

print("="*70)
print("ТЕСТИРОВАНИЕ СТРАТЕГИЙ #14, #16, #18 (ПРЯМОЙ ВЫЗОВ)")
print("="*70)

outcomes, odds_array = load_real_odds_outcomes()
print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# Параметры
fixed_fraction_params = [
    {'fixed_percent': 1.0},
    {'fixed_percent': 2.5},
    {'fixed_percent': 5.0},
]

proportional_kelly_params = [
    {'risk': 2.5, 'confidence': 0.5, 'max_percent': 5.0},
    {'risk': 2.0, 'confidence': 0.7, 'max_percent': 10.0},
    {'risk': 1.5, 'confidence': 0.9, 'max_percent': 15.0},
]

target_based_params = [
    {'target_bankroll_percent': 150.0, 'aggressive_pct': 2.0, 'conservative_pct': 0.5},
    {'target_bankroll_percent': 200.0, 'aggressive_pct': 3.0, 'conservative_pct': 1.0},
    {'target_bankroll_percent': 300.0, 'aggressive_pct': 5.0, 'conservative_pct': 1.5},
]

results = []

# #14 Fixed Fraction
print("\n" + "="*70)
print("#14 FIXED FRACTION STRATEGY")
print("="*70)

for params in fixed_fraction_params:
    for apply_var in [False, True]:
        var_suffix = "_with_variation" if apply_var else ""
        name = f"fixed_fraction_{params['fixed_percent']}pct{var_suffix}"
        
        print(f"\n{name}: {params}, var={apply_var}")
        
        bankroll, bet_history, min_bet, max_bet, avg_bet = fixed_fraction_strategy_with_real_odds(
            outcomes, odds_array, **params, apply_variation=apply_var
        )
        
        metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
        
        result = {
            'strategy_name': name,
            'base_strategy': 'fixed_fraction',
            'strategy_params': params,
            'with_variation': 'Yes' if apply_var else 'No',
            'description': f"Fixed Fraction: {params['fixed_percent']}% fixed. {'With variation' if apply_var else 'No variation'}",
            'avg_bet_pct': avg_bet,
            'min_bet_pct': min_bet,
            'max_bet_pct': max_bet,
            **metrics
        }
        
        save_results_to_csv(result)
        results.append(result)
        
        print(f"  Profit: {metrics['avg_profit_pct']:.2f}%, DD>50%: {metrics['drawdown_50_pct']:.2f}%")

# #16 Proportional Kelly
print("\n" + "="*70)
print("#16 PROPORTIONAL KELLY STRATEGY")
print("="*70)

for params in proportional_kelly_params:
    for apply_var in [False, True]:
        var_suffix = "_with_variation" if apply_var else ""
        name = f"proportional_kelly_r{params['risk']}_c{params['confidence']}{var_suffix}"
        
        print(f"\n{name}: {params}, var={apply_var}")
        
        bankroll, bet_history, min_bet, max_bet, avg_bet = proportional_kelly_strategy_with_real_odds(
            outcomes, odds_array, **params, apply_variation=apply_var
        )
        
        metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
        
        result = {
            'strategy_name': name,
            'base_strategy': 'proportional_kelly',
            'strategy_params': params,
            'with_variation': 'Yes' if apply_var else 'No',
            'description': f"Proportional Kelly: risk={params['risk']}, conf={params['confidence']}. {'With variation' if apply_var else 'No variation'}",
            'avg_bet_pct': avg_bet,
            'min_bet_pct': min_bet,
            'max_bet_pct': max_bet,
            **metrics
        }
        
        save_results_to_csv(result)
        results.append(result)
        
        print(f"  Profit: {metrics['avg_profit_pct']:.2f}%, DD>50%: {metrics['drawdown_50_pct']:.2f}%")

# #18 Target-Based
print("\n" + "="*70)
print("#18 TARGET-BASED STRATEGY")
print("="*70)

for params in target_based_params:
    for apply_var in [False, True]:
        var_suffix = "_with_variation" if apply_var else ""
        name = f"target_based_t{params['target_bankroll_percent']}_a{params['aggressive_pct']}_c{params['conservative_pct']}{var_suffix}"
        
        print(f"\n{name}: {params}, var={apply_var}")
        
        bankroll, bet_history, min_bet, max_bet, avg_bet = target_based_strategy_with_real_odds(
            outcomes, odds_array, **params, apply_variation=apply_var
        )
        
        metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
        
        result = {
            'strategy_name': name,
            'base_strategy': 'target_based',
            'strategy_params': params,
            'with_variation': 'Yes' if apply_var else 'No',
            'description': f"Target-Based: target={params['target_bankroll_percent']}%, agg={params['aggressive_pct']}, cons={params['conservative_pct']}. {'With variation' if apply_var else 'No variation'}",
            'avg_bet_pct': avg_bet,
            'min_bet_pct': min_bet,
            'max_bet_pct': max_bet,
            **metrics
        }
        
        save_results_to_csv(result)
        results.append(result)
        
        print(f"  Profit: {metrics['avg_profit_pct']:.2f}%, DD>50%: {metrics['drawdown_50_pct']:.2f}%")

print("\n" + "="*70)
print("СВОДКА")
print("="*70)
print(f"\n✅ Добавлено {len(results)} новых стратегий в results.csv")
print(f"Теперь всего: ~{86 + len(results)} стратегий")

# Краткая сводка
for base_strat in ['fixed_fraction', 'proportional_kelly', 'target_based']:
    strat_results = [r for r in results if r['base_strategy'] == base_strat]
    avg_profit = np.mean([r['avg_profit_pct'] for r in strat_results])
    avg_dd50 = np.mean([r['drawdown_50_pct'] for r in strat_results])
    print(f"\n{base_strat}: avg profit={avg_profit:.1f}%, avg DD>50%={avg_dd50:.2f}%")
