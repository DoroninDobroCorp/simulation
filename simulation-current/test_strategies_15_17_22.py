"""
Тестирование стратегий #15, #17, #22.
- #15 Anti-Martingale
- #17 Volatility-Adjusted
- #22 Win/Loss Streak Aware
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    anti_martingale_strategy_with_real_odds,
    volatility_adjusted_strategy_with_real_odds,
    streak_aware_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

print("="*70)
print("ТЕСТИРОВАНИЕ СТРАТЕГИЙ #15, #17, #22")
print("="*70)

outcomes, odds_array = load_real_odds_outcomes()
print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# #15 Anti-Martingale: 3 набора параметров
anti_martingale_params = [
    {'base_percent': 1.0, 'multiplier': 1.5, 'max_percent': 5.0, 'max_streak': 3},   # Консервативная
    {'base_percent': 1.5, 'multiplier': 1.7, 'max_percent': 10.0, 'max_streak': 4},  # Умеренная
    {'base_percent': 2.0, 'multiplier': 2.0, 'max_percent': 15.0, 'max_streak': 5},  # Агрессивная
]

# #17 Volatility-Adjusted: 3 набора параметров
volatility_params = [
    {'base_percent': 2.0, 'lookback': 30, 'volatility_factor': 0.5},   # Слабая реакция
    {'base_percent': 2.5, 'lookback': 50, 'volatility_factor': 1.0},   # Средняя реакция
    {'base_percent': 3.0, 'lookback': 70, 'volatility_factor': 1.5},   # Сильная реакция
]

# #22 Streak Aware: 3 набора параметров
streak_params = [
    {'base_percent': 2.0, 'win_streak_multiplier': 1.2, 'loss_streak_divider': 1.3, 'max_multiplier': 2.0},  # Консервативная
    {'base_percent': 2.5, 'win_streak_multiplier': 1.3, 'loss_streak_divider': 1.4, 'max_multiplier': 3.0},  # Умеренная
    {'base_percent': 3.0, 'win_streak_multiplier': 1.5, 'loss_streak_divider': 1.5, 'max_multiplier': 4.0},  # Агрессивная
]

results = []
total = 18
count = 0

print("\n" + "="*70)
print("#15 ANTI-MARTINGALE")
print("="*70)

for params in anti_martingale_params:
    for apply_var in [False, True]:
        count += 1
        var_suffix = "_with_variation" if apply_var else ""
        name = f"anti_martingale_b{params['base_percent']}_m{params['multiplier']}_s{params['max_streak']}{var_suffix}"
        
        print(f"\n[{count}/{total}] {name}")
        
        bankroll, bet_history, min_bet, max_bet, avg_bet = anti_martingale_strategy_with_real_odds(
            outcomes, odds_array, **params, apply_variation=apply_var
        )
        
        metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
        
        result = {
            'strategy_name': name,
            'base_strategy': 'anti_martingale',
            'strategy_params': params,
            'with_variation': 'Yes' if apply_var else 'No',
            'description': f"Anti-Martingale: base={params['base_percent']}%, mult={params['multiplier']}, max_streak={params['max_streak']}. {'With var' if apply_var else 'No var'}",
            'avg_bet_pct': avg_bet,
            'min_bet_pct': min_bet,
            'max_bet_pct': max_bet,
            **metrics
        }
        
        save_results_to_csv(result)
        results.append(result)
        print(f"  ✅ Profit: {metrics['avg_profit_pct']:.1f}%, DD>50%: {metrics['drawdown_50_pct']:.2f}%, Bankrupt: {metrics['bankrupt_pct']:.2f}%")

print("\n" + "="*70)
print("#17 VOLATILITY-ADJUSTED")
print("="*70)

for params in volatility_params:
    for apply_var in [False, True]:
        count += 1
        var_suffix = "_with_variation" if apply_var else ""
        name = f"volatility_adjusted_b{params['base_percent']}_lb{params['lookback']}_vf{params['volatility_factor']}{var_suffix}"
        
        print(f"\n[{count}/{total}] {name}")
        
        bankroll, bet_history, min_bet, max_bet, avg_bet = volatility_adjusted_strategy_with_real_odds(
            outcomes, odds_array, **params, apply_variation=apply_var
        )
        
        metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
        
        result = {
            'strategy_name': name,
            'base_strategy': 'volatility_adjusted',
            'strategy_params': params,
            'with_variation': 'Yes' if apply_var else 'No',
            'description': f"Volatility-Adjusted: base={params['base_percent']}%, lookback={params['lookback']}, vol_factor={params['volatility_factor']}. {'With var' if apply_var else 'No var'}",
            'avg_bet_pct': avg_bet,
            'min_bet_pct': min_bet,
            'max_bet_pct': max_bet,
            **metrics
        }
        
        save_results_to_csv(result)
        results.append(result)
        print(f"  ✅ Profit: {metrics['avg_profit_pct']:.1f}%, DD>50%: {metrics['drawdown_50_pct']:.2f}%, Bankrupt: {metrics['bankrupt_pct']:.2f}%")

print("\n" + "="*70)
print("#22 STREAK AWARE")
print("="*70)

for params in streak_params:
    for apply_var in [False, True]:
        count += 1
        var_suffix = "_with_variation" if apply_var else ""
        name = f"streak_aware_b{params['base_percent']}_wm{params['win_streak_multiplier']}_ld{params['loss_streak_divider']}{var_suffix}"
        
        print(f"\n[{count}/{total}] {name}")
        
        bankroll, bet_history, min_bet, max_bet, avg_bet = streak_aware_strategy_with_real_odds(
            outcomes, odds_array, **params, apply_variation=apply_var
        )
        
        metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
        
        result = {
            'strategy_name': name,
            'base_strategy': 'streak_aware',
            'strategy_params': params,
            'with_variation': 'Yes' if apply_var else 'No',
            'description': f"Streak Aware: base={params['base_percent']}%, win_mult={params['win_streak_multiplier']}, loss_div={params['loss_streak_divider']}. {'With var' if apply_var else 'No var'}",
            'avg_bet_pct': avg_bet,
            'min_bet_pct': min_bet,
            'max_bet_pct': max_bet,
            **metrics
        }
        
        save_results_to_csv(result)
        results.append(result)
        print(f"  ✅ Profit: {metrics['avg_profit_pct']:.1f}%, DD>50%: {metrics['drawdown_50_pct']:.2f}%, Bankrupt: {metrics['bankrupt_pct']:.2f}%")

print("\n" + "="*70)
print("ИТОГИ")
print("="*70)
print(f"\n✅ Добавлено 18 новых записей")
print(f"Теперь всего: ~{104 + 18} = 122 стратегий")

for base in ['anti_martingale', 'volatility_adjusted', 'streak_aware']:
    strat_results = [r for r in results if r['base_strategy'] == base]
    avg_profit = np.mean([r['avg_profit_pct'] for r in strat_results])
    avg_dd50 = np.mean([r['drawdown_50_pct'] for r in strat_results])
    print(f"\n{base}: avg_profit={avg_profit:.1f}%, avg_dd50={avg_dd50:.2f}%")
