"""
Добавление оптимизированных стратегий из отчета с ТОЧНЫМИ параметрами.
Каждая по 2 раза: с вариацией и без.
"""

import numpy as np
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    linear_roi_odds_strategy_with_real_odds,
    linear_roi_strategy_with_real_odds,
    adaptive_constant_profit_strategy_with_real_odds,
    sqrt_roi_strategy_with_real_odds,
    adaptive_strategy_with_real_odds,
    dynamic_kelly_strategy_with_real_odds,
    linear_scaled_strategy_with_real_odds,
    calculate_metrics_with_odds,
    save_results_to_csv
)

outcomes, odds_array = load_real_odds_outcomes()
print(f"Загружено {outcomes.shape[0]} симуляций\n")

# ТОЧНЫЕ параметры из отчета (отмеченные ! в отчете)
optimized_strategies = [
    # Conservative - Linear ROI-Odds (ROI: 111.50%, DD: 20.89%)
    ('linear_roi_odds', 'CONSERVATIVE', linear_roi_odds_strategy_with_real_odds, {
        'base_roi': 9.263, 
        'base_percent': 1.669, 
        'max_percent': 8.464, 
        'odds_penalty_factor': 0.7,
        'min_odds': 1.41, 
        'max_odds': 3.65
    }),
    
    # Cautious - Linear ROI-Odds (ROI: 208.90%, DD: 31.62%, DD>50%: 2%)
    ('linear_roi_odds', 'CAUTIOUS', linear_roi_odds_strategy_with_real_odds, {
        'base_roi': 11.003, 
        'base_percent': 2.84, 
        'max_percent': 6.789, 
        'odds_penalty_factor': 0.7,
        'min_odds': 1.552, 
        'max_odds': 4.034
    }),
    
    # Balanced - Linear ROI-Odds (ROI: 281.56%, DD: 36.28%, DD>50%: 6.8%)
    ('linear_roi_odds', 'BALANCED', linear_roi_odds_strategy_with_real_odds, {
        'base_roi': 5.649, 
        'base_percent': 1.533, 
        'max_percent': 15.089, 
        'odds_penalty_factor': 0.7,
        'min_odds': 1.686, 
        'max_odds': 4.627
    }),
    
    # Conservative - Linear ROI (ROI: 105.69%)
    ('linear_roi', 'CONSERVATIVE', linear_roi_strategy_with_real_odds, {
        'base_roi': 5.164, 
        'base_percent': 0.614, 
        'max_percent': 27.211
    }),
    
    # Conservative - Adaptive Constant Profit (ROI: 99.23%)
    ('adaptive_constant_profit', 'CONSERVATIVE', adaptive_constant_profit_strategy_with_real_odds, {
        'min_roi': 5.052, 
        'max_roi': 30.648, 
        'min_target_pct': 0.276, 
        'max_target_pct': 8.094,
        'max_bet_percent': 15.0
    }),
    
    # Cautious - Adaptive Constant Profit (ROI: 218.30%, DD>50%: 3.2%)
    ('adaptive_constant_profit', 'CAUTIOUS', adaptive_constant_profit_strategy_with_real_odds, {
        'min_roi': 0.117, 
        'max_roi': 31.685, 
        'min_target_pct': 0.203, 
        'max_target_pct': 7.41,
        'max_bet_percent': 15.0
    }),
    
    # Balanced - Adaptive Constant Profit (ROI: 292.31%, DD>50%: 11%)
    ('adaptive_constant_profit', 'BALANCED', adaptive_constant_profit_strategy_with_real_odds, {
        'min_roi': 2.474, 
        'max_roi': 34.033, 
        'min_target_pct': 1.443, 
        'max_target_pct': 6.059,
        'max_bet_percent': 15.0
    }),
    
    # Balanced - Sqrt ROI (ROI: 280.23%, DD>50%: 11%)
    ('sqrt_roi', 'BALANCED', sqrt_roi_strategy_with_real_odds, {
        'base_roi': 5.703, 
        'base_percent': 1.439, 
        'max_percent': 28.148
    }),
    
    # Risky - Adaptive (ROI: 2602.50%, DD>50%: 65.4%) - но ограничим
    ('adaptive', 'RISKY_LIMITED', adaptive_strategy_with_real_odds, {
        'base_percent': 2.476, 
        'max_percent': 16.839,
        'min_roi': 4.933, 
        'max_roi': 14.367
    }),
    
    # Cautious - Dynamic Kelly (ROI: 211.58%, DD>50%: 3.4%)
    ('dynamic_kelly', 'CAUTIOUS', dynamic_kelly_strategy_with_real_odds, {
        'risk': 4.264, 
        'min_fraction': 0.166, 
        'max_fraction': 0.552, 
        'min_roi': 5.369, 
        'max_roi': 24.12
    }),
]

print("="*80)
print("ДОБАВЛЕНИЕ ОПТИМИЗИРОВАННЫХ СТРАТЕГИЙ ИЗ ОТЧЕТА")
print("="*80)
print(f"Будет добавлено: {len(optimized_strategies)} × 2 = {len(optimized_strategies)*2} вариантов\n")

count = 0
total = len(optimized_strategies) * 2
added = 0

for strategy_name, profile, strategy_func, params in optimized_strategies:
    for apply_var in [False, True]:
        count += 1
        var_str = "Yes" if apply_var else "No"
        var_suffix = "_var" if apply_var else ""
        
        # Генерируем имя
        params_short = "_".join([f"{k[:3]}{v:.1f}" for k, v in list(params.items())[:3]])
        name = f"{strategy_name}_OPT_{profile}{var_suffix}"
        
        print(f"[{count}/{total}] {name[:65]:<65}", end=' ', flush=True)
        
        try:
            bankroll, bet_history, min_bet, max_bet, avg_bet = strategy_func(
                outcomes, odds_array, **params, apply_variation=apply_var
            )
            
            metrics = calculate_metrics_with_odds(bankroll, bet_history, odds_array)
            
            # Проверяем на адекватность
            if max_bet > 50 or metrics['avg_profit_pct'] < 5:
                print(f"⚠️  Skip (max={max_bet:.0f}%, profit={metrics['avg_profit_pct']:.0f}%)")
                continue
            
            result = {
                'strategy_name': name,
                'base_strategy': strategy_name,
                'strategy_params': params,
                'with_variation': var_str,
                'description': f"OPTIMIZED {profile}: {strategy_name}",
                'avg_bet_pct': avg_bet,
                'min_bet_pct': min_bet,
                'max_bet_pct': max_bet,
                **metrics
            }
            
            save_results_to_csv(result)
            added += 1
            print(f"✅ +{metrics['avg_profit_pct']:.0f}% DD50:{metrics['drawdown_50_pct']:.1f}%")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")

print("\n" + "="*80)
print(f"✅ Добавлено {added}/{total} оптимизированных вариантов")
print("="*80)
