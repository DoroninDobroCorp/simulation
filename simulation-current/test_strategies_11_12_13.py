"""
Тестирование стратегий #11, #12, #13 в 18 вариациях.

#11: Linear Scaled Strategy
#12: Linear ROI-Odds Strategy  
#13: Adaptive Constant Profit Strategy

По 3 набора параметров × 2 режима (с/без вариации) = 6 тестов каждая
Итого: 3 стратегии × 6 вариаций = 18 новых записей
"""

from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import run_strategy_with_real_odds, save_results_to_csv

print("="*70)
print("ТЕСТИРОВАНИЕ СТРАТЕГИЙ #11, #12, #13")
print("="*70)

print("\nЗагрузка симуляций с реальными коэффициентами...")
outcomes, odds_array = load_real_odds_outcomes()

print(f"Загружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# Параметры для тестирования
linear_scaled_params = [
    {'min_roi': 5.0, 'max_roi': 15.0, 'min_percent': 0.5, 'max_percent': 3.0},   # Консервативная
    {'min_roi': 3.0, 'max_roi': 20.0, 'min_percent': 1.0, 'max_percent': 7.0},   # Умеренная
    {'min_roi': 1.0, 'max_roi': 25.0, 'min_percent': 1.5, 'max_percent': 12.0},  # Агрессивная
]

linear_roi_odds_params = [
    {'base_roi': 7.0, 'base_percent': 1.0, 'odds_penalty_factor': 0.5, 'max_percent': 5.0, 'min_odds': 1.5, 'max_odds': 5.0},   # Консервативная
    {'base_roi': 5.0, 'base_percent': 1.5, 'odds_penalty_factor': 0.7, 'max_percent': 10.0, 'min_odds': 1.5, 'max_odds': 5.0},  # Умеренная
    {'base_roi': 3.0, 'base_percent': 2.0, 'odds_penalty_factor': 0.8, 'max_percent': 15.0, 'min_odds': 1.3, 'max_odds': 6.0},  # Агрессивная
]

adaptive_constant_profit_params = [
    {'min_roi': 5.0, 'max_roi': 15.0, 'min_target_pct': 0.5, 'max_target_pct': 1.5, 'max_bet_percent': 8.0},   # Консервативная
    {'min_roi': 3.0, 'max_roi': 20.0, 'min_target_pct': 1.0, 'max_target_pct': 3.0, 'max_bet_percent': 12.0},  # Умеренная
    {'min_roi': 1.0, 'max_roi': 25.0, 'min_target_pct': 1.5, 'max_target_pct': 5.0, 'max_bet_percent': 18.0},  # Агрессивная
]

strategies_to_test = [
    ('linear_scaled', '#11 Linear Scaled', linear_scaled_params),
    ('linear_roi_odds', '#12 Linear ROI-Odds', linear_roi_odds_params),
    ('adaptive_constant_profit', '#13 Adaptive Constant Profit', adaptive_constant_profit_params),
]

print("="*70)
print("ПАРАМЕТРЫ ТЕСТИРОВАНИЯ:")
print("="*70)

print("\n#11 Linear Scaled Strategy:")
for i, params in enumerate(linear_scaled_params, 1):
    print(f"  {i}. roi=[{params['min_roi']}-{params['max_roi']}], bet=[{params['min_percent']}-{params['max_percent']}]%")

print("\n#12 Linear ROI-Odds Strategy:")
for i, params in enumerate(linear_roi_odds_params, 1):
    print(f"  {i}. base_roi={params['base_roi']}, base%={params['base_percent']}, penalty={params['odds_penalty_factor']}, max={params['max_percent']}%")

print("\n#13 Adaptive Constant Profit Strategy:")
for i, params in enumerate(adaptive_constant_profit_params, 1):
    print(f"  {i}. roi=[{params['min_roi']}-{params['max_roi']}], target=[{params['min_target_pct']}-{params['max_target_pct']}]%, max_bet={params['max_bet_percent']}%")

print("\nКаждый набор тестируется:")
print("  - БЕЗ вариации (идеальные условия)")
print("  - С вариацией (реалистичные условия, 30%-115%)")
print(f"\nИтого: 3 стратегии × 3 параметра × 2 режима = 18 тестов")
print("="*70)

results = []
total_tests = 18
current_test = 0

for strategy_name, strategy_desc, params_list in strategies_to_test:
    print(f"\n{'='*70}")
    print(f"СТРАТЕГИЯ: {strategy_desc}")
    print(f"{'='*70}")
    
    for params in params_list:
        current_test += 1
        print(f"\n[{current_test}/{total_tests}] БЕЗ вариации: {params}")
        
        # БЕЗ вариации
        result = run_strategy_with_real_odds(
            strategy_name,
            outcomes,
            odds_array,
            apply_variation=False,
            **params
        )
        save_results_to_csv(result)
        results.append(result)
        
        current_test += 1
        print(f"\n[{current_test}/{total_tests}] С вариацией: {params}")
        
        # С вариацией
        result = run_strategy_with_real_odds(
            strategy_name,
            outcomes,
            odds_array,
            apply_variation=True,
            **params
        )
        save_results_to_csv(result)
        results.append(result)

print("\n" + "="*70)
print("ИТОГОВОЕ СРАВНЕНИЕ ПО СТРАТЕГИЯМ")
print("="*70)

for strategy_name, strategy_desc, _ in strategies_to_test:
    print(f"\n{strategy_desc}:")
    print(f"{'Вариант':<50} {'Var':<5} {'Profit%':<10} {'DD>50%':<8} {'Bankrupt%':<10}")
    print("-"*70)
    
    strategy_results = [r for r in results if r['base_strategy'] == strategy_name]
    for r in strategy_results:
        var = 'Yes' if r['with_variation'] else 'No'
        variant_name = r['strategy_name'].replace(f"{strategy_name}_", "").replace("_with_variation", "")
        if len(variant_name) > 49:
            variant_name = variant_name[:46] + "..."
        print(f"{variant_name:<50} {var:<5} "
              f"{r['avg_profit_pct']:<10.2f} {r['drawdown_50_pct']:<8.2f} {r['bankrupt_pct']:<10.2f}")

print("\n✅ Готово! 18 новых записей добавлены в results.csv")
print(f"Всего строк в results.csv: {68 + 18 + 1} = 87 (с учетом заголовка)")
