"""
Тестирование стратегий #2, #3, #4 в 18 вариациях.

#2: Linear ROI Strategy
#3: Square Root ROI Strategy  
#4: Logarithmic ROI Strategy

По 3 набора параметров × 2 режима (с/без вариации) = 6 тестов каждая
Итого: 3 стратегии × 6 вариаций = 18 новых записей
"""

from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import run_strategy_with_real_odds, save_results_to_csv

print("="*70)
print("ТЕСТИРОВАНИЕ СТРАТЕГИЙ #2, #3, #4")
print("="*70)

print("\nЗагрузка симуляций с реальными коэффициентами...")
outcomes, odds_array = load_real_odds_outcomes()

print(f"Загружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# 3 набора параметров для каждой ROI-based стратегии
# Параметры подобраны для разных профилей риска
roi_params = [
    {'base_roi': 5.0, 'base_percent': 1.0, 'max_percent': 5.0},   # Консервативная
    {'base_roi': 7.0, 'base_percent': 1.5, 'max_percent': 10.0},  # Умеренная
    {'base_roi': 10.0, 'base_percent': 2.0, 'max_percent': 15.0}, # Агрессивная
]

strategies_to_test = [
    ('linear_roi', '#2 Linear ROI'),
    ('sqrt_roi', '#3 Square Root ROI'),
    ('log_roi', '#4 Logarithmic ROI'),
]

print("="*70)
print("ПАРАМЕТРЫ ТЕСТИРОВАНИЯ:")
print("="*70)
for strategy_name, strategy_desc in strategies_to_test:
    print(f"\n{strategy_desc}:")
    for i, params in enumerate(roi_params, 1):
        print(f"  {i}. base_roi={params['base_roi']}, base_pct={params['base_percent']}, max={params['max_percent']}%")

print("\nКаждый набор тестируется:")
print("  - БЕЗ вариации (идеальные условия)")
print("  - С вариацией (реалистичные условия, 30%-115%)")
print(f"\nИтого: 3 стратегии × 3 параметра × 2 режима = 18 тестов")
print("="*70)

results = []
total_tests = len(strategies_to_test) * len(roi_params) * 2
current_test = 0

for strategy_name, strategy_desc in strategies_to_test:
    print(f"\n{'='*70}")
    print(f"СТРАТЕГИЯ: {strategy_desc}")
    print(f"{'='*70}")
    
    for params in roi_params:
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

for strategy_name, strategy_desc in strategies_to_test:
    print(f"\n{strategy_desc}:")
    print(f"{'Параметры':<40} {'Var':<5} {'ROI%':<8} {'Profit%':<10} {'Bankrupt%':<11} {'DD>50%':<8}")
    print("-"*70)
    
    strategy_results = [r for r in results if r['base_strategy'] == strategy_name]
    for r in strategy_results:
        var = 'Yes' if r['with_variation'] else 'No'
        params = r['strategy_params']
        param_str = f"br{params['base_roi']}_bp{params['base_percent']}_max{params['max_percent']}"
        print(f"{param_str:<40} {var:<5} "
              f"{r['avg_roi_from_turnover']:<8.2f} {r['avg_profit_pct']:<10.2f} "
              f"{r['bankrupt_pct']:<11.2f} {r['drawdown_50_pct']:<8.2f}")

print("\n✅ Готово! 18 новых записей добавлены в results.csv")
print(f"Всего строк в results.csv: {14 + 18 + 1} (с учетом заголовка)")
