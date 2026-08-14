"""
Тестирование стратегий #8, #9, #10 в 18 вариациях.

#8: Dynamic Kelly Strategy
#9: Exponential ROI Strategy  
#10: Hybrid Strategy

По 3 набора параметров × 2 режима (с/без вариации) = 6 тестов каждая
Итого: 3 стратегии × 6 вариаций = 18 новых записей
"""

from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import run_strategy_with_real_odds, save_results_to_csv

print("="*70)
print("ТЕСТИРОВАНИЕ СТРАТЕГИЙ #8, #9, #10")
print("="*70)

print("\nЗагрузка симуляций с реальными коэффициентами...")
outcomes, odds_array = load_real_odds_outcomes()

print(f"Загружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# Параметры для тестирования
dynamic_kelly_params = [
    {'risk': 2.5, 'min_fraction': 0.05, 'max_fraction': 0.25, 'min_roi': 3.0, 'max_roi': 15.0},  # Консервативная
    {'risk': 2.0, 'min_fraction': 0.1, 'max_fraction': 0.5, 'min_roi': 2.0, 'max_roi': 20.0},    # Умеренная
    {'risk': 1.5, 'min_fraction': 0.2, 'max_fraction': 0.8, 'min_roi': 1.0, 'max_roi': 25.0},    # Агрессивная
]

exponential_roi_params = [
    {'base_roi': 7.0, 'base_percent': 0.5, 'factor': 0.05, 'max_percent': 5.0},   # Консервативная
    {'base_roi': 5.0, 'base_percent': 1.0, 'factor': 0.1, 'max_percent': 10.0},   # Умеренная
    {'base_roi': 3.0, 'base_percent': 1.5, 'factor': 0.15, 'max_percent': 15.0},  # Агрессивная
]

hybrid_params = [
    {'base_percent': 1.0, 'max_percent': 5.0, 'min_roi': 3.0, 'max_roi': 15.0, 'min_odds': 1.5, 'max_odds': 5.0, 
     'roi_weight': 0.8, 'odds_weight': 0.2},  # Консервативная - больше веса ROI
    {'base_percent': 1.5, 'max_percent': 10.0, 'min_roi': 2.0, 'max_roi': 20.0, 'min_odds': 1.3, 'max_odds': 6.0,
     'roi_weight': 0.7, 'odds_weight': 0.3},  # Умеренная - сбалансированные веса
    {'base_percent': 2.0, 'max_percent': 15.0, 'min_roi': 1.0, 'max_roi': 25.0, 'min_odds': 1.2, 'max_odds': 8.0,
     'roi_weight': 0.6, 'odds_weight': 0.4},  # Агрессивная - больше учет odds
]

strategies_to_test = [
    ('dynamic_kelly', '#8 Dynamic Kelly', dynamic_kelly_params),
    ('exponential_roi', '#9 Exponential ROI', exponential_roi_params),
    ('hybrid', '#10 Hybrid', hybrid_params),
]

print("="*70)
print("ПАРАМЕТРЫ ТЕСТИРОВАНИЯ:")
print("="*70)

print("\n#8 Dynamic Kelly Strategy:")
for i, params in enumerate(dynamic_kelly_params, 1):
    print(f"  {i}. risk={params['risk']}, frac=[{params['min_fraction']}-{params['max_fraction']}], roi=[{params['min_roi']}-{params['max_roi']}]")

print("\n#9 Exponential ROI Strategy:")
for i, params in enumerate(exponential_roi_params, 1):
    print(f"  {i}. base_roi={params['base_roi']}, base_pct={params['base_percent']}, factor={params['factor']}, max={params['max_percent']}%")

print("\n#10 Hybrid Strategy:")
for i, params in enumerate(hybrid_params, 1):
    print(f"  {i}. base={params['base_percent']}%, max={params['max_percent']}%, weights={params['roi_weight']}/{params['odds_weight']}")

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
print(f"Всего строк в results.csv: {50 + 18 + 1} = 69 (с учетом заголовка)")
