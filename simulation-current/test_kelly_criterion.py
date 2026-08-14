"""
Тестирование Kelly Criterion стратегии в 6 вариациях.

3 набора параметров × 2 режима (с/без вариации) = 6 тестов
"""

from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import run_strategy_with_real_odds, save_results_to_csv

print("="*70)
print("ТЕСТИРОВАНИЕ KELLY CRITERION СТРАТЕГИИ")
print("="*70)

print("\nЗагрузка симуляций с реальными коэффициентами...")
outcomes, odds_array = load_real_odds_outcomes()

print(f"Загружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Диапазон коэффициентов: {odds_array.min():.2f} - {odds_array.max():.2f}")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# 3 набора параметров Kelly Criterion
# Из документа: risk=2.0 по умолчанию, kelly_fraction обычно 0.25-0.5
kelly_params = [
    {'risk': 2.0, 'kelly_fraction': 0.25},  # Консервативная
    {'risk': 2.0, 'kelly_fraction': 0.50},  # Умеренная
    {'risk': 1.5, 'kelly_fraction': 0.50},  # Более агрессивная
]

print("="*70)
print("ПАРАМЕТРЫ ТЕСТИРОВАНИЯ:")
print("="*70)
for i, params in enumerate(kelly_params, 1):
    print(f"{i}. risk={params['risk']}, kelly_fraction={params['kelly_fraction']}")
print("\nКаждый набор тестируется:")
print("  - БЕЗ вариации (идеальные условия)")
print("  - С вариацией (реалистичные условия, 30%-115%)")
print("="*70)

results = []

for params in kelly_params:
    # БЕЗ вариации
    result = run_strategy_with_real_odds(
        'kelly_criterion',
        outcomes,
        odds_array,
        apply_variation=False,
        **params
    )
    save_results_to_csv(result)
    results.append(result)
    
    # С вариацией
    result = run_strategy_with_real_odds(
        'kelly_criterion',
        outcomes,
        odds_array,
        apply_variation=True,
        **params
    )
    save_results_to_csv(result)
    results.append(result)

print("\n" + "="*70)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("="*70)
print(f"{'Стратегия':<40} {'Var':<5} {'ROI%':<8} {'Profit%':<10} {'Bankrupt%':<11} {'DD>50%':<8}")
print("-"*70)
for r in results:
    var = 'Yes' if r['with_variation'] else 'No'
    print(f"{r['strategy_name']:<40} {var:<5} "
          f"{r['avg_roi_from_turnover']:<8.2f} {r['avg_profit_pct']:<10.2f} "
          f"{r['bankrupt_pct']:<11.2f} {r['drawdown_50_pct']:<8.2f}")

print("\n✅ Готово! Результаты добавлены в results.csv")
print("Используйте compare_with_without_variation.py для детального анализа")
