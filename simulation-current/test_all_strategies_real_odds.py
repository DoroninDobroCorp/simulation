from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import run_strategy_with_real_odds, save_results_to_csv

print("Загрузка симуляций с реальными коэффициентами...")
outcomes, odds_array = load_real_odds_outcomes()

print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Диапазон коэффициентов: {odds_array.min():.2f} - {odds_array.max():.2f}")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# Стратегии для тестирования
bet_sizes = [1.0, 1.5, 2.0, 5.0]

print("="*70)
print("ПРОГОН ВСЕХ СТРАТЕГИЙ С РЕАЛЬНЫМИ КОЭФФИЦИЕНТАМИ")
print("="*70)

for bet_size in bet_sizes:
    # Без вариации
    result = run_strategy_with_real_odds(
        'dynamic_percentage', 
        outcomes, 
        odds_array, 
        bet_size, 
        apply_variation=False
    )
    save_results_to_csv(result)
    
    # С вариацией (30%-115%)
    result = run_strategy_with_real_odds(
        'dynamic_percentage',
        outcomes,
        odds_array,
        bet_size,
        apply_variation=True
    )
    save_results_to_csv(result)

print("\n" + "="*70)
print("ИТОГО")
print("="*70)
print(f"Протестировано {len(bet_sizes) * 2} стратегий")
print("Результаты сохранены в results.csv")
print("\n✅ Готово!")
