"""
Скрипт для быстрого добавления новой стратегии в results.csv
"""
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import run_strategy_with_real_odds, save_results_to_csv

print("Загрузка симуляций с реальными коэффициентами...")
outcomes, odds_array = load_real_odds_outcomes()

print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Средний коэффициент: {odds_array.mean():.2f}\n")

# НАСТРОЙТЕ ЗДЕСЬ:
BET_SIZE = 1.5  # процент от текущего банка

print("="*70)
print(f"ДОБАВЛЕНИЕ СТРАТЕГИИ: dynamic_percentage_{BET_SIZE}%")
print("="*70)

# Без вариации
print("\n1. БЕЗ вариации размера ставок:")
result = run_strategy_with_real_odds(
    'dynamic_percentage', 
    outcomes, 
    odds_array, 
    BET_SIZE, 
    apply_variation=False
)
save_results_to_csv(result)

# С вариацией (30%-115%)
print("\n2. С вариацией размера ставок (30%-115%):")
result = run_strategy_with_real_odds(
    'dynamic_percentage',
    outcomes,
    odds_array,
    BET_SIZE,
    apply_variation=True
)
save_results_to_csv(result)

print("\n✅ Стратегии добавлены в results.csv!")
