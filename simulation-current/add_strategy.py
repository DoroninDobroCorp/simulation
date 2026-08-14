from generate_simulations import load_outcomes
from run_strategies import run_strategy, save_results_to_csv
from config import ODDS, INITIAL_BANKROLL

print("Загрузка исходов ставок...")
outcomes = load_outcomes()

print(f"\nДобавление новой стратегии в results.csv")

result = run_strategy('dynamic_percentage', outcomes, bet_size_pct=1.5)
save_results_to_csv(result)

print("\n✅ Стратегия добавлена!")
