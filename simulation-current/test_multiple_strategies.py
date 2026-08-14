from generate_simulations import load_outcomes
from run_strategies import run_strategy, save_results_to_csv, save_results_to_markdown
from config import ODDS, INITIAL_BANKROLL

print("Загрузка исходов ставок...")
outcomes = load_outcomes()

print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
print(f"Начальный банкролл: {INITIAL_BANKROLL}")
print(f"Коэффициент: {ODDS}\n")

results = []

strategies_to_test = [
    ('flat', 1.0),
    ('flat', 2.0),
    ('flat', 5.0),
    ('dynamic_percentage', 1.0),
    ('dynamic_percentage', 2.0),
    ('dynamic_percentage', 5.0),
]

for strategy_name, bet_size in strategies_to_test:
    result = run_strategy(strategy_name, outcomes, bet_size_pct=bet_size)
    results.append(result)
    save_results_to_csv(result)

save_results_to_markdown(results)

print("\n" + "="*70)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("="*70)
print(f"{'Стратегия':<20} {'Bet%':<6} {'ROI%':<7} {'AvgProfit%':<12} {'Bankrupt%':<11} {'DD>50%':<8}")
print("-"*70)
for r in results:
    print(f"{r['strategy_name']:<20} {r['params']['bet_size_pct']:<6.1f} "
          f"{r['avg_roi_from_turnover']:<7.2f} {r['avg_profit_pct']:<12.2f} "
          f"{r['bankrupt_pct']:<11.2f} {r['drawdown_50_pct']:<8.2f}")
