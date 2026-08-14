from generate_simulations import load_outcomes
from run_strategies import run_strategy, save_results_to_csv
from config import ODDS, INITIAL_BANKROLL

print("Загрузка исходов...")
outcomes = load_outcomes()

print(f"\nТестирование ROI для разных стратегий")
print(f"Ожидаемый ROI с оборота: ~7%")
print(f"Начальный банкролл: {INITIAL_BANKROLL}")
print(f"Коэффициент: {ODDS}\n")

results = []

print("=" * 70)
print("FLAT стратегия - фиксированный % от начального банка")
print("=" * 70)
result = run_strategy('flat', outcomes, bet_size_pct=2.0)
results.append(result)
save_results_to_csv(result)

print("\n" + "=" * 70)
print("DYNAMIC стратегия - фиксированный % от текущего банка")
print("=" * 70)
result = run_strategy('dynamic_percentage', outcomes, bet_size_pct=2.0)
results.append(result)
save_results_to_csv(result)

print("\n" + "=" * 70)
print("СРАВНЕНИЕ ROI")
print("=" * 70)
for r in results:
    print(f"{r['strategy_name']:20s} | ROI: {r['avg_roi_from_turnover']:>6.2f}% | Avg Profit: {r['avg_profit_pct']:>8.2f}%")
