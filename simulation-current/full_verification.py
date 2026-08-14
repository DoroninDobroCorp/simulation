"""
ПОЛНАЯ ПРОВЕРКА всех аспектов рекомендованных стратегий.
"""

import numpy as np
import pandas as pd
from generate_real_odds_simulations import load_real_odds_outcomes
from run_strategies_real_odds import (
    adaptive_strategy_with_real_odds,
    calculate_metrics_with_odds
)
from config import INITIAL_BANKROLL

outcomes, odds_array = load_real_odds_outcomes()

print("="*90)
print("🔍 ПОЛНАЯ ПРОВЕРКА РЕКОМЕНДОВАННЫХ СТРАТЕГИЙ")
print("="*90)

# === ТЕСТ 1: ADAPTIVE_OPT_RISKY_LIMITED ===
print("\n" + "="*90)
print("ТЕСТ 1: ADAPTIVE_OPT_RISKY_LIMITED (безопасная)")
print("="*90)

params1 = {
    'base_percent': 2.476,
    'max_percent': 16.839,
    'min_roi': 4.933,
    'max_roi': 14.367,
    'min_odds': 1.5,
    'max_odds': 5.0,
    'apply_variation': False
}

print("\n📝 Параметры:")
for k, v in params1.items():
    if k != 'apply_variation':
        print(f"   {k}: {v}")

br1, bh1, min_bet1, max_bet1, avg_bet1 = adaptive_strategy_with_real_odds(
    outcomes, odds_array, **params1
)
m1 = calculate_metrics_with_odds(br1, bh1, odds_array)

print("\n📊 Результаты из CSV:")
df_cons = pd.read_csv('results_conservative_DD50.csv')
csv_row = df_cons[df_cons['strategy'] == 'adaptive_OPT_RISKY_LIMITED'].iloc[0]
print(f"   Profit: {csv_row['avg_profit_%']:.1f}%")
print(f"   DD>50%: {csv_row['dd>50_%']:.2f}%")
print(f"   Max bet: {csv_row['max_bet_%']:.2f}%")

print("\n📊 Результаты пересчета:")
print(f"   Profit: {m1['avg_profit_pct']:.1f}%")
print(f"   DD>50%: {m1['drawdown_50_pct']:.2f}%")
print(f"   Max bet: {max_bet1:.2f}%")

if abs(csv_row['avg_profit_%'] - m1['avg_profit_pct']) < 0.1:
    print("✅ Результаты совпадают!")
else:
    print(f"❌ НЕСОВПАДЕНИЕ! Разница: {abs(csv_row['avg_profit_%'] - m1['avg_profit_pct']):.2f}%")

# Детальная проверка логики
print("\n🔬 ПРОВЕРКА ЛОГИКИ (первые 20 ставок, симуляция #0):")
print(f"{'#':<4} {'Bank':<8} {'Peak':<8} {'DD%':<6} {'Modifier':<10} {'Bet%':<6} {'Result':<6}")
print("-"*70)

roi_pct = 7.0
for i in range(20):
    current_bank = br1[0, i]
    peak_bank = np.max(br1[0, :i+1])
    odds = odds_array[i]
    
    # Расчет базовой ставки
    norm_roi = np.clip((roi_pct - params1['min_roi']) / (params1['max_roi'] - params1['min_roi']), 0, 1)
    norm_odds = np.clip((odds - params1['min_odds']) / (params1['max_odds'] - params1['min_odds']), 0, 1)
    combined = np.sqrt(norm_roi) * (1 - 0.5 * norm_odds)
    base_bet_pct = params1['base_percent'] * combined
    
    # Адаптивный модификатор
    dd_from_peak = (peak_bank - current_bank) / peak_bank if peak_bank > 0 else 0
    modifier = 1.0
    if dd_from_peak > 0.20:
        modifier = 0.75
    if dd_from_peak > 0.40:
        modifier = 0.5
    
    final_bet_pct = base_bet_pct * modifier
    bet_amount = bh1[0, i]
    actual_pct = (bet_amount / current_bank * 100) if current_bank > 0 else 0
    
    result = "WIN" if outcomes[0, i] else "LOSS"
    
    # Проверка что модификатор работает
    modifier_str = f"{modifier:.2f}"
    if dd_from_peak > 0.20:
        modifier_str += " ✓DD>20%"
    
    print(f"{i+1:<4} {current_bank:>7.0f} {peak_bank:>7.0f} {dd_from_peak*100:>5.1f} {modifier_str:<10} {actual_pct:>5.2f} {result:<6}")

# Проверка банкротства
print("\n🔍 ПРОВЕРКА: Были ли случаи банкротства?")
bankrupt_sims = np.where(np.any(br1 <= 0, axis=1))[0]
print(f"   Банкротств: {len(bankrupt_sims)} из {br1.shape[0]} симуляций")

if len(bankrupt_sims) > 0:
    print(f"\n   Примеры банкротств (первые 3):")
    for sim_idx in bankrupt_sims[:3]:
        bankrupt_at = np.where(br1[sim_idx] <= 0)[0][0]
        print(f"   Симуляция #{sim_idx}: банкротство на ставке #{bankrupt_at}")
        print(f"      Банк перед: {br1[sim_idx, bankrupt_at]:.2f}")
        print(f"      Ставка: {bh1[sim_idx, bankrupt_at]:.2f}")
        
        # Проверяем что после банкротства ставки = 0
        if bankrupt_at < bh1.shape[1] - 1:
            next_bets = bh1[sim_idx, bankrupt_at+1:bankrupt_at+5]
            print(f"      Следующие 4 ставки: {next_bets}")
            if np.all(next_bets == 0):
                print(f"      ✅ После банкротства ставки = 0")
            else:
                print(f"      ❌ БАГ! После банкротства продолжает ставить!")

# Проверка что max_bet адекватный
print(f"\n🔍 ПРОВЕРКА max_bet: {max_bet1:.2f}%")
if max_bet1 > 20:
    print(f"   ❌ ПОДОЗРИТЕЛЬНО! Ожидаем до 16.839% (max_percent)")
    # Найдем где была эта ставка
    max_bet_sims, max_bet_idx = np.where(bh1 / br1[:, :-1] * 100 > 20)
    if len(max_bet_sims) > 0:
        sim = max_bet_sims[0]
        bet_i = max_bet_idx[0]
        print(f"   Найдено в симуляции #{sim}, ставка #{bet_i}")
        print(f"   Банк: {br1[sim, bet_i]:.6f}")
        print(f"   Ставка: {bh1[sim, bet_i]:.2f}")
        print(f"   Процент: {bh1[sim, bet_i] / br1[sim, bet_i] * 100:.1f}%")
else:
    print(f"   ✅ В норме (< 20%)")

# === ТЕСТ 2: ADAPTIVE_EXTREME_ROI16173 ===
print("\n" + "="*90)
print("ТЕСТ 2: ADAPTIVE_EXTREME_ROI16173 (агрессивная)")
print("="*90)

params2 = {
    'base_percent': 2.777,
    'max_percent': 19.534,
    'min_roi': 2.732,
    'max_roi': 25.191,
    'min_odds': 1.5,
    'max_odds': 5.0,
    'apply_variation': False
}

print("\n📝 Параметры:")
for k, v in params2.items():
    if k != 'apply_variation':
        print(f"   {k}: {v}")

br2, bh2, min_bet2, max_bet2, avg_bet2 = adaptive_strategy_with_real_odds(
    outcomes, odds_array, **params2
)
m2 = calculate_metrics_with_odds(br2, bh2, odds_array)

print("\n📊 Результаты из CSV:")
df_agg = pd.read_csv('results_aggressive_bankrupt10.csv')
csv_row2 = df_agg[df_agg['strategy'] == 'adaptive_EXTREME_ROI16173'].iloc[0]
print(f"   Profit: {csv_row2['avg_profit_%']:.1f}%")
print(f"   DD>50%: {csv_row2['dd>50_%']:.2f}%")
print(f"   Bankrupt: {csv_row2['bankrupt_%']:.2f}%")

print("\n📊 Результаты пересчета:")
print(f"   Profit: {m2['avg_profit_pct']:.1f}%")
print(f"   DD>50%: {m2['drawdown_50_pct']:.2f}%")
print(f"   Bankrupt: {m2['bankrupt_pct']:.2f}%")

if abs(csv_row2['avg_profit_%'] - m2['avg_profit_pct']) < 0.1:
    print("✅ Результаты совпадают!")
else:
    print(f"❌ НЕСОВПАДЕНИЕ! Разница: {abs(csv_row2['avg_profit_%'] - m2['avg_profit_pct']):.2f}%")

print("\n" + "="*90)
print("ИТОГОВАЯ ПРОВЕРКА:")
print("="*90)

# Сравнение стратегий
print(f"\n1. БЕЗОПАСНАЯ (RISKY_LIMITED):")
print(f"   Profit: +{m1['avg_profit_pct']:.1f}%")
print(f"   Risk: DD>50% {m1['drawdown_50_pct']:.2f}%, Bankrupt {m1['bankrupt_pct']:.2f}%")
print(f"   Ставка: avg={avg_bet1:.2f}%, max={max_bet1:.2f}%")

print(f"\n2. АГРЕССИВНАЯ (EXTREME_ROI16173):")
print(f"   Profit: +{m2['avg_profit_pct']:.1f}%")
print(f"   Risk: DD>50% {m2['drawdown_50_pct']:.2f}%, Bankrupt {m2['bankrupt_pct']:.2f}%")
print(f"   Ставка: avg={avg_bet2:.2f}%, max={max_bet2:.2f}%")

# Неожиданность?
if m2['avg_profit_pct'] > m1['avg_profit_pct'] and m2['bankrupt_pct'] <= m1['bankrupt_pct']:
    print(f"\n❓ СТРАННО: EXTREME дает больше прибыли ({m2['avg_profit_pct']:.1f}% vs {m1['avg_profit_pct']:.1f}%)")
    print(f"   но риски такие же или ниже!")
    print(f"   Возможно RISKY_LIMITED не оптимален?")

print("\n" + "="*90)
print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
print("="*90)
