"""
ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ KELLY CRITERION

В текущем коде проблема:
    win_prob = 1.0 / odds  → Kelly = 0% (нет edge)
    
Правильная формула:
    win_prob = (TARGET_ROI + 1.0) / odds
"""

import numpy as np
from config import INITIAL_BANKROLL, TARGET_ROI


def kelly_correct_realistic(outcomes, odds_array, kelly_fraction=1.0, 
                            recalc_min=30, recalc_max=70):
    """
    Kelly Criterion с ПРАВИЛЬНОЙ формулой вероятности
    
    Args:
        outcomes: массив результатов (num_sims, num_bets)
        odds_array: массив коэффициентов (num_bets,)
        kelly_fraction: доля от полного Kelly (0.25 = четверть, 1.0 = полный)
        recalc_min/max: интервал пересчета базового банка
    
    Returns:
        bankroll_history: история банкролла (num_sims, num_bets + 1)
    """
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        
        np.random.seed(sim_idx + 100)
        
        while current_pos < num_bets:
            # Случайный период до пересчета
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                odds = odds_array[i]
                
                # ✅ ПРАВИЛЬНАЯ ФОРМУЛА!
                # Для ROI = 7% и odds = 2.76:
                # win_prob = 1.07 / 2.76 = 0.388 (38.8%)
                # Это РЕАЛЬНАЯ вероятность, дающая ROI=7%
                win_prob = (TARGET_ROI + 1.0) / odds
                
                # Kelly formula: f = (b*p - q) / b
                # где b = odds - 1, p = win_prob, q = 1 - win_prob
                b = odds - 1
                p = win_prob
                q = 1 - p
                
                # Считаем Kelly
                if b > 0:
                    kelly_f = (b * p - q) / b
                else:
                    kelly_f = 0
                
                # Защита от отрицательных значений
                kelly_f = max(0, kelly_f)
                
                # Применяем fraction (например, 0.25 = четверть Kelly)
                bet_pct = kelly_f * kelly_fraction * 100
                
                # Ограничение 10%
                bet_pct = min(bet_pct, 10.0)
                
                # Размер ставки
                bet_size = base_bank * bet_pct / 100
                bet_size = min(bet_size, current_bank, current_bank * 0.10)
                
                # Применяем вариацию 35-115%
                variation = np.random.uniform(0.35, 1.15)
                bet_size = bet_size * variation
                bet_size = min(bet_size, current_bank * 0.10)
                
                # Результат ставки
                if outcomes[sim_idx, i]:
                    current_bank += bet_size * (odds - 1)
                else:
                    current_bank -= bet_size
                
                bankroll_history[sim_idx, i + 1] = current_bank
            
            if current_bank < 1:
                break
            
            # Обновляем базовый банк для следующего периода
            base_bank = current_bank
            current_pos = period_end
    
    return bankroll_history


def calculate_metrics_quick(bankroll_history):
    """Быстрый расчет метрик"""
    num_sims = bankroll_history.shape[0]
    
    # Банкротство
    bankrupt_count = np.sum(np.any(bankroll_history < 1.0, axis=1))
    bankrupt_pct = bankrupt_count / num_sims * 100
    
    # Просадки
    peaks = np.maximum.accumulate(bankroll_history, axis=1)
    drawdowns_pct = (bankroll_history - peaks) / peaks * 100
    
    dd50_count = np.sum(np.any(drawdowns_pct <= -50, axis=1))
    dd80_count = np.sum(np.any(drawdowns_pct <= -80, axis=1))
    
    # Прибыль
    final_bankrolls = bankroll_history[:, -1]
    profit_pcts = (final_bankrolls - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    
    return {
        'profit': np.mean(profit_pcts),
        'bankrupt': bankrupt_pct,
        'dd50': dd50_count / num_sims * 100,
        'dd80': dd80_count / num_sims * 100,
        'worst_dd': np.min(np.min(drawdowns_pct, axis=1))
    }


if __name__ == '__main__':
    """Быстрый тест"""
    from generate_real_odds_simulations import load_real_odds_outcomes
    
    print("="*80)
    print("🧪 ТЕСТ ПРАВИЛЬНОГО KELLY")
    print("="*80)
    
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    print(f"Средний коэффициент: {np.mean(odds_array):.2f}")
    print(f"Целевой ROI: {TARGET_ROI*100:.1f}%")
    
    # Тестируем разные fractions
    fractions = [0.25, 0.5, 1.0, 1.5, 2.0]
    
    print(f"\n{'Fraction':<12} {'Profit':<12} {'Bankrupt':<12} {'DD>50%':<12} {'Worst DD'}")
    print("-"*80)
    
    for frac in fractions:
        br = kelly_correct_realistic(outcomes, odds_array, kelly_fraction=frac)
        m = calculate_metrics_quick(br)
        
        print(f"{frac:<12.2f} +{m['profit']:<11.0f} {m['bankrupt']:<12.2f} {m['dd50']:<12.1f} {m['worst_dd']:.1f}%")
    
    print("\n" + "="*80)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*80)
