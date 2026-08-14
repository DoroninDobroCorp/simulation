"""
ЛОГАРИФМИЧЕСКАЯ ФОРМУЛА (из backend/calculator)

Точная реализация формулы:
1. edge = min(realROI, 8%)
2. edgeDecimal = edge / 100
3. logFactor = 1 - (1 / (odds / (1 + edgeDecimal)))
4. betSizePercent = log₁₀(logFactor) / log₁₀(10^(-defaultRisk))
5. betSizePercent = min(betSizePercent, maxBetPercent / 100)
6. betSize = betSizePercent × defaultBank
"""

import numpy as np
from config import INITIAL_BANKROLL, TARGET_ROI


def logarithmic_kelly_realistic(outcomes, odds_array, default_risk=15.0, max_bet_percent=10.0,
                                recalc_min=30, recalc_max=70):
    """
    Логарифмическая формула с реалистичным пересчетом банка
    
    Args:
        outcomes: массив результатов (num_sims, num_bets)
        odds_array: массив коэффициентов (num_bets,)
        default_risk: параметр риска (обычно 15.0)
        max_bet_percent: максимальный % ставки (10.0 = 10%)
        recalc_min/max: интервал пересчета банка
    
    Returns:
        bankroll_history: история банкролла (num_sims, num_bets + 1)
    """
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    # Предварительный расчет для каждого odds
    log_base = np.log10(10 ** (-default_risk))  # = -default_risk
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        
        np.random.seed(sim_idx + 300)  # Уникальный seed
        
        while current_pos < num_bets:
            # Случайный период до пересчета
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                odds = odds_array[i]
                
                # ===== ЛОГАРИФМИЧЕСКАЯ ФОРМУЛА =====
                
                # Шаг 1: edge (ограничено 8%)
                edge = min(TARGET_ROI * 100, 8.0)
                edge_decimal = edge / 100
                
                # Шаг 2: logFactor
                # logFactor = 1 - (1 / (odds / (1 + edge_decimal)))
                # Упрощенно: logFactor = 1 - (1 + edge_decimal) / odds
                log_factor = 1 - (1 + edge_decimal) / odds
                
                # Шаг 3: betSizePercent через логарифм
                if log_factor > 0:
                    # betSizePercent = log₁₀(logFactor) / log₁₀(10^-defaultRisk)
                    bet_size_percent = np.log10(log_factor) / log_base
                    
                    # Шаг 4: ограничение max_bet_percent
                    bet_size_percent = min(bet_size_percent, max_bet_percent / 100)
                    
                    # Защита от отрицательных
                    bet_size_percent = max(0, bet_size_percent)
                else:
                    # Если logFactor <= 0, ставка = 0
                    bet_size_percent = 0
                
                # Размер ставки от базового банка
                bet_size = base_bank * bet_size_percent
                bet_size = min(bet_size, current_bank, current_bank * max_bet_percent / 100)
                
                # Применяем вариацию 35-115%
                variation = np.random.uniform(0.35, 1.15)
                bet_size = bet_size * variation
                bet_size = min(bet_size, current_bank * max_bet_percent / 100)
                
                # Округление до 5 (как в оригинале)
                if bet_size >= 5:
                    bet_size = round(bet_size / 5) * 5
                
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
    """Тест логарифмической формулы"""
    from generate_real_odds_simulations import load_real_odds_outcomes
    
    print("="*100)
    print("🧪 ТЕСТ ЛОГАРИФМИЧЕСКОЙ ФОРМУЛЫ")
    print("="*100)
    
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    print(f"Средний коэффициент: {np.mean(odds_array):.2f}")
    print(f"Целевой ROI: {TARGET_ROI*100:.1f}%")
    
    # Тестируем с разными defaultRisk
    risk_values = [10.0, 12.0, 15.0, 18.0, 20.0]
    
    print(f"\n{'defaultRisk':<15} {'Profit':<12} {'Bankrupt':<12} {'DD>50%':<12} {'Worst DD'}")
    print("-"*100)
    
    results = []
    
    for risk in risk_values:
        br = logarithmic_kelly_realistic(
            outcomes, odds_array,
            default_risk=risk,
            max_bet_percent=10.0
        )
        m = calculate_metrics_quick(br)
        
        print(f"{risk:<15.1f} +{m['profit']:<11.0f} {m['bankrupt']:<12.2f} {m['dd50']:<12.1f} {m['worst_dd']:.1f}%")
        
        results.append({
            'defaultRisk': risk,
            'profit': m['profit'],
            'bankrupt': m['bankrupt'],
            'dd50': m['dd50'],
            'dd80': m['dd80'],
            'worst_dd': m['worst_dd']
        })
    
    print("\n" + "="*100)
    print("✅ ТЕСТ ЗАВЕРШЕН")
    print("="*100)
    
    # Сохраняем результаты
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('logarithmic_kelly_test.csv', index=False)
    print("\n📁 Результаты сохранены в: logarithmic_kelly_test.csv")
