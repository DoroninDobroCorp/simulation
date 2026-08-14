"""
РЕАЛИСТИЧНАЯ СИМУЛЯЦИЯ
Пересчет банкролла раз в 30-70 ставок (случайно), а не каждую ставку!
"""

import numpy as np
from config import INITIAL_BANKROLL, TARGET_ROI


def adaptive_constant_profit_realistic(outcomes, odds_array, 
                                       min_roi=3.0, max_roi=20.0,
                                       min_target_pct=0.5, max_target_pct=3.0, 
                                       max_bet_percent=15.0, 
                                       apply_variation=False,
                                       recalc_min=30, recalc_max=70):
    """
    Adaptive Constant Profit с РЕАЛИСТИЧНЫМ пересчетом банкролла.
    
    Ключевое отличие: Банкролл пересчитывается НЕ каждую ставку,
    а раз в recalc_min-recalc_max ставок (случайно).
    
    Это имитирует реальность: букмекер не знает твой текущий банк после каждой ставки!
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        min_roi, max_roi: диапазон ROI для интерполяции
        min_target_pct: минимальная целевая прибыль в % при min_roi
        max_target_pct: максимальная целевая прибыль в % при max_roi
        max_bet_percent: максимальный процент ставки от банка
        apply_variation: применять ли реалистичную вариацию размера ставок
        recalc_min, recalc_max: диапазон для периодичности пересчета банкролла
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    # История банкролла и ставок
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Интерполяция целевой прибыли по ROI
    if max_roi > min_roi:
        roi_factor = (roi_pct - min_roi) / (max_roi - min_roi)
        roi_factor = np.clip(roi_factor, 0, 1)
    else:
        roi_factor = 0.5
    
    target_profit_pct = min_target_pct + (max_target_pct - min_target_pct) * roi_factor
    
    # Симулируем для КАЖДОЙ симуляции отдельно (т.к. периоды пересчета случайны)
    for sim_idx in range(num_sims):
        current_position = 0
        current_bank = INITIAL_BANKROLL
        
        # Генерируем случайные периоды пересчета для этой симуляции
        np.random.seed(sim_idx + 42)  # Для воспроизводимости
        recalc_periods = []
        while current_position < num_bets:
            period = np.random.randint(recalc_min, recalc_max + 1)
            recalc_periods.append(period)
            current_position += period
        
        # Обнуляем позицию
        current_position = 0
        base_bank = INITIAL_BANKROLL  # Базовый банк для расчета ставок
        
        for period_idx, period_length in enumerate(recalc_periods):
            # Определяем конец периода
            period_end = min(current_position + period_length, num_bets)
            
            # Рассчитываем ставки для этого периода на основе base_bank
            for i in range(current_position, period_end):
                odds = odds_array[i]
                
                # Целевая прибыль от БАЗОВОГО банка (не меняется внутри периода)
                target_profit = base_bank * target_profit_pct / 100
                
                # Ставка для получения этой прибыли
                if odds > 1.05:
                    bet_size = target_profit / (odds - 1)
                else:
                    bet_size = base_bank * 0.01
                
                # Ограничиваем максимум от БАЗОВОГО банка
                max_bet = base_bank * max_bet_percent / 100
                bet_size = min(bet_size, max_bet)
                
                # Применяем вариацию если нужно (РЕАЛЬНЫЕ УСЛОВИЯ: 35-115%)
                if apply_variation:
                    variation = np.random.uniform(0.35, 1.15)
                    bet_size = bet_size * variation
                
                # Ограничиваем от ТЕКУЩЕГО банка (безопасность)
                if current_bank <= 0:
                    bet_size = 0
                else:
                    bet_size = min(bet_size, current_bank)
                    bet_size = min(bet_size, current_bank * 0.10)  # Не больше 10%
                
                bet_history[sim_idx, i] = bet_size
                
                # Применяем результат ставки
                if outcomes[sim_idx, i]:  # WIN
                    win_amount = bet_size * (odds - 1)
                    current_bank += win_amount
                else:  # LOSS
                    current_bank -= bet_size
                
                bankroll_history[sim_idx, i + 1] = current_bank
                
                # Проверка банкротства
                if current_bank < 1.0:
                    # Обнуляем оставшиеся ставки
                    bet_history[sim_idx, i+1:] = 0
                    bankroll_history[sim_idx, i+2:] = current_bank
                    break
            
            # Проверяем банкротство
            if current_bank < 1.0:
                break
            
            # ПЕРЕСЧЕТ БАЗОВОГО БАНКА для следующего периода
            base_bank = current_bank
            current_position = period_end
    
    # Рассчитываем метрики ставок
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll_history[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct


def calculate_metrics_realistic(bankroll_history, bet_history, odds_array):
    """
    Рассчитывает метрики для реалистичной симуляции.
    """
    num_sims = bankroll_history.shape[0]
    
    # Банкротство (bank < 1 = фактическое банкротство)
    bankrupt_count = np.sum(np.any(bankroll_history < 1.0, axis=1))
    bankrupt_pct = bankrupt_count / num_sims * 100
    
    # Просадки от пика
    peaks = np.maximum.accumulate(bankroll_history, axis=1)
    drawdowns_pct = (bankroll_history - peaks) / peaks * 100
    
    max_drawdown_per_sim = np.min(drawdowns_pct, axis=1)
    
    threshold_20_count = np.sum(np.any(drawdowns_pct <= -20, axis=1))
    threshold_50_count = np.sum(np.any(drawdowns_pct <= -50, axis=1))
    threshold_80_count = np.sum(np.any(drawdowns_pct <= -80, axis=1))
    
    # Прибыль
    final_bankrolls = bankroll_history[:, -1]
    final_profits = final_bankrolls - INITIAL_BANKROLL
    profit_pcts = final_profits / INITIAL_BANKROLL * 100
    
    # ROI от оборота
    total_bets = np.sum(bet_history, axis=1)
    roi_from_turnover = np.zeros(num_sims)
    valid_turnover = total_bets > 0
    roi_from_turnover[valid_turnover] = (final_profits[valid_turnover] / total_bets[valid_turnover]) * 100
    
    return {
        'bankrupt_pct': bankrupt_pct,
        'drawdown_20_pct': threshold_20_count / num_sims * 100,
        'drawdown_50_pct': threshold_50_count / num_sims * 100,
        'drawdown_80_pct': threshold_80_count / num_sims * 100,
        'avg_profit_pct': np.mean(profit_pcts),
        'min_profit_pct': np.min(profit_pcts),
        'max_profit_pct': np.max(profit_pcts),
        'avg_max_drawdown_pct': np.mean(max_drawdown_per_sim),
        'worst_drawdown_pct': np.min(max_drawdown_per_sim),
        'avg_roi_from_turnover': np.mean(roi_from_turnover),
        'min_roi_from_turnover': np.min(roi_from_turnover),
        'max_roi_from_turnover': np.max(roi_from_turnover),
    }


if __name__ == "__main__":
    print("Тестирование реалистичной симуляции...")
    
    from generate_real_odds_simulations import load_real_odds_outcomes
    
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"Загружено: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    
    # Тест k=0.5 без вариации
    print("\nТест k=0.5 без вариации:")
    br, bh, _, max_bet, avg_bet = adaptive_constant_profit_realistic(
        outcomes, odds_array,
        min_roi=4.733, max_roi=23.005,
        min_target_pct=3.982 * 0.5,
        max_target_pct=13.078 * 0.5,
        max_bet_percent=20.0 * 0.5,
        apply_variation=False,
        recalc_min=30, recalc_max=70
    )
    
    metrics = calculate_metrics_realistic(br, bh, odds_array)
    print(f"  Profit: +{metrics['avg_profit_pct']:.1f}%")
    print(f"  Bankrupt: {metrics['bankrupt_pct']:.2f}%")
    print(f"  DD>50%: {metrics['drawdown_50_pct']:.1f}%")
    print(f"  Avg bet: {avg_bet:.2f}%")
    print(f"  Max bet: {max_bet:.2f}%")
    
    print("\nТест k=0.5 с вариацией:")
    br2, bh2, _, max_bet2, avg_bet2 = adaptive_constant_profit_realistic(
        outcomes, odds_array,
        min_roi=4.733, max_roi=23.005,
        min_target_pct=3.982 * 0.5,
        max_target_pct=13.078 * 0.5,
        max_bet_percent=20.0 * 0.5,
        apply_variation=True,
        recalc_min=30, recalc_max=70
    )
    
    metrics2 = calculate_metrics_realistic(br2, bh2, odds_array)
    print(f"  Profit: +{metrics2['avg_profit_pct']:.1f}%")
    print(f"  Bankrupt: {metrics2['bankrupt_pct']:.2f}%")
    print(f"  DD>50%: {metrics2['drawdown_50_pct']:.1f}%")
    print(f"  Avg bet: {avg_bet2:.2f}%")
    print(f"  Max bet: {max_bet2:.2f}%")
    
    print("\n✅ Реалистичная симуляция работает!")
