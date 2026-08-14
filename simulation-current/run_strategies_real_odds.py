"""
=============================================================================
СИСТЕМА ВАРИАЦИИ РАЗМЕРА СТАВОК
=============================================================================

ВАЖНО ДЛЯ ПОНИМАНИЯ:

1. РАСЧЕТНАЯ СТАВКА (Ideal Bet):
   - Каждая стратегия рассчитывает "идеальный" размер ставки по своей формуле
   - Например, Kelly рекомендует 2.5% от банка, dynamic_percentage - 1.5%
   
2. РЕАЛЬНАЯ СТАВКА (Actual Bet):
   - В реальности не всегда получается поставить точно расчетную сумму
   - Причины: лимиты букмекера, доступные суммы, округления и т.д.
   
3. ВАРИАЦИЯ (apply_variation=True):
   - Имитирует реальность: случайный коэффициент от 35% до 115%
   - STAKE_VARIATIONS = [0.35, 0.40, 0.45, ..., 1.10, 1.15]
   - Реальная ставка = Расчетная × случайный_коэффициент
   
4. ПРИМЕР:
   Стратегия рассчитала: 100 единиц
   - БЕЗ вариации: ставим ровно 100
   - С вариацией: ставим случайно от 30 до 115 (например, 85 или 105)

5. КАК ПРОГОНЯТЬ СТРАТЕГИИ:
   
   # Без вариации (идеальные условия):
   result = run_strategy_with_real_odds(
       'kelly_criterion', outcomes, odds, 
       risk=2.0, kelly_fraction=0.5,
       apply_variation=False
   )
   
   # С вариацией (реалистичные условия):
   result = run_strategy_with_real_odds(
       'kelly_criterion', outcomes, odds,
       risk=2.0, kelly_fraction=0.5, 
       apply_variation=True
   )

6. ЭФФЕКТ ВАРИАЦИИ (см. compare_with_without_variation.py):
   - Снижает риски (меньше DD, меньше сливов)
   - Снижает прибыль на ~20-30%
   - ROI остается ~7% в обоих случаях

=============================================================================
"""

import numpy as np
import csv
import os
from datetime import datetime
from generate_real_odds_simulations import load_real_odds_outcomes
from config import INITIAL_BANKROLL, RANDOM_SEED, TARGET_ROI

# Вариация размера ставок: от 35% до 115% с шагом 5% (РЕАЛЬНЫЕ УСЛОВИЯ РАБОТЫ)
STAKE_VARIATIONS = np.arange(0.35, 1.16, 0.05)  # [0.35, 0.40, 0.45, ... 1.10, 1.15]

def apply_realistic_stake_variation(bet_amounts, seed_offset=0):
    """
    Применяет реалистичную вариацию к размерам ставок.
    
    Args:
        bet_amounts: numpy array (num_sims, num_bets) с расчетными размерами ставок
        seed_offset: смещение для генератора случайных чисел
    
    Returns:
        numpy array с реальными размерами ставок после вариации
    """
    np.random.seed(RANDOM_SEED + seed_offset)
    num_sims, num_bets = bet_amounts.shape
    
    # Для каждой ставки случайно выбираем коэффициент вариации
    variation_factors = np.random.choice(STAKE_VARIATIONS, size=(num_sims, num_bets))
    
    # Применяем вариацию
    real_bet_amounts = bet_amounts * variation_factors
    
    return real_bet_amounts, variation_factors

def calculate_metrics_with_odds(bankroll_history, bet_history, odds_array):
    """
    Рассчитывает метрики с учетом переменных коэффициентов.
    
    Args:
        bankroll_history: numpy array (num_sims, num_bets + 1)
        bet_history: numpy array (num_sims, num_bets)
        odds_array: numpy array (num_bets,) с коэффициентами
    
    Returns:
        dict с метриками
    """
    num_sims = bankroll_history.shape[0]
    
    # Банкротство (ИСПРАВЛЕНО: bank < 1 = фактическое банкротство, работать нельзя)
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
    final_profits_pct = final_profits / INITIAL_BANKROLL * 100
    
    # ROI с оборота
    total_turnover = np.sum(bet_history, axis=1)
    roi_from_turnover = (final_profits / total_turnover) * 100
    
    return {
        'bankrupt_pct': bankrupt_pct,
        'drawdown_20_pct': threshold_20_count / num_sims * 100,
        'drawdown_50_pct': threshold_50_count / num_sims * 100,
        'drawdown_80_pct': threshold_80_count / num_sims * 100,
        'avg_profit_pct': np.mean(final_profits_pct),
        'min_profit_pct': np.min(final_profits_pct),
        'max_profit_pct': np.max(final_profits_pct),
        'avg_max_drawdown_pct': np.mean(max_drawdown_per_sim),
        'worst_drawdown_pct': np.min(max_drawdown_per_sim),
        'avg_roi_from_turnover': np.mean(roi_from_turnover),
        'min_roi_from_turnover': np.min(roi_from_turnover),
        'max_roi_from_turnover': np.max(roi_from_turnover)
    }

def kelly_criterion_strategy_with_real_odds(outcomes, odds_array, risk=2.0, kelly_fraction=1.0, apply_variation=False):
    """
    Kelly Criterion стратегия с реальными коэффициентами.
    
    Формула Келли: f = (b×p - q) / b
    где:
    - p = вероятность выигрыша = (1 + ROI) / odds
    - b = odds - 1 (выигрыш на единицу ставки)
    - q = 1 - p (вероятность проигрыша)
    - f = доля банка для ставки
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        risk: делитель для консервативности (чем больше, тем консервативнее)
        kelly_fraction: доля от полного Келли (0.25-0.5 рекомендуется)
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Рассчитываем идеальные размеры ставок по Келли
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Вероятность выигрыша из TARGET_ROI
        p = (1 + TARGET_ROI) / odds
        p = np.clip(p, 0, 1)  # Ограничиваем [0, 1]
        
        # Формула Келли
        b = odds - 1
        q = 1 - p
        kelly_f = (b * p - q) / b if b > 0 else 0
        kelly_f = max(0, kelly_f)  # Не ставим если edge отрицательный
        
        # Применяем параметры консервативности
        kelly_f = kelly_f * kelly_fraction / risk
        
        # Ограничиваем максимум 10% банка
        kelly_f = min(kelly_f, 0.10)
        
        bet_amount = current_bankroll * kelly_f
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=1)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def linear_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, max_percent=10.0, apply_variation=False):
    """
    Linear ROI Strategy: размер ставки линейно пропорционален ROI.
    
    Формула: bet% = base_percent × (ROI / base_ROI)
    Ограничение: не более max_percent
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_roi: базовый ROI для нормализации (обычно 5%)
        base_percent: процент ставки при base_roi (обычно 1%)
        max_percent: максимальный процент ставки
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # ROI для данного коэффициента
        roi = (TARGET_ROI * 100)  # переводим в проценты
        
        # Линейное масштабирование
        bet_pct = base_percent * (roi / base_roi)
        bet_pct = min(bet_pct, max_percent)  # Ограничиваем максимум
        bet_pct = max(0, bet_pct)  # Не меньше 0
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=2)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def sqrt_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, max_percent=10.0, apply_variation=False):
    """
    Square Root ROI Strategy: размер ставки пропорционален √(ROI/base_ROI).
    
    Формула: bet% = base_percent × √(ROI / base_ROI)
    Ограничение: не более max_percent
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_roi: базовый ROI для нормализации
        base_percent: процент ставки при base_roi
        max_percent: максимальный процент ставки
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # ROI для данного коэффициента
        roi = (TARGET_ROI * 100)  # переводим в проценты
        
        # Масштабирование через квадратный корень
        roi_ratio = roi / base_roi
        bet_pct = base_percent * np.sqrt(max(0, roi_ratio))
        bet_pct = min(bet_pct, max_percent)
        bet_pct = max(0, bet_pct)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=3)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def log_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, max_percent=10.0, apply_variation=False):
    """
    Logarithmic ROI Strategy: размер ставки растет логарифмически с ROI.
    
    Формула: bet% = base_percent × log(ROI/base_ROI + 1)
    Ограничение: не более max_percent
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_roi: базовый ROI для нормализации
        base_percent: процент ставки при base_roi
        max_percent: максимальный процент ставки
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # ROI для данного коэффициента
        roi = (TARGET_ROI * 100)  # переводим в проценты
        
        # Логарифмическое масштабирование
        roi_ratio = roi / base_roi
        bet_pct = base_percent * np.log(roi_ratio + 1)
        bet_pct = min(bet_pct, max_percent)
        bet_pct = max(0, bet_pct)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=4)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def constant_profit_strategy_with_real_odds(outcomes, odds_array, target_profit_pct=1.0, max_percent=10.0, apply_variation=False):
    """
    Constant Profit Strategy: размер ставки подбирается для получения фиксированной прибыли.
    
    Формула: bet = target_profit / (odds - 1)
    Ограничение: не более max_percent от банка
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        target_profit_pct: целевая прибыль в % от текущего банка
        max_percent: максимальный процент ставки
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Целевая прибыль в абсолютных единицах
        target_profit = current_bankroll * target_profit_pct / 100
        
        # Ставка для получения этой прибыли
        if odds > 1:
            bet_amount = target_profit / (odds - 1)
        else:
            bet_amount = current_bankroll * 0.01  # Минимальная ставка
        
        # Ограничиваем максимум
        max_bet = current_bankroll * max_percent / 100
        bet_amount = np.minimum(bet_amount, max_bet)
        
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=5)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def combined_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, max_percent=10.0, 
                                                min_roi=3.0, max_roi=15.0, min_odds=1.5, max_odds=5.0, 
                                                apply_variation=False):
    """
    Combined ROI-Odds Strategy: учитывает одновременно ROI и коэффициент.
    
    Формула: bet% = base_percent × √(norm_ROI) × (1 - 0.5×norm_odds)
    где norm_ROI и norm_odds нормализованы в [0,1]
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_percent: базовый процент ставки
        max_percent: максимальный процент ставки
        min_roi, max_roi: диапазон для нормализации ROI
        min_odds, max_odds: диапазон для нормализации odds
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Нормализация ROI в [0, 1]
        norm_roi = np.clip((roi_pct - min_roi) / (max_roi - min_roi), 0, 1)
        
        # Нормализация odds в [0, 1]
        norm_odds = np.clip((odds - min_odds) / (max_odds - min_odds), 0, 1)
        
        # Комбинированный фактор
        roi_factor = np.sqrt(norm_roi)  # Сглаживание ROI
        odds_penalty = 1 - 0.5 * norm_odds  # Штраф за высокие odds
        combined = roi_factor * odds_penalty
        
        # Размер ставки
        bet_pct = base_percent * combined
        bet_pct = min(bet_pct, max_percent)
        bet_pct = max(0, bet_pct)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=6)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def adaptive_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, max_percent=10.0, 
                                      min_roi=3.0, max_roi=15.0, min_odds=1.5, max_odds=5.0,
                                      apply_variation=False):
    """
    Adaptive Strategy: как Combined, но снижает агрессивность при просадках.
    
    Модификаторы:
    - Если банк < 80% от пика → ставка ×0.75
    - Если банк < 60% от пика → ставка ×0.5
    - Дополнительно ограничивает max_percent при просадке от начального банка
    
    ИСПРАВЛЕНО: Теперь рассчитывает ставки В ПРОЦЕССЕ симуляции, а не заранее!
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_percent: базовый процент ставки
        max_percent: максимальный процент ставки
        min_roi, max_roi: диапазон для нормализации ROI
        min_odds, max_odds: диапазон для нормализации odds
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Бегущий пик банка (инкрементально, чтобы не пересчитывать accumulate каждый шаг)
    peak_bankroll = bankroll[:, 0].copy()
    
    # Рассчитываем ставки В ПРОЦЕССЕ симуляции чтобы учесть реальные просадки
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Обновляем пиковый банк до текущей ставки
        peak_bankroll = np.maximum(peak_bankroll, current_bankroll)
        
        # Нормализация ROI в [0, 1]
        norm_roi = np.clip((roi_pct - min_roi) / (max_roi - min_roi), 0, 1)
        
        # Нормализация odds в [0, 1]
        norm_odds = np.clip((odds - min_odds) / (max_odds - min_odds), 0, 1)
        
        # Базовая комбинированная ставка
        roi_factor = np.sqrt(norm_roi)
        odds_penalty = 1 - 0.5 * norm_odds
        combined = roi_factor * odds_penalty
        
        bet_pct = base_percent * combined
        
        # Адаптивные модификаторы при просадке от пика
        drawdown_from_peak = (peak_bankroll - current_bankroll) / np.maximum(peak_bankroll, 1e-10)
        drawdown_modifier = np.ones(num_sims)
        drawdown_modifier = np.where(drawdown_from_peak > 0.20, 0.75, drawdown_modifier)
        drawdown_modifier = np.where(drawdown_from_peak > 0.40, 0.5, drawdown_modifier)
        
        bet_pct_array = bet_pct * drawdown_modifier
        
        # Дополнительное ограничение при просадке от начального банка
        current_max_percent = np.full(num_sims, max_percent)
        drawdown_from_initial = (INITIAL_BANKROLL - current_bankroll) / INITIAL_BANKROLL
        current_max_percent = np.where(drawdown_from_initial > 0.30, max_percent * 0.5, current_max_percent)
        current_max_percent = np.where(drawdown_from_initial > 0.40, max_percent * 0.25, current_max_percent)
        
        bet_pct_array = np.minimum(bet_pct_array, current_max_percent)
        bet_pct_array = np.maximum(bet_pct_array, 0)
        
        bet_amount = current_bankroll * bet_pct_array / 100
        bet_history_ideal[:, i] = bet_amount
        
        # Симулируем СРАЗУ результат ставки для следующей итерации
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=7)
        
        # Пересимулируем с вариациями
        bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
        for i in range(num_bets):
            current_bankroll = bankroll[:, i]
            bet_amount = bet_history[:, i]
            odds = odds_array[i]
            
            win_amount = bet_amount * (odds - 1)
            loss_amount = bet_amount
            
            bankroll[:, i + 1] = current_bankroll + np.where(
                outcomes[:, i],
                win_amount,
                -loss_amount
            )
    else:
        bet_history = bet_history_ideal
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def dynamic_kelly_strategy_with_real_odds(outcomes, odds_array, risk=2.0, min_fraction=0.1, max_fraction=0.5, 
                                          min_roi=3.0, max_roi=15.0, apply_variation=False):
    """
    Dynamic Kelly Strategy: Kelly с динамической фракцией в зависимости от ROI.
    
    Формула: 
    1. Базовая Kelly: f = (b×p - q) / b / risk
    2. ROI factor: roi_factor = (ROI - min_roi) / (max_roi - min_roi)
    3. Dynamic fraction: fraction = min_frac + (max_frac - min_frac) × roi_factor
    4. Итоговая ставка: f × fraction
    
    При высоком ROI использует большую фракцию (агрессивнее).
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        risk: делитель для консервативности (чем больше, тем консервативнее)
        min_fraction: минимальная фракция Kelly при min_roi
        max_fraction: максимальная фракция Kelly при max_roi
        min_roi, max_roi: диапазон ROI для интерполяции фракции
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Вычисляем roi_factor для интерполяции фракции
    roi_factor = np.clip((roi_pct - min_roi) / (max_roi - min_roi), 0, 1)
    
    # Динамическая фракция Kelly
    dynamic_fraction = min_fraction + (max_fraction - min_fraction) * roi_factor
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Вероятность выигрыша из ROI
        p = (1 + TARGET_ROI) / odds
        p = np.clip(p, 0.01, 0.99)
        
        q = 1 - p
        b = odds - 1
        
        # Базовая Kelly
        kelly_bet = (b * p - q) / b / risk
        kelly_bet = np.maximum(kelly_bet, 0)
        
        # Применяем динамическую фракцию
        bet_pct = kelly_bet * dynamic_fraction
        
        # Ограничиваем максимум 10%
        bet_pct = min(bet_pct, 10.0)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=8)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def exponential_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, 
                                             factor=0.1, max_percent=10.0, apply_variation=False):
    """
    Exponential ROI Strategy: ставка растёт экспоненциально с ростом ROI.
    
    Формула: bet% = base_percent × exp(factor × (ROI - base_ROI))
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_roi: базовый ROI для нормализации
        base_percent: процент ставки при base_roi
        factor: контролирует скорость экспоненциального роста (обычно 0.05-0.15)
        max_percent: максимальный процент ставки
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        
        # Экспоненциальное масштабирование
        exponent = factor * (roi_pct - base_roi)
        bet_pct = base_percent * np.exp(exponent)
        
        # Ограничиваем максимум
        bet_pct = min(bet_pct, max_percent)
        bet_pct = max(0, bet_pct)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=9)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def hybrid_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, max_percent=10.0,
                                    min_roi=3.0, max_roi=15.0, min_odds=1.5, max_odds=5.0,
                                    roi_weight=0.7, odds_weight=0.3, apply_variation=False):
    """
    Hybrid Strategy: взвешенная комбинация нормализованных ROI и odds.
    
    Формула: 
    1. norm_ROI = (ROI - min) / (max - min)
    2. norm_odds = 1 - (odds - min) / (max - min)  [инвертированно!]
    3. combined = roi_weight × norm_ROI + odds_weight × norm_odds
    4. bet% = base_percent × combined
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_percent: базовый процент ставки
        max_percent: максимальный процент ставки
        min_roi, max_roi: диапазон для нормализации ROI
        min_odds, max_odds: диапазон для нормализации odds
        roi_weight: вес ROI в комбинации (обычно 0.7)
        odds_weight: вес odds в комбинации (обычно 0.3)
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Нормализация ROI в [0, 1]
        norm_roi = np.clip((roi_pct - min_roi) / (max_roi - min_roi), 0, 1)
        
        # Нормализация odds в [0, 1] (инвертированно - низкие odds лучше)
        norm_odds = 1 - np.clip((odds - min_odds) / (max_odds - min_odds), 0, 1)
        
        # Взвешенная комбинация
        combined = roi_weight * norm_roi + odds_weight * norm_odds
        
        # Размер ставки
        bet_pct = base_percent * combined
        bet_pct = min(bet_pct, max_percent)
        bet_pct = max(0, bet_pct)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=10)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def linear_scaled_strategy_with_real_odds(outcomes, odds_array, min_roi=3.0, max_roi=20.0, 
                                          min_percent=1.0, max_percent=7.0, apply_variation=False):
    """
    Linear Scaled Strategy: прямое линейное отображение ROI в процент ставки.
    
    Формула: bet% = min% + (max% - min%) × (ROI - min_ROI) / (max_ROI - min_ROI)
    
    Простейший линейный mapping без учета odds.
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        min_roi, max_roi: диапазон ROI для интерполяции
        min_percent, max_percent: диапазон процента ставки
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Линейная интерполяция ROI → bet%
    if max_roi > min_roi:
        roi_factor = (roi_pct - min_roi) / (max_roi - min_roi)
        roi_factor = np.clip(roi_factor, 0, 1)
    else:
        roi_factor = 0.5
    
    bet_pct = min_percent + (max_percent - min_percent) * roi_factor
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=11)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def linear_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, 
                                             max_percent=10.0, odds_penalty_factor=0.7,
                                             min_odds=1.5, max_odds=5.0, apply_variation=False):
    """
    Linear ROI-Odds Strategy: линейная зависимость от ROI с коррекцией на odds.
    
    Формула: 
    1. Базовая ставка: bet% = base% × (ROI / base_ROI)
    2. Штраф за высокие odds: odds_factor = 1 - odds_penalty_factor × norm_odds
    3. Итоговая ставка: bet% × odds_factor
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        base_roi: базовый ROI для нормализации
        base_percent: процент ставки при base_roi
        max_percent: максимальный процент ставки
        odds_penalty_factor: коэффициент штрафа за высокие odds (обычно 0.5-0.8)
        min_odds, max_odds: диапазон для нормализации odds
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Базовая ставка линейно от ROI
        bet_pct = base_percent * (roi_pct / base_roi)
        
        # Нормализация odds в [0, 1]
        norm_odds = np.clip((odds - min_odds) / (max_odds - min_odds), 0, 1)
        
        # Штраф за высокие odds
        odds_factor = 1 - odds_penalty_factor * norm_odds
        
        # Применяем коррекцию
        bet_pct = bet_pct * odds_factor
        
        # Ограничиваем
        bet_pct = min(bet_pct, max_percent)
        bet_pct = max(0, bet_pct)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=12)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def adaptive_constant_profit_strategy_with_real_odds(outcomes, odds_array, min_roi=3.0, max_roi=20.0,
                                                      min_target_pct=0.5, max_target_pct=3.0, 
                                                      max_bet_percent=15.0, apply_variation=False):
    """
    Adaptive Constant Profit Strategy: целевая прибыль масштабируется по ROI.
    
    Формула:
    1. target% = min% + (max% - min%) × (ROI - min_ROI) / (max_ROI - min_ROI)
    2. bet = (target% × bank) / (odds - 1)
    3. Ограничение: не более max_bet_percent
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        min_roi, max_roi: диапазон ROI для интерполяции
        min_target_pct: минимальная целевая прибыль в % при min_roi
        max_target_pct: максимальная целевая прибыль в % при max_roi
        max_bet_percent: максимальный процент ставки от банка
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    # Интерполяция целевой прибыли по ROI
    if max_roi > min_roi:
        roi_factor = (roi_pct - min_roi) / (max_roi - min_roi)
        roi_factor = np.clip(roi_factor, 0, 1)
    else:
        roi_factor = 0.5
    
    target_profit_pct = min_target_pct + (max_target_pct - min_target_pct) * roi_factor
    
    # Рассчитываем идеальные размеры ставок
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Целевая прибыль в абсолютных единицах
        target_profit = current_bankroll * target_profit_pct / 100
        
        # Ставка для получения этой прибыли
        # ИСПРАВЛЕНО: добавлена защита от низких коэффициентов
        if odds > 1.05:  # Минимальный коэффициент для безопасности
            bet_amount = target_profit / (odds - 1)
        else:
            bet_amount = current_bankroll * 0.01  # Минимальная ставка при низких odds
        
        # Ограничиваем максимум
        max_bet = current_bankroll * max_bet_percent / 100
        bet_amount = np.minimum(bet_amount, max_bet)
        
        bet_history_ideal[:, i] = bet_amount
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=13)
    else:
        bet_history = bet_history_ideal
    
    # Симулируем с реальными ставками
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        # ИСПРАВЛЕНО: Если банк <= 0, обнуляем ставку (банкротство!)
        bet_amount = np.where(current_bankroll <= 0, 0, bet_amount)
        
        # КРИТИЧНО: Ставка не может быть больше текущего банка!
        bet_amount = np.minimum(bet_amount, current_bankroll)
        
        # ДОПОЛНИТЕЛЬНО: Ограничение максимум 10% от текущего банка
        max_allowed = current_bankroll * 0.10  # 10% от текущего
        bet_amount = np.minimum(bet_amount, max_allowed)
        
        bet_history[:, i] = bet_amount  # Сохраняем скорректированную ставку
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def fixed_fraction_strategy_with_real_odds(outcomes, odds_array, fixed_percent=2.0, apply_variation=False):
    """
    Fixed Fraction Strategy: всегда фиксированный процент от текущего банка.
    
    Простейшая стратегия: bet = fixed_percent × bank
    Не зависит от ROI или odds. Невозможно разориться.
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        fixed_percent: фиксированный процент ставки (обычно 1-5%)
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = current_bankroll * fixed_percent / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=14)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def proportional_kelly_strategy_with_real_odds(outcomes, odds_array, risk=2.0, confidence=0.7, max_percent=10.0, apply_variation=False):
    """
    Proportional Kelly Strategy: Kelly с коэффициентом уверенности.
    
    Формула: kelly_bet × confidence
    где confidence (0-1) отражает уверенность в оценке вероятностей.
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        risk: делитель для консервативности
        confidence: коэффициент уверенности 0-1 (обычно 0.5-0.9)
        max_percent: максимальный процент ставки
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        p = (1 + TARGET_ROI) / odds
        p = np.clip(p, 0.01, 0.99)
        q = 1 - p
        b = odds - 1
        
        kelly_bet = (b * p - q) / b / risk
        kelly_bet = np.maximum(kelly_bet, 0)
        
        # Применяем confidence
        bet_pct = kelly_bet * confidence
        bet_pct = min(bet_pct, max_percent)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=16)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def target_based_strategy_with_real_odds(outcomes, odds_array, target_bankroll_percent=200.0, 
                                          aggressive_pct=3.0, conservative_pct=1.0, apply_variation=False):
    """
    Target-Based Strategy: переключение агрессивности по достижении цели.
    
    Пока банк < target → aggressive_pct
    После достижения цели → conservative_pct
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        target_bankroll_percent: целевой размер банка в % от начального (обычно 150-300%)
        aggressive_pct: процент ставки до достижения цели
        conservative_pct: процент ставки после достижения цели
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    target_bankroll = INITIAL_BANKROLL * target_bankroll_percent / 100
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        
        # Выбираем процент в зависимости от достижения цели
        bet_pct = np.where(current_bankroll < target_bankroll, aggressive_pct, conservative_pct)
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal, seed_offset=18)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def anti_martingale_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, multiplier=1.5, 
                                             max_percent=10.0, max_streak=5, apply_variation=False):
    """
    Anti-Martingale Strategy: увеличение ставки после ВЫИГРЫША.
    
    После выигрыша: bet *= multiplier (до max_streak раз)
    После проигрыша: возврат к base_percent
    
    Безопаснее чем Martingale - ограничивает убытки, максимизирует прибыль на winning streaks.
    
    Args:
        outcomes: numpy array (num_sims, num_bets)
        odds_array: numpy array (num_bets,)
        base_percent: базовая ставка после проигрыша
        multiplier: множитель после выигрыша (обычно 1.5-2.0)
        max_percent: максимальная ставка в %
        max_streak: максимальная длина серии умножений
        apply_variation: применять вариацию 30%-115%
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    current_bet_pct = np.full(num_sims, base_percent, dtype=float)
    win_streak = np.zeros(num_sims, dtype=int)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = current_bankroll * current_bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
        
        won = outcomes[:, i]
        
        # После выигрыша: увеличиваем ставку
        win_streak = np.where(won, win_streak + 1, 0)
        can_increase = win_streak < max_streak
        current_bet_pct = np.where(won & can_increase, 
                                   np.minimum(current_bet_pct * multiplier, max_percent),
                                   np.where(won, current_bet_pct, base_percent))
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=15)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def volatility_adjusted_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0, 
                                                 lookback=50, volatility_factor=1.0, apply_variation=False):
    """
    Volatility-Adjusted Strategy: снижение ставок при высокой волатильности.
    
    Формула: bet = base_percent / (1 + volatility_factor × std_dev)
    где std_dev рассчитывается по последним lookback результатам.
    
    Args:
        outcomes: numpy array (num_sims, num_bets)
        odds_array: numpy array (num_bets,)
        base_percent: базовая ставка в спокойные периоды
        lookback: количество последних результатов для расчета волатильности
        volatility_factor: коэффициент чувствительности к волатильности
        apply_variation: применять вариацию 30%-115%
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        
        # Расчет волатильности по последним результатам
        if i < lookback:
            volatility = 0.0
        else:
            # Берем последние lookback результатов (profit/loss в %)
            recent_changes = np.zeros((num_sims, lookback))
            for j in range(lookback):
                idx = i - lookback + j
                prev_bank = bankroll[:, idx]
                next_bank = bankroll[:, idx + 1]
                recent_changes[:, j] = (next_bank - prev_bank) / np.maximum(prev_bank, 1e-10) * 100
            
            volatility = np.std(recent_changes, axis=1)
        
        # Корректировка ставки
        adjustment = 1 / (1 + volatility_factor * volatility / 100)
        bet_pct = base_percent * adjustment
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=17)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def streak_aware_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0, 
                                         win_streak_multiplier=1.2, loss_streak_divider=1.3,
                                         max_multiplier=3.0, apply_variation=False):
    """
    Win/Loss Streak Aware Strategy: адаптация к сериям побед/поражений.
    
    После каждой победы: bet *= win_streak_multiplier
    После каждого поражения: bet /= loss_streak_divider
    Возврат к base при смене направления.
    
    Args:
        outcomes: numpy array (num_sims, num_bets)
        odds_array: numpy array (num_bets,)
        base_percent: базовая ставка
        win_streak_multiplier: множитель при победной серии
        loss_streak_divider: делитель при проигрышной серии
        max_multiplier: максимальное увеличение относительно base
        apply_variation: применять вариацию 30%-115%
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    current_bet_pct = np.full(num_sims, base_percent, dtype=float)
    last_outcome = np.full(num_sims, -1, dtype=int)  # -1=unknown, 0=loss, 1=win
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = current_bankroll * current_bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
        
        won = outcomes[:, i].astype(int)
        
        # Определяем: продолжение серии или смена
        same_direction = (last_outcome == won)
        changed_direction = (last_outcome != -1) & ~same_direction
        
        # Корректируем ставку
        # Win streak продолжается: увеличиваем
        current_bet_pct = np.where(same_direction & (won == 1),
                                   np.minimum(current_bet_pct * win_streak_multiplier, base_percent * max_multiplier),
                                   current_bet_pct)
        
        # Loss streak продолжается: уменьшаем
        current_bet_pct = np.where(same_direction & (won == 0),
                                   np.maximum(current_bet_pct / loss_streak_divider, base_percent / max_multiplier),
                                   current_bet_pct)
        
        # Смена направления: возврат к base
        current_bet_pct = np.where(changed_direction, base_percent, current_bet_pct)
        
        last_outcome = won
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=22)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def sharpe_optimized_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0, 
                                              lookback=100, risk_free_rate=0.0, apply_variation=False):
    """
    Sharpe Ratio Optimization: размер ставки максимизирует Sharpe Ratio.
    
    Sharpe = (Return - RiskFreeRate) / Volatility
    Рассчитывается по последним lookback ставкам.
    
    Args:
        outcomes: numpy array (num_sims, num_bets)
        odds_array: numpy array (num_bets,)
        base_percent: базовая ставка
        lookback: окно для расчета Sharpe Ratio
        risk_free_rate: безрисковая ставка (обычно 0)
        apply_variation: применять вариацию 30%-115%
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        
        if i < lookback:
            # Недостаточно истории - используем базовую ставку
            bet_pct = base_percent
        else:
            # Расчет returns за lookback период
            returns = np.zeros((num_sims, lookback))
            for j in range(lookback):
                idx = i - lookback + j
                prev = bankroll[:, idx]
                curr = bankroll[:, idx + 1]
                returns[:, j] = (curr - prev) / np.maximum(prev, 1e-10)
            
            # Sharpe Ratio: (mean_return - rf) / std_return
            mean_return = np.mean(returns, axis=1)
            std_return = np.std(returns, axis=1)
            
            # Если низкий Sharpe - снижаем агрессию, если высокий - увеличиваем
            sharpe = (mean_return - risk_free_rate) / np.maximum(std_return, 1e-6)
            
            # Нормализуем sharpe к multiplier: хороший sharpe > 1 → увеличиваем ставку
            # Плохой sharpe < 0 → уменьшаем ставку
            multiplier = 1 + np.clip(sharpe, -0.5, 1.0)  # Диапазон [0.5, 2.0]
            bet_pct = base_percent * multiplier
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=19)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def bayesian_kelly_strategy_with_real_odds(outcomes, odds_array, prior_mean=0.378, prior_std=0.05,
                                           risk_factor=2.0, max_percent=10.0, apply_variation=False):
    """
    Bayesian Kelly: Kelly с учетом неопределенности в оценке вероятности.
    
    Вместо точечной оценки p используется распределение N(prior_mean, prior_std).
    При высокой неопределенности автоматически снижается размер ставки.
    
    Args:
        outcomes: numpy array (num_sims, num_bets)
        odds_array: numpy array (num_bets,)
        prior_mean: априорная вероятность выигрыша (для ROI=7%, odds~2.76 → p~0.378)
        prior_std: стандартное отклонение неопределенности
        risk_factor: делитель Kelly для консерватизма
        max_percent: максимальная ставка
        apply_variation: применять вариацию 30%-115%
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        odds = odds_array[i]
        
        # Используем prior с неопределенностью
        # При высокой неопределенности снижаем ставку
        p_mean = prior_mean
        p_std = prior_std
        
        # Conservative adjustment: снижаем p на uncertainty
        p_conservative = p_mean - p_std  # Консервативная оценка
        p_conservative = np.clip(p_conservative, 0.01, 0.99)
        
        q = 1 - p_conservative
        b = odds - 1
        
        # Kelly с консервативной p
        kelly_bet = (b * p_conservative - q) / b / risk_factor
        kelly_bet = np.maximum(kelly_bet, 0)
        kelly_bet = min(kelly_bet, max_percent)
        
        bet_amount = current_bankroll * kelly_bet / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=21)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct, max_bet_pct, avg_bet_pct = np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def multi_objective_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                             w_profit=0.5, w_drawdown=0.3, w_volatility=0.2,
                                             lookback=50, apply_variation=False):
    """
    Multi-Objective Optimization: баланс между прибылью, просадкой и волатильностью.
    
    Score = w_profit×norm_profit - w_drawdown×norm_dd - w_volatility×norm_vol
    Размер ставки корректируется для максимизации score.
    
    Args:
        outcomes, odds_array: данные
        base_percent: базовая ставка
        w_profit, w_drawdown, w_volatility: веса целей (должны сумма = 1.0)
        lookback: окно для расчета метрик
        apply_variation: применять вариацию
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        
        if i < lookback:
            bet_pct = base_percent
        else:
            # Расчет profit, DD, vol за lookback
            profit_pct = (bankroll[:, i] - bankroll[:, i-lookback]) / bankroll[:, i-lookback] * 100
            
            # Drawdown
            max_in_period = np.maximum.accumulate(bankroll[:, i-lookback:i+1], axis=1)
            dd = (max_in_period - bankroll[:, i-lookback:i+1]) / max_in_period * 100
            max_dd = np.max(dd, axis=1)
            
            # Volatility
            returns = np.diff(bankroll[:, i-lookback:i+1], axis=1) / bankroll[:, i-lookback:i]
            volatility = np.std(returns, axis=1) * 100
            
            # Нормализация
            profit_norm = np.clip(profit_pct / 50, 0, 2)  # 0-100% profit → 0-2
            dd_norm = np.clip(max_dd / 30, 0, 2)           # 0-60% DD → 0-2
            vol_norm = np.clip(volatility / 5, 0, 2)       # 0-10% vol → 0-2
            
            # Multi-objective score
            score = w_profit * profit_norm - w_drawdown * dd_norm - w_volatility * vol_norm
            
            # Корректируем ставку: хороший score → увеличиваем, плохой → уменьшаем
            multiplier = 1 + np.clip(score, -0.5, 1.0)
            bet_pct = base_percent * multiplier
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=20)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        bankroll[:, i + 1] = bankroll[:, i] + np.where(outcomes[:, i], bet_amount * (odds - 1), -bet_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        valid_mask = bankroll[:, i] > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / bankroll[valid_mask, i]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    min_bet_pct, max_bet_pct, avg_bet_pct = (np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)) if len(valid_bets) > 0 else (0, 0, 0)
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def portfolio_theory_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                              rebalance_frequency=100, apply_variation=False):
    """
    Portfolio Theory Approach: распределение между высокими/средними/низкими odds.
    
    Упрощенная версия: делим ставки на 3 категории по odds и балансируем между ними.
    
    Args:
        outcomes, odds_array: данные
        base_percent: базовая ставка
        rebalance_frequency: как часто перебалансировать
        apply_variation: применять вариацию
    """
    num_sims, num_bets = outcomes.shape
    
    # Категоризация odds: low (<2), med (2-3.5), high (>3.5)
    odds_categories = np.where(odds_array < 2.0, 0, np.where(odds_array < 3.5, 1, 2))
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Веса категорий (можно динамически корректировать, но для простоты фиксируем)
    weights = np.array([0.4, 0.4, 0.2])  # Low, Med, High odds
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        cat = odds_categories[i]
        
        # Ставка зависит от категории и веса
        bet_pct = base_percent * weights[cat] / 0.3  # Нормализация
        
        bet_amount = current_bankroll * bet_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=23)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        bankroll[:, i + 1] = bankroll[:, i] + np.where(outcomes[:, i], bet_amount * (odds - 1), -bet_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        valid_mask = bankroll[:, i] > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / bankroll[valid_mask, i]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    min_bet_pct, max_bet_pct, avg_bet_pct = (np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)) if len(valid_bets) > 0 else (0, 0, 0)
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def ml_adaptive_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                         learning_rate=0.01, lookback=30, apply_variation=False):
    """
    ML Adaptive Strategy: адаптация на основе успешности предыдущих ставок.
    
    Упрощенная версия: корректируем ставку на основе win_rate последних N ставок
    с экспоненциальным сглаживанием (похоже на простое ML).
    
    Args:
        outcomes, odds_array: данные
        base_percent: базовая ставка
        learning_rate: скорость адаптации (0-1)
        lookback: окно для анализа
        apply_variation: применять вариацию
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Adaptive bet size per simulation
    adaptive_pct = np.full(num_sims, base_percent, dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        
        if i >= lookback:
            # Win rate последних lookback ставок
            recent_outcomes = outcomes[:, i-lookback:i]
            win_rate = np.mean(recent_outcomes, axis=1)
            
            # Expected win rate для нашего ROI
            expected_wr = TARGET_ROI + 1 / np.mean(odds_array)
            
            # Корректируем: если win_rate > expected → увеличиваем, иначе → уменьшаем
            performance = (win_rate - expected_wr) / expected_wr
            
            # Адаптация с learning rate
            adjustment = 1 + learning_rate * performance
            adaptive_pct = adaptive_pct * adjustment
            adaptive_pct = np.clip(adaptive_pct, base_percent * 0.5, base_percent * 2.0)
        
        bet_amount = current_bankroll * adaptive_pct / 100
        bet_history_ideal[:, i] = bet_amount
    
    if apply_variation:
        bet_history, _ = apply_realistic_stake_variation(bet_history_ideal, seed_offset=24)
    else:
        bet_history = bet_history_ideal
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    for i in range(num_bets):
        bet_amount = bet_history[:, i]
        odds = odds_array[i]
        bankroll[:, i + 1] = bankroll[:, i] + np.where(outcomes[:, i], bet_amount * (odds - 1), -bet_amount)
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        valid_mask = bankroll[:, i] > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / bankroll[valid_mask, i]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    min_bet_pct, max_bet_pct, avg_bet_pct = (np.min(valid_bets), np.max(valid_bets), np.mean(valid_bets)) if len(valid_bets) > 0 else (0, 0, 0)
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def dynamic_percentage_strategy_with_real_odds(outcomes, odds_array, bet_size_pct, apply_variation=False):
    """
    НАСТОЯЩАЯ Dynamic стратегия: меняет процент в зависимости от результатов!
    
    Логика:
    - Базовый процент = bet_size_pct
    - Если банк > 120% от начального → увеличиваем на 20% (агрессивнее)
    - Если банк < 80% от начального → уменьшаем на 30% (осторожнее)
    - Если банк < 60% от начального → уменьшаем на 50% (очень осторожно)
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        odds_array: numpy array (num_bets,) с коэффициентами
        bet_size_pct: базовый процент от текущего банка
        apply_variation: применять ли реалистичную вариацию размера ставок
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history_ideal = np.zeros((num_sims, num_bets), dtype=float)
    
    # Рассчитываем размеры ставок ДИНАМИЧЕСКИ в процессе симуляции
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        
        # Динамический модификатор в зависимости от состояния банка
        bank_ratio = current_bankroll / INITIAL_BANKROLL
        dynamic_modifier = np.ones(num_sims)
        
        # Если в плюсе - ставим больше
        dynamic_modifier = np.where(bank_ratio > 1.2, 1.2, dynamic_modifier)
        
        # Если в минусе - ставим меньше
        dynamic_modifier = np.where(bank_ratio < 0.8, 0.7, dynamic_modifier)
        dynamic_modifier = np.where(bank_ratio < 0.6, 0.5, dynamic_modifier)
        
        # Рассчитываем ставку с учетом модификатора
        actual_bet_pct = bet_size_pct * dynamic_modifier
        bet_amount = np.maximum(current_bankroll * actual_bet_pct / 100, 0)
        bet_history_ideal[:, i] = bet_amount
        
        # Симулируем результат СРАЗУ для следующей итерации
        odds = odds_array[i]
        win_amount = bet_amount * (odds - 1)
        loss_amount = bet_amount
        bankroll[:, i + 1] = current_bankroll + np.where(outcomes[:, i], win_amount, -loss_amount)
    
    # Применяем вариацию если нужно
    if apply_variation:
        bet_history, variation_factors = apply_realistic_stake_variation(bet_history_ideal)
        
        # Пересимулируем с вариациями
        bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
        for i in range(num_bets):
            current_bankroll = bankroll[:, i]
            bet_amount = bet_history[:, i]
            odds = odds_array[i]
            
            win_amount = bet_amount * (odds - 1)
            loss_amount = bet_amount
            
            bankroll[:, i + 1] = current_bankroll + np.where(
                outcomes[:, i],
                win_amount,
                -loss_amount
            )
    else:
        bet_history = bet_history_ideal
    
    # Рассчитываем проценты от текущего банка
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll_before_bet = bankroll[:, i]
        valid_mask = current_bankroll_before_bet > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll_before_bet[valid_mask]) * 100
    
    valid_bets = bet_pct_from_current[bet_pct_from_current > 0]
    if len(valid_bets) > 0:
        min_bet_pct = np.min(valid_bets)
        max_bet_pct = np.max(valid_bets)
        avg_bet_pct = np.mean(valid_bets)
    else:
        min_bet_pct = max_bet_pct = avg_bet_pct = 0
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def run_strategy_with_real_odds(strategy_name, outcomes, odds_array, apply_variation=False, **strategy_params):
    """Прогоняет стратегию с реальными коэффициентами."""
    variation_suffix = "_with_variation" if apply_variation else ""
    
    # Формируем имя стратегии
    if strategy_name == 'kelly_criterion':
        risk = strategy_params.get('risk', 2.0)
        kelly_fraction = strategy_params.get('kelly_fraction', 1.0)
        unique_name = f"{strategy_name}_r{risk}_f{kelly_fraction}{variation_suffix}"
        param_desc = f"risk={risk}, fraction={kelly_fraction}"
    elif strategy_name == 'dynamic_percentage':
        bet_size_pct = strategy_params.get('bet_size_pct', 1.5)
        unique_name = f"{strategy_name}_{bet_size_pct}%{variation_suffix}"
        param_desc = f"bet_size={bet_size_pct}%"
    elif strategy_name in ['linear_roi', 'sqrt_roi', 'log_roi']:
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_br{base_roi}_bp{base_percent}_max{max_percent}{variation_suffix}"
        param_desc = f"base_roi={base_roi}, base_pct={base_percent}, max={max_percent}"
    elif strategy_name == 'constant_profit':
        target_profit_pct = strategy_params.get('target_profit_pct', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_tp{target_profit_pct}_max{max_percent}{variation_suffix}"
        param_desc = f"target_profit={target_profit_pct}%, max={max_percent}"
    elif strategy_name in ['combined_roi_odds', 'adaptive']:
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        unique_name = f"{strategy_name}_bp{base_percent}_max{max_percent}_roi{min_roi}-{max_roi}{variation_suffix}"
        param_desc = f"base={base_percent}%, max={max_percent}%, roi_range={min_roi}-{max_roi}"
    elif strategy_name == 'dynamic_kelly':
        risk = strategy_params.get('risk', 2.0)
        min_fraction = strategy_params.get('min_fraction', 0.1)
        max_fraction = strategy_params.get('max_fraction', 0.5)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        unique_name = f"{strategy_name}_r{risk}_f{min_fraction}-{max_fraction}_roi{min_roi}-{max_roi}{variation_suffix}"
        param_desc = f"risk={risk}, frac={min_fraction}-{max_fraction}, roi={min_roi}-{max_roi}"
    elif strategy_name == 'exponential_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        factor = strategy_params.get('factor', 0.1)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_br{base_roi}_bp{base_percent}_f{factor}_max{max_percent}{variation_suffix}"
        param_desc = f"base_roi={base_roi}, base_pct={base_percent}, factor={factor}, max={max_percent}"
    elif strategy_name == 'hybrid':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        roi_weight = strategy_params.get('roi_weight', 0.7)
        odds_weight = strategy_params.get('odds_weight', 0.3)
        unique_name = f"{strategy_name}_bp{base_percent}_max{max_percent}_w{roi_weight}-{odds_weight}{variation_suffix}"
        param_desc = f"base={base_percent}%, max={max_percent}%, weights={roi_weight}/{odds_weight}"
    elif strategy_name == 'linear_scaled':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_percent = strategy_params.get('min_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 7.0)
        unique_name = f"{strategy_name}_roi{min_roi}-{max_roi}_bet{min_percent}-{max_percent}{variation_suffix}"
        param_desc = f"roi=[{min_roi}-{max_roi}], bet=[{min_percent}-{max_percent}]%"
    elif strategy_name == 'linear_roi_odds':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        odds_penalty_factor = strategy_params.get('odds_penalty_factor', 0.7)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_br{base_roi}_bp{base_percent}_pen{odds_penalty_factor}_max{max_percent}{variation_suffix}"
        param_desc = f"base_roi={base_roi}, base%={base_percent}, penalty={odds_penalty_factor}, max={max_percent}"
    elif strategy_name == 'adaptive_constant_profit':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_target_pct = strategy_params.get('min_target_pct', 0.5)
        max_target_pct = strategy_params.get('max_target_pct', 3.0)
        max_bet_percent = strategy_params.get('max_bet_percent', 15.0)
        unique_name = f"{strategy_name}_roi{min_roi}-{max_roi}_t{min_target_pct}-{max_target_pct}_max{max_bet_percent}{variation_suffix}"
        param_desc = f"roi=[{min_roi}-{max_roi}], target=[{min_target_pct}-{max_target_pct}]%, max={max_bet_percent}"
    else:
        unique_name = f"{strategy_name}{variation_suffix}"
        param_desc = str(strategy_params)
    
    print(f"\n{'='*70}")
    print(f"СТРАТЕГИЯ: {unique_name}")
    print(f"Параметры: {param_desc}, Вариация: {'Да (30%-115%)' if apply_variation else 'Нет'}")
    print(f"{'='*70}")
    
    if strategy_name == 'dynamic_percentage':
        bet_size_pct = strategy_params.get('bet_size_pct', 1.5)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            dynamic_percentage_strategy_with_real_odds(outcomes, odds_array, bet_size_pct, apply_variation)
    elif strategy_name == 'kelly_criterion':
        risk = strategy_params.get('risk', 2.0)
        kelly_fraction = strategy_params.get('kelly_fraction', 1.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            kelly_criterion_strategy_with_real_odds(outcomes, odds_array, risk, kelly_fraction, apply_variation)
    elif strategy_name == 'linear_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            linear_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, apply_variation)
    elif strategy_name == 'sqrt_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            sqrt_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, apply_variation)
    elif strategy_name == 'log_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            log_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, apply_variation)
    elif strategy_name == 'constant_profit':
        target_profit_pct = strategy_params.get('target_profit_pct', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            constant_profit_strategy_with_real_odds(outcomes, odds_array, target_profit_pct, max_percent, apply_variation)
    elif strategy_name == 'combined_roi_odds':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            combined_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_percent, max_percent, 
                                                       min_roi, max_roi, min_odds, max_odds, apply_variation)
    elif strategy_name == 'adaptive':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            adaptive_strategy_with_real_odds(outcomes, odds_array, base_percent, max_percent, 
                                             min_roi, max_roi, min_odds, max_odds, apply_variation)
    elif strategy_name == 'dynamic_kelly':
        risk = strategy_params.get('risk', 2.0)
        min_fraction = strategy_params.get('min_fraction', 0.1)
        max_fraction = strategy_params.get('max_fraction', 0.5)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            dynamic_kelly_strategy_with_real_odds(outcomes, odds_array, risk, min_fraction, max_fraction,
                                                  min_roi, max_roi, apply_variation)
    elif strategy_name == 'exponential_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        factor = strategy_params.get('factor', 0.1)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            exponential_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, factor, max_percent, apply_variation)
    elif strategy_name == 'hybrid':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        roi_weight = strategy_params.get('roi_weight', 0.7)
        odds_weight = strategy_params.get('odds_weight', 0.3)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            hybrid_strategy_with_real_odds(outcomes, odds_array, base_percent, max_percent,
                                          min_roi, max_roi, min_odds, max_odds, roi_weight, odds_weight, apply_variation)
    elif strategy_name == 'linear_scaled':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_percent = strategy_params.get('min_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 7.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            linear_scaled_strategy_with_real_odds(outcomes, odds_array, min_roi, max_roi, min_percent, max_percent, apply_variation)
    elif strategy_name == 'linear_roi_odds':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        odds_penalty_factor = strategy_params.get('odds_penalty_factor', 0.7)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            linear_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, 
                                                     odds_penalty_factor, min_odds, max_odds, apply_variation)
    elif strategy_name == 'adaptive_constant_profit':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_target_pct = strategy_params.get('min_target_pct', 0.5)
        max_target_pct = strategy_params.get('max_target_pct', 3.0)
        max_bet_percent = strategy_params.get('max_bet_percent', 15.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            adaptive_constant_profit_strategy_with_real_odds(outcomes, odds_array, min_roi, max_roi,
                                                             min_target_pct, max_target_pct, max_bet_percent, apply_variation)
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy_name}")
    
    metrics = calculate_metrics_with_odds(bankroll_history, bet_history, odds_array)
    
    print(f"\n📊 SIZING:")
    print(f"  Средняя ставка: {avg_bet_pct:.2f}% от текущего банка")
    print(f"  Min: {min_bet_pct:.2f}%, Max: {max_bet_pct:.2f}%")
    
    print(f"\n💰 РЕЗУЛЬТАТЫ:")
    print(f"  ROI с оборота: {metrics['avg_roi_from_turnover']:>7.2f}%")
    print(f"  Средняя прибыль: {metrics['avg_profit_pct']:>8.2f}%")
    
    print(f"\n⚠️  РИСКИ:")
    print(f"  Слито: {metrics['bankrupt_pct']:.2f}%")
    print(f"  DD>20%: {metrics['drawdown_20_pct']:.2f}%")
    print(f"  DD>50%: {metrics['drawdown_50_pct']:.2f}%")
    print(f"  DD>80%: {metrics['drawdown_80_pct']:.2f}%")
    
    # Описание стратегии
    if strategy_name == 'kelly_criterion':
        risk = strategy_params.get('risk', 2.0)
        kelly_fraction = strategy_params.get('kelly_fraction', 1.0)
        description = (f"Kelly Criterion: risk={risk}, fraction={kelly_fraction}. "
                      f"Формула: f = (b×p - q) / b / risk × fraction, где p из ROI={TARGET_ROI*100:.1f}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией размера ставки 30%-115% от расчетного.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'dynamic_percentage':
        bet_size_pct = strategy_params.get('bet_size_pct', 1.5)
        description = (f"Dynamic Percentage: {bet_size_pct}% от текущего банка. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией размера ставки 30%-115% от расчетного.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'linear_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Linear ROI: bet% = {base_percent} × (ROI/{base_roi}), max {max_percent}%. "
                      f"ROI={TARGET_ROI*100:.1f}%. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'sqrt_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Square Root ROI: bet% = {base_percent} × √(ROI/{base_roi}), max {max_percent}%. "
                      f"ROI={TARGET_ROI*100:.1f}%. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'log_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Logarithmic ROI: bet% = {base_percent} × log(ROI/{base_roi} + 1), max {max_percent}%. "
                      f"ROI={TARGET_ROI*100:.1f}%. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'constant_profit':
        target_profit_pct = strategy_params.get('target_profit_pct', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Constant Profit: ставка = {target_profit_pct}% / (odds-1), max {max_percent}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'combined_roi_odds':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        description = (f"Combined ROI-Odds: bet% = {base_percent} × √(norm_ROI) × (1-0.5×norm_odds), max {max_percent}%. "
                      f"ROI range [{min_roi}-{max_roi}]. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'adaptive':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Adaptive: Combined с адаптивными модификаторами при просадках. "
                      f"Base {base_percent}%, max {max_percent}%. DD>20%: ×0.75, DD>40%: ×0.5. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'dynamic_kelly':
        risk = strategy_params.get('risk', 2.0)
        min_fraction = strategy_params.get('min_fraction', 0.1)
        max_fraction = strategy_params.get('max_fraction', 0.5)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        description = (f"Dynamic Kelly: Kelly с динамической фракцией [{min_fraction}-{max_fraction}] по ROI[{min_roi}-{max_roi}]. "
                      f"Risk={risk}. При высоком ROI агрессивнее. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'exponential_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        factor = strategy_params.get('factor', 0.1)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Exponential ROI: bet% = {base_percent} × exp({factor} × (ROI-{base_roi})), max {max_percent}%. "
                      f"Очень агрессивная при высоких ROI. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'hybrid':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        roi_weight = strategy_params.get('roi_weight', 0.7)
        odds_weight = strategy_params.get('odds_weight', 0.3)
        description = (f"Hybrid: взвешенная комбинация norm_ROI и norm_odds, веса {roi_weight}/{odds_weight}. "
                      f"Base {base_percent}%, max {max_percent}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'linear_scaled':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_percent = strategy_params.get('min_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 7.0)
        description = (f"Linear Scaled: прямая интерполяция ROI[{min_roi}-{max_roi}] → bet[{min_percent}-{max_percent}]%. "
                      f"Простейший линейный mapping. ROI={TARGET_ROI*100:.1f}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'linear_roi_odds':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        odds_penalty_factor = strategy_params.get('odds_penalty_factor', 0.7)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Linear ROI-Odds: bet = {base_percent} × (ROI/{base_roi}) × (1 - {odds_penalty_factor}×norm_odds), max {max_percent}%. "
                      f"Линейная по ROI с коррекцией на odds. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'adaptive_constant_profit':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_target_pct = strategy_params.get('min_target_pct', 0.5)
        max_target_pct = strategy_params.get('max_target_pct', 3.0)
        max_bet_percent = strategy_params.get('max_bet_percent', 15.0)
        description = (f"Adaptive Constant Profit: целевая прибыль [{min_target_pct}-{max_target_pct}]% по ROI[{min_roi}-{max_roi}], max_bet {max_bet_percent}%. "
                      f"bet = target / (odds-1). "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    else:
        description = f"{strategy_name} with params {strategy_params}"
    
    result = {
        'strategy_name': unique_name,
        'base_strategy': strategy_name,
        'strategy_params': strategy_params,
        'with_variation': apply_variation,
        'description': description,
        'avg_bet_pct': avg_bet_pct,
        'min_bet_pct': min_bet_pct,
        'max_bet_pct': max_bet_pct,
        **metrics
    }
    
    return result

def save_results_to_csv(result, filename='results.csv'):
    """Сохраняет результат в CSV (append режим)."""
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'strategy', 'with_variation', 'avg_bet_%', 'min_bet_%', 'max_bet_%',
                'roi_%', 'avg_profit_%', 'min_profit_%', 'max_profit_%',
                'bankrupt_%', 'dd>20_%', 'dd>50_%', 'dd>80_%',
                'avg_maxdd_%', 'worst_dd_%',
                'timestamp', 'description'
            ])
        
        # Правильная обработка with_variation (может быть bool или string)
        with_var = result['with_variation']
        if isinstance(with_var, str):
            with_var_str = with_var  # Уже строка 'Yes' или 'No'
        else:
            with_var_str = 'Yes' if with_var else 'No'  # Преобразуем bool
        
        writer.writerow([
            result['strategy_name'],
            with_var_str,
            f"{result['avg_bet_pct']:.2f}",
            f"{result['min_bet_pct']:.2f}",
            f"{result['max_bet_pct']:.2f}",
            f"{result['avg_roi_from_turnover']:.2f}",
            f"{result['avg_profit_pct']:.2f}",
            f"{result['min_profit_pct']:.2f}",
            f"{result['max_profit_pct']:.2f}",
            f"{result['bankrupt_pct']:.2f}",
            f"{result['drawdown_20_pct']:.2f}",
            f"{result['drawdown_50_pct']:.2f}",
            f"{result['drawdown_80_pct']:.2f}",
            f"{result['avg_max_drawdown_pct']:.2f}",
            f"{result['worst_drawdown_pct']:.2f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            result['description']
        ])
    
    print(f"✅ Результаты добавлены в {filename}")

if __name__ == '__main__':
    print("Загрузка симуляций с реальными коэффициентами...")
    outcomes, odds_array = load_real_odds_outcomes()
    
    print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
    print(f"Диапазон коэффициентов: {odds_array.min():.2f} - {odds_array.max():.2f}")
    
    # Тестируем dynamic_percentage 1.5% с и без вариации
    strategies_to_test = [
        ('dynamic_percentage', 1.5, False),
        ('dynamic_percentage', 1.5, True),
    ]
    
    for strategy_name, bet_size, apply_variation in strategies_to_test:
        result = run_strategy_with_real_odds(strategy_name, outcomes, odds_array, bet_size, apply_variation)
        save_results_to_csv(result)
    
    print("\n✅ Готово! Результаты в results_real_odds.csv")
