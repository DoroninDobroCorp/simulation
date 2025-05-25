import numpy as np
import math

def calculate_kelly_bet(odds, roi, bank, risk=2.0, kelly_fraction=1.0):
    """
    Классическая стратегия Келли.
    Ограничивает ставку максимумом 8% от банка.
    """
    if roi < 0:
        return 0
    roi = min(roi, 8.0)
    edge_decimal = roi / 100.0
    capped_odds = min(odds, 2.5)
    p = (1 / capped_odds) * (1 + edge_decimal)
    q = 1 - p
    adjusted_risk = risk * 1.1
    p = p / adjusted_risk
    b = capped_odds - 1
    kelly_stake = 0
    if p > 0 and b > 0:
        kelly_stake = (b * p - q) / b
    kelly_stake *= kelly_fraction
    kelly_stake = max(0, kelly_stake)
    max_stake_percent = 0.08  # максимум 8% от банка
    kelly_stake = min(kelly_stake, max_stake_percent)
    if kelly_stake < 0.001:
        return 0
    bet_size = kelly_stake * bank
    bet_size = min(bet_size, bank * max_stake_percent)
    return bet_size

def calculate_linear_roi_bet(odds, roi, bank, base_roi=5.0, base_percent=1.0, max_percent=10.0):
    """
    Линейная стратегия ROI.
    Ставка пропорциональна отношению ROI к базовому значению.
    """
    if roi <= 0:
        return 0
    bet_percent = base_percent * (roi / base_roi)
    bet_percent = min(bet_percent, max_percent)
    return bet_percent * bank / 100

def calculate_sqrt_roi_bet(odds, roi, bank, base_roi=5.0, base_percent=1.0, max_percent=10.0):
    """
    Стратегия с квадратным корнем от ROI.
    Делает ставку более консервативной при низком ROI.
    """
    if roi <= 0:
        return 0
    bet_percent = base_percent * np.sqrt(roi / base_roi)
    bet_percent = min(bet_percent, max_percent)
    return bet_percent * bank / 100

def calculate_log_roi_bet(odds, roi, bank, base_roi=5.0, base_percent=1.0, max_percent=10.0):
    """
    Логарифмическая стратегия ROI.
    Снижает рост ставки при увеличении ROI.
    """
    if roi <= 0:
        return 0
    log_ratio = np.log(roi / base_roi + 1)
    bet_percent = base_percent * log_ratio
    bet_percent = min(bet_percent, max_percent)
    return bet_percent * bank / 100

def calculate_constant_profit_bet(odds, roi, bank, target_profit_percent=2.0):
    """
    Стратегия с постоянной прибылью.
    Размер ставки определяется как целевая прибыль, делённая на (odds - 1).
    """
    if roi <= 0 or odds <= 1.0:
        return 0
    target_profit = (target_profit_percent / 100) * bank
    bet_size = target_profit / (odds - 1.0)
    max_bet = 0.1 * bank  # не более 10% от банка
    return min(bet_size, max_bet)

def calculate_combined_roi_odds_bet(odds, roi, bank, min_odds, max_odds, min_roi, max_roi,
                                   base_percent=1.0, max_percent=10.0):
    """
    Комбинированная ROI-Odds стратегия.
    Нормализует ROI и коэффициент и комбинирует их для расчёта ставки.
    """
    if roi <= 0 or odds <= 1:
        return 0
    normalized_roi = min(1.0, max(0.0, (roi - min_roi) / (max_roi - min_roi)))
    normalized_odds = min(1.0, max(0.0, (odds - min_odds) / (max_odds - min_odds)))
    roi_factor = np.sqrt(normalized_roi)
    odds_factor = 1.0 - 0.5 * normalized_odds
    combined_factor = roi_factor * odds_factor
    bet_percent = base_percent + (max_percent - base_percent) * combined_factor
    return bet_percent * bank / 100

def calculate_adaptive_bet(odds, roi, bank, initial_bank, max_bank, min_odds, max_odds, min_roi, max_roi,
                           base_percent=1.0, max_percent=10.0):
    """
    Адаптивная стратегия.
    Начинается как комбинированная стратегия, но корректируется в зависимости от динамики банка.
    """
    base_bet = calculate_combined_roi_odds_bet(
        odds, roi, bank, min_odds, max_odds, min_roi, max_roi,
        base_percent, max_percent
    )
    bank_ratio = bank / max_bank if max_bank > 0 else 1
    if bank_ratio < 0.8:
        base_bet *= 0.75
    if bank_ratio < 0.6:
        base_bet *= 0.5
    initial_ratio = bank / initial_bank if initial_bank > 0 else 1
    if initial_ratio < 0.7:
        base_bet = min(base_bet, bank * max_percent * 0.5 / 100)
    if initial_ratio < 0.6:
        base_bet = min(base_bet, bank * max_percent * 0.25 / 100)
    return base_bet

def calculate_dynamic_kelly_bet(odds, roi, bank, risk=2.0, min_fraction=0.1, max_fraction=0.5,
                                min_roi=3.0, max_roi=20.0):
    """
    Dynamic Kelly – вариант Келли с динамической фракцией, зависящей от ROI.
    """
    if roi <= 0:
        return 0
    edge_decimal = roi / 100
    log_factor = 1 - (1 / (odds / (1 + edge_decimal)))
    if log_factor <= 0:
        return 0
    bet_size_percent = np.log10(log_factor) / np.log10(np.power(10, -risk))
    if bet_size_percent < 0 or bet_size_percent > 1:
        return 0
    if max_roi > min_roi:
        roi_factor = min(1.0, max(0.0, (roi - min_roi) / (max_roi - min_roi)))
    else:
        roi_factor = 0
    roi_fraction = min_fraction + (max_fraction - min_fraction) * roi_factor
    bet_size_percent *= roi_fraction
    return bet_size_percent * bank

def calculate_exp_roi_bet(odds, roi, bank, base_roi=5.0, base_percent=1.0, max_percent=7.0, factor=0.1):
    """
    Экспоненциальная ROI стратегия.
    Ставка растёт экспоненциально с ростом ROI относительно базового значения.
    """
    if roi <= 0:
        return 0
    bet_percent = base_percent * math.exp(factor * (roi - base_roi))
    bet_percent = min(bet_percent, max_percent)
    return bet_percent * bank / 100

def calculate_hybrid_bet(odds, roi, bank, base_percent=1.0, max_percent=5.0, 
                         roi_weight=0.7, odds_weight=0.3, 
                         min_roi=3.0, max_roi=20.0, min_odds=1.5, max_odds=10.0):
    """
    Гибридная стратегия.
    Комбинирует нормализованные показатели ROI и коэффициента с использованием весов.
    """
    if roi <= 0:
        return 0
    norm_roi = 0.5
    if max_roi > min_roi:
        norm_roi = min(1.0, max(0.0, (roi - min_roi) / (max_roi - min_roi)))
    norm_odds = 0.5
    if max_odds > min_odds:
        norm_odds = 1 - min(1.0, max(0.0, (odds - min_odds) / (max_odds - min_odds)))
    combined_factor = roi_weight * norm_roi + odds_weight * norm_odds
    bet_percent = base_percent + (max_percent - base_percent) * combined_factor
    return bet_percent * bank / 100

def calculate_linear_scaled_bet(odds, roi, bank, min_roi=3.0, max_roi=20.0, min_percent=1.0, max_percent=7.0):
    """
    Линейно масштабируемая стратегия.
    Прямолинейное отображение ROI в диапазон процентов ставки.
    """
    if roi <= 0:
        return 0
    roi = max(min(roi, max_roi), min_roi)
    stake_percent = min_percent
    if max_roi > min_roi:
        stake_percent = min_percent + (max_percent - min_percent) * (roi - min_roi) / (max_roi - min_roi)
    return stake_percent * bank / 100

def calculate_linear_roi_odds_bet(odds, roi, bank, base_roi=5.0, base_percent=1.0, max_percent=10.0,
                                  min_odds=1.5, max_odds=3.5):
    """
    Линейная ROI-Odds стратегия.
    Сначала рассчитывается линейный фактор от ROI, затем корректируется с учетом нормализованных odds.
    """
    if roi <= 0:
        return 0
    roi_factor = roi / base_roi
    odds_factor = 1.0
    if odds > min_odds and max_odds > min_odds:
        normalized_odds = min(1.0, (odds - min_odds) / (max_odds - min_odds))
        odds_factor = 1.0 - (0.7 * normalized_odds)
    bet_percent = base_percent * roi_factor * odds_factor
    bet_percent = min(bet_percent, max_percent)
    return bet_percent * bank / 100

def calculate_adaptive_constant_profit_bet(odds, roi, bank, min_roi=0.0, max_roi=20.0, 
                                           min_profit_percent=1.0, max_profit_percent=7.0):
    """
    Адаптивная стратегия постоянной прибыли.
    Целевая прибыль (в процентах от банка) зависит от ROI.
    """
    if roi <= 0 or odds <= 1.0:
        return 0
    roi = max(min(roi, max_roi), min_roi)
    target_profit_percent = min_profit_percent + (max_profit_percent - min_profit_percent) * (roi - min_roi) / (max_roi - min_roi)
    target_profit = target_profit_percent * bank / 100
    bet_size = target_profit / (odds - 1.0)
    return bet_size