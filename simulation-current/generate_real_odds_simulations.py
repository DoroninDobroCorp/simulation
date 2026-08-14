import numpy as np
import pandas as pd
from config import NUM_SIMULATIONS, TARGET_ROI, RANDOM_SEED

def load_real_odds(filename='matching_teams_clean.csv'):
    """Загружает реальные коэффициенты из CSV."""
    df = pd.read_csv(filename)
    odds = df['coefficient'].values
    print(f"Загружено {len(odds)} реальных коэффициентов")
    print(f"Min коэф: {odds.min():.2f}, Max коэф: {odds.max():.2f}, Avg: {odds.mean():.2f}")
    return odds

def generate_bet_outcomes_with_real_odds(real_odds, num_simulations=NUM_SIMULATIONS, target_roi=TARGET_ROI):
    """
    Генерирует исходы ставок с реальными коэффициентами и заданным ROI.
    
    Args:
        real_odds: numpy array с реальными коэффициентами
        num_simulations: количество симуляций
        target_roi: целевой ROI (например, 0.07 для 7%)
    
    Returns:
        tuple: (outcomes, odds_per_bet)
            outcomes: numpy array (num_simulations, num_bets) с True/False
            odds_per_bet: numpy array (num_bets,) с коэффициентами для каждой ставки
    """
    np.random.seed(RANDOM_SEED)
    
    num_bets = len(real_odds)
    
    # Для каждого коэффициента рассчитываем вероятность выигрыша с учетом ROI
    # ROI = p × odds - 1
    # p = (1 + ROI) / odds
    win_probabilities = (1 + target_roi) / real_odds
    
    # Ограничиваем вероятности диапазоном [0, 1]
    win_probabilities = np.clip(win_probabilities, 0, 1)
    
    # Генерируем исходы для каждой симуляции
    outcomes = np.zeros((num_simulations, num_bets), dtype=bool)
    for i in range(num_bets):
        outcomes[:, i] = np.random.random(num_simulations) < win_probabilities[i]
    
    return outcomes, real_odds

def save_real_odds_outcomes(outcomes, odds, filename='bet_outcomes_real_odds.npz'):
    """Сохраняет исходы и коэффициенты."""
    np.savez_compressed(filename, outcomes=outcomes, odds=odds)
    print(f"\nСохранено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
    print(f"Размер файла: {outcomes.nbytes / 1024 / 1024:.2f} MB")
    
    # Проверяем средний ROI
    total_profit = 0
    total_turnover = 0
    for i in range(len(odds)):
        wins = np.sum(outcomes[:, i])
        total = outcomes.shape[0]
        profit = wins * (odds[i] - 1) - (total - wins)
        total_profit += profit
        total_turnover += total
    
    avg_roi = (total_profit / total_turnover) * 100
    print(f"Средний ROI с оборота: {avg_roi:.2f}% (ожидаемый: {TARGET_ROI*100:.2f}%)")

def load_real_odds_outcomes(filename='bet_outcomes_real_odds.npz'):
    """Загружает исходы и коэффициенты из файла."""
    data = np.load(filename)
    return data['outcomes'], data['odds']

if __name__ == '__main__':
    print("="*70)
    print("ГЕНЕРАЦИЯ СИМУЛЯЦИЙ С РЕАЛЬНЫМИ КОЭФФИЦИЕНТАМИ")
    print("="*70)
    
    real_odds = load_real_odds()
    
    print(f"\nГенерация {NUM_SIMULATIONS} симуляций с реальными коэффициентами...")
    print(f"Целевой ROI: {TARGET_ROI*100:.2f}%")
    
    outcomes, odds = generate_bet_outcomes_with_real_odds(real_odds)
    save_real_odds_outcomes(outcomes, odds)
    
    print("\n✅ Готово!")
