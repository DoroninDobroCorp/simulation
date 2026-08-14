import numpy as np
from config import NUM_SIMULATIONS, BETS_PER_SIMULATION, WIN_RATE, RANDOM_SEED

def generate_bet_outcomes():
    """
    Генерирует фиксированные исходы ставок для всех симуляций.
    
    Returns:
        numpy.ndarray: Массив формы (NUM_SIMULATIONS, BETS_PER_SIMULATION)
                      где True = выигрыш, False = проигрыш
    """
    np.random.seed(RANDOM_SEED)
    
    outcomes = np.random.random((NUM_SIMULATIONS, BETS_PER_SIMULATION)) < WIN_RATE
    
    return outcomes

def save_outcomes(outcomes, filename='bet_outcomes.npy'):
    """Сохраняет исходы в компактный numpy формат."""
    np.save(filename, outcomes)
    print(f"Сохранено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
    print(f"Размер файла: {outcomes.nbytes / 1024 / 1024:.2f} MB")
    print(f"Средняя проходимость: {outcomes.mean():.4f} (ожидаемая: {WIN_RATE:.4f})")

def load_outcomes(filename='bet_outcomes.npy'):
    """Загружает исходы из файла."""
    return np.load(filename)

if __name__ == '__main__':
    print(f"Генерация {NUM_SIMULATIONS} симуляций по {BETS_PER_SIMULATION} ставок...")
    print(f"Параметры: win_rate = {WIN_RATE:.4f} ({WIN_RATE*100:.2f}%)")
    
    outcomes = generate_bet_outcomes()
    save_outcomes(outcomes)
