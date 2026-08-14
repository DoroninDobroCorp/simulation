"""
Модуль для симуляции серии ставок с учётом выигрышей/проигрышей.

Симулирует серию ставок с отслеживанием ключевых показателей:
- Итоговый банк
- Прирост банка (процентное изменение от начального)
- Средний размер ставки и процент от начального банка
- Максимальная просадка от максимума и от начального банка
"""

import numpy as np
from distribution_generator import DistributionGenerator
import importlib
import bet_strategies
import matplotlib.pyplot as plt


class BetSimulator:
    """
    Класс для симуляции серии ставок.
    """
    
    def __init__(self, 
                 initial_bank=10000, 
                 strategy_func=None,
                 strategy_params=None,
                 distribution_generator=None,
                 max_drawdown_pct=80,
                 min_bank_pct=20):
        """
        Инициализация симулятора ставок.
        
        Args:
            initial_bank: Начальный банк
            strategy_func: Функция стратегии расчета ставки
            strategy_params: Параметры стратегии
            distribution_generator: Генератор распределений для коэффициентов и ROI
            max_drawdown_pct: Максимальная просадка в % от пика (для досрочного завершения)
            min_bank_pct: Минимальный % банка от пика (для досрочного завершения)
        """
        self.initial_bank = initial_bank
        self.bank = initial_bank
        self.strategy_func = strategy_func
        self.strategy_params = strategy_params or {}
        self.distribution_generator = distribution_generator or DistributionGenerator()
        self.max_drawdown_pct = max_drawdown_pct
        self.min_bank_pct = min_bank_pct
        
        # История изменения банка и ставок
        self.bank_history = [initial_bank]
        self.bet_sizes = []
        self.odds_history = []
        self.roi_history = []
        self.win_loss_history = []
        
        # Метрики
        self.max_bank = initial_bank
        self.max_drawdown_from_peak = 0
        self.max_drawdown_from_initial = 0
        self.early_stop = False
        self.early_stop_reason = ""
        
    def reset(self):
        """
        Сброс симулятора к начальному состоянию.
        """
        self.bank = self.initial_bank
        self.bank_history = [self.initial_bank]
        self.bet_sizes = []
        self.odds_history = []
        self.roi_history = []
        self.win_loss_history = []
        self.max_bank = self.initial_bank
        self.max_drawdown_from_peak = 0
        self.max_drawdown_from_initial = 0
        self.early_stop = False
        self.early_stop_reason = ""
    
    def set_strategy(self, strategy_func, strategy_params=None):
        """
        Установка стратегии расчета ставок.
        
        Args:
            strategy_func: Функция стратегии
            strategy_params: Параметры стратегии
        """
        self.strategy_func = strategy_func
        self.strategy_params = strategy_params or {}
    
    def calculate_bet_size(self, odds, roi):
        """
        Расчет размера ставки с использованием выбранной стратегии.
        
        Args:
            odds: Коэффициент ставки
            roi: ROI ставки
            
        Returns:
            Размер ставки
        """
        # Если стратегия не задана, ставка составляет 1% от банка
        if self.strategy_func is None:
            return self.bank * 0.01
        
        # Передаем параметры в стратегию
        params = {
            'odds': odds,
            'roi': roi,
            'bank': self.bank,
            **self.strategy_params
        }
        
        # Вызываем функцию стратегии с распакованными параметрами
        bet_size = self.strategy_func(**params)
        
        # Ограничиваем размер ставки банком
        bet_size = min(bet_size, self.bank)
        
        return bet_size
    
    def simulate_bet(self, odds=None, roi=None):
        """
        Симуляция одной ставки.
        
        Args:
            odds: Коэффициент ставки (если None, генерируется)
            roi: ROI ставки (если None, генерируется)
            
        Returns:
            Результат ставки (выигрыш или проигрыш)
        """
        if odds is None:
            odds = self.distribution_generator.generate_odds(1)[0]
        
        if roi is None:
            roi = self.distribution_generator.generate_roi(1)[0]
        
        # Сохраняем историю коэффициентов и ROI
        self.odds_history.append(odds)
        self.roi_history.append(roi)
        
        # Рассчитываем размер ставки
        bet_size = self.calculate_bet_size(odds, roi)
        self.bet_sizes.append(bet_size)
        
        # Определяем вероятность выигрыша на основе ROI и коэффициента
        # ROI = (win_prob * odds - 1) * 100
        # Решая относительно win_prob: win_prob = (1 + ROI/100) / odds
        # Это корректная формула для расчёта вероятности из ROI
        win_probability = (1 + roi / 100) / odds
        
        # Ограничиваем вероятность диапазоном [0, 1] для безопасности
        win_probability = max(0.0, min(1.0, win_probability))
        
        # Определяем, выиграла ли ставка
        is_win = np.random.random() < win_probability
        self.win_loss_history.append(is_win)
        
        # Обновляем банк
        if is_win:
            profit = bet_size * (odds - 1)
            self.bank += profit
        else:
            self.bank -= bet_size
        
        # Обновляем историю банка
        self.bank_history.append(self.bank)
        
        # Обновляем максимальный банк
        if self.bank > self.max_bank:
            self.max_bank = self.bank
        
        # Вычисляем текущую просадку от пика
        if self.max_bank > 0:
            current_drawdown_from_peak = (self.max_bank - self.bank) / self.max_bank * 100
            self.max_drawdown_from_peak = max(self.max_drawdown_from_peak, current_drawdown_from_peak)
        
        # Вычисляем текущую просадку от начального банка
        if self.initial_bank > 0:
            current_drawdown_from_initial = (self.initial_bank - self.bank) / self.initial_bank * 100
            self.max_drawdown_from_initial = max(self.max_drawdown_from_initial, current_drawdown_from_initial)
        
        # Проверяем, не нужно ли досрочно завершить симуляцию
        if self.bank <= 0:
            self.early_stop = True
            self.early_stop_reason = "Банк равен нулю или меньше"
        elif self.max_bank > 0 and self.bank / self.max_bank * 100 < self.min_bank_pct:
            self.early_stop = True
            self.early_stop_reason = f"Банк упал ниже {self.min_bank_pct}% от пика"
        
        return is_win
    
    def simulate_series(self, num_bets=1500):
        """
        Симуляция серии ставок.
        
        Args:
            num_bets: Количество ставок для симуляции
            
        Returns:
            Словарь с результатами симуляции
        """
        self.reset()
        
        for _ in range(num_bets):
            self.simulate_bet()
            
            # Проверка на досрочное завершение
            if self.early_stop:
                break
        
        # Вычисление метрик
        final_bank = self.bank
        bank_growth_pct = (final_bank - self.initial_bank) / self.initial_bank * 100
        
        avg_bet_size = np.mean(self.bet_sizes) if self.bet_sizes else 0
        avg_bet_pct = avg_bet_size / self.initial_bank * 100
        
        num_wins = sum(self.win_loss_history)
        num_losses = len(self.win_loss_history) - num_wins
        win_rate = num_wins / len(self.win_loss_history) if self.win_loss_history else 0
        
        # Формирование результата
        results = {
            'initial_bank': self.initial_bank,
            'final_bank': final_bank,
            'bank_growth_pct': bank_growth_pct,
            'avg_bet_size': avg_bet_size,
            'avg_bet_pct': avg_bet_pct,
            'max_drawdown_from_peak': self.max_drawdown_from_peak,
            'max_drawdown_from_initial': self.max_drawdown_from_initial,
            'num_bets': len(self.win_loss_history),
            'win_rate': win_rate,
            'early_stop': self.early_stop,
            'early_stop_reason': self.early_stop_reason,
            'bank_history': self.bank_history.copy(),
            'bet_sizes': self.bet_sizes.copy(),
            'odds_history': self.odds_history.copy(),
            'roi_history': self.roi_history.copy(),
            'win_loss_history': self.win_loss_history.copy(),
        }
        
        return results
    
    def run_multiple_simulations(self, num_simulations=500, num_bets=1500):
        """
        Запуск множественных симуляций и агрегация результатов.
        
        Args:
            num_simulations: Количество симуляций
            num_bets: Количество ставок в каждой симуляции
            
        Returns:
            Агрегированные результаты симуляций
        """
        all_results = []
        
        for _ in range(num_simulations):
            results = self.simulate_series(num_bets)
            all_results.append(results)
        
        # Агрегация ключевых метрик
        final_banks = [r['final_bank'] for r in all_results]
        bank_growths = [r['bank_growth_pct'] for r in all_results]
        max_drawdowns_from_peak = [r['max_drawdown_from_peak'] for r in all_results]
        max_drawdowns_from_initial = [r['max_drawdown_from_initial'] for r in all_results]
        early_stops = [r['early_stop'] for r in all_results]
        
        # Подсчет просадок, превышающих пороговые значения
        drawdowns_over_50_pct = sum(1 for d in max_drawdowns_from_peak if d > 50)
        drawdowns_over_80_pct = sum(1 for d in max_drawdowns_from_peak if d > 80)
        
        # Формирование агрегированного результата
        aggregated_results = {
            'num_simulations': num_simulations,
            'avg_final_bank': np.mean(final_banks),
            'median_final_bank': np.median(final_banks),
            'min_final_bank': min(final_banks),
            'max_final_bank': max(final_banks),
            'avg_bank_growth_pct': np.mean(bank_growths),
            'median_bank_growth_pct': np.median(bank_growths),
            'avg_max_drawdown_from_peak': np.mean(max_drawdowns_from_peak),
            'median_max_drawdown_from_peak': np.median(max_drawdowns_from_peak),
            'avg_max_drawdown_from_initial': np.mean(max_drawdowns_from_initial),
            'pct_drawdowns_over_50': drawdowns_over_50_pct / num_simulations * 100,
            'pct_drawdowns_over_80': drawdowns_over_80_pct / num_simulations * 100,
            'pct_early_stops': sum(early_stops) / num_simulations * 100,
            'all_results': all_results
        }
        
        return aggregated_results
    
    def plot_bank_history(self, results=None):
        """
        Построение графика истории изменения банка.
        
        Args:
            results: Результаты симуляции (если None, используется последняя симуляция)
        """
        if results is None:
            bank_history = self.bank_history
            initial_bank = self.initial_bank
        else:
            bank_history = results['bank_history']
            initial_bank = results['initial_bank']
        
        plt.figure(figsize=(12, 6))
        plt.plot(bank_history, label='Банк')
        plt.axhline(y=initial_bank, color='r', linestyle='--', label='Начальный банк')
        plt.xlabel('Номер ставки')
        plt.ylabel('Банк')
        plt.title('История изменения банка')
        plt.legend()
        plt.grid(True)
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Получение функции стратегии из модуля bet_strategies
    strategy_func = bet_strategies.calculate_kelly_bet
    
    # Параметры стратегии
    strategy_params = {'risk': 2.0, 'kelly_fraction': 0.5}
    
    # Создание симулятора
    simulator = BetSimulator(
        initial_bank=10000,
        strategy_func=strategy_func,
        strategy_params=strategy_params
    )
    
    # Запуск серии ставок
    results = simulator.simulate_series(num_bets=100)
    
    # Вывод результатов
    print(f"Начальный банк: {results['initial_bank']}")
    print(f"Итоговый банк: {results['final_bank']:.2f}")
    print(f"Прирост банка: {results['bank_growth_pct']:.2f}%")
    print(f"Средний размер ставки: {results['avg_bet_size']:.2f}")
    print(f"Максимальная просадка от пика: {results['max_drawdown_from_peak']:.2f}%")
    print(f"Максимальная просадка от начального банка: {results['max_drawdown_from_initial']:.2f}%")
    
    # Построение графика истории банка
    simulator.plot_bank_history(results) 