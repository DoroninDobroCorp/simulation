"""
Модуль для генерации распределений коэффициентов и ROI.

Генерирует:
- Коэффициенты (odds) в диапазоне от 1.5 до 3.5 со смещением вправо, средний кэф 2.8
- ROI на основе смешанного распределения с тремя диапазонами:
  - Обычный: 3-6% (вероятность 85%)
  - Средний: 6-12% (вероятность 10%)
  - Редкий: 12-20% (вероятность 5%)
"""

import numpy as np
from scipy import stats


class DistributionGenerator:
    """
    Класс для генерации распределений коэффициентов и ROI.
    """
    
    def __init__(self, 
                 odds_min=1.5, 
                 odds_max=3.5, 
                 odds_mean=2.8,
                 normal_roi_range=(3.0, 6.0),
                 medium_roi_range=(6.0, 12.0),
                 rare_roi_range=(12.0, 20.0),
                 roi_weights=(0.85, 0.10, 0.05)):
        """
        Инициализация генератора распределений.
        
        Args:
            odds_min: Минимальное значение коэффициента
            odds_max: Максимальное значение коэффициента
            odds_mean: Среднее значение коэффициента (смещение)
            normal_roi_range: Диапазон обычного ROI (низкий)
            medium_roi_range: Диапазон среднего ROI
            rare_roi_range: Диапазон редкого ROI (высокий)
            roi_weights: Весовые коэффициенты для каждого диапазона ROI
        """
        self.odds_min = odds_min
        self.odds_max = odds_max
        self.odds_mean = odds_mean
        
        self.normal_roi_range = normal_roi_range
        self.medium_roi_range = medium_roi_range
        self.rare_roi_range = rare_roi_range
        self.roi_weights = roi_weights
        
        # Настройка распределения для коэффициентов
        # Используем бета-распределение для создания смещенного распределения
        odds_range = odds_max - odds_min
        # ИСПРАВЛЕНО: Beta(2, 2.2) давала средний кэф 2.45 вместо 2.8. Параметры
        # выводятся из нужного среднего: mean(Beta(a, b)) = a / (a + b).
        # При [1.5, 3.5] и среднем 2.8 это Beta(2.6, 1.4) — масса смещена вправо.
        target = (odds_mean - odds_min) / odds_range
        if not 0 < target < 1:
            raise ValueError("odds_mean должен лежать строго между odds_min и odds_max")
        concentration = 4.0
        self.alpha = target * concentration
        self.beta = (1 - target) * concentration
        # Коэффициент масштабирования для достижения среднего значения 2.8
        self.scale_factor = odds_range
        self.location = odds_min

    def generate_odds(self, size=1):
        """
        Генерирует коэффициенты ставок со смещением вправо.
        
        Args:
            size: Количество генерируемых коэффициентов
            
        Returns:
            Массив коэффициентов
        """
        # Генерация из бета-распределения
        beta_samples = np.random.beta(self.alpha, self.beta, size=size)
        # Масштабирование к нужному диапазону
        odds = self.location + beta_samples * self.scale_factor
        
        return odds
    
    def generate_roi(self, size=1):
        """
        Генерирует значения ROI на основе смешанного распределения.
        
        Args:
            size: Количество генерируемых значений ROI
            
        Returns:
            Массив значений ROI
        """
        # Генерация категорий (какой диапазон ROI выбрать) на основе весов
        categories = np.random.choice(3, size=size, p=self.roi_weights)
        
        # Инициализация массива ROI
        roi_values = np.zeros(size)
        
        # Заполнение значений ROI в зависимости от категории
        for i, category in enumerate(categories):
            if category == 0:  # Обычный ROI (3-6%)
                roi_values[i] = np.random.uniform(
                    self.normal_roi_range[0], 
                    self.normal_roi_range[1]
                )
            elif category == 1:  # Средний ROI (6-12%)
                roi_values[i] = np.random.uniform(
                    self.medium_roi_range[0], 
                    self.medium_roi_range[1]
                )
            else:  # Редкий ROI (12-20%)
                roi_values[i] = np.random.uniform(
                    self.rare_roi_range[0], 
                    self.rare_roi_range[1]
                )
        
        return roi_values


# Пример использования
if __name__ == "__main__":
    generator = DistributionGenerator()
    
    # Генерируем 1000 коэффициентов и ROI для демонстрации
    odds_sample = generator.generate_odds(1000)
    roi_sample = generator.generate_roi(1000)
    
    print(f"Коэффициенты (odds): Среднее = {np.mean(odds_sample):.2f}, "
          f"Мин = {np.min(odds_sample):.2f}, Макс = {np.max(odds_sample):.2f}")
    
    print(f"ROI: Среднее = {np.mean(roi_sample):.2f}%, "
          f"Мин = {np.min(roi_sample):.2f}%, Макс = {np.max(roi_sample):.2f}%")
    
    # Распределение по диапазонам ROI
    normal_roi = len([r for r in roi_sample if 3.0 <= r <= 6.0])
    medium_roi = len([r for r in roi_sample if 6.0 < r <= 12.0])
    rare_roi = len([r for r in roi_sample if r > 12.0])
    
    print(f"Распределение ROI: "
          f"Обычный (3-6%): {normal_roi/10:.1f}%, "
          f"Средний (6-12%): {medium_roi/10:.1f}%, "
          f"Редкий (12-20%): {rare_roi/10:.1f}%") 