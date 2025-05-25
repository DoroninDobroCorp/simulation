"""
Модуль для визуализации результатов симуляций и оптимизации.

Предоставляет функции для построения различных графиков:
- История изменения банка
- Сравнительные диаграммы стратегий
- Графики размеров ставок
- Визуализация распределений результатов
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class Visualizer:
    """
    Класс для визуализации результатов симуляций и оптимизации.
    """
    
    @staticmethod
    def plot_bank_history(bank_history, initial_bank=None, figsize=(10, 6), return_figure=False):
        """
        Построение графика истории изменения банка.
        
        Args:
            bank_history: Список с историей изменения банка
            initial_bank: Начальный банк (для отображения горизонтальной линии)
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(bank_history, label='Банк')
        
        if initial_bank is not None:
            ax.axhline(y=initial_bank, color='r', linestyle='--', label='Начальный банк')
        
        ax.set_xlabel('Номер ставки')
        ax.set_ylabel('Банк')
        ax.set_title('История изменения банка')
        ax.legend()
        ax.grid(True)
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_multiple_bank_histories(bank_histories, labels=None, initial_bank=None, 
                                   figsize=(10, 6), return_figure=False):
        """
        Построение графиков историй изменения банка для нескольких симуляций.
        
        Args:
            bank_histories: Список списков с историями изменения банка
            labels: Список меток для каждой истории
            initial_bank: Начальный банк (для отображения горизонтальной линии)
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is None:
            labels = [f'Симуляция {i+1}' for i in range(len(bank_histories))]
        
        for i, history in enumerate(bank_histories):
            ax.plot(history, label=labels[i])
        
        if initial_bank is not None:
            ax.axhline(y=initial_bank, color='r', linestyle='--', label='Начальный банк')
        
        ax.set_xlabel('Номер ставки')
        ax.set_ylabel('Банк')
        ax.set_title('Сравнение историй изменения банка')
        ax.legend()
        ax.grid(True)
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_bet_sizes(bet_sizes, figsize=(10, 6), return_figure=False):
        """
        Построение графика размеров ставок.
        
        Args:
            bet_sizes: Список размеров ставок
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(bet_sizes)
        ax.set_xlabel('Номер ставки')
        ax.set_ylabel('Размер ставки')
        ax.set_title('Размеры ставок')
        ax.grid(True)
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_roi_distribution(roi_values, figsize=(10, 6), return_figure=False):
        """
        Построение гистограммы распределения ROI.
        
        Args:
            roi_values: Список значений ROI
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Настраиваем количество бинов для лучшей визуализации
        num_bins = min(30, max(10, len(roi_values) // 50))
        
        ax.hist(roi_values, bins=num_bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('ROI (%)')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение ROI')
        ax.grid(True, alpha=0.3)
        
        # Добавляем вертикальные линии для разных диапазонов ROI
        ax.axvline(x=6.0, color='r', linestyle='--', label='Граница обычного/среднего ROI')
        ax.axvline(x=12.0, color='g', linestyle='--', label='Граница среднего/редкого ROI')
        ax.legend()
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_odds_distribution(odds_values, figsize=(10, 6), return_figure=False):
        """
        Построение гистограммы распределения коэффициентов.
        
        Args:
            odds_values: Список значений коэффициентов
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Настраиваем количество бинов для лучшей визуализации
        num_bins = min(30, max(10, len(odds_values) // 50))
        
        ax.hist(odds_values, bins=num_bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Коэффициент')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение коэффициентов')
        ax.grid(True, alpha=0.3)
        
        # Добавляем вертикальную линию для среднего значения
        mean_odds = np.mean(odds_values)
        ax.axvline(x=mean_odds, color='r', linestyle='--', 
                   label=f'Средний коэффициент: {mean_odds:.2f}')
        ax.legend()
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_final_bank_distribution(final_banks, initial_bank=None, figsize=(10, 6), return_figure=False):
        """
        Построение гистограммы распределения итоговых банков.
        
        Args:
            final_banks: Список итоговых банков
            initial_bank: Начальный банк (для отображения вертикальной линии)
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Настраиваем количество бинов для лучшей визуализации
        num_bins = min(30, max(10, len(final_banks) // 5))
        
        ax.hist(final_banks, bins=num_bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Итоговый банк')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение итоговых банков')
        ax.grid(True, alpha=0.3)
        
        # Добавляем вертикальные линии
        mean_bank = np.mean(final_banks)
        median_bank = np.median(final_banks)
        
        ax.axvline(x=mean_bank, color='r', linestyle='--', 
                   label=f'Средний банк: {mean_bank:.2f}')
        ax.axvline(x=median_bank, color='g', linestyle=':', 
                   label=f'Медианный банк: {median_bank:.2f}')
        
        if initial_bank is not None:
            ax.axvline(x=initial_bank, color='b', linestyle='-', 
                       label=f'Начальный банк: {initial_bank}')
        
        ax.legend()
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_drawdown_distribution(drawdowns, thresholds=[50, 80], figsize=(10, 6), return_figure=False):
        """
        Построение гистограммы распределения просадок.
        
        Args:
            drawdowns: Список максимальных просадок
            thresholds: Пороговые значения просадок для отображения вертикальных линий
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Настраиваем количество бинов для лучшей визуализации
        num_bins = min(30, max(10, len(drawdowns) // 5))
        
        ax.hist(drawdowns, bins=num_bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Максимальная просадка (%)')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение максимальных просадок')
        ax.grid(True, alpha=0.3)
        
        # Добавляем вертикальные линии для пороговых значений
        for threshold in thresholds:
            ax.axvline(x=threshold, color='r', linestyle='--', 
                       label=f'Порог {threshold}%')
            
            # Вычисляем процент просадок, превышающих порог
            pct_over_threshold = sum(1 for d in drawdowns if d > threshold) / len(drawdowns) * 100
            
            # Добавляем текст с процентом
            y_pos = ax.get_ylim()[1] * 0.9
            ax.text(threshold + 2, y_pos, f'{pct_over_threshold:.1f}% > {threshold}%', 
                    color='r', fontsize=10)
        
        ax.legend()
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_strategies_comparison(strategies_results, metric='bank_growth_pct', 
                                 figsize=(12, 8), return_figure=False):
        """
        Построение сравнительной диаграммы стратегий.
        
        Args:
            strategies_results: Словарь с результатами для разных стратегий
            metric: Метрика для сравнения ('bank_growth_pct', 'avg_max_drawdown_from_peak', и т.д.)
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        # Формируем данные для графика
        strategies = []
        metrics = []
        risk_profiles = []
        
        for strategy_name, risk_profile_results in strategies_results.items():
            for risk_profile, result in risk_profile_results.items():
                strategies.append(strategy_name)
                risk_profiles.append(risk_profile)
                metrics.append(result['metrics'][metric])
        
        # Создаем DataFrame для удобства построения графика
        df = pd.DataFrame({
            'Стратегия': strategies,
            'Профиль риска': risk_profiles,
            'Метрика': metrics
        })
        
        # Определяем названия метрик для подписей
        metric_names = {
            'bank_growth_pct': 'Прирост банка (%)',
            'avg_final_bank': 'Средний итоговый банк',
            'avg_max_drawdown_from_peak': 'Средняя макс. просадка от пика (%)',
            'pct_drawdowns_over_50': '% симуляций с просадкой >50%',
            'pct_drawdowns_over_80': '% симуляций с просадкой >80%',
            'pct_early_stops': '% досрочных остановок'
        }
        
        metric_name = metric_names.get(metric, metric)
        
        # Строим график
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(x='Стратегия', y='Метрика', hue='Профиль риска', data=df, ax=ax)
        
        ax.set_xlabel('Стратегия')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Сравнение стратегий по метрике: {metric_name}')
        
        # Поворачиваем подписи на оси X для лучшей читаемости
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if return_figure:
            return fig
        else:
            plt.show()
    
    @staticmethod
    def plot_risk_profiles_comparison(strategies_results, risk_profiles=None, metrics=None,
                                    figsize=(15, 10), return_figure=False):
        """
        Построение сравнительной диаграммы для профилей риска.
        
        Args:
            strategies_results: Словарь с результатами для разных стратегий
            risk_profiles: Список профилей риска для сравнения (если None, все профили)
            metrics: Список метрик для сравнения (если None, основные метрики)
            figsize: Размер графика (ширина, высота)
            return_figure: Возвращать фигуру вместо отображения графика
            
        Returns:
            Фигура с графиком, если return_figure=True
        """
        # Если профили не указаны, используем все имеющиеся
        all_profiles = set()
        for strategy_results in strategies_results.values():
            all_profiles.update(strategy_results.keys())
        
        if risk_profiles is None:
            risk_profiles = sorted(all_profiles)
        
        # Если метрики не указаны, используем основные
        if metrics is None:
            metrics = ['bank_growth_pct', 'avg_max_drawdown_from_peak', 'pct_drawdowns_over_50']
        
        # Определяем названия метрик для подписей
        metric_names = {
            'bank_growth_pct': 'Прирост банка (%)',
            'avg_final_bank': 'Средний итоговый банк',
            'avg_max_drawdown_from_peak': 'Средняя макс. просадка от пика (%)',
            'pct_drawdowns_over_50': '% симуляций с просадкой >50%',
            'pct_drawdowns_over_80': '% симуляций с просадкой >80%',
            'pct_early_stops': '% досрочных остановок'
        }
        
        # Определяем количество графиков
        num_metrics = len(metrics)
        
        # Создаем подграфики
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize)
        if num_metrics == 1:
            axes = [axes]  # Для одной метрики axes не будет списком
        
        for i, metric in enumerate(metrics):
            # Формируем данные для графика
            data = []
            
            for risk_profile in risk_profiles:
                for strategy_name, risk_profile_results in strategies_results.items():
                    if risk_profile in risk_profile_results:
                        result = risk_profile_results[risk_profile]
                        data.append({
                            'Стратегия': strategy_name,
                            'Профиль риска': risk_profile,
                            'Метрика': result['metrics'][metric]
                        })
            
            # Создаем DataFrame для удобства построения графика
            if data:
                df = pd.DataFrame(data)
                
                # Строим график
                sns.barplot(x='Профиль риска', y='Метрика', hue='Стратегия', data=df, ax=axes[i])
                
                axes[i].set_xlabel('Профиль риска')
                axes[i].set_ylabel(metric_names.get(metric, metric))
                axes[i].set_title(f'Сравнение профилей риска по метрике: {metric_names.get(metric, metric)}')
                
                # Настраиваем легенду для компактности
                axes[i].legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if return_figure:
            return fig
        else:
            plt.show()


# Фигура PyQt для отображения графиков в GUI
class MplCanvas(FigureCanvas):
    """
    Класс для встраивания matplotlib фигур в PyQt.
    """
    def __init__(self, fig=None, width=5, height=4, dpi=100):
        if fig is None:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
        else:
            self.fig = fig
            self.axes = self.fig.axes[0] if self.fig.axes else self.fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(self.fig)


# Пример использования
if __name__ == "__main__":
    # Генерируем тестовые данные
    bank_history = [10000] + [10000 + 1000 * np.sin(i / 10) for i in range(1, 101)]
    bet_sizes = [100 + 50 * np.sin(i / 5) for i in range(100)]
    roi_values = np.random.normal(5, 2, 1000)
    odds_values = np.random.normal(2.8, 0.5, 1000)
    
    # Тестируем различные графики
    Visualizer.plot_bank_history(bank_history, initial_bank=10000)
    Visualizer.plot_bet_sizes(bet_sizes)
    Visualizer.plot_roi_distribution(roi_values)
    Visualizer.plot_odds_distribution(odds_values) 