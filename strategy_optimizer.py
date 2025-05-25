"""
Модуль для оптимизации параметров стратегий.

Выполняет подбор оптимальных параметров для различных стратегий 
с учетом разных уровней риска (консервативный, осторожный, сбалансированный, 
рискованный, экстремальный).
"""

import numpy as np
import bet_strategies
from bet_simulator import BetSimulator
from distribution_generator import DistributionGenerator
import inspect
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import random
import gc
import json
import os


# Определение риск-профилей
RISK_PROFILES = {
    'Conservative': {'pct_drawdowns_over_50': 0.0, 'pct_drawdowns_over_80': 0.0},
    'Cautious': {'pct_drawdowns_over_50': 3.5, 'pct_drawdowns_over_80': 0.0},
    'Balanced': {'pct_drawdowns_over_50': 10.0, 'pct_drawdowns_over_80': 0.0},
    'Risky': {'pct_drawdowns_over_50': float('inf'), 'pct_drawdowns_over_80': 5.0},
    'Crazy': {'pct_drawdowns_over_50': float('inf'), 'pct_drawdowns_over_80': 25.0},
    'Extreme': {'pct_drawdowns_over_50': float('inf'), 'pct_drawdowns_over_80': 50.0}
}


class StrategyOptimizer:
    """
    Класс для оптимизации параметров стратегий ставок.
    """
    
    def __init__(self, 
                 initial_bank=10000, 
                 num_bets=1500, 
                 num_simulations=500,
                 distribution_generator=None):
        """
        Инициализация оптимизатора стратегий.
        
        Args:
            initial_bank: Начальный банк
            num_bets: Количество ставок в каждой симуляции
            num_simulations: Количество симуляций для оценки метрик
            distribution_generator: Генератор распределений для коэффициентов и ROI
        """
        self.initial_bank = initial_bank
        self.num_bets = num_bets
        self.num_simulations = num_simulations
        self.distribution_generator = distribution_generator or DistributionGenerator()
        
        # Копируем глобальную переменную RISK_PROFILES в атрибут класса
        self.RISK_PROFILES = RISK_PROFILES
        
        # Словари для хранения лучших параметров для каждой стратегии и профиля риска
        self.best_params = {}
    
    def get_strategy_functions(self):
        """
        Получает список функций стратегий из модуля bet_strategies.
        
        Returns:
            Словарь с именами функций и самими функциями
        """
        strategy_funcs = {}
        
        # Получаем все функции из модуля bet_strategies
        for name, func in inspect.getmembers(bet_strategies, inspect.isfunction):
            if name.startswith('calculate_'):
                strategy_funcs[name] = func
        
        return strategy_funcs
    
    def get_parameter_ranges(self, strategy_name, risk_profile_name=None):
        """
        Определяет диапазоны параметров для оптимизации в зависимости от стратегии.
        
        Args:
            strategy_name: Имя стратегии
            risk_profile_name: Имя профиля риска (не используется в новом подходе)
            
        Returns:
            Словарь с диапазонами параметров для оптимизации
        """
        # Базовые диапазоны параметров для каждой стратегии
        parameter_ranges = {
            'calculate_kelly_bet': {
                'risk': (1.0, 6.0),
                'kelly_fraction': (0.1, 2.0)
            },
            'calculate_linear_roi_bet': {
                'base_roi': (1.0, 12.0),
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 30.0)
            },
            'calculate_sqrt_roi_bet': {
                'base_roi': (1.0, 12.0),
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 30.0)
            },
            'calculate_log_roi_bet': {
                'base_roi': (1.0, 12.0),
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 30.0)
            },
            'calculate_constant_profit_bet': {
                'target_profit_percent': (0.3, 10.0)
            },
            'calculate_combined_roi_odds_bet': {
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 30.0),
                'min_odds': (1.2, 2.0),
                'max_odds': (3.0, 6.0),
                'min_roi': (1.0, 6.0),
                'max_roi': (10.0, 35.0)
            },
            'calculate_adaptive_bet': {
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 30.0),
                'min_odds': (1.2, 2.0),
                'max_odds': (3.0, 6.0),
                'min_roi': (1.0, 6.0),
                'max_roi': (10.0, 35.0)
            },
            'calculate_dynamic_kelly_bet': {
                'risk': (1.0, 6.0),
                'min_fraction': (0.05, 0.5),
                'max_fraction': (0.3, 2.0),
                'min_roi': (1.0, 6.0),
                'max_roi': (10.0, 35.0)
            },
            'calculate_exp_roi_bet': {
                'base_roi': (1.0, 12.0),
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 25.0),
                'factor': (0.05, 0.6)
            },
            'calculate_hybrid_bet': {
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 25.0),
                'roi_weight': (0.3, 0.97),
                'odds_weight': (0.03, 0.7),
                'min_roi': (1.0, 6.0),
                'max_roi': (10.0, 35.0),
                'min_odds': (1.2, 2.0),
                'max_odds': (3.0, 6.0)
            },
            'calculate_linear_scaled_bet': {
                'min_roi': (1.0, 6.0),
                'max_roi': (10.0, 35.0),
                'min_percent': (0.1, 4.0),
                'max_percent': (2.0, 25.0)
            },
            'calculate_linear_roi_odds_bet': {
                'base_roi': (1.0, 12.0),
                'base_percent': (0.1, 4.0),
                'max_percent': (2.0, 30.0),
                'min_odds': (1.2, 2.0),
                'max_odds': (3.0, 6.0)
            },
            'calculate_adaptive_constant_profit_bet': {
                'min_roi': (0.0, 6.0),
                'max_roi': (10.0, 35.0),
                'min_profit_percent': (0.1, 4.0),
                'max_profit_percent': (2.0, 20.0)
            }
        }
        
        return parameter_ranges.get(strategy_name, {})
    
    def adaptive_parameter_tuning(self, strategy_func, parameter_name, initial_value, lower_bound, upper_bound, 
                              fixed_params=None, target_metric='roi', target_value=None, tolerance=0.5,
                              risk_profile_name=None):
        """
        Адаптивная настройка параметра для стратегии.
        
        Args:
            strategy_func: Функция стратегии
            parameter_name: Имя настраиваемого параметра
            initial_value: Начальное значение параметра
            lower_bound: Нижняя граница диапазона поиска
            upper_bound: Верхняя граница диапазона поиска
            fixed_params: Фиксированные параметры стратегии
            target_metric: Название целевой метрики
            target_value: Целевое значение метрики
            tolerance: Допустимое отклонение от целевого значения
            risk_profile_name: Имя профиля риска
            
        Returns:
            Кортеж (оптимальное значение параметра, результаты симуляции)
        """
        # Эта функция больше не используется в новом подходе
        pass
    
    def optimize_strategy(self, strategy_name, risk_profile_name, fixed_params=None):
        """
        Оптимизирует параметры стратегии для соответствия профилю риска.
        
        Args:
            strategy_name: Имя стратегии для оптимизации
            risk_profile_name: Имя профиля риска
            fixed_params: Фиксированные параметры стратегии (опционально)
            
        Returns:
            Кортеж (оптимальные параметры, метрики результатов)
        """
        # Эта функция больше не используется в новом подходе
        pass
    
    def run_optimization(self, strategies_to_optimize=None, risk_profiles_to_optimize=None, 
                    progress_callback=None, log_callback=None, intermediate_save_path='optimization_progress.json'):
        """
        Запускает оптимизацию выбранных стратегий с широким перебором параметров.
        
        Args:
            strategies_to_optimize: Список имен стратегий для оптимизации (если None, то все стратегии)
            risk_profiles_to_optimize: Список имен профилей риска для оптимизации (если None, то все профили)
            progress_callback: Функция обратного вызова для отображения прогресса
            log_callback: Функция обратного вызова для логирования
            intermediate_save_path: Путь для сохранения промежуточных результатов
            
        Returns:
            Словарь с результатами оптимизации
        """
        # Получаем все функции стратегий
        strategy_funcs = self.get_strategy_functions()
        
        # Определяем, какие стратегии оптимизировать
        if strategies_to_optimize is None:
            strategies_to_optimize = list(strategy_funcs.keys())
        else:
            # Фильтруем только существующие стратегии
            strategies_to_optimize = [s for s in strategies_to_optimize if s in strategy_funcs]
        
        # Определяем, какие профили риска оптимизировать
        if risk_profiles_to_optimize is None:
            risk_profiles_to_optimize = list(self.RISK_PROFILES.keys())
        else:
            # Фильтруем только существующие профили
            risk_profiles_to_optimize = [p for p in risk_profiles_to_optimize if p in self.RISK_PROFILES]
        
        if log_callback:
            log_callback(f"Оптимизация для профилей: {', '.join(risk_profiles_to_optimize)}")
            log_callback(f"Оптимизация для стратегий: {', '.join(strategies_to_optimize)}")
        
        # Функция для обработки прогресса
        def update_progress(message):
            if progress_callback:
                try:
                    progress_callback(message)
                except Exception as e:
                    print(f"Ошибка при вызове progress_callback: {e}")
        
        # Словарь для хранения всех результатов оптимизации
        optimization_results = {}
        
        # Проверяем, есть ли промежуточные результаты
        previously_completed_strategies = set()
        if os.path.exists(intermediate_save_path):
            try:
                with open(intermediate_save_path, 'r', encoding='utf-8') as f:
                    saved_results = json.load(f)
                    
                # Загружаем ранее сохраненные результаты
                optimization_results = saved_results.get('optimization_results', {})
                previously_completed_strategies = set(saved_results.get('completed_strategies', []))
                
                if log_callback:
                    log_callback(f"Загружены промежуточные результаты для {len(previously_completed_strategies)} стратегий")
                update_progress(f"Загружены результаты для {len(previously_completed_strategies)} стратегий")
            except Exception as e:
                if log_callback:
                    log_callback(f"Ошибка при загрузке промежуточных результатов: {e}")
                # В случае ошибки начинаем с нуля
                optimization_results = {}
                previously_completed_strategies = set()
        
        # Для каждой стратегии генерируем и тестируем случайные наборы параметров
        num_strategies = len(strategies_to_optimize)
        for strategy_idx, strategy_name in enumerate(strategies_to_optimize):
            # Пропускаем стратегии, которые уже были обработаны
            if strategy_name in previously_completed_strategies:
                if log_callback:
                    log_callback(f"Стратегия {strategy_name} уже обработана, пропускаем")
                update_progress(f"Пропускаем {strategy_name} (уже обработана) [{strategy_idx+1}/{num_strategies}]")
                continue
            
            update_progress(f"Генерация параметров для {strategy_name} [{strategy_idx+1}/{num_strategies}]...")
            
            if log_callback:
                log_callback(f"Начинаем оптимизацию стратегии {strategy_name}")
            
            # Получаем функцию стратегии
            strategy_func = strategy_funcs[strategy_name]
            
            # Получаем базовые диапазоны параметров
            parameter_ranges = self.get_parameter_ranges(strategy_name)
            if not parameter_ranges:
                if log_callback:
                    log_callback(f"Нет диапазонов параметров для стратегии {strategy_name}")
                continue
            
            # Список для хранения всех результатов симуляций для этой стратегии
            strategy_results = []
            
            # Генерируем 100 случайных наборов параметров для широкого охвата
            num_parameter_sets = 100
            for param_set_idx in range(num_parameter_sets):
                update_progress(f"Симуляция {strategy_name}: {param_set_idx+1}/{num_parameter_sets}")
                
                # Генерируем случайные параметры в пределах диапазонов
                params = {}
                for param_name, (min_val, max_val) in parameter_ranges.items():
                    # Для некоторых параметров лучше использовать логарифмическое распределение
                    if param_name in ['risk', 'kelly_fraction', 'factor']:
                        # Логарифмическое распределение для более равномерного покрытия разных порядков величины
                        import math
                        log_min = math.log(max(min_val, 0.001))  # Защита от log(0)
                        log_max = math.log(max_val)
                        log_val = log_min + random.random() * (log_max - log_min)
                        params[param_name] = math.exp(log_val)
                    else:
                        # Равномерное распределение для большинства параметров
                        params[param_name] = min_val + random.random() * (max_val - min_val)
                
                # Специальная обработка для стратегии calculate_adaptive_bet
                if strategy_name == 'calculate_adaptive_bet':
                    params['initial_bank'] = self.initial_bank
                    params['max_bank'] = self.initial_bank * 10
                
                # Выполняем симуляцию с этими параметрами
                try:
                    results = self.simulate_with_params(strategy_func, params)
                    if results:
                        # Сохраняем параметры вместе с результатами для последующего анализа
                        simulation_data = {
                            'strategy_name': strategy_name,
                            'params': params,
                            'metrics': results
                        }
                        strategy_results.append(simulation_data)
                except Exception as e:
                    if log_callback:
                        log_callback(f"Ошибка при симуляции {strategy_name} с параметрами {params}: {e}")
            
            # Теперь для каждого профиля риска выбираем наилучшие параметры из тех, что мы сгенерировали
            optimization_results[strategy_name] = {}
            
            for risk_profile_name in risk_profiles_to_optimize:
                if log_callback:
                    log_callback(f"Анализ результатов для стратегии {strategy_name}, профиль {risk_profile_name}")
                
                # Получаем целевые ограничения для профиля риска
                risk_profile = self.RISK_PROFILES[risk_profile_name]
                max_pct_drawdowns_over_50 = risk_profile.get('pct_drawdowns_over_50', float('inf'))
                max_pct_drawdowns_over_80 = risk_profile.get('pct_drawdowns_over_80', float('inf'))
                
                # Отбираем симуляции, соответствующие профилю риска
                valid_simulations = []
                for sim in strategy_results:
                    metrics = sim['metrics']
                    
                    # Проверяем соответствие ограничениям профиля риска
                    # Conservative и Cautious строго требуют соблюдения ограничений
                    is_strict_profile = risk_profile_name in ['Conservative', 'Cautious']
                    
                    # Для Conservative и Cautious требуем точного соответствия
                    if is_strict_profile:
                        if metrics['pct_drawdowns_over_50'] > max_pct_drawdowns_over_50:
                            continue
                        if metrics['pct_drawdowns_over_80'] > max_pct_drawdowns_over_80:
                            continue
                    else:
                        # Для остальных профилей допускаем небольшое отклонение
                        # Но сильно штрафуем за несоответствие целевым значениям
                        if metrics['pct_drawdowns_over_50'] > max_pct_drawdowns_over_50 * 1.1 and max_pct_drawdowns_over_50 < float('inf'):
                            continue
                        if metrics['pct_drawdowns_over_80'] > max_pct_drawdowns_over_80 * 1.1 and max_pct_drawdowns_over_80 < float('inf'):
                            continue
                    
                    # Расчет балла качества для профиля риска
                    score = 0
                    
                    # Учитываем ROI (чем выше, тем лучше)
                    score += metrics['roi']
                    
                    # Штрафуем за отклонение от целевых значений просадок
                    # Для профилей Risky, Crazy и Extreme хотим приблизиться к целевому значению просадок
                    if risk_profile_name in ['Risky', 'Crazy', 'Extreme']:
                        if max_pct_drawdowns_over_80 < float('inf'):
                            target_diff = abs(metrics['pct_drawdowns_over_80'] - max_pct_drawdowns_over_80)
                            score -= target_diff * 5  # Штраф за отклонение
                    
                    # Для Conservative, Cautious и Balanced хотим минимизировать просадки
                    else:
                        score -= metrics['pct_drawdowns_over_50'] * 2
                        score -= metrics['pct_drawdowns_over_80'] * 5
                    
                    # Штрафуем за досрочные остановки
                    score -= metrics['pct_early_stops'] * 2
                    
                    # Добавляем симуляцию с рассчитанным баллом
                    valid_simulations.append((sim, score))
                
                # Если нашли хотя бы одну подходящую симуляцию
                if valid_simulations:
                    # Сортируем по баллу (от большего к меньшему)
                    valid_simulations.sort(key=lambda x: x[1], reverse=True)
                    
                    # Берем лучший результат
                    best_sim, best_score = valid_simulations[0]
                    
                    # Формируем результат
                    optimization_results[strategy_name][risk_profile_name] = {
                        'strategy_name': strategy_name,
                        'risk_profile': risk_profile_name,
                        'optimal_params': best_sim['params'],
                        'metrics': best_sim['metrics']
                    }
                    
                    if log_callback:
                        log_callback(f"Для стратегии {strategy_name}, профиль {risk_profile_name}:")
                        log_callback(f"  Найдены оптимальные параметры: {best_sim['params']}")
                        log_callback(f"  ROI: {best_sim['metrics']['roi']:.2f}%")
                        log_callback(f"  Макс. просадка: {best_sim['metrics']['avg_max_drawdown_from_peak']:.2f}%")
                        log_callback(f"  Просадки >50%: {best_sim['metrics']['pct_drawdowns_over_50']:.1f}%")
                        log_callback(f"  Просадки >80%: {best_sim['metrics']['pct_drawdowns_over_80']:.1f}%")
                else:
                    if log_callback:
                        log_callback(f"Для стратегии {strategy_name}, профиль {risk_profile_name} не найдено подходящих параметров")
            
            # После обработки стратегии, сохраняем промежуточные результаты и очищаем память
            # Добавляем стратегию в список завершенных
            previously_completed_strategies.add(strategy_name)
            
            # Создаем словарь для сохранения
            save_data = {
                'optimization_results': optimization_results,
                'completed_strategies': list(previously_completed_strategies)
            }
            
            # Сохраняем промежуточные результаты
            try:
                with open(intermediate_save_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, default=str)
                
                if log_callback:
                    log_callback(f"Промежуточные результаты сохранены в {intermediate_save_path}")
            except Exception as e:
                if log_callback:
                    log_callback(f"Ошибка при сохранении промежуточных результатов: {e}")
            
            # Очищаем уже ненужные данные из памяти
            strategy_results.clear()
            gc.collect()  # Явно вызываем сборщик мусора
            
            if log_callback:
                log_callback(f"Завершена обработка стратегии {strategy_name} [{strategy_idx+1}/{num_strategies}]")
        
        # Сохраняем результаты оптимизации
        self.best_params = {}
        for strategy_name, profiles in optimization_results.items():
            self.best_params[strategy_name] = profiles
        
        update_progress("Оптимизация завершена")
        return optimization_results
    
    def generate_optimization_report(self):
        """
        Генерирует отчет о результатах оптимизации.
        
        Returns:
            Строка с отчетом
        """
        if not self.best_params:
            return "Оптимизация еще не выполнена."
        
        report = "Отчет об оптимизации параметров стратегий\n"
        report += "=" * 50 + "\n\n"
        
        # Для каждого профиля риска находим лучшие стратегии
        for profile_name in self.RISK_PROFILES.keys():
            report += f"Профиль риска: {profile_name}\n"
            report += "-" * 30 + "\n"
            
            # Собираем результаты для всех стратегий с данным профилем
            profile_results = []
            for strategy_name, profiles in self.best_params.items():
                if profile_name in profiles:
                    result = profiles[profile_name]
                    profile_results.append(result)
            
            # Сортируем по ROI (приросту банка)
            if profile_results:
                profile_results.sort(key=lambda x: x['metrics']['roi'], reverse=True)
                
                # Выводим топ-3 стратегии или все, если их меньше 3
                report += "Лучшие стратегии:\n"
                top_n = min(3, len(profile_results))
                for i, result in enumerate(profile_results[:top_n], 1):
                    strategy = result['strategy_name']
                    params = result['optimal_params']
                    metrics = result['metrics']
                    
                    # Округляем значения параметров для лучшей читаемости
                    readable_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, float):
                            readable_params[param_name] = round(param_value, 3)
                        else:
                            readable_params[param_name] = param_value
                    
                    report += f"{i}. {strategy}\n"
                    report += f"   Параметры: {readable_params}\n"
                    report += f"   ROI: {metrics['roi']:.2f}%\n"
                    report += f"   Средний итоговый банк: {metrics['avg_final_bank']:.2f}\n"
                    report += f"   Средняя макс. просадка от пика: {metrics['avg_max_drawdown_from_peak']:.2f}%\n"
                    report += f"   % симуляций с просадкой >50%: {metrics['pct_drawdowns_over_50']:.2f}%\n"
                    report += f"   % симуляций с просадкой >80%: {metrics['pct_drawdowns_over_80']:.2f}%\n"
                    report += f"   % досрочных остановок: {metrics['pct_early_stops']:.2f}%\n\n"
            else:
                report += "Нет результатов для данного профиля риска.\n\n"
        
        return report

    def generate_detailed_optimization_report(self):
        """
        Генерирует подробный отчет о результатах оптимизации для всех стратегий и профилей риска.
        
        Returns:
            Строка с подробным отчетом
        """
        if not self.best_params:
            return "Оптимизация еще не выполнена."
        
        report = "Подробный отчет об оптимизации параметров стратегий\n"
        report += "=" * 70 + "\n\n"
        
        # Для каждого профиля риска выводим все найденные стратегии
        for profile_name in self.RISK_PROFILES.keys():
            report += f"Профиль риска: {profile_name}\n"
            report += "=" * 50 + "\n"
            
            # Получаем ограничения профиля риска для справки
            risk_limits = self.RISK_PROFILES[profile_name]
            report += f"Ограничения профиля: "
            if risk_limits.get('pct_drawdowns_over_50', float('inf')) < float('inf'):
                report += f"просадки >50% макс. {risk_limits['pct_drawdowns_over_50']:.1f}%, "
            if risk_limits.get('pct_drawdowns_over_80', float('inf')) < float('inf'):
                report += f"просадки >80% макс. {risk_limits['pct_drawdowns_over_80']:.1f}%"
            report += "\n\n"
            
            # Собираем результаты для всех стратегий с данным профилем
            profile_results = []
            for strategy_name, profiles in self.best_params.items():
                if profile_name in profiles:
                    result = profiles[profile_name]
                    profile_results.append(result)
            
            # Сортируем по ROI (от большего к меньшему)
            if profile_results:
                profile_results.sort(key=lambda x: x['metrics']['roi'], reverse=True)
                
                # Выводим все стратегии
                for i, result in enumerate(profile_results, 1):
                    strategy = result['strategy_name']
                    params = result['optimal_params']
                    metrics = result['metrics']
                    
                    # Округляем значения параметров для лучшей читаемости
                    readable_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, float):
                            readable_params[param_name] = round(param_value, 3)
                        else:
                            readable_params[param_name] = param_value
                    
                    # Выводим порядковый номер, имя стратегии и соответствие целевым показателям
                    report += f"{i}. {strategy}"
                    
                    # Добавляем маркеры соответствия/несоответствия целевым значениям просадок
                    meets_drawdown_50 = True
                    meets_drawdown_80 = True
                    
                    if risk_limits.get('pct_drawdowns_over_50', float('inf')) < float('inf'):
                        meets_drawdown_50 = metrics['pct_drawdowns_over_50'] <= risk_limits['pct_drawdowns_over_50'] * 1.1
                    
                    if risk_limits.get('pct_drawdowns_over_80', float('inf')) < float('inf'):
                        if profile_name in ['Risky', 'Crazy', 'Extreme']:
                            # Для этих профилей мы хотим приблизиться к целевому значению, а не быть ниже него
                            target = risk_limits['pct_drawdowns_over_80']
                            actual = metrics['pct_drawdowns_over_80']
                            meets_drawdown_80 = abs(actual - target) <= target * 0.2  # В пределах 20% от целевого
                        else:
                            meets_drawdown_80 = metrics['pct_drawdowns_over_80'] <= risk_limits['pct_drawdowns_over_80'] * 1.1
                    
                    # Добавляем индикаторы соответствия целевым значениям
                    indicators = []
                    if not meets_drawdown_50:
                        indicators.append("⚠️ >50%")
                    if not meets_drawdown_80:
                        indicators.append("⚠️ >80%")
                    
                    if indicators:
                        report += f" {''.join(indicators)}\n"
                    else:
                        report += f" ✅\n"
                    
                    report += f"   Параметры: {readable_params}\n"
                    report += f"   ROI: {metrics['roi']:.2f}%\n"
                    report += f"   Средний итоговый банк: {metrics['avg_final_bank']:.2f}\n"
                    report += f"   Средняя макс. просадка от пика: {metrics['avg_max_drawdown_from_peak']:.2f}%\n"
                    report += f"   % симуляций с просадкой >50%: {metrics['pct_drawdowns_over_50']:.2f}%\n"
                    report += f"   % симуляций с просадкой >80%: {metrics['pct_drawdowns_over_80']:.2f}%\n"
                    report += f"   % досрочных остановок: {metrics['pct_early_stops']:.2f}%\n\n"
            else:
                report += "Нет результатов для данного профиля риска.\n\n"
            
            report += "\n" + "-" * 70 + "\n\n"
        
        return report

    def generate_report(self, report_type='summary', output_format='text'):
        """
        Генерирует отчет о результатах оптимизации в выбранном формате.
        
        Args:
            report_type: Тип отчета ('summary' для краткого или 'detailed' для подробного)
            output_format: Формат вывода ('text' для обычного текста)
            
        Returns:
            Строка с отчетом в выбранном формате
        """
        if report_type == 'detailed':
            report = self.generate_detailed_optimization_report()
        else:  # По умолчанию краткий отчет
            report = self.generate_optimization_report()
        
        # Для текущей версии поддерживается только текстовый формат
        # В будущем можно добавить поддержку JSON, CSV, HTML и т.д.
        return report

    def simulate_with_params(self, strategy_func, params):
        """
        Выполняет симуляцию с заданными параметрами.
        
        Args:
            strategy_func: Функция стратегии
            params: Параметры стратегии
            
        Returns:
            Словарь с результатами симуляции
        """
        try:
            # Убедимся, что все параметры имеют правильный тип
            converted_params = {}
            for key, value in params.items():
                if isinstance(value, str):
                    try:
                        if value.lower() in ['inf', 'infinity']:
                            converted_params[key] = float('inf')
                        else:
                            converted_params[key] = float(value)
                    except (ValueError, TypeError):
                        converted_params[key] = value  # Оставляем как есть, если не можем преобразовать
                else:
                    converted_params[key] = value
            
            # Создаем симулятор
            simulator = BetSimulator(
                initial_bank=self.initial_bank,
                strategy_func=strategy_func,
                strategy_params=converted_params,
                distribution_generator=self.distribution_generator
            )
            
            # Запускаем множественные симуляции
            results = simulator.run_multiple_simulations(
                num_bets=self.num_bets,
                num_simulations=self.num_simulations
            )
            
            # Дополнительная проверка и обработка результатов
            if results:
                # Добавляем расчетный ROI
                bank_growth_pct = results.get('avg_bank_growth_pct', 0)
                initial_investment = 100  # Представляем первоначальную инвестицию как 100%
                final_value = initial_investment + bank_growth_pct
                roi = ((final_value / initial_investment) - 1) * 100
                results['roi'] = roi
                
                # Дополнительные проверки для выявления недопустимых результатов
                if results.get('pct_early_stops', 0) > 95:  # Если более 95% симуляций завершились досрочно
                    results['roi'] = -100  # Штрафуем такие результаты
                
                # Проверяем, что все ключевые метрики являются числами
                for key in ['avg_final_bank', 'avg_bank_growth_pct', 'avg_max_drawdown_from_peak', 
                           'pct_drawdowns_over_50', 'pct_drawdowns_over_80', 'pct_early_stops', 'roi']:
                    if key in results and isinstance(results[key], str):
                        try:
                            results[key] = float(results[key])
                        except (ValueError, TypeError):
                            results[key] = 0  # Устанавливаем безопасное значение по умолчанию
            
            return results
        
        except Exception as e:
            print(f"Ошибка при симуляции с параметрами {params}: {str(e)}")
            return None


# Пример использования
if __name__ == "__main__":
    # Создание оптимизатора
    optimizer = StrategyOptimizer(
        initial_bank=10000,
        num_bets=100,  # Уменьшенное значение для быстрого примера
        num_simulations=50  # Уменьшенное значение для быстрого примера
    )
    
    # Запуск оптимизации для одной стратегии и одного профиля риска
    results = optimizer.run_optimization(
        strategies_to_optimize=['calculate_kelly_bet'],
        risk_profiles_to_optimize=['Balanced'],
        progress_callback=print
    )
    
    # Использование нового универсального метода для получения отчетов
    
    # Краткий отчет (топ-3 стратегии для каждого профиля риска)
    print("Генерация краткого отчета...")
    summary_report = optimizer.generate_report(report_type='summary')
    print(summary_report)
    
    # Полный отчет (все стратегии для всех профилей риска)
    print("\nГенерация полного отчета со всеми стратегиями...")
    detailed_report = optimizer.generate_report(report_type='detailed')
    
    # Пример: сохранение полного отчета в файл
    print("Сохранение полного отчета в файл 'detailed_optimization_report.txt'")
    with open('detailed_optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(detailed_report) 