"""
Модуль с графическим интерфейсом пользователя для системы оптимизации банкролл-менеджмента.
"""

import os
import sys
import inspect
import json
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Настраиваем matplotlib для работы с PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, 
    QComboBox, QTextEdit, QProgressBar, QCheckBox, QGroupBox, QMessageBox,
    QListWidget, QScrollArea, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
import bet_strategies
from distribution_generator import DistributionGenerator
from bet_simulator import BetSimulator
from strategy_optimizer import StrategyOptimizer, RISK_PROFILES
from visualization import Visualizer

# Класс для отображения графиков matplotlib в PyQt
class MplCanvas(FigureCanvasQTAgg):
    """
    Класс для отображения графиков matplotlib в PyQt.
    """
    def __init__(self, width=5, height=4, dpi=100, fig=None):
        if fig is None:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
        else:
            self.fig = fig
            self.axes = fig.gca()
        
        super(MplCanvas, self).__init__(self.fig)

# Класс для выполнения симуляции в отдельном потоке
class SimulationWorker(QThread):
    """
    Рабочий поток для выполнения симуляции ставок.
    """
    finished = pyqtSignal(dict)  # Сигнал с результатами симуляции
    progress = pyqtSignal(int)   # Сигнал с прогрессом выполнения
    
    def __init__(self, initial_bank, strategy_func, strategy_params, num_simulations, num_bets):
        super().__init__()
        self.initial_bank = initial_bank
        self.strategy_func = strategy_func
        self.strategy_params = strategy_params
        self.num_simulations = num_simulations
        self.num_bets = num_bets
        
    def run(self):
        """
        Выполнение симуляций в отдельном потоке.
        """
        # Создание симулятора
        simulator = BetSimulator(
            initial_bank=self.initial_bank,
            strategy_func=self.strategy_func,
            strategy_params=self.strategy_params
        )
        
        # Запуск множественных симуляций с обновлением прогресса
        results = {}
        for i in range(self.num_simulations):
            sim_result = simulator.simulate_series(self.num_bets)
            
            # Обновление прогресса
            progress = int((i + 1) / self.num_simulations * 100)
            self.progress.emit(progress)
            
            # Если нужно прервать выполнение
            if self.isInterruptionRequested():
                break
        
        # Агрегация результатов множественных симуляций
        aggregated_results = simulator.run_multiple_simulations(
            num_simulations=self.num_simulations,
            num_bets=self.num_bets
        )
        
        # Отправка сигнала о завершении
        self.finished.emit(aggregated_results)


# Класс для выполнения оптимизации в отдельном потоке
class OptimizationWorker(QThread):
    """
    Рабочий поток для оптимизации параметров стратегий.
    """
    finished = pyqtSignal(dict)      # Сигнал с результатами оптимизации
    progress = pyqtSignal(int)       # Сигнал с прогрессом выполнения
    log_message = pyqtSignal(str)    # Сигнал с сообщением для лога
    
    def __init__(self, initial_bank, num_bets, num_simulations, 
                 strategies_to_optimize, risk_profiles_to_optimize):
        super().__init__()
        self.initial_bank = initial_bank
        self.num_bets = num_bets
        self.num_simulations = num_simulations
        self.strategies_to_optimize = strategies_to_optimize
        self.risk_profiles_to_optimize = risk_profiles_to_optimize
        
    def run(self):
        """
        Выполнение оптимизации в отдельном потоке.
        """
        # Создание оптимизатора
        optimizer = StrategyOptimizer(
            initial_bank=self.initial_bank,
            num_bets=self.num_bets,
            num_simulations=self.num_simulations
        )
        
        # Функция обратного вызова для логирования
        def progress_callback(message):
            self.log_message.emit(message)
        
        # Запуск оптимизации
        results = optimizer.run_optimization(
            strategies_to_optimize=self.strategies_to_optimize,
            risk_profiles_to_optimize=self.risk_profiles_to_optimize,
            progress_callback=progress_callback
        )
        
        # Отправка сигнала о завершении
        self.finished.emit(results)


class MainWindow(QMainWindow):
    """
    Главное окно приложения.
    """
    
    def __init__(self):
        super().__init__()
        
        # Настройка основного окна
        self.setWindowTitle("Система оптимизации банкролл-менеджмента")
        self.setMinimumSize(1024, 768)
        
        # Создание центрального виджета
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Создание основного макета
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Создание вкладок
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Создание вкладок для разных разделов приложения
        self.create_simulation_tab()
        self.create_optimization_tab()
        self.create_visualization_tab()
        
        # Инициализация данных
        self.distribution_generator = DistributionGenerator()
        self.simulation_results = None
        self.optimization_results = None
        
        # Инициализация рабочих потоков
        self.simulation_worker = None
        
    def create_simulation_tab(self):
        """
        Создание вкладки для симуляции ставок.
        """
        self.simulation_tab = QWidget()
        self.tabs.addTab(self.simulation_tab, "Симуляция")
        
        # Создание основного макета для вкладки
        layout = QVBoxLayout(self.simulation_tab)
        
        # Создание группы для параметров симуляции
        params_group = QGroupBox("Параметры симуляции")
        params_layout = QFormLayout()
        
        # Поле для начального банка
        self.initial_bank_input = QSpinBox()
        self.initial_bank_input.setRange(1000, 1000000)
        self.initial_bank_input.setValue(10000)
        self.initial_bank_input.setSingleStep(1000)
        params_layout.addRow("Начальный банк:", self.initial_bank_input)
        
        # Поле для количества ставок
        self.num_bets_input = QSpinBox()
        self.num_bets_input.setRange(100, 10000)
        self.num_bets_input.setValue(1500)
        self.num_bets_input.setSingleStep(100)
        params_layout.addRow("Количество ставок:", self.num_bets_input)
        
        # Поле для количества симуляций
        self.num_simulations_input = QSpinBox()
        self.num_simulations_input.setRange(10, 1000)
        self.num_simulations_input.setValue(500)
        self.num_simulations_input.setSingleStep(10)
        params_layout.addRow("Количество симуляций:", self.num_simulations_input)
        
        # Выбор стратегии
        self.strategy_combo = QComboBox()
        # Получение всех доступных стратегий из модуля bet_strategies
        self.strategy_functions = {}
        for name in dir(bet_strategies):
            if name.startswith('calculate_'):
                self.strategy_combo.addItem(name)
                self.strategy_functions[name] = getattr(bet_strategies, name)
        
        params_layout.addRow("Стратегия:", self.strategy_combo)
        
        # Группа для параметров стратегии
        self.strategy_params_group = QGroupBox("Параметры стратегии")
        self.strategy_params_layout = QFormLayout(self.strategy_params_group)
        
        # Словарь для полей ввода параметров
        self.param_inputs = {}
        
        # Обновление параметров при изменении стратегии
        self.strategy_combo.currentTextChanged.connect(self.update_strategy_params)
        
        # Завершение настройки группы параметров
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        layout.addWidget(self.strategy_params_group)
        
        # Добавление кнопок для запуска и остановки симуляции
        buttons_layout = QHBoxLayout()
        
        self.start_simulation_button = QPushButton("Запустить симуляцию")
        self.start_simulation_button.clicked.connect(self.start_simulation)
        buttons_layout.addWidget(self.start_simulation_button)
        
        self.stop_simulation_button = QPushButton("Остановить симуляцию")
        self.stop_simulation_button.clicked.connect(self.stop_simulation)
        self.stop_simulation_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_simulation_button)
        
        layout.addLayout(buttons_layout)
        
        # Прогресс-бар для отображения прогресса симуляции
        self.simulation_progress = QProgressBar()
        layout.addWidget(self.simulation_progress)
        
        # Группа для отображения результатов
        results_group = QGroupBox("Результаты симуляции")
        results_layout = QVBoxLayout(results_group)
        
        # Текстовое поле для вывода результатов
        self.simulation_results_text = QTextEdit()
        self.simulation_results_text.setReadOnly(True)
        results_layout.addWidget(self.simulation_results_text)
        
        # Добавление области с графиком банка
        self.bank_history_canvas = MplCanvas(width=8, height=4)
        results_layout.addWidget(self.bank_history_canvas)
        
        layout.addWidget(results_group)
        
        # Первоначальное обновление параметров стратегии
        self.update_strategy_params()
    
    def update_strategy_params(self):
        """
        Обновление полей для параметров выбранной стратегии.
        """
        # Очистка текущих полей
        for i in reversed(range(self.strategy_params_layout.count())):
            self.strategy_params_layout.itemAt(i).widget().deleteLater()
        
        self.param_inputs.clear()
        
        # Получение текущей стратегии
        strategy_name = self.strategy_combo.currentText()
        if not strategy_name:
            return
        
        # Получение функции стратегии
        strategy_func = self.strategy_functions.get(strategy_name)
        if not strategy_func:
            return
        
        # Анализ параметров функции стратегии
        sig = inspect.signature(strategy_func)
        for param_name, param in sig.parameters.items():
            # Пропускаем основные параметры (odds, roi, bank)
            if param_name in ['odds', 'roi', 'bank']:
                continue
            
            # Создаем элемент управления для параметра
            if param_name.endswith('_percent'):
                # Для параметров с процентами создаем QDoubleSpinBox с диапазоном 0-100
                param_input = QDoubleSpinBox()
                param_input.setRange(0.1, 100.0)
                param_input.setValue(1.0)
                param_input.setSingleStep(0.1)
                param_input.setSuffix('%')
            elif param_name == 'risk':
                # Для параметра риска создаем QDoubleSpinBox с диапазоном 1-10
                param_input = QDoubleSpinBox()
                param_input.setRange(1.0, 10.0)
                param_input.setValue(2.0)
                param_input.setSingleStep(0.1)
            elif param_name == 'kelly_fraction':
                # Для фракции Келли создаем QDoubleSpinBox с диапазоном 0-1
                param_input = QDoubleSpinBox()
                param_input.setRange(0.1, 1.0)
                param_input.setValue(0.5)
                param_input.setSingleStep(0.05)
            elif param_name.startswith('min_') or param_name.startswith('max_'):
                # Для минимальных и максимальных значений
                param_input = QDoubleSpinBox()
                if 'roi' in param_name:
                    param_input.setRange(0.0, 30.0)
                    param_input.setValue(5.0 if 'min' in param_name else 20.0)
                elif 'odds' in param_name:
                    param_input.setRange(1.1, 10.0)
                    param_input.setValue(1.5 if 'min' in param_name else 3.5)
                else:
                    param_input.setRange(0.0, 100.0)
                    param_input.setValue(1.0 if 'min' in param_name else 10.0)
                param_input.setSingleStep(0.1)
            elif 'factor' in param_name or 'weight' in param_name:
                # Для факторов и весов
                param_input = QDoubleSpinBox()
                param_input.setRange(0.01, 1.0)
                param_input.setValue(0.5)
                param_input.setSingleStep(0.01)
            else:
                # Для других параметров
                param_input = QDoubleSpinBox()
                param_input.setRange(0.0, 100.0)
                param_input.setValue(5.0)
                param_input.setSingleStep(0.1)
            
            self.strategy_params_layout.addRow(f"{param_name}:", param_input)
            self.param_inputs[param_name] = param_input
    
    def get_strategy_params(self):
        """
        Получение параметров стратегии из полей ввода.
        """
        params = {}
        for param_name, param_input in self.param_inputs.items():
            params[param_name] = param_input.value()
            
            # Преобразование процентов
            if param_name.endswith('_percent'):
                # Если параметр задан в процентах, но не содержит "percent" в имени,
                # то преобразуем его из процентов в долю
                if not param_input.suffix() == '%':
                    params[param_name] /= 100.0
        
        return params
    
    def start_simulation(self):
        """
        Запуск симуляции ставок.
        """
        # Получение параметров симуляции
        initial_bank = self.initial_bank_input.value()
        num_bets = self.num_bets_input.value()
        num_simulations = self.num_simulations_input.value()
        
        # Получение выбранной стратегии
        strategy_name = self.strategy_combo.currentText()
        strategy_func = self.strategy_functions.get(strategy_name)
        
        # Получение параметров стратегии
        strategy_params = self.get_strategy_params()
        
        # Блокировка элементов управления на время симуляции
        self.start_simulation_button.setEnabled(False)
        self.stop_simulation_button.setEnabled(True)
        self.simulation_progress.setValue(0)
        
        # Создание и запуск рабочего потока для симуляции
        self.simulation_worker = SimulationWorker(
            initial_bank=initial_bank,
            strategy_func=strategy_func,
            strategy_params=strategy_params,
            num_simulations=num_simulations,
            num_bets=num_bets
        )
        
        # Подключение сигналов
        self.simulation_worker.progress.connect(self.update_simulation_progress)
        self.simulation_worker.finished.connect(self.simulation_finished)
        
        # Запуск потока
        self.simulation_worker.start()
    
    def stop_simulation(self):
        """
        Остановка симуляции.
        """
        if self.simulation_worker and self.simulation_worker.isRunning():
            self.simulation_worker.requestInterruption()
            self.simulation_worker.wait()
        
        self.start_simulation_button.setEnabled(True)
        self.stop_simulation_button.setEnabled(False)
    
    def update_simulation_progress(self, value):
        """
        Обновление индикатора прогресса симуляции.
        """
        self.simulation_progress.setValue(value)
    
    def simulation_finished(self, results):
        """
        Обработка завершения симуляции.
        """
        # Сохранение результатов
        self.simulation_results = results
        
        # Разблокировка элементов управления
        self.start_simulation_button.setEnabled(True)
        self.stop_simulation_button.setEnabled(False)
        
        # Отображение результатов в текстовом поле
        self.show_simulation_results()
        
        # Отображение графика истории банка
        self.plot_bank_history()
    
    def show_simulation_results(self):
        """
        Отображение результатов симуляции в текстовом поле.
        """
        if not self.simulation_results:
            return
        
        results = self.simulation_results
        
        text = "Результаты симуляции:\n\n"
        text += f"Количество симуляций: {results['num_simulations']}\n"
        text += f"Начальный банк: {results['all_results'][0]['initial_bank']}\n"
        text += f"Средний итоговый банк: {results['avg_final_bank']:.2f}\n"
        text += f"Медианный итоговый банк: {results['median_final_bank']:.2f}\n"
        text += f"Минимальный итоговый банк: {results['min_final_bank']:.2f}\n"
        text += f"Максимальный итоговый банк: {results['max_final_bank']:.2f}\n"
        text += f"Средний прирост банка: {results['avg_bank_growth_pct']:.2f}%\n"
        text += f"Медианный прирост банка: {results['median_bank_growth_pct']:.2f}%\n"
        text += f"Средняя макс. просадка от пика: {results['avg_max_drawdown_from_peak']:.2f}%\n"
        text += f"Медианная макс. просадка от пика: {results['median_max_drawdown_from_peak']:.2f}%\n"
        text += f"% симуляций с просадкой >50%: {results['pct_drawdowns_over_50']:.2f}%\n"
        text += f"% симуляций с просадкой >80%: {results['pct_drawdowns_over_80']:.2f}%\n"
        text += f"% досрочных остановок: {results['pct_early_stops']:.2f}%\n"
        
        self.simulation_results_text.setText(text)
    
    def plot_bank_history(self):
        """
        Отображение графика истории банка.
        """
        if not self.simulation_results or not self.simulation_results.get('all_results'):
            return
        
        # Получаем историю банка из первой симуляции
        bank_history = self.simulation_results['all_results'][0]['bank_history']
        initial_bank = self.simulation_results['all_results'][0]['initial_bank']
        
        # Очистка предыдущего графика
        self.bank_history_canvas.axes.clear()
        
        # Построение нового графика
        self.bank_history_canvas.axes.plot(bank_history, label='Банк')
        self.bank_history_canvas.axes.axhline(y=initial_bank, color='r', linestyle='--', label='Начальный банк')
        self.bank_history_canvas.axes.set_xlabel('Номер ставки')
        self.bank_history_canvas.axes.set_ylabel('Банк')
        self.bank_history_canvas.axes.set_title('История изменения банка (первая симуляция)')
        self.bank_history_canvas.axes.legend()
        self.bank_history_canvas.axes.grid(True)
        
        # Обновление графика
        self.bank_history_canvas.draw()
        
    def create_optimization_tab(self):
        """
        Создание вкладки для оптимизации параметров стратегий.
        """
        self.optimization_tab = QWidget()
        self.tabs.addTab(self.optimization_tab, "Оптимизация")

        # Создание основного макета для вкладки
        layout = QVBoxLayout(self.optimization_tab)
        
        # Создание группы для параметров оптимизации
        params_group = QGroupBox("Параметры оптимизации")
        params_layout = QFormLayout()
        
        # Поле для начального банка
        self.opt_initial_bank_input = QSpinBox()
        self.opt_initial_bank_input.setRange(1000, 1000000)
        self.opt_initial_bank_input.setValue(10000)
        self.opt_initial_bank_input.setSingleStep(1000)
        params_layout.addRow("Начальный банк:", self.opt_initial_bank_input)
        
        # Поле для количества ставок
        self.opt_num_bets_input = QSpinBox()
        self.opt_num_bets_input.setRange(100, 10000)
        self.opt_num_bets_input.setValue(1500)
        self.opt_num_bets_input.setSingleStep(100)
        params_layout.addRow("Количество ставок:", self.opt_num_bets_input)
        
        # Поле для количества симуляций
        self.opt_num_simulations_input = QSpinBox()
        self.opt_num_simulations_input.setRange(10, 1000)
        self.opt_num_simulations_input.setValue(500)
        self.opt_num_simulations_input.setSingleStep(10)
        params_layout.addRow("Количество симуляций:", self.opt_num_simulations_input)
        
        # Группа для выбора стратегий
        strategies_group = QGroupBox("Стратегии для оптимизации")
        strategies_layout = QVBoxLayout(strategies_group)
        
        # Получение всех доступных стратегий
        self.opt_strategy_checkboxes = {}
        for name in dir(bet_strategies):
            if name.startswith('calculate_'):
                checkbox = QCheckBox(name)
                checkbox.setChecked(False)
                strategies_layout.addWidget(checkbox)
                self.opt_strategy_checkboxes[name] = checkbox
        
        # Кнопки для выбора всех/ни одной стратегии
        strategies_buttons_layout = QHBoxLayout()
        self.select_all_strategies_button = QPushButton("Выбрать все стратегии")
        self.select_all_strategies_button.clicked.connect(self.select_all_strategies)
        strategies_buttons_layout.addWidget(self.select_all_strategies_button)
        
        self.deselect_all_strategies_button = QPushButton("Снять выбор стратегий")
        self.deselect_all_strategies_button.clicked.connect(self.deselect_all_strategies)
        strategies_buttons_layout.addWidget(self.deselect_all_strategies_button)
        
        strategies_layout.addLayout(strategies_buttons_layout)
        
        # Область прокрутки для стратегий
        strategies_scroll = QScrollArea()
        strategies_scroll.setWidgetResizable(True)
        strategies_widget = QWidget()
        strategies_widget.setLayout(strategies_layout)
        strategies_scroll.setWidget(strategies_widget)
        
        # Группа для выбора профилей риска
        risk_profiles_group = QGroupBox("Профили риска для оптимизации")
        risk_profiles_layout = QVBoxLayout(risk_profiles_group)
        
        # Чекбоксы для профилей риска
        self.opt_risk_profile_checkboxes = {}
        for profile_name in RISK_PROFILES.keys():
            checkbox = QCheckBox(profile_name)
            checkbox.setChecked(False)
            risk_profiles_layout.addWidget(checkbox)
            self.opt_risk_profile_checkboxes[profile_name] = checkbox
        
        # Кнопки для выбора всех/ни одного профиля
        risk_profiles_buttons_layout = QHBoxLayout()
        select_all_profiles_button = QPushButton("Выбрать все профили")
        select_all_profiles_button.clicked.connect(self.select_all_risk_profiles)
        risk_profiles_buttons_layout.addWidget(select_all_profiles_button)
        
        deselect_all_profiles_button = QPushButton("Снять выбор профилей")
        deselect_all_profiles_button.clicked.connect(self.deselect_all_risk_profiles)
        risk_profiles_buttons_layout.addWidget(deselect_all_profiles_button)
        
        risk_profiles_layout.addLayout(risk_profiles_buttons_layout)
        
        # Завершение настройки группы параметров
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Добавление групп выбора стратегий и профилей риска
        options_layout = QHBoxLayout()
        options_layout.addWidget(strategies_scroll)
        options_layout.addWidget(risk_profiles_group)
        layout.addLayout(options_layout)
        
        # Добавление кнопок для запуска и остановки оптимизации
        buttons_layout = QHBoxLayout()
        
        self.start_optimization_button = QPushButton("Запустить оптимизацию")
        self.start_optimization_button.clicked.connect(self.start_optimization)
        buttons_layout.addWidget(self.start_optimization_button)
        
        self.stop_optimization_button = QPushButton("Остановить оптимизацию")
        self.stop_optimization_button.clicked.connect(self.stop_optimization)
        self.stop_optimization_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_optimization_button)
        
        layout.addLayout(buttons_layout)
        
        # Прогресс-бар для отображения прогресса оптимизации
        self.optimization_progress = QProgressBar()
        layout.addWidget(self.optimization_progress)
        
        # Область для отображения лога
        log_group = QGroupBox("Лог оптимизации")
        log_layout = QVBoxLayout(log_group)
        
        self.optimization_log = QTextEdit()
        self.optimization_log.setReadOnly(True)
        log_layout.addWidget(self.optimization_log)
        
        layout.addWidget(log_group)
        
        # Область для отображения результатов с прокруткой
        results_group = QGroupBox("Результаты оптимизации")
        results_layout = QVBoxLayout(results_group)
        
        # Создаем область прокрутки для результатов
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setMinimumHeight(300)
        
        # Создаем виджет с текстом результатов
        results_widget = QWidget()
        results_widget_layout = QVBoxLayout(results_widget)
        
        self.optimization_results_text = QTextEdit()
        self.optimization_results_text.setReadOnly(True)
        self.optimization_results_text.setMinimumHeight(250)
        results_widget_layout.addWidget(self.optimization_results_text)
        
        # Добавляем виджет в область прокрутки
        results_scroll.setWidget(results_widget)
        results_layout.addWidget(results_scroll)
        
        layout.addWidget(results_group)
    
    def select_all_strategies(self):
        """
        Выбор всех стратегий для оптимизации.
        """
        for checkbox in self.opt_strategy_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_strategies(self):
        """
        Снятие выбора со всех стратегий.
        """
        for checkbox in self.opt_strategy_checkboxes.values():
            checkbox.setChecked(False)
    
    def select_all_risk_profiles(self):
        """
        Выбор всех профилей риска для оптимизации.
        """
        for checkbox in self.opt_risk_profile_checkboxes.values():
            checkbox.setChecked(True)
    
    def deselect_all_risk_profiles(self):
        """
        Снятие выбора со всех профилей риска.
        """
        for checkbox in self.opt_risk_profile_checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_strategies(self):
        """
        Получение списка выбранных стратегий для оптимизации.
        """
        selected_strategies = []
        for name, checkbox in self.opt_strategy_checkboxes.items():
            if checkbox.isChecked():
                selected_strategies.append(name)
        return selected_strategies
    
    def get_selected_risk_profiles(self):
        """
        Получение списка выбранных профилей риска для оптимизации.
        """
        selected_profiles = []
        for name, checkbox in self.opt_risk_profile_checkboxes.items():
            if checkbox.isChecked():
                selected_profiles.append(name)
        return selected_profiles
    
    def start_optimization(self):
        """
        Запуск оптимизации параметров стратегий.
        """
        # Получение параметров оптимизации
        initial_bank = self.opt_initial_bank_input.value()
        num_bets = self.opt_num_bets_input.value()
        num_simulations = self.opt_num_simulations_input.value()
        
        # Получение выбранных стратегий и профилей риска
        selected_strategies = self.get_selected_strategies()
        selected_risk_profiles = self.get_selected_risk_profiles()
        
        if not selected_strategies:
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы одну стратегию для оптимизации.")
            return
        
        if not selected_risk_profiles:
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы один профиль риска для оптимизации.")
            return
        
        # Блокировка элементов управления на время оптимизации
        self.start_optimization_button.setEnabled(False)
        self.stop_optimization_button.setEnabled(True)
        self.optimization_progress.setValue(0)
        self.optimization_log.clear()
        
        # Создание и запуск оптимизатора в отдельном потоке
        self.optimization_worker = OptimizationWorker(
            initial_bank=initial_bank,
            num_bets=num_bets,
            num_simulations=num_simulations,
            strategies_to_optimize=selected_strategies,
            risk_profiles_to_optimize=selected_risk_profiles
        )
        
        # Подключение сигналов
        self.optimization_worker.progress.connect(self.update_optimization_progress)
        self.optimization_worker.log_message.connect(self.update_optimization_log)
        self.optimization_worker.finished.connect(self.optimization_finished)
        
        # Запуск потока
        self.optimization_worker.start()
    
    def stop_optimization(self):
        """
        Остановка оптимизации.
        """
        if self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.requestInterruption()
            self.optimization_worker.wait()
        
        self.start_optimization_button.setEnabled(True)
        self.stop_optimization_button.setEnabled(False)
    
    def update_optimization_progress(self, value):
        """
        Обновление индикатора прогресса оптимизации.
        """
        self.optimization_progress.setValue(value)
    
    def update_optimization_log(self, message):
        """
        Обновление лога оптимизации.
        """
        self.optimization_log.append(message)
    
    def optimization_finished(self, results):
        """
        Обработка завершения оптимизации.
        """
        # Сохранение результатов
        self.optimization_results = results
        
        # Разблокировка элементов управления
        self.start_optimization_button.setEnabled(True)
        self.stop_optimization_button.setEnabled(False)
        
        # Отображение результатов в текстовом поле
        self.show_optimization_results()
    
    def show_optimization_results(self):
        """
        Отображение результатов оптимизации в текстовом поле.
        """
        if not self.optimization_results:
            return
        
        # Создание оптимизатора только для генерации отчета
        optimizer = StrategyOptimizer()
        optimizer.best_params = self.optimization_results
        
        # Генерация отчета
        report = optimizer.generate_optimization_report()
        
        # Отображение отчета
        self.optimization_results_text.setText(report)
    
    def create_visualization_tab(self):
        """
        Создание вкладки для визуализации результатов.
        """
        self.visualization_tab = QWidget()
        self.tabs.addTab(self.visualization_tab, "Визуализация")
        
        # Создание основного макета для вкладки
        layout = QVBoxLayout(self.visualization_tab)
        
        # Группа для выбора источника данных
        source_group = QGroupBox("Источник данных")
        source_layout = QVBoxLayout(source_group)
        
        # Радиокнопки для выбора источника данных
        self.simulation_source_radio = QCheckBox("Результаты симуляции")
        self.simulation_source_radio.setChecked(False)
        self.simulation_source_radio.toggled.connect(self.update_visualization_source)
        source_layout.addWidget(self.simulation_source_radio)
        
        self.optimization_source_radio = QCheckBox("Результаты оптимизации")
        self.optimization_source_radio.setChecked(False)
        self.optimization_source_radio.toggled.connect(self.update_visualization_source)
        source_layout.addWidget(self.optimization_source_radio)
        
        layout.addWidget(source_group)
        
        # Группа для выбора типа визуализации
        viz_type_group = QGroupBox("Тип визуализации")
        viz_type_layout = QVBoxLayout(viz_type_group)
        
        # Комбо-бокс для выбора типа визуализации
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItem("История банка")
        self.viz_type_combo.addItem("Распределение итоговых банков")
        self.viz_type_combo.addItem("Распределение ROI")
        self.viz_type_combo.addItem("Распределение коэффициентов")
        self.viz_type_combo.addItem("Распределение просадок")
        self.viz_type_combo.addItem("Сравнение стратегий")
        viz_type_layout.addWidget(self.viz_type_combo)
        
        # Группа для параметров визуализации
        self.viz_params_group = QGroupBox("Параметры визуализации")
        self.viz_params_layout = QFormLayout(self.viz_params_group)
        
        # Словарь для полей ввода параметров визуализации
        self.viz_param_inputs = {}
        
        # Обновление параметров при изменении типа визуализации
        self.viz_type_combo.currentTextChanged.connect(self.update_visualization_params)
        
        viz_type_layout.addWidget(self.viz_params_group)
        layout.addWidget(viz_type_group)
        
        # Кнопка для создания визуализации
        self.create_viz_button = QPushButton("Создать визуализацию")
        self.create_viz_button.clicked.connect(self.create_visualization)
        layout.addWidget(self.create_viz_button)
        
        # Область для отображения графика
        self.viz_canvas = MplCanvas(width=10, height=8)
        layout.addWidget(self.viz_canvas)
        
        # Первоначальное обновление параметров визуализации
        self.update_visualization_source()
        self.update_visualization_params()
    
    def update_visualization_source(self):
        """
        Обновление параметров в зависимости от выбранного источника данных.
        """
        # Проверка наличия данных
        if self.simulation_source_radio.isChecked() and not self.simulation_results:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов симуляции. Сначала выполните симуляцию.")
            self.simulation_source_radio.setChecked(False)
            return
        
        if self.optimization_source_radio.isChecked() and not self.optimization_results:
            QMessageBox.warning(self, "Предупреждение", "Нет результатов оптимизации. Сначала выполните оптимизацию.")
            self.optimization_source_radio.setChecked(False)
            return
        
        # Обновление списка типов визуализации в зависимости от источника
        self.viz_type_combo.clear()
        
        if self.simulation_source_radio.isChecked():
            self.viz_type_combo.addItem("История банка")
            self.viz_type_combo.addItem("Распределение итоговых банков")
            self.viz_type_combo.addItem("Распределение ROI")
            self.viz_type_combo.addItem("Распределение коэффициентов")
            self.viz_type_combo.addItem("Распределение просадок")
        
        if self.optimization_source_radio.isChecked():
            self.viz_type_combo.addItem("Сравнение стратегий")
            self.viz_type_combo.addItem("Сравнение профилей риска")
        
        # Обновление параметров визуализации
        self.update_visualization_params()
    
    def update_visualization_params(self):
        """
        Обновление параметров в зависимости от выбранного типа визуализации.
        """
        # Очистка текущих полей
        for i in reversed(range(self.viz_params_layout.count())):
            self.viz_params_layout.itemAt(i).widget().deleteLater()
        
        self.viz_param_inputs.clear()
        
        # Получение текущего типа визуализации
        viz_type = self.viz_type_combo.currentText()
        if not viz_type:
            return
        
        # Параметры в зависимости от типа визуализации
        if viz_type == "История банка":
            # Выбор симуляции для отображения
            num_simulations = len(self.simulation_results.get('all_results', []))
            if num_simulations > 0:
                param_input = QSpinBox()
                param_input.setRange(1, num_simulations)
                param_input.setValue(1)
                self.viz_params_layout.addRow("Номер симуляции:", param_input)
                self.viz_param_inputs['simulation_index'] = param_input
        
        elif viz_type == "Распределение просадок":
            # Выбор типа просадки
            param_input = QComboBox()
            param_input.addItem("От пика")
            param_input.addItem("От начального банка")
            self.viz_params_layout.addRow("Тип просадки:", param_input)
            self.viz_param_inputs['drawdown_type'] = param_input
            
            # Пороговые значения
            param_input = QLineEdit("50, 80")
            self.viz_params_layout.addRow("Пороговые значения (через запятую):", param_input)
            self.viz_param_inputs['thresholds'] = param_input
        
        elif viz_type == "Сравнение стратегий":
            # Выбор метрики для сравнения
            param_input = QComboBox()
            param_input.addItem("Прирост банка (%)")
            param_input.addItem("Средний итоговый банк")
            param_input.addItem("Средняя макс. просадка от пика (%)")
            param_input.addItem("% симуляций с просадкой >50%")
            param_input.addItem("% симуляций с просадкой >80%")
            self.viz_params_layout.addRow("Метрика для сравнения:", param_input)
            self.viz_param_inputs['metric'] = param_input
        
        elif viz_type == "Сравнение профилей риска":
            # Выбор метрик для сравнения
            param_input = QLineEdit("bank_growth_pct, avg_max_drawdown_from_peak")
            self.viz_params_layout.addRow("Метрики (через запятую):", param_input)
            self.viz_param_inputs['metrics'] = param_input
    
    def create_visualization(self):
        """
        Создание и отображение выбранной визуализации.
        """
        # Проверка выбора источника данных
        if not self.simulation_source_radio.isChecked() and not self.optimization_source_radio.isChecked():
            QMessageBox.warning(self, "Предупреждение", "Выберите источник данных для визуализации.")
            return
        
        # Получение типа визуализации
        viz_type = self.viz_type_combo.currentText()
        if not viz_type:
            return
        
        # Очистка текущего графика
        self.viz_canvas.axes.clear()
        
        # Создание визуализации в зависимости от типа
        fig = None
        
        if self.simulation_source_radio.isChecked():
            if viz_type == "История банка":
                # Получение индекса симуляции
                sim_index = self.viz_param_inputs.get('simulation_index', QSpinBox()).value() - 1
                sim_index = max(0, min(sim_index, len(self.simulation_results['all_results']) - 1))
                
                # Получение данных для визуализации
                bank_history = self.simulation_results['all_results'][sim_index]['bank_history']
                initial_bank = self.simulation_results['all_results'][sim_index]['initial_bank']
                
                # Создание графика
                fig = Visualizer.plot_bank_history(bank_history, initial_bank, return_figure=True)
            
            elif viz_type == "Распределение итоговых банков":
                # Получение данных для визуализации
                final_banks = [r['final_bank'] for r in self.simulation_results['all_results']]
                initial_bank = self.simulation_results['all_results'][0]['initial_bank']
                
                # Создание графика
                fig = Visualizer.plot_final_bank_distribution(final_banks, initial_bank, return_figure=True)
            
            elif viz_type == "Распределение ROI":
                # Получение данных для визуализации
                all_roi = []
                for result in self.simulation_results['all_results']:
                    all_roi.extend(result['roi_history'])
                
                # Создание графика
                fig = Visualizer.plot_roi_distribution(all_roi, return_figure=True)
            
            elif viz_type == "Распределение коэффициентов":
                # Получение данных для визуализации
                all_odds = []
                for result in self.simulation_results['all_results']:
                    all_odds.extend(result['odds_history'])
                
                # Создание графика
                fig = Visualizer.plot_odds_distribution(all_odds, return_figure=True)
            
            elif viz_type == "Распределение просадок":
                # Получение типа просадки
                drawdown_type = self.viz_param_inputs.get('drawdown_type', QComboBox()).currentText()
                
                # Получение пороговых значений
                thresholds_text = self.viz_param_inputs.get('thresholds', QLineEdit()).text()
                try:
                    thresholds = [float(t.strip()) for t in thresholds_text.split(',')]
                except ValueError:
                    thresholds = [50, 80]
                
                # Получение данных для визуализации
                if drawdown_type == "От пика":
                    drawdowns = [r['max_drawdown_from_peak'] for r in self.simulation_results['all_results']]
                else:
                    drawdowns = [r['max_drawdown_from_initial'] for r in self.simulation_results['all_results']]
                
                # Создание графика
                fig = Visualizer.plot_drawdown_distribution(drawdowns, thresholds, return_figure=True)
        
        elif self.optimization_source_radio.isChecked():
            if viz_type == "Сравнение стратегий":
                # Получение метрики для сравнения
                metric_name = self.viz_param_inputs.get('metric', QComboBox()).currentText()
                
                # Отображаемое имя метрики -> имя в данных
                metric_mapping = {
                    "Прирост банка (%)": "bank_growth_pct",
                    "Средний итоговый банк": "avg_final_bank",
                    "Средняя макс. просадка от пика (%)": "avg_max_drawdown_from_peak",
                    "% симуляций с просадкой >50%": "pct_drawdowns_over_50",
                    "% симуляций с просадкой >80%": "pct_drawdowns_over_80"
                }
                
                metric = metric_mapping.get(metric_name, "bank_growth_pct")
                
                # Создание графика
                fig = Visualizer.plot_strategies_comparison(self.optimization_results, metric, return_figure=True)
            
            elif viz_type == "Сравнение профилей риска":
                # Получение метрик для сравнения
                metrics_text = self.viz_param_inputs.get('metrics', QLineEdit()).text()
                try:
                    metrics = [m.strip() for m in metrics_text.split(',')]
                except:
                    metrics = ["bank_growth_pct", "avg_max_drawdown_from_peak"]
                
                # Создание графика
                fig = Visualizer.plot_risk_profiles_comparison(self.optimization_results, metrics=metrics, return_figure=True)
        
        # Если график был создан, отображаем его
        if fig:
            # Заменяем текущий canvas новым
            new_canvas = MplCanvas(fig=fig)
            
            # Находим индекс canvas в layout
            layout = self.visualization_tab.layout()
            for i in range(layout.count()):
                if layout.itemAt(i).widget() == self.viz_canvas:
                    # Удаляем старый canvas
                    layout.itemAt(i).widget().deleteLater()
                    # Добавляем новый canvas
                    layout.insertWidget(i, new_canvas)
                    # Обновляем ссылку на текущий canvas
                    self.viz_canvas = new_canvas
                    break


# Главная функция запуска приложения
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 