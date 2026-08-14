🇬🇧 [English](#-english) | 🇷🇺 [Русский](#-русский)

---

# 🇬🇧 English

<div align="center">

# 🎰 Bankroll Management Simulator

**A modular system for simulating and optimizing bankroll management strategies**

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_strategies.py)

[Features](#-features) •
[Quick Start](#-quick-start) •
[Strategies](#-strategies) •
[Architecture](#-architecture) •
[Documentation](#-documentation)

</div>

---

## 📋 About

A system for simulating betting sequences and finding optimal parameters for bankroll management strategies. This project is intended for **educational and research purposes** — exploring mathematical risk management models, the Kelly criterion, and Monte Carlo simulations.

### Key Objectives

- 📊 Simulate betting sequences while tracking bankroll, drawdowns, and growth
- 🔬 Compare 12+ bet sizing strategies
- ⚙️ Automatically find optimal strategy parameters
- 📈 Visualize results and perform comparative analysis

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Monte Carlo simulation** | Multiple simulation runs to assess strategy robustness |
| **12+ strategies** | Kelly, linear, logarithmic, exponential, hybrid, and more |
| **Parameter optimizer** | Automatic parameter search across 4 risk profiles |
| **GUI application** | Graphical interface built with PyQt5 for running simulations |
| **Web simulator** | HTML versions with interactive charts (Chart.js) |
| **Visualization** | Bankroll history, drawdowns, distributions, strategy comparison charts |
| **Automated tests** | Validation of calculations and input checks |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/DoroninDobroCorp/simulation.git
cd simulation
pip install -r requirements.txt
```

### Launch the GUI Application

```bash
python main.py
```

### Run Tests

```bash
python test_strategies.py
```

### Quick Simulation from Code

```python
from bet_simulator import BetSimulator
from bet_strategies import calculate_kelly_bet

simulator = BetSimulator(
    initial_bank=10000,
    strategy_func=calculate_kelly_bet,
    strategy_params={'risk': 2.0, 'kelly_fraction': 0.5}
)

results = simulator.simulate_series(num_bets=1500)
print(f"Final bankroll: {results['final_bank']:.2f}")
print(f"Growth: {results['bank_growth_pct']:.1f}%")
print(f"Max drawdown: {results['max_drawdown_from_peak']:.1f}%")
```

### Multiple Simulations (Monte Carlo)

```python
results = simulator.run_multiple_simulations(
    num_simulations=500,
    num_bets=1500
)
print(f"Average final bankroll: {results['avg_final_bank']:.2f}")
print(f"Median growth: {results['median_bank_growth_pct']:.1f}%")
```

## 🎯 Strategies

<details>
<summary><b>All 12 implemented strategies</b></summary>

| Strategy | Description | Profile |
|----------|-------------|---------|
| `calculate_kelly_bet` | Classic Kelly criterion | Universal |
| `calculate_dynamic_kelly_bet` | Kelly with dynamic ROI-based fraction | Universal |
| `calculate_linear_roi_bet` | Linear ROI dependency | Simple |
| `calculate_sqrt_roi_bet` | Square root of ROI | Conservative |
| `calculate_log_roi_bet` | Logarithmic ROI dependency | Conservative |
| `calculate_exp_roi_bet` | Exponential ROI dependency | Aggressive |
| `calculate_constant_profit_bet` | Fixed target profit | Simple |
| `calculate_combined_roi_odds_bet` | Combined ROI and odds | Advanced |
| `calculate_adaptive_bet` | Adapts to bankroll dynamics | Advanced |
| `calculate_hybrid_bet` | Hybrid with ROI/odds weights | Advanced |
| `calculate_linear_scaled_bet` | Linear ROI → % scaling | Simple |
| `calculate_linear_roi_odds_bet` | Linear ROI with odds adjustment | Advanced |
| `calculate_adaptive_constant_profit_bet` | Adaptive constant profit | Advanced |

</details>

### Kelly Strategy Configuration Example

```python
# Conservative (half-Kelly, high risk divisor)
params = {'risk': 3.0, 'kelly_fraction': 0.5}

# Balanced (recommended)
params = {'risk': 2.0, 'kelly_fraction': 0.5}

# Aggressive (full Kelly)
params = {'risk': 1.0, 'kelly_fraction': 1.0}
```

## 🏗 Architecture

```
simulation/
├── main.py                    # Entry point (launches GUI)
├── bet_simulator.py           # Core: betting sequence simulation
├── bet_strategies.py          # 12+ bet sizing strategies
├── distribution_generator.py  # Odds and ROI distribution generation
├── strategy_optimizer.py      # Strategy parameter optimization
├── visualization.py           # Charts and visualization (matplotlib)
├── gui.py                     # GUI application (PyQt5)
├── test_strategies.py         # Automated tests
├── sim.html                   # Web simulator (basic)
├── simulation.html            # Web simulator (extended)
├── supersim.html              # Web simulator (full)
├── requirements.txt           # Python dependencies
└── optimization_report.txt    # Parameter optimization results
```

### Modules

- **`bet_simulator.py`** — `BetSimulator` class: single bet and series simulation, metric calculation (drawdown, growth, win rate), Monte Carlo multiple runs
- **`bet_strategies.py`** — Bet sizing functions with input data validation
- **`distribution_generator.py`** — Odds generation (beta distribution, mean≈2.8) and ROI (mixture: 85% normal / 10% medium / 5% rare)
- **`strategy_optimizer.py`** — Parameter grid search across 4 risk profiles (Conservative, Cautious, Moderate, Aggressive)
- **`visualization.py`** — Matplotlib charts: bankroll history, distributions, strategy comparison

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [`tz.md`](tz.md) | Project technical specification |
| [`CHANGELOG.md`](CHANGELOG.md) | Changelog |
| [`REFACTORING_REPORT.md`](REFACTORING_REPORT.md) | Detailed refactoring report |
| [`SUMMARY.md`](SUMMARY.md) | Summary of changes |
| [`optimization_report.txt`](optimization_report.txt) | Parameter optimization results |

## 🔧 Requirements

- **Python** 3.6+
- **NumPy** ≥ 1.20.0
- **SciPy** ≥ 1.7.0
- **Matplotlib** ≥ 3.5.0
- **PyQt5** ≥ 5.15.0
- **Pandas** ≥ 1.3.0
- **Seaborn** ≥ 0.11.0

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-strategy`)
3. Commit your changes (`git commit -m 'Add amazing strategy'`)
4. Push to the branch (`git push origin feature/amazing-strategy`)
5. Open a Pull Request

Before submitting a PR, make sure all tests pass:
```bash
python test_strategies.py
```

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## ⚠️ Disclaimer

> This project was created **solely for educational and research purposes** to study mathematical risk management models. The authors are not responsible for the use of this software in real financial operations.

---

# 🇷🇺 Русский

<div align="center">

# 🎰 Bankroll Management Simulator

**Модульная система симуляции и оптимизации стратегий управления банкроллом**

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_strategies.py)

[Возможности](#-возможности) •
[Быстрый старт](#-быстрый-старт) •
[Стратегии](#-стратегии) •
[Архитектура](#-архитектура) •
[Документация](#-документация)

</div>

---

## 📋 О проекте

Система для моделирования серий ставок и поиска оптимальных параметров стратегий банкролл-менеджмента. Проект предназначен для **образовательных и исследовательских целей** — изучения математических моделей управления рисками, критерия Келли и Монте-Карло симуляций.

### Ключевые задачи

- 📊 Симуляция серий ставок с отслеживанием банка, просадок и прироста
- 🔬 Сравнение 12+ стратегий расчёта размера ставки
- ⚙️ Автоматический подбор оптимальных параметров стратегий
- 📈 Визуализация результатов и сравнительный анализ

## ✨ Возможности

| Функция | Описание |
|---------|----------|
| **Монте-Карло симуляция** | Множественные прогоны серий ставок для оценки устойчивости стратегии |
| **12+ стратегий** | Келли, линейная, логарифмическая, экспоненциальная, гибридная и другие |
| **Оптимизатор параметров** | Автоматический перебор параметров для 4 профилей риска |
| **GUI-приложение** | Графический интерфейс на PyQt5 для запуска симуляций |
| **Веб-симулятор** | HTML-версии с интерактивными графиками (Chart.js) |
| **Визуализация** | Графики банка, просадок, распределений, сравнения стратегий |
| **Автотесты** | Проверка корректности расчётов и валидации |

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/DoroninDobroCorp/simulation.git
cd simulation
pip install -r requirements.txt
```

### Запуск GUI-приложения

```bash
python main.py
```

### Запуск тестов

```bash
python test_strategies.py
```

### Быстрая симуляция из кода

```python
from bet_simulator import BetSimulator
from bet_strategies import calculate_kelly_bet

simulator = BetSimulator(
    initial_bank=10000,
    strategy_func=calculate_kelly_bet,
    strategy_params={'risk': 2.0, 'kelly_fraction': 0.5}
)

results = simulator.simulate_series(num_bets=1500)
print(f"Итоговый банк: {results['final_bank']:.2f}")
print(f"Прирост: {results['bank_growth_pct']:.1f}%")
print(f"Макс. просадка: {results['max_drawdown_from_peak']:.1f}%")
```

### Множественные симуляции (Монте-Карло)

```python
results = simulator.run_multiple_simulations(
    num_simulations=500,
    num_bets=1500
)
print(f"Средний итоговый банк: {results['avg_final_bank']:.2f}")
print(f"Медианный прирост: {results['median_bank_growth_pct']:.1f}%")
```

## 🎯 Стратегии

<details>
<summary><b>Все 12 реализованных стратегий</b></summary>

| Стратегия | Описание | Профиль |
|-----------|----------|---------|
| `calculate_kelly_bet` | Классический критерий Келли | Универсальная |
| `calculate_dynamic_kelly_bet` | Келли с динамической фракцией от ROI | Универсальная |
| `calculate_linear_roi_bet` | Линейная зависимость от ROI | Простая |
| `calculate_sqrt_roi_bet` | Квадратный корень от ROI | Консервативная |
| `calculate_log_roi_bet` | Логарифмическая зависимость от ROI | Консервативная |
| `calculate_exp_roi_bet` | Экспоненциальная зависимость от ROI | Агрессивная |
| `calculate_constant_profit_bet` | Постоянная целевая прибыль | Простая |
| `calculate_combined_roi_odds_bet` | Комбинация ROI и коэффициента | Продвинутая |
| `calculate_adaptive_bet` | Адаптивная к динамике банка | Продвинутая |
| `calculate_hybrid_bet` | Гибридная с весами ROI/odds | Продвинутая |
| `calculate_linear_scaled_bet` | Линейное масштабирование ROI → % | Простая |
| `calculate_linear_roi_odds_bet` | Линейная ROI с поправкой на odds | Продвинутая |
| `calculate_adaptive_constant_profit_bet` | Адаптивная постоянная прибыль | Продвинутая |

</details>

### Пример настройки стратегии Келли

```python
# Консервативная (half-Kelly, high risk divisor)
params = {'risk': 3.0, 'kelly_fraction': 0.5}

# Сбалансированная (рекомендуется)
params = {'risk': 2.0, 'kelly_fraction': 0.5}

# Агрессивная (full Kelly)
params = {'risk': 1.0, 'kelly_fraction': 1.0}
```

## 🏗 Архитектура

```
simulation/
├── main.py                    # Точка входа (запуск GUI)
├── bet_simulator.py           # Ядро: симуляция серий ставок
├── bet_strategies.py          # 12+ стратегий расчёта размера ставки
├── distribution_generator.py  # Генерация распределений odds и ROI
├── strategy_optimizer.py      # Оптимизация параметров стратегий
├── visualization.py           # Графики и визуализация (matplotlib)
├── gui.py                     # GUI-приложение (PyQt5)
├── test_strategies.py         # Автоматические тесты
├── sim.html                   # Веб-симулятор (базовый)
├── simulation.html            # Веб-симулятор (расширенный)
├── supersim.html              # Веб-симулятор (полный)
├── requirements.txt           # Python-зависимости
└── optimization_report.txt    # Результаты оптимизации параметров
```

### Модули

- **`bet_simulator.py`** — Класс `BetSimulator`: симуляция одиночных ставок и серий, расчёт метрик (просадка, прирост, win rate), множественные прогоны Монте-Карло
- **`bet_strategies.py`** — Функции расчёта размера ставки с валидацией входных данных
- **`distribution_generator.py`** — Генерация коэффициентов (бета-распределение, mean≈2.8) и ROI (смешанное: 85% обычный / 10% средний / 5% редкий)
- **`strategy_optimizer.py`** — Перебор параметров стратегий по 4 профилям риска (Conservative, Cautious, Moderate, Aggressive)
- **`visualization.py`** — Matplotlib-графики: история банка, распределения, сравнение стратегий

## 📖 Документация

| Документ | Описание |
|----------|----------|
| [`tz.md`](tz.md) | Техническое задание проекта |
| [`CHANGELOG.md`](CHANGELOG.md) | История изменений |
| [`REFACTORING_REPORT.md`](REFACTORING_REPORT.md) | Детальный отчёт о рефакторинге |
| [`SUMMARY.md`](SUMMARY.md) | Краткая сводка изменений |
| [`optimization_report.txt`](optimization_report.txt) | Результаты оптимизации параметров |

## 🔧 Требования

- **Python** 3.6+
- **NumPy** ≥ 1.20.0
- **SciPy** ≥ 1.7.0
- **Matplotlib** ≥ 3.5.0
- **PyQt5** ≥ 5.15.0
- **Pandas** ≥ 1.3.0
- **Seaborn** ≥ 0.11.0

## 🤝 Contributing

1. Fork проекта
2. Создайте feature-ветку (`git checkout -b feature/amazing-strategy`)
3. Закоммитьте изменения (`git commit -m 'Add amazing strategy'`)
4. Push в ветку (`git push origin feature/amazing-strategy`)
5. Откройте Pull Request

Перед отправкой PR убедитесь, что тесты проходят:
```bash
python test_strategies.py
```

## 📄 Лицензия

Распространяется под лицензией MIT. Подробнее — [LICENSE](LICENSE).

## ⚠️ Дисклеймер

> Данный проект создан **исключительно в образовательных и исследовательских целях** для изучения математических моделей управления рисками. Авторы не несут ответственности за использование данного ПО в реальных финансовых операциях.
