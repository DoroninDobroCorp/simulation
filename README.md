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
