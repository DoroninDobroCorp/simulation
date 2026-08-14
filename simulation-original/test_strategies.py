#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Модуль для тестирования критических функций стратегий ставок.
"""

import sys
import numpy as np
from bet_strategies import (
    calculate_kelly_bet,
    calculate_linear_roi_bet,
    calculate_sqrt_roi_bet,
    calculate_constant_profit_bet
)
from bet_simulator import BetSimulator
from distribution_generator import DistributionGenerator


def test_win_probability_calculation():
    """Тест корректности расчёта вероятности выигрыша из ROI."""
    print("=" * 60)
    print("ТЕСТ: Расчёт вероятности выигрыша из ROI")
    print("=" * 60)
    
    test_cases = [
        # (odds, roi, expected_win_prob)
        (2.0, 5.0, 0.525),   # ROI=5% при коэффициенте 2.0 => p = 1.05/2 = 0.525
        (3.0, 10.0, 0.367),  # ROI=10% при коэффициенте 3.0 => p = 1.10/3 = 0.367
        (1.5, 3.0, 0.687),   # ROI=3% при коэффициенте 1.5 => p = 1.03/1.5 = 0.687
    ]
    
    all_passed = True
    for odds, roi, expected_prob in test_cases:
        # Формула: win_prob = (1 + ROI/100) / odds
        calculated_prob = (1 + roi / 100) / odds
        
        # Проверяем с точностью до 0.001
        if abs(calculated_prob - expected_prob) < 0.001:
            print(f"✅ PASS: odds={odds}, roi={roi}% => p={calculated_prob:.3f}")
        else:
            print(f"❌ FAIL: odds={odds}, roi={roi}% => p={calculated_prob:.3f} (ожидалось {expected_prob:.3f})")
            all_passed = False
    
    print()
    return all_passed


def test_kelly_strategy():
    """Тест стратегии Келли."""
    print("=" * 60)
    print("ТЕСТ: Стратегия Келли")
    print("=" * 60)
    
    bank = 10000
    
    test_cases = [
        # (odds, roi, risk, kelly_fraction, should_bet)
        (2.0, 5.0, 2.0, 0.5, True),   # Положительное математическое ожидание
        (2.0, -2.0, 2.0, 0.5, False), # Отрицательный ROI => не ставим
        (1.5, 0.5, 2.0, 0.5, False),  # Слишком низкий ROI
        (3.0, 15.0, 1.5, 1.0, True),  # Высокий ROI => агрессивная ставка
    ]
    
    all_passed = True
    for odds, roi, risk, kelly_fraction, should_bet in test_cases:
        bet_size = calculate_kelly_bet(odds, roi, bank, risk, kelly_fraction)
        
        if should_bet:
            if bet_size > 0:
                bet_pct = (bet_size / bank) * 100
                print(f"✅ PASS: odds={odds}, roi={roi}%, risk={risk} => ставка {bet_size:.2f} ({bet_pct:.2f}%)")
            else:
                print(f"❌ FAIL: odds={odds}, roi={roi}%, risk={risk} => ставка должна быть > 0, получили {bet_size:.2f}")
                all_passed = False
        else:
            if bet_size == 0:
                print(f"✅ PASS: odds={odds}, roi={roi}%, risk={risk} => ставка 0 (корректно)")
            else:
                print(f"❌ FAIL: odds={odds}, roi={roi}%, risk={risk} => ставка должна быть 0, получили {bet_size:.2f}")
                all_passed = False
    
    print()
    return all_passed


def test_strategies_validation():
    """Тест валидации входных данных для стратегий."""
    print("=" * 60)
    print("ТЕСТ: Валидация входных данных")
    print("=" * 60)
    
    # Тестируем с некорректными данными
    invalid_cases = [
        # (odds, roi, bank, strategy_name)
        (-1.0, 5.0, 10000, "kelly"),      # Отрицательный коэффициент
        (2.0, -10.0, 10000, "kelly"),     # Отрицательный ROI
        (2.0, 5.0, 0, "kelly"),           # Нулевой банк
        (0, 5.0, 10000, "linear_roi"),    # Нулевой коэффициент
    ]
    
    strategies = {
        "kelly": calculate_kelly_bet,
        "linear_roi": calculate_linear_roi_bet,
        "sqrt_roi": calculate_sqrt_roi_bet,
    }
    
    all_passed = True
    for odds, roi, bank, strategy_name in invalid_cases:
        strategy_func = strategies.get(strategy_name)
        if not strategy_func:
            continue
        
        bet_size = strategy_func(odds, roi, bank)
        
        if bet_size == 0:
            print(f"✅ PASS: {strategy_name} с некорректными данными (odds={odds}, roi={roi}, bank={bank}) => возвращает 0")
        else:
            print(f"❌ FAIL: {strategy_name} с некорректными данными (odds={odds}, roi={roi}, bank={bank}) => должна вернуть 0, получили {bet_size}")
            all_passed = False
    
    print()
    return all_passed


def test_simulation_integrity():
    """Тест целостности симуляции."""
    print("=" * 60)
    print("ТЕСТ: Целостность симуляции")
    print("=" * 60)
    
    # Создаём симулятор с простой стратегией
    initial_bank = 10000
    simulator = BetSimulator(
        initial_bank=initial_bank,
        strategy_func=calculate_kelly_bet,
        strategy_params={'risk': 2.0, 'kelly_fraction': 0.5}
    )
    
    # Запускаем короткую симуляцию
    results = simulator.simulate_series(num_bets=100)
    
    all_passed = True
    
    # Проверяем базовые инварианты
    if results['initial_bank'] == initial_bank:
        print(f"✅ PASS: Начальный банк корректен: {results['initial_bank']}")
    else:
        print(f"❌ FAIL: Начальный банк некорректен: {results['initial_bank']} (ожидалось {initial_bank})")
        all_passed = False
    
    if results['final_bank'] >= 0:
        print(f"✅ PASS: Итоговый банк неотрицателен: {results['final_bank']:.2f}")
    else:
        print(f"❌ FAIL: Итоговый банк отрицателен: {results['final_bank']:.2f}")
        all_passed = False
    
    if 0 <= results['win_rate'] <= 1:
        print(f"✅ PASS: Win rate в допустимом диапазоне: {results['win_rate']:.2%}")
    else:
        print(f"❌ FAIL: Win rate вне диапазона [0, 1]: {results['win_rate']:.2%}")
        all_passed = False
    
    if results['max_drawdown_from_peak'] >= 0:
        print(f"✅ PASS: Просадка неотрицательна: {results['max_drawdown_from_peak']:.2f}%")
    else:
        print(f"❌ FAIL: Просадка отрицательна: {results['max_drawdown_from_peak']:.2f}%")
        all_passed = False
    
    # Проверяем длины массивов
    if len(results['bank_history']) == results['num_bets'] + 1:
        print(f"✅ PASS: Длина bank_history корректна: {len(results['bank_history'])}")
    else:
        print(f"❌ FAIL: Длина bank_history некорректна: {len(results['bank_history'])} (ожидалось {results['num_bets'] + 1})")
        all_passed = False
    
    print()
    return all_passed


def test_roi_from_probability():
    """Тест обратного преобразования: вероятность -> ROI."""
    print("=" * 60)
    print("ТЕСТ: Обратное преобразование (вероятность -> ROI)")
    print("=" * 60)
    
    test_cases = [
        # (odds, win_probability, expected_roi)
        (2.0, 0.525, 5.0),
        (3.0, 0.367, 10.1),  # С учётом погрешности округления
        (1.5, 0.687, 3.05),
    ]
    
    all_passed = True
    for odds, win_prob, expected_roi in test_cases:
        # Формула: ROI = (win_prob * odds - 1) * 100
        calculated_roi = (win_prob * odds - 1) * 100
        
        # Проверяем с точностью до 0.2%
        if abs(calculated_roi - expected_roi) < 0.2:
            print(f"✅ PASS: odds={odds}, p={win_prob:.3f} => ROI={calculated_roi:.2f}%")
        else:
            print(f"❌ FAIL: odds={odds}, p={win_prob:.3f} => ROI={calculated_roi:.2f}% (ожидалось {expected_roi:.2f}%)")
            all_passed = False
    
    print()
    return all_passed


def run_all_tests():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ЗАПУСК ВСЕХ ТЕСТОВ")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Расчёт вероятности выигрыша", test_win_probability_calculation()))
    results.append(("Обратное преобразование", test_roi_from_probability()))
    results.append(("Стратегия Келли", test_kelly_strategy()))
    results.append(("Валидация входных данных", test_strategies_validation()))
    results.append(("Целостность симуляции", test_simulation_integrity()))
    
    # Итоговый отчёт
    print("=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Пройдено: {passed}/{total} тестов")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        return 0
    else:
        print(f"\n⚠️  ВНИМАНИЕ: {total - passed} тест(ов) провалено!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
