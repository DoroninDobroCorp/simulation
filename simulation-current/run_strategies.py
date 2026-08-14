import numpy as np
import csv
import os
from datetime import datetime
from generate_simulations import load_outcomes
from config import ODDS, INITIAL_BANKROLL

def calculate_metrics(bankroll_history, bet_history):
    """
    Рассчитывает метрики для результатов симуляций.
    
    Args:
        bankroll_history: numpy array формы (num_simulations, num_bets + 1)
        bet_history: numpy array формы (num_simulations, num_bets) - размеры ставок
    
    Returns:
        dict с метриками
    """
    num_sims = bankroll_history.shape[0]
    
    bankrupt_count = np.sum(np.any(bankroll_history <= 0, axis=1))
    bankrupt_pct = bankrupt_count / num_sims * 100
    
    peaks = np.maximum.accumulate(bankroll_history, axis=1)
    drawdowns_pct = (bankroll_history - peaks) / peaks * 100
    
    max_drawdown_per_sim = np.min(drawdowns_pct, axis=1)
    
    threshold_20_count = np.sum(np.any(drawdowns_pct <= -20, axis=1))
    threshold_50_count = np.sum(np.any(drawdowns_pct <= -50, axis=1))
    threshold_80_count = np.sum(np.any(drawdowns_pct <= -80, axis=1))
    
    final_bankrolls = bankroll_history[:, -1]
    final_profits = final_bankrolls - INITIAL_BANKROLL
    final_profits_pct = final_profits / INITIAL_BANKROLL * 100
    
    total_turnover = np.sum(bet_history, axis=1)
    roi_from_turnover = (final_profits / total_turnover) * 100
    
    return {
        'bankrupt_pct': bankrupt_pct,
        'drawdown_20_pct': threshold_20_count / num_sims * 100,
        'drawdown_50_pct': threshold_50_count / num_sims * 100,
        'drawdown_80_pct': threshold_80_count / num_sims * 100,
        'avg_profit_pct': np.mean(final_profits_pct),
        'min_profit_pct': np.min(final_profits_pct),
        'max_profit_pct': np.max(final_profits_pct),
        'avg_max_drawdown_pct': np.mean(max_drawdown_per_sim),
        'worst_drawdown_pct': np.min(max_drawdown_per_sim),
        'avg_roi_from_turnover': np.mean(roi_from_turnover),
        'min_roi_from_turnover': np.min(roi_from_turnover),
        'max_roi_from_turnover': np.max(roi_from_turnover)
    }

def flat_strategy(outcomes, bet_size_pct):
    """
    Flat стратегия: фиксированный процент от начального банка.
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        bet_size_pct: размер ставки в % от начального банка
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    bet_amount = INITIAL_BANKROLL * bet_size_pct / 100
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history = np.full((num_sims, num_bets), bet_amount, dtype=float)
    
    for i in range(num_bets):
        win_amount = bet_amount * (ODDS - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = bankroll[:, i] + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        valid_mask = current_bankroll > 0
        bet_pct_from_current[valid_mask, i] = (bet_amount / current_bankroll[valid_mask]) * 100
    
    min_bet_pct = np.min(bet_pct_from_current[bet_pct_from_current > 0])
    max_bet_pct = np.max(bet_pct_from_current[np.isfinite(bet_pct_from_current)])
    avg_bet_pct = np.mean(bet_pct_from_current[bet_pct_from_current > 0])
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def dynamic_percentage_strategy(outcomes, bet_size_pct):
    """
    Dynamic стратегия: фиксированный процент от текущего банка.
    
    Args:
        outcomes: numpy array (num_sims, num_bets) с True/False
        bet_size_pct: размер ставки в % от текущего банка
    
    Returns:
        tuple: (bankroll_history, bet_history, min_bet_pct, max_bet_pct)
    """
    num_sims, num_bets = outcomes.shape
    
    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history = np.zeros((num_sims, num_bets), dtype=float)
    
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        bet_amount = np.maximum(current_bankroll * bet_size_pct / 100, 0)
        bet_history[:, i] = bet_amount
        
        win_amount = bet_amount * (ODDS - 1)
        loss_amount = bet_amount
        
        bankroll[:, i + 1] = current_bankroll + np.where(
            outcomes[:, i],
            win_amount,
            -loss_amount
        )
    
    bet_pct_from_current = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        current_bankroll = bankroll[:, i]
        valid_mask = current_bankroll > 0
        bet_pct_from_current[valid_mask, i] = (bet_history[valid_mask, i] / current_bankroll[valid_mask]) * 100
    
    min_bet_pct = np.min(bet_pct_from_current[bet_pct_from_current > 0])
    max_bet_pct = np.max(bet_pct_from_current[np.isfinite(bet_pct_from_current)])
    avg_bet_pct = np.mean(bet_pct_from_current[bet_pct_from_current > 0])
    
    return bankroll, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct

def generate_strategy_name(base_name, params):
    """Генерирует уникальное имя стратегии."""
    bet_size = params.get('bet_size_pct', 'N/A')
    return f"{base_name}_{bet_size}%"

def generate_strategy_description(base_name, params):
    """Генерирует описание правил стратегии."""
    bet_size = params.get('bet_size_pct', 0)
    bet_amount = INITIAL_BANKROLL * bet_size / 100
    
    if base_name == 'flat':
        return (f"Flat стратегия: фиксированная абсолютная ставка {bet_amount:.0f} единиц "
                f"({bet_size}% от начального банка {INITIAL_BANKROLL}). "
                f"Размер ставки не меняется независимо от текущего банкролла. "
                f"При росте банка процент ставки от текущего уменьшается, при падении - увеличивается.")
    elif base_name == 'dynamic_percentage':
        return (f"Dynamic стратегия: фиксированный процент {bet_size}% от текущего банка. "
                f"Размер ставки пересчитывается перед каждой ставкой как {bet_size}% от текущего баланса. "
                f"При росте банка ставка растет пропорционально (компаунд эффект), "
                f"при падении - уменьшается (защита от разорения).")
    else:
        return f"Стратегия {base_name} с параметрами {params}"

def run_strategy(strategy_name, outcomes, **strategy_params):
    """Прогоняет стратегию и выводит детальный отчет."""
    unique_name = generate_strategy_name(strategy_name, strategy_params)
    description = generate_strategy_description(strategy_name, strategy_params)
    
    print(f"\n{'='*70}")
    print(f"СТРАТЕГИЯ: {unique_name}")
    print(f"Параметры: {strategy_params}")
    print(f"{'='*70}")
    
    if strategy_name == 'flat':
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = flat_strategy(
            outcomes, 
            strategy_params['bet_size_pct']
        )
    elif strategy_name == 'dynamic_percentage':
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = dynamic_percentage_strategy(
            outcomes,
            strategy_params['bet_size_pct']
        )
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy_name}")
    
    metrics = calculate_metrics(bankroll_history, bet_history)
    
    print(f"\n📊 SIZING МЕТРИКИ:")
    print(f"  Средняя ставка: {avg_bet_pct:.2f}% от текущего банка")
    print(f"  Min ставка:     {min_bet_pct:.2f}% от текущего банка")
    print(f"  Max ставка:     {max_bet_pct:.2f}% от текущего банка")
    
    print(f"\n💰 РЕЗУЛЬТАТЫ:")
    print(f"  ROI с оборота (средний): {metrics['avg_roi_from_turnover']:>7.2f}%")
    print(f"  Средняя прибыль:         {metrics['avg_profit_pct']:>8.2f}%")
    print(f"  Мин прибыль:             {metrics['min_profit_pct']:>8.2f}%")
    print(f"  Макс прибыль:            {metrics['max_profit_pct']:>8.2f}%")
    
    print(f"\n⚠️  РИСКИ:")
    print(f"  Слито балансов (≤0):       {metrics['bankrupt_pct']:>6.2f}%")
    print(f"  Просадка >20% от пика:     {metrics['drawdown_20_pct']:>6.2f}%")
    print(f"  Просадка >50% от пика:     {metrics['drawdown_50_pct']:>6.2f}%")
    print(f"  Просадка >80% от пика:     {metrics['drawdown_80_pct']:>6.2f}%")
    print(f"  Средняя макс просадка:     {metrics['avg_max_drawdown_pct']:>6.2f}%")
    print(f"  Худшая просадка:           {metrics['worst_drawdown_pct']:>6.2f}%")
    
    print(f"{'='*70}\n")
    
    result = {
        'strategy_name': unique_name,
        'base_strategy': strategy_name,
        'params': strategy_params,
        'description': description,
        'avg_bet_pct': avg_bet_pct,
        'min_bet_pct': min_bet_pct,
        'max_bet_pct': max_bet_pct,
        **metrics
    }
    
    return result

def save_results_to_csv(result, filename='results.csv'):
    """Сохраняет результат в CSV файл (append режим)."""
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'strategy', 'avg_bet_%', 'min_bet_%', 'max_bet_%',
                'roi_%', 'avg_profit_%', 'min_profit_%', 'max_profit_%',
                'bankrupt_%', 'dd>20_%', 'dd>50_%', 'dd>80_%',
                'avg_maxdd_%', 'worst_dd_%',
                'timestamp', 'description'
            ])
        
        writer.writerow([
            result['strategy_name'],
            f"{result['avg_bet_pct']:.2f}",
            f"{result['min_bet_pct']:.2f}",
            f"{result['max_bet_pct']:.2f}",
            f"{result['avg_roi_from_turnover']:.2f}",
            f"{result['avg_profit_pct']:.2f}",
            f"{result['min_profit_pct']:.2f}",
            f"{result['max_profit_pct']:.2f}",
            f"{result['bankrupt_pct']:.2f}",
            f"{result['drawdown_20_pct']:.2f}",
            f"{result['drawdown_50_pct']:.2f}",
            f"{result['drawdown_80_pct']:.2f}",
            f"{result['avg_max_drawdown_pct']:.2f}",
            f"{result['worst_drawdown_pct']:.2f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            result['description']
        ])
    
    print(f"✅ Результаты добавлены в {filename}")

def save_results_to_markdown(results, filename='results.md'):
    """Создает/обновляет Markdown отчет со всеми результатами."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 📊 Результаты симуляций BRM стратегий\n\n")
        f.write(f"**Параметры симуляции:**\n")
        f.write(f"- Коэффициент: {ODDS}\n")
        f.write(f"- Начальный банкролл: {INITIAL_BANKROLL}\n")
        f.write(f"- Количество симуляций: 10000\n")
        f.write(f"- Ставок в симуляции: 1000\n\n")
        f.write("---\n\n")
        
        f.write("## Сравнительная таблица\n\n")
        f.write("| Стратегия | Avg Bet% | ROI | Avg Profit | Bankrupt | DD>20% | DD>50% | DD>80% |\n")
        f.write("|-----------|----------|-----|------------|----------|--------|--------|--------|\n")
        
        for r in results:
            f.write(f"| {r['strategy_name']} | {r['avg_bet_pct']:.2f}% | "
                   f"{r['avg_roi_from_turnover']:.2f}% | {r['avg_profit_pct']:.2f}% | "
                   f"{r['bankrupt_pct']:.2f}% | {r['drawdown_20_pct']:.2f}% | "
                   f"{r['drawdown_50_pct']:.2f}% | {r['drawdown_80_pct']:.2f}% |\n")
        
        f.write("\n---\n\n")
        f.write("## Детальные результаты\n\n")
        
        for r in results:
            f.write(f"### {r['strategy_name']}\n\n")
            f.write(f"**📋 Описание:**\n{r['description']}\n\n")
            f.write("**💰 Результаты:**\n")
            f.write(f"- ROI с оборота: {r['avg_roi_from_turnover']:.2f}%\n")
            f.write(f"- Средняя прибыль: {r['avg_profit_pct']:.2f}%\n")
            f.write(f"- Мин прибыль: {r['min_profit_pct']:.2f}%\n")
            f.write(f"- Макс прибыль: {r['max_profit_pct']:.2f}%\n\n")
            f.write("**⚠️ Риски:**\n")
            f.write(f"- Слито балансов: {r['bankrupt_pct']:.2f}%\n")
            f.write(f"- Просадка >20% от пика: {r['drawdown_20_pct']:.2f}%\n")
            f.write(f"- Просадка >50% от пика: {r['drawdown_50_pct']:.2f}%\n")
            f.write(f"- Просадка >80% от пика: {r['drawdown_80_pct']:.2f}%\n")
            f.write(f"- Средняя макс просадка: {r['avg_max_drawdown_pct']:.2f}%\n")
            f.write(f"- Худшая просадка: {r['worst_drawdown_pct']:.2f}%\n\n")
            f.write("---\n\n")
    
    print(f"✅ Markdown отчет обновлен: {filename}")

if __name__ == '__main__':
    print("Загрузка исходов ставок...")
    outcomes = load_outcomes()
    
    print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
    print(f"Начальный банкролл: {INITIAL_BANKROLL}")
    print(f"Коэффициент: {ODDS}")
    
    results = []
    
    result = run_strategy('dynamic_percentage', outcomes, bet_size_pct=2.0)
    results.append(result)
    save_results_to_csv(result)
    
    save_results_to_markdown(results)
