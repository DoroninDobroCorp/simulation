"""
ПАРАЛЛЕЛЬНЫЕ БОЕВЫЕ ЭКСПЕРИМЕНТЫ С ВОЗМОЖНОСТЬЮ ПРОДОЛЖЕНИЯ
- 8 ядер
- Автосохранение каждые 50 тестов
- Можно прервать и продолжить с места остановки
- Прогресс-бар
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import os

from generate_real_odds_simulations import load_real_odds_outcomes
from realistic_simulation import adaptive_constant_profit_realistic, calculate_metrics_realistic
from test_all_strategies_realistic import (
    test_linear_roi_realistic, 
    test_exponential_roi_realistic,
    test_anti_martingale_realistic,
    test_sqrt_roi_realistic,
    test_fixed_fraction_realistic
)

# Глобальные данные для работников
outcomes_global = None
odds_array_global = None

def init_worker(outcomes, odds_array):
    """Инициализация данных в каждом воркере"""
    global outcomes_global, odds_array_global
    outcomes_global = outcomes
    odds_array_global = odds_array


def test_adaptive_cp(params):
    """Тест adaptive_constant_profit"""
    k = params['k']
    task_id = params.get('task_id', '')
    
    br, _, _, _, avg_bet = adaptive_constant_profit_realistic(
        outcomes_global, odds_array_global,
        min_roi=4.733,
        max_roi=23.005,
        min_target_pct=3.982 * k,
        max_target_pct=13.078 * k,
        max_bet_percent=20.0 * k,
        apply_variation=True,  # БОЕВЫЕ УСЛОВИЯ: 35-115%
        recalc_min=30, recalc_max=70  # Пересчет раз в 30-70 ставок
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
        'task_id': task_id,
        'strategy': 'adaptive_constant_profit',
        'params': f'k={k:.2f}_var',
        'k': k,
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    }


def test_anti_mart(params):
    """Тест anti-martingale"""
    base, mult = params['base'], params['mult']
    task_id = params.get('task_id', '')
    
    br = test_anti_martingale_realistic(
        outcomes_global, odds_array_global,
        base_percent=base,
        multiplier=mult,
        max_percent=10.0,
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
        'task_id': task_id,
        'strategy': 'anti_martingale',
        'params': f'base={base}%_m={mult}',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    }


def test_linear(params):
    """Тест linear_roi"""
    base = params['base']
    
    br = test_linear_roi_realistic(
        outcomes_global, odds_array_global,
        base_roi=5.0,
        base_percent=base,
        max_percent=10.0,
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
        'strategy': 'linear_roi',
        'params': f'base={base}%',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    }


def test_exponential(params):
    """Тест exponential_roi"""
    base, exp = params['base'], params['exp']
    
    br = test_exponential_roi_realistic(
        outcomes_global, odds_array_global,
        base_roi=5.0,
        base_percent=base,
        exponent=exp,
        max_percent=10.0,
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
        'strategy': 'exponential_roi',
        'params': f'base={base}%_e={exp}',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    }


def test_sqrt(params):
    """Тест sqrt_roi"""
    base = params['base']
    
    br = test_sqrt_roi_realistic(
        outcomes_global, odds_array_global,
        base_roi=5.0,
        base_percent=base,
        max_percent=10.0,
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
        'strategy': 'sqrt_roi',
        'params': f'base={base}%',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    }


def test_fixed(params):
    """Тест fixed_fraction"""
    pct = params['pct']
    
    br = test_fixed_fraction_realistic(
        outcomes_global, odds_array_global,
        fixed_percent=pct,
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
        'strategy': 'fixed_fraction',
        'params': f'{pct}%',
        'profit': m['avg_profit_pct'],
        'bankrupt': m['bankrupt_pct'],
        'dd50': m['drawdown_50_pct'],
        'dd80': m['drawdown_80_pct'],
        'worst_dd': m['worst_drawdown_pct']
    }


def create_task_id(task_type, params):
    """Создает уникальный ID задания"""
    if task_type == 'adaptive':
        return f"adaptive_k={params['k']:.2f}"
    elif task_type == 'anti_mart':
        return f"anti_base={params['base']}_m={params['mult']}"
    elif task_type == 'linear':
        return f"linear_base={params['base']}"
    elif task_type == 'exponential':
        return f"exp_base={params['base']}_e={params['exp']}"
    elif task_type == 'sqrt':
        return f"sqrt_base={params['base']}"
    elif task_type == 'fixed':
        return f"fixed_pct={params['pct']}"
    return ""


if __name__ == '__main__':
    print("="*100)
    print("⚔️ БОЕВЫЕ ЭКСПЕРИМЕНТЫ (8 ЯДЕР, ВОЗОБНОВЛЯЕМЫЕ)")
    print("="*100)
    
    # Загружаем данные
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    num_workers = min(16, cpu_count())  # Используем до 16 (с гиперпотоком)
    print(f"Процессоров: {cpu_count()}, используем: {num_workers}")
    
    print("\n⚙️  БОЕВЫЕ УСЛОВИЯ:")
    print("   ✅ Вариация ставок: 35-115% (рандом)")
    print("   ✅ Пересчет банка: раз в 30-70 ставок (рандом)")
    print("   ✅ Max ставка: 10% от текущего банка")
    print("   ✅ Автосохранение: каждые 50 тестов")
    
    # =============================================================================
    # ГЕНЕРАЦИЯ ЗАДАНИЙ
    # =============================================================================
    print("\n📋 Генерация заданий...")
    
    tasks = []
    
    # 1. adaptive_constant_profit - широкий диапазон
    k_values = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 
                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    for k in k_values:
        task_id = create_task_id('adaptive', {'k': k})
        tasks.append((task_id, 'adaptive', test_adaptive_cp, {'k': k}))
    
    # 2. anti_martingale
    base_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
    mult_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0]
    
    for base in base_values:
        for mult in mult_values:
            task_id = create_task_id('anti_mart', {'base': base, 'mult': mult})
            tasks.append((task_id, 'anti_mart', test_anti_mart, {'base': base, 'mult': mult}))
    
    # 3. linear_roi
    base_values_linear = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    for base in base_values_linear:
        task_id = create_task_id('linear', {'base': base})
        tasks.append((task_id, 'linear', test_linear, {'base': base}))
    
    # 4. exponential_roi
    base_values_exp = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    exp_values = [1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    
    for base in base_values_exp:
        for exp in exp_values:
            task_id = create_task_id('exponential', {'base': base, 'exp': exp})
            tasks.append((task_id, 'exponential', test_exponential, {'base': base, 'exp': exp}))
    
    # 5. sqrt_roi
    for base in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        task_id = create_task_id('sqrt', {'base': base})
        tasks.append((task_id, 'sqrt', test_sqrt, {'base': base}))
    
    # 6. fixed_fraction
    for pct in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
        task_id = create_task_id('fixed', {'pct': pct})
        tasks.append((task_id, 'fixed', test_fixed, {'pct': pct}))
    
    total_tasks = len(tasks)
    print(f"✅ Всего заданий: {total_tasks}")
    
    # =============================================================================
    # ЗАГРУЗКА ПРЕДЫДУЩИХ РЕЗУЛЬТАТОВ
    # =============================================================================
    progress_file = 'combat_progress.csv'
    completed_ids = set()
    results = []
    
    if os.path.exists(progress_file):
        print(f"\n📂 Найден файл прогресса: {progress_file}")
        df_prev = pd.read_csv(progress_file)
        results = df_prev.to_dict('records')
        
        # Восстанавливаем task_id из params
        for r in results:
            # Создаем примерный ID (не идеально, но для большинства случаев работает)
            completed_ids.add(r['params'])
        
        print(f"   Загружено результатов: {len(results)}")
        print(f"   Осталось тестов: {total_tasks - len(results)}")
    else:
        print(f"\n🆕 Начинаем с нуля")
    
    # Фильтруем уже выполненные задания
    tasks_to_do = [(tid, ttype, tfunc, tparams) for tid, ttype, tfunc, tparams in tasks 
                   if tid not in completed_ids]
    
    if len(tasks_to_do) == 0:
        print("\n✅ Все задания уже выполнены!")
    else:
        print(f"\n🚀 Запуск {len(tasks_to_do)} тестов на 8 ядрах...\n")
        
        start_time = time.time()
        
        # =============================================================================
        # ПАРАЛЛЕЛЬНОЕ ВЫПОЛНЕНИЕ
        # =============================================================================
        with Pool(processes=8, initializer=init_worker, initargs=(outcomes, odds_array)) as pool:
            task_functions = [task[2] for task in tasks_to_do]
            task_params = [task[3] for task in tasks_to_do]
            
            for i, result in enumerate(tqdm(
                pool.imap(lambda args: args[0](args[1]), zip(task_functions, task_params)),
                total=len(tasks_to_do),
                desc="Прогресс",
                unit="тест"
            )):
                results.append(result)
                
                # Автосохранение каждые 50 тестов
                if (i + 1) % 50 == 0:
                    df_temp = pd.DataFrame(results)
                    df_temp.to_csv(progress_file, index=False)
                    print(f"\n💾 Сохранено {len(results)} результатов")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Тесты завершены за {elapsed/60:.1f} минут")
    
    # =============================================================================
    # ФИНАЛЬНОЕ СОХРАНЕНИЕ И АНАЛИЗ
    # =============================================================================
    print("\n" + "="*100)
    print("💾 ФИНАЛЬНОЕ СОХРАНЕНИЕ")
    print("="*100)
    
    df = pd.DataFrame(results).sort_values('profit', ascending=False)
    df.to_csv('combat_results_final.csv', index=False)
    
    print(f"\nВсего результатов: {len(df)}")
    
    # Категории
    zero = df[df['bankrupt'] == 0.0].sort_values('profit', ascending=False)
    agg = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)].sort_values('profit', ascending=False)
    mega = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)].sort_values('profit', ascending=False)
    
    print(f"\n📊 Статистика:")
    print(f"   🛡️  0% банкротств: {len(zero)} вариантов")
    print(f"   ⚡ До 5% банкротств: {len(agg)} вариантов")
    print(f"   🔥 5-25% банкротств: {len(mega)} вариантов")
    
    # =============================================================================
    # ЛУЧШИЕ В КАЖДОЙ КАТЕГОРИИ
    # =============================================================================
    print("\n" + "="*100)
    print("🎯 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ")
    print("="*100)
    
    if len(zero) > 0:
        print("\n🛡️  БЕЗ СЛИВОВ (0% банкротств) - ТОП-3:")
        for i, (_, r) in enumerate(zero.head(3).iterrows(), 1):
            print(f"   {i}. {r['strategy']:<25} {r['params']:<25} +{r['profit']:>6.0f}% | DD>50: {r['dd50']:>5.1f}%")
    
    if len(agg) > 0:
        print("\n⚡ АГРЕССИВНАЯ (до 5%) - ТОП-3:")
        for i, (_, r) in enumerate(agg.head(3).iterrows(), 1):
            print(f"   {i}. {r['strategy']:<25} {r['params']:<25} +{r['profit']:>6.0f}% | B: {r['bankrupt']:>5.2f}%")
    
    if len(mega) > 0:
        print("\n🔥 МЕГА АГРЕССИВНАЯ (5-25%) - ТОП-3:")
        for i, (_, r) in enumerate(mega.head(3).iterrows(), 1):
            print(f"   {i}. {r['strategy']:<25} {r['params']:<25} +{r['profit']:>6.0f}% | B: {r['bankrupt']:>5.2f}%")
    
    # Лучшая в каждой категории
    print("\n" + "="*100)
    print("💎 ЛУЧШАЯ В КАЖДОЙ КАТЕГОРИИ")
    print("="*100)
    
    if len(zero) > 0:
        best = zero.iloc[0]
        print(f"\n🥇 БЕЗ СЛИВОВ:")
        print(f"   {best['strategy']} ({best['params']})")
        print(f"   Прибыль: +{best['profit']:.0f}%")
        print(f"   Банкротств: 0%")
        print(f"   DD>50%: {best['dd50']:.1f}% | DD>80%: {best['dd80']:.1f}% | Worst: {best['worst_dd']:.1f}%")
    
    if len(agg) > 0:
        best = agg.iloc[0]
        print(f"\n🥈 АГРЕССИВНАЯ:")
        print(f"   {best['strategy']} ({best['params']})")
        print(f"   Прибыль: +{best['profit']:.0f}%")
        print(f"   Банкротств: {best['bankrupt']:.2f}%")
        print(f"   DD>50%: {best['dd50']:.1f}% | DD>80%: {best['dd80']:.1f}%")
    
    if len(mega) > 0:
        best = mega.iloc[0]
        print(f"\n🥉 МЕГА АГРЕССИВНАЯ:")
        print(f"   {best['strategy']} ({best['params']})")
        print(f"   Прибыль: +{best['profit']:.0f}%")
        print(f"   Банкротств: {best['bankrupt']:.2f}%")
        print(f"   DD>50%: {best['dd50']:.1f}% | DD>80%: {best['dd80']:.1f}%")
    
    print("\n" + "="*100)
    print("✅ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print(f"📁 Финальные результаты: combat_results_final.csv")
    print(f"📁 Прогресс для продолжения: {progress_file}")
    print("="*100)
