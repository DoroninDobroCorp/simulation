"""
ПАРАЛЛЕЛЬНЫЕ БОЕВЫЕ ЭКСПЕРИМЕНТЫ
- 8 ядер для скорости
- Прогресс-бар
- Автосохранение каждые 50 тестов
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

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
    
    br, _, _, _, avg_bet = adaptive_constant_profit_realistic(
        outcomes_global, odds_array_global,
        min_roi=4.733,
        max_roi=23.005,
        min_target_pct=3.982 * k,
        max_target_pct=13.078 * k,
        max_bet_percent=20.0 * k,
        apply_variation=True,
        recalc_min=30, recalc_max=70
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
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
    
    br = test_anti_martingale_realistic(
        outcomes_global, odds_array_global,
        base_percent=base,
        multiplier=mult,
        max_percent=10.0
    )
    
    m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
    
    return {
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
        max_percent=10.0
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
        max_percent=10.0
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
        max_percent=10.0
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
        fixed_percent=pct
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


if __name__ == '__main__':
    print("="*100)
    print("⚔️ ПАРАЛЛЕЛЬНЫЕ БОЕВЫЕ ЭКСПЕРИМЕНТЫ (8 ЯДЕР)")
    print("="*100)
    
    # Загружаем данные
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    print(f"Процессоров: {cpu_count()}, используем: 8")
    
    # =============================================================================
    # ГЕНЕРАЦИЯ ЗАДАНИЙ
    # =============================================================================
    print("\n📋 Генерация заданий...")
    
    tasks = []
    
    # 1. adaptive_constant_profit - много значений k
    k_values = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 
                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0]
    
    for k in k_values:
        tasks.append(('adaptive', test_adaptive_cp, {'k': k}))
    
    # 2. anti_martingale
    base_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]
    mult_values = [1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]
    
    for base in base_values:
        for mult in mult_values:
            tasks.append(('anti_mart', test_anti_mart, {'base': base, 'mult': mult}))
    
    # 3. linear_roi
    base_values_linear = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0]
    
    for base in base_values_linear:
        tasks.append(('linear', test_linear, {'base': base}))
    
    # 4. exponential_roi
    base_values_exp = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
    exp_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]
    
    for base in base_values_exp:
        for exp in exp_values:
            tasks.append(('exponential', test_exponential, {'base': base, 'exp': exp}))
    
    # 5. sqrt_roi
    for base in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]:
        tasks.append(('sqrt', test_sqrt, {'base': base}))
    
    # 6. fixed_fraction
    for pct in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0]:
        tasks.append(('fixed', test_fixed, {'pct': pct}))
    
    total_tasks = len(tasks)
    print(f"✅ Всего заданий: {total_tasks}")
    print(f"   - adaptive_constant_profit: {len([t for t in tasks if t[0] == 'adaptive'])}")
    print(f"   - anti_martingale: {len([t for t in tasks if t[0] == 'anti_mart'])}")
    print(f"   - linear_roi: {len([t for t in tasks if t[0] == 'linear'])}")
    print(f"   - exponential_roi: {len([t for t in tasks if t[0] == 'exponential'])}")
    print(f"   - sqrt_roi: {len([t for t in tasks if t[0] == 'sqrt'])}")
    print(f"   - fixed_fraction: {len([t for t in tasks if t[0] == 'fixed'])}")
    
    print(f"\n⏱️  Примерное время: ~{total_tasks * 2 / 60:.0f} минут")
    
    # =============================================================================
    # ПАРАЛЛЕЛЬНОЕ ВЫПОЛНЕНИЕ
    # =============================================================================
    print("\n🚀 Запуск параллельных тестов...\n")
    
    results = []
    start_time = time.time()
    
    with Pool(processes=8, initializer=init_worker, initargs=(outcomes, odds_array)) as pool:
        # Параллельный запуск с прогресс-баром
        task_functions = [task[1] for task in tasks]
        task_params = [task[2] for task in tasks]
        
        for i, result in enumerate(tqdm(
            pool.imap(lambda args: args[0](args[1]), zip(task_functions, task_params)),
            total=total_tasks,
            desc="Прогресс",
            unit="тест"
        )):
            results.append(result)
            
            # Автосохранение каждые 50 тестов
            if (i + 1) % 50 == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv('combat_results_progress.csv', index=False)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Все тесты завершены за {elapsed/60:.1f} минут")
    
    # =============================================================================
    # СОХРАНЕНИЕ И АНАЛИЗ
    # =============================================================================
    print("\n" + "="*100)
    print("💾 СОХРАНЕНИЕ И АНАЛИЗ")
    print("="*100)
    
    df = pd.DataFrame(results).sort_values('profit', ascending=False)
    df.to_csv('combat_results_final.csv', index=False)
    
    print(f"\nВсего протестировано: {len(df)} комбинаций")
    
    # Категории
    zero = df[df['bankrupt'] == 0.0].sort_values('profit', ascending=False)
    agg = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)].sort_values('profit', ascending=False)
    mega = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)].sort_values('profit', ascending=False)
    extreme = df[df['bankrupt'] > 25.0].sort_values('profit', ascending=False)
    
    print(f"\n🛡️  0% банкротств: {len(zero)} вариантов")
    print(f"⚡ До 5% банкротств: {len(agg)} вариантов")
    print(f"🔥 5-25% банкротств: {len(mega)} вариантов")
    print(f"🚨 >25% банкротств: {len(extreme)} вариантов")
    
    # =============================================================================
    # ЛУЧШИЕ В КАЖДОЙ КАТЕГОРИИ
    # =============================================================================
    print("\n" + "="*100)
    print("💎 ЛУЧШИЕ В КАЖДОЙ КАТЕГОРИИ")
    print("="*100)
    
    if len(zero) > 0:
        print("\n🥇 0% БАНКРОТСТВ (ТОП-5):")
        for i, (_, r) in enumerate(zero.head(5).iterrows(), 1):
            print(f"   {i}. {r['strategy']:<25} {r['params']:<25} +{r['profit']:>6.0f}% | DD>50: {r['dd50']:>5.1f}% | Worst: {r['worst_dd']:>6.1f}%")
    
    if len(agg) > 0:
        print("\n🥈 АГРЕССИВНАЯ до 5% (ТОП-5):")
        for i, (_, r) in enumerate(agg.head(5).iterrows(), 1):
            print(f"   {i}. {r['strategy']:<25} {r['params']:<25} +{r['profit']:>6.0f}% | B: {r['bankrupt']:>5.2f}% | DD>50: {r['dd50']:>5.1f}%")
    
    if len(mega) > 0:
        print("\n🥉 МЕГА АГРЕССИВНАЯ 5-25% (ТОП-5):")
        for i, (_, r) in enumerate(mega.head(5).iterrows(), 1):
            print(f"   {i}. {r['strategy']:<25} {r['params']:<25} +{r['profit']:>6.0f}% | B: {r['bankrupt']:>5.2f}% | DD>50: {r['dd50']:>5.1f}%")
    
    # =============================================================================
    # ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ
    # =============================================================================
    print("\n" + "="*100)
    print("🎯 ФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ (БОЕВЫЕ УСЛОВИЯ)")
    print("="*100)
    
    if len(zero) > 0:
        best = zero.iloc[0]
        print(f"\n🛡️  БЕЗ СЛИВОВ (0% банкротств):")
        print(f"    Стратегия: {best['strategy']}")
        print(f"    Параметры: {best['params']}")
        print(f"    Прибыль: +{best['profit']:.0f}%")
        print(f"    DD>50%: {best['dd50']:.1f}%")
        print(f"    DD>80%: {best['dd80']:.1f}%")
        print(f"    Worst DD: {best['worst_dd']:.1f}%")
    
    if len(agg) > 0:
        best = agg.iloc[0]
        print(f"\n⚡ АГРЕССИВНАЯ (до 5% банкротств):")
        print(f"    Стратегия: {best['strategy']}")
        print(f"    Параметры: {best['params']}")
        print(f"    Прибыль: +{best['profit']:.0f}%")
        print(f"    Банкротств: {best['bankrupt']:.2f}%")
        print(f"    DD>50%: {best['dd50']:.1f}%")
        print(f"    DD>80%: {best['dd80']:.1f}%")
    
    if len(mega) > 0:
        best = mega.iloc[0]
        print(f"\n🔥 МЕГА АГРЕССИВНАЯ (5-25% банкротств):")
        print(f"    Стратегия: {best['strategy']}")
        print(f"    Параметры: {best['params']}")
        print(f"    Прибыль: +{best['profit']:.0f}%")
        print(f"    Банкротств: {best['bankrupt']:.2f}%")
        print(f"    DD>50%: {best['dd50']:.1f}%")
        print(f"    DD>80%: {best['dd80']:.1f}%")
    
    print("\n" + "="*100)
    print("✅ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print(f"📁 Результаты: combat_results_final.csv")
    print("="*100)
