"""
БОЕВЫЕ ЭКСПЕРИМЕНТЫ - ФИНАЛЬНАЯ ВЕРСИЯ
- Максимум процессов (16)
- Автосохранение каждые 20 тестов
- Простая и надежная логика возобновления
- Прогресс-бар
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import os
import json

from generate_real_odds_simulations import load_real_odds_outcomes
from realistic_simulation import adaptive_constant_profit_realistic, calculate_metrics_realistic
from test_all_strategies_realistic import (
    test_linear_roi_realistic, 
    test_exponential_roi_realistic,
    test_anti_martingale_realistic,
    test_sqrt_roi_realistic,
    test_fixed_fraction_realistic
)
from kelly_correct import kelly_correct_realistic

# Глобальные данные
outcomes_global = None
odds_array_global = None

def init_worker(outcomes, odds_array):
    global outcomes_global, odds_array_global
    outcomes_global = outcomes
    odds_array_global = odds_array


def run_single_test(args):
    """Универсальная функция для запуска любого теста"""
    strategy, params = args
    
    try:
        if strategy == 'adaptive_constant_profit':
            k = params['k']
            br, _, _, _, _ = adaptive_constant_profit_realistic(
                outcomes_global, odds_array_global,
                min_roi=4.733, max_roi=23.005,
                min_target_pct=3.982 * k,
                max_target_pct=13.078 * k,
                max_bet_percent=20.0 * k,
                apply_variation=True,
                recalc_min=30, recalc_max=70
            )
            param_str = f'k={k:.2f}_var'
            
        elif strategy == 'anti_martingale':
            base, mult = params['base'], params['mult']
            br = test_anti_martingale_realistic(
                outcomes_global, odds_array_global,
                base_percent=base, multiplier=mult, max_percent=10.0,
                recalc_min=30, recalc_max=70
            )
            param_str = f'base={base}%_m={mult}'
            
        elif strategy == 'linear_roi':
            base = params['base']
            br = test_linear_roi_realistic(
                outcomes_global, odds_array_global,
                base_roi=5.0, base_percent=base, max_percent=10.0,
                recalc_min=30, recalc_max=70
            )
            param_str = f'base={base}%'
            
        elif strategy == 'exponential_roi':
            base, exp = params['base'], params['exp']
            br = test_exponential_roi_realistic(
                outcomes_global, odds_array_global,
                base_roi=5.0, base_percent=base, exponent=exp, max_percent=10.0,
                recalc_min=30, recalc_max=70
            )
            param_str = f'base={base}%_e={exp}'
            
        elif strategy == 'sqrt_roi':
            base = params['base']
            br = test_sqrt_roi_realistic(
                outcomes_global, odds_array_global,
                base_roi=5.0, base_percent=base, max_percent=10.0,
                recalc_min=30, recalc_max=70
            )
            param_str = f'base={base}%'
            
        elif strategy == 'fixed_fraction':
            pct = params['pct']
            br = test_fixed_fraction_realistic(
                outcomes_global, odds_array_global,
                fixed_percent=pct,
                recalc_min=30, recalc_max=70
            )
            param_str = f'{pct}%'
            
        elif strategy == 'kelly_correct':
            frac = params['fraction']
            br = kelly_correct_realistic(
                outcomes_global, odds_array_global,
                kelly_fraction=frac,
                recalc_min=30, recalc_max=70
            )
            param_str = f'frac={frac}'
        
        m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
        
        return {
            'strategy': strategy,
            'params': param_str,
            'params_json': json.dumps(params),  # Для точного сравнения
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct']
        }
    except Exception as e:
        return {
            'strategy': strategy,
            'params': str(params),
            'params_json': json.dumps(params),
            'profit': 0,
            'bankrupt': 100,
            'dd50': 0,
            'dd80': 0,
            'worst_dd': -100,
            'error': str(e)
        }


if __name__ == '__main__':
    print("="*100)
    print("⚔️ БОЕВЫЕ ЭКСПЕРИМЕНТЫ - ФИНАЛЬНАЯ ВЕРСИЯ")
    print("="*100)
    
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    
    num_workers = min(16, cpu_count() * 2)  # Используем гиперпотоки
    print(f"CPU ядер: {cpu_count()}, используем процессов: {num_workers}")
    
    print("\n⚙️  БОЕВЫЕ УСЛОВИЯ:")
    print("   ✅ Вариация ставок: 35-115% (рандом каждая ставка)")
    print("   ✅ Пересчет банка: раз в 30-70 ставок (рандом)")
    print("   ✅ Max ставка: 10% от текущего банка (жестко)")
    print("   ✅ Автосохранение: каждые 20 тестов")
    
    # Генерация заданий
    print("\n📋 Генерация заданий...")
    tasks = []
    
    # 1. adaptive_constant_profit
    k_values = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3, 
                0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0]
    for k in k_values:
        tasks.append(('adaptive_constant_profit', {'k': k}))
    
    # 2. anti_martingale
    for base in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]:
        for mult in [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.5, 3.0]:
            tasks.append(('anti_martingale', {'base': base, 'mult': mult}))
    
    # 3. linear_roi
    for base in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        tasks.append(('linear_roi', {'base': base}))
    
    # 4. exponential_roi
    for base in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        for exp in [1.3, 1.5, 1.7, 2.0, 2.5, 3.0]:
            tasks.append(('exponential_roi', {'base': base, 'exp': exp}))
    
    # 5. sqrt_roi
    for base in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        tasks.append(('sqrt_roi', {'base': base}))
    
    # 6. fixed_fraction
    for pct in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
        tasks.append(('fixed_fraction', {'pct': pct}))
    
    # 7. kelly_correct (ПРАВИЛЬНЫЙ Kelly!)
    for frac in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
                 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]:
        tasks.append(('kelly_correct', {'fraction': frac}))
    
    total_tasks = len(tasks)
    print(f"✅ Всего заданий: {total_tasks}")
    
    # Загрузка прогресса
    progress_file = 'combat_progress.csv'
    results = []
    completed = set()
    
    if os.path.exists(progress_file):
        df_prev = pd.read_csv(progress_file)
        results = df_prev.to_dict('records')
        for r in results:
            completed.add(r['params_json'])
        print(f"\n📂 Загружено результатов: {len(results)}")
    
    # Фильтруем выполненные
    tasks_to_do = [(s, p) for s, p in tasks if json.dumps(p) not in completed]
    print(f"🚀 Осталось выполнить: {len(tasks_to_do)}")
    
    if len(tasks_to_do) == 0:
        print("\n✅ Все задания выполнены!")
    else:
        start_time = time.time()
        
        with Pool(processes=num_workers, initializer=init_worker, initargs=(outcomes, odds_array)) as pool:
            for i, result in enumerate(tqdm(
                pool.imap_unordered(run_single_test, tasks_to_do),
                total=len(tasks_to_do),
                desc="Тесты",
                unit="шт"
            )):
                results.append(result)
                
                # Автосохранение каждые 20
                if (i + 1) % 20 == 0:
                    pd.DataFrame(results).to_csv(progress_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Завершено за {elapsed/60:.1f} мин ({elapsed/len(tasks_to_do):.1f} сек/тест)")
    
    # Финальное сохранение
    df = pd.DataFrame(results).sort_values('profit', ascending=False)
    df.to_csv('combat_results_final.csv', index=False)
    
    print("\n" + "="*100)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*100)
    
    # Фильтруем ошибки
    df = df[df['profit'] > 0]
    
    zero = df[df['bankrupt'] == 0.0].sort_values('profit', ascending=False)
    agg = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)].sort_values('profit', ascending=False)
    mega = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)].sort_values('profit', ascending=False)
    
    print(f"\nВсего: {len(df)} успешных тестов")
    print(f"  🛡️  0% банкротств: {len(zero)}")
    print(f"  ⚡ 0-5% банкротств: {len(agg)}")
    print(f"  🔥 5-25% банкротств: {len(mega)}")
    
    print("\n" + "="*100)
    print("🎯 ТРИ ЛУЧШИЕ СТРАТЕГИИ")
    print("="*100)
    
    if len(zero) > 0:
        best = zero.iloc[0]
        print(f"\n🛡️  БЕЗ СЛИВОВ (0%):")
        print(f"   {best['strategy']} ({best['params']})")
        print(f"   +{best['profit']:.0f}% | DD>50: {best['dd50']:.1f}% | Worst: {best['worst_dd']:.1f}%")
    
    if len(agg) > 0:
        best = agg.iloc[0]
        print(f"\n⚡ АГРЕССИВНАЯ (до 5%):")
        print(f"   {best['strategy']} ({best['params']})")
        print(f"   +{best['profit']:.0f}% | B: {best['bankrupt']:.2f}% | DD>50: {best['dd50']:.1f}%")
    
    if len(mega) > 0:
        best = mega.iloc[0]
        print(f"\n🔥 МЕГА АГРЕССИВНАЯ (5-25%):")
        print(f"   {best['strategy']} ({best['params']})")
        print(f"   +{best['profit']:.0f}% | B: {best['bankrupt']:.2f}% | DD>50: {best['dd50']:.1f}%")
    
    print("\n" + "="*100)
    print("✅ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print(f"📁 Результаты: combat_results_final.csv")
    print("="*100)
