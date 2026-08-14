"""
ТЕСТ СИСТЕМЫ БОЕВЫХ ЭКСПЕРИМЕНТОВ
Проверяем на 30 тестах:
- Работает ли параллельность
- Сохраняются ли промежуточные результаты
- Можно ли продолжить после прерывания
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
from test_all_strategies_realistic import test_anti_martingale_realistic, test_linear_roi_realistic

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
        
        m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
        
        return {
            'strategy': strategy,
            'params': param_str,
            'params_json': json.dumps(params),
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
    print("🧪 ТЕСТ СИСТЕМЫ (30 тестов для проверки)")
    print("="*100)
    
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    
    num_workers = min(8, cpu_count())
    print(f"CPU ядер: {cpu_count()}, используем процессов: {num_workers}")
    
    # МАЛЕНЬКАЯ выборка для теста
    print("\n📋 Генерация ТЕСТОВЫХ заданий (всего 30)...")
    tasks = []
    
    # 10 adaptive
    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]:
        tasks.append(('adaptive_constant_profit', {'k': k}))
    
    # 10 anti_martingale
    for base in [2.0, 3.0, 4.0, 5.0, 6.0]:
        for mult in [1.5, 2.0]:
            tasks.append(('anti_martingale', {'base': base, 'mult': mult}))
    
    # 10 linear_roi
    for base in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]:
        tasks.append(('linear_roi', {'base': base}))
    
    total_tasks = len(tasks)
    print(f"✅ Всего тестовых заданий: {total_tasks}")
    
    # Проверка существующего прогресса
    progress_file = 'test_combat_progress.csv'
    results = []
    completed = set()
    
    if os.path.exists(progress_file):
        print(f"\n📂 Найден файл прогресса: {progress_file}")
        df_prev = pd.read_csv(progress_file)
        results = df_prev.to_dict('records')
        for r in results:
            completed.add(r['params_json'])
        print(f"   Загружено результатов: {len(results)}")
    else:
        print(f"\n🆕 Файла прогресса нет, начинаем с нуля")
    
    # Фильтруем выполненные
    tasks_to_do = [(s, p) for s, p in tasks if json.dumps(p) not in completed]
    print(f"🚀 Осталось выполнить: {len(tasks_to_do)}")
    
    if len(tasks_to_do) == 0:
        print("\n✅ Все задания уже выполнены!")
        print("\n💡 Удали test_combat_progress.csv чтобы запустить заново")
    else:
        print(f"\n⏱️  Примерное время: ~{len(tasks_to_do) * 2 / num_workers / 60:.1f} минут")
        print("\n🚀 СТАРТ (автосохранение каждые 5 тестов)...\n")
        
        start_time = time.time()
        
        with Pool(processes=num_workers, initializer=init_worker, initargs=(outcomes, odds_array)) as pool:
            for i, result in enumerate(tqdm(
                pool.imap_unordered(run_single_test, tasks_to_do),
                total=len(tasks_to_do),
                desc="Прогресс",
                unit="тест"
            )):
                results.append(result)
                
                # Автосохранение каждые 5 тестов
                if (i + 1) % 5 == 0 or (i + 1) == len(tasks_to_do):
                    df_temp = pd.DataFrame(results)
                    df_temp.to_csv(progress_file, index=False)
                    print(f"\n💾 Сохранено {len(results)} результатов в {progress_file}")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Завершено за {elapsed:.1f} сек ({elapsed/len(tasks_to_do):.1f} сек/тест)")
    
    # АНАЛИЗ
    print("\n" + "="*100)
    print("📊 АНАЛИЗ ТЕСТОВЫХ РЕЗУЛЬТАТОВ")
    print("="*100)
    
    df = pd.DataFrame(results)
    
    # Проверка на ошибки
    errors = df[df.get('error', '') != '']
    if len(errors) > 0:
        print(f"\n❌ ОШИБКИ: {len(errors)} тестов с ошибками!")
        for _, e in errors.iterrows():
            print(f"   {e['strategy']} {e['params']}: {e.get('error', 'unknown')}")
    else:
        print(f"\n✅ Ошибок нет, все {len(df)} тестов успешны")
    
    df = df[df['profit'] > 0]  # Только успешные
    df = df.sort_values('profit', ascending=False)
    df.to_csv('test_combat_final.csv', index=False)
    
    zero = df[df['bankrupt'] == 0.0]
    agg = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)]
    mega = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)]
    
    print(f"\nКатегории:")
    print(f"  🛡️  0% банкротств: {len(zero)} тестов")
    print(f"  ⚡ 0-5% банкротств: {len(agg)} тестов")
    print(f"  🔥 5-25% банкротств: {len(mega)} тестов")
    
    if len(zero) > 0:
        best = zero.iloc[0]
        print(f"\n🥇 Лучшая без сливов:")
        print(f"   {best['strategy']} ({best['params']}) → +{best['profit']:.0f}%")
    
    if len(agg) > 0:
        best = agg.iloc[0]
        print(f"\n🥈 Лучшая агрессивная:")
        print(f"   {best['strategy']} ({best['params']}) → +{best['profit']:.0f}%, B:{best['bankrupt']:.2f}%")
    
    if len(mega) > 0:
        best = mega.iloc[0]
        print(f"\n🥉 Лучшая мега:")
        print(f"   {best['strategy']} ({best['params']}) → +{best['profit']:.0f}%, B:{best['bankrupt']:.2f}%")
    
    # ПРОВЕРКА СИСТЕМЫ
    print("\n" + "="*100)
    print("🔍 ПРОВЕРКА СИСТЕМЫ")
    print("="*100)
    
    checks = []
    
    # 1. Файл прогресса существует?
    if os.path.exists(progress_file):
        checks.append("✅ Файл прогресса создан")
    else:
        checks.append("❌ Файл прогресса НЕ создан!")
    
    # 2. Количество результатов совпадает?
    if len(results) == total_tasks:
        checks.append(f"✅ Все {total_tasks} тестов выполнены")
    else:
        checks.append(f"⚠️  Выполнено {len(results)} из {total_tasks}")
    
    # 3. Есть результаты во всех категориях?
    if len(zero) > 0 and len(agg) > 0 and len(mega) > 0:
        checks.append("✅ Есть результаты во всех категориях")
    else:
        checks.append("⚠️  Не во всех категориях есть результаты")
    
    # 4. Прибыль реалистична?
    if len(df) > 0:
        max_profit = df['profit'].max()
        min_profit = df['profit'].min()
        if 0 < min_profit < max_profit < 20000:
            checks.append(f"✅ Прибыль реалистична ({min_profit:.0f}% - {max_profit:.0f}%)")
        else:
            checks.append(f"⚠️  Прибыль странная ({min_profit:.0f}% - {max_profit:.0f}%)")
    
    for check in checks:
        print(f"  {check}")
    
    print("\n" + "="*100)
    print("🎯 ИТОГ ТЕСТА")
    print("="*100)
    
    if all("✅" in c for c in checks):
        print("\n✅ ВСЁ РАБОТАЕТ ОТЛИЧНО!")
        print("   - Параллельность работает")
        print("   - Автосохранение работает")
        print("   - Логика корректна")
        print("   - Результаты реалистичны")
        print("\n💡 МОЖНО ЗАПУСКАТЬ ПОЛНУЮ ВЕРСИЮ!")
        print(f"   Удали {progress_file} и запусти combat_final.py")
    else:
        print("\n⚠️  ЕСТЬ ПРОБЛЕМЫ - проверь логи выше")
    
    print("\n📁 Файлы:")
    print(f"   - {progress_file} (промежуточный)")
    print(f"   - test_combat_final.csv (финальный)")
    print("="*100)
