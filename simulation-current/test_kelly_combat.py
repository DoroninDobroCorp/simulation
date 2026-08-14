"""
БЫСТРЫЙ ТЕСТ ПРАВИЛЬНОГО KELLY
Протестируем только Kelly и добавим к существующим результатам
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import json
import os

from generate_real_odds_simulations import load_real_odds_outcomes
from kelly_correct import kelly_correct_realistic, calculate_metrics_quick
from realistic_simulation import calculate_metrics_realistic

# Глобальные данные
outcomes_global = None
odds_array_global = None

def init_worker(outcomes, odds_array):
    global outcomes_global, odds_array_global
    outcomes_global = outcomes
    odds_array_global = odds_array


def run_kelly_test(args):
    """Тест одного Kelly варианта"""
    fraction = args
    
    try:
        br = kelly_correct_realistic(
            outcomes_global, odds_array_global,
            kelly_fraction=fraction,
            recalc_min=30, recalc_max=70
        )
        
        m = calculate_metrics_realistic(br, np.zeros_like(outcomes_global, dtype=float), odds_array_global)
        
        return {
            'strategy': 'kelly_correct',
            'params': f'frac={fraction}',
            'params_json': json.dumps({'fraction': fraction}),
            'profit': m['avg_profit_pct'],
            'bankrupt': m['bankrupt_pct'],
            'dd50': m['drawdown_50_pct'],
            'dd80': m['drawdown_80_pct'],
            'worst_dd': m['worst_drawdown_pct']
        }
    except Exception as e:
        return {
            'strategy': 'kelly_correct',
            'params': f'frac={fraction}',
            'params_json': json.dumps({'fraction': fraction}),
            'profit': 0,
            'bankrupt': 100,
            'dd50': 0,
            'dd80': 0,
            'worst_dd': -100,
            'error': str(e)
        }


if __name__ == '__main__':
    print("="*100)
    print("⚔️ ТЕСТ ПРАВИЛЬНОГО KELLY - БОЕВЫЕ УСЛОВИЯ")
    print("="*100)
    
    outcomes, odds_array = load_real_odds_outcomes()
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    
    num_workers = min(16, cpu_count() * 2)
    print(f"CPU ядер: {cpu_count()}, используем процессов: {num_workers}")
    
    print("\n⚙️  БОЕВЫЕ УСЛОВИЯ:")
    print("   ✅ Вариация ставок: 35-115% (рандом каждая ставка)")
    print("   ✅ Пересчет банка: раз в 30-70 ставок (рандом)")
    print("   ✅ Max ставка: 10% от текущего банка (жестко)")
    
    # Kelly fractions для теста
    fractions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
                 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
    
    print(f"\n📋 Тестируем {len(fractions)} вариантов Kelly Criterion")
    print(f"⏱️  Примерное время: ~{len(fractions) * 3 / num_workers / 60:.1f} минут")
    
    start_time = time.time()
    results = []
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(outcomes, odds_array)) as pool:
        for result in tqdm(
            pool.imap_unordered(run_kelly_test, fractions),
            total=len(fractions),
            desc="Kelly тесты",
            unit="шт"
        ):
            results.append(result)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Завершено за {elapsed/60:.1f} мин ({elapsed/len(fractions):.1f} сек/тест)")
    
    # Сохраняем
    df_kelly = pd.DataFrame(results).sort_values('profit', ascending=False)
    df_kelly.to_csv('kelly_results.csv', index=False)
    
    print("\n" + "="*100)
    print("📊 РЕЗУЛЬТАТЫ KELLY")
    print("="*100)
    
    # Категории
    zero = df_kelly[df_kelly['bankrupt'] == 0.0].sort_values('profit', ascending=False)
    agg = df_kelly[(df_kelly['bankrupt'] > 0) & (df_kelly['bankrupt'] <= 5.0)].sort_values('profit', ascending=False)
    
    print(f"\n🛡️  0% банкротств: {len(zero)} вариантов")
    print(f"⚡ До 5% банкротств: {len(agg)} вариантов")
    
    if len(zero) > 0:
        print(f"\n📊 ТОП-10 БЕЗ СЛИВОВ:")
        print(f"\n{'#':<4} {'Fraction':<12} {'Profit':<12} {'DD>50%':<10} {'DD>80%':<10} {'Worst DD'}")
        print("-"*100)
        for i, (_, r) in enumerate(zero.head(10).iterrows(), 1):
            frac = json.loads(r['params_json'])['fraction']
            print(f"{i:<4} {frac:<12.2f} +{r['profit']:<11.0f} {r['dd50']:<10.1f} {r['dd80']:<10.1f} {r['worst_dd']:.1f}%")
        
        best = zero.iloc[0]
        best_frac = json.loads(best['params_json'])['fraction']
        print(f"\n💎 ЛУЧШИЙ Kelly:")
        print(f"   Fraction: {best_frac}")
        print(f"   Profit: +{best['profit']:.0f}%")
        print(f"   Bankrupt: 0%")
        print(f"   Worst DD: {best['worst_dd']:.1f}%")
    
    # Объединяем с существующими
    print("\n" + "="*100)
    print("📁 ОБЪЕДИНЕНИЕ С СУЩЕСТВУЮЩИМИ РЕЗУЛЬТАТАМИ")
    print("="*100)
    
    if os.path.exists('combat_results_final.csv'):
        df_old = pd.read_csv('combat_results_final.csv')
        print(f"   Старых результатов: {len(df_old)}")
        print(f"   Новых Kelly: {len(df_kelly)}")
        
        df_combined = pd.concat([df_old, df_kelly], ignore_index=True).sort_values('profit', ascending=False)
        df_combined.to_csv('combat_results_with_kelly.csv', index=False)
        
        print(f"   ✅ Объединено: {len(df_combined)} результатов")
        print(f"   ✅ Сохранено в: combat_results_with_kelly.csv")
    else:
        print(f"   ⚠️  Файл combat_results_final.csv не найден")
        print(f"   ✅ Сохранены только Kelly: kelly_results.csv")
    
    print("\n" + "="*100)
    print("✅ ТЕСТ ЗАВЕРШЕН!")
    print("="*100)
