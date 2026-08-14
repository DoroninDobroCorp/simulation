"""
Проверка стратегий на дубликаты - находим стратегии с идентичными результатами.
"""

import pandas as pd
import numpy as np

# Читаем консервативные стратегии
df = pd.read_csv('results_conservative_DD50.csv')

print("="*80)
print("🔍 ПОИСК СТРАТЕГИЙ С ОДИНАКОВЫМИ РЕЗУЛЬТАТАМИ")
print("="*80)

# Удаляем вариации - сравниваем только базовые
df_no_var = df[df['with_variation'] == 'No'].copy()

print(f"\nВсего стратегий без вариаций: {len(df_no_var)}")

# Группируем по ключевым метрикам
key_columns = ['avg_profit_%', 'dd>20_%', 'dd>50_%', 'dd>80_%', 'bankrupt_%', 
               'avg_bet_%', 'roi_%']

# Округляем для сравнения (до 2 знаков)
for col in key_columns:
    df_no_var[f'{col}_rounded'] = df_no_var[col].round(2)

# Группируем по округленным значениям
rounded_cols = [f'{col}_rounded' for col in key_columns]
groups = df_no_var.groupby(rounded_cols)

print("\n" + "="*80)
print("🔍 ГРУППЫ СТРАТЕГИЙ С ИДЕНТИЧНЫМИ РЕЗУЛЬТАТАМИ:")
print("="*80)

duplicates_found = 0

for name, group in groups:
    if len(group) > 1:
        duplicates_found += 1
        print(f"\n{'='*80}")
        print(f"ГРУППА {duplicates_found}: {len(group)} стратегий с ОДИНАКОВЫМИ результатами")
        print(f"{'='*80}")
        print(f"Profit: {group.iloc[0]['avg_profit_%']:.2f}%")
        print(f"DD>50%: {group.iloc[0]['dd>50_%']:.2f}%")
        print(f"Bankrupt: {group.iloc[0]['bankrupt_%']:.2f}%")
        print(f"\nСтратегии:")
        for i, (idx, row) in enumerate(group.iterrows(), 1):
            print(f"  {i}. {row['strategy']}")

if duplicates_found == 0:
    print("\n✅ Дубликатов не найдено! Все стратегии уникальны.")
else:
    print(f"\n⚠️ Найдено {duplicates_found} групп дубликатов!")
    print("\nЭто могут быть:")
    print("  1. Баги в коде (стратегии реально одинаковые)")
    print("  2. Разные подходы дающие случайно одинаковый результат")
    print("  3. Разные параметры одной стратегии дающие одинаковый результат")

# Точная проверка на 100% совпадение (до 6 знаков)
print("\n" + "="*80)
print("🔍 ТОЧНАЯ ПРОВЕРКА (до 6 знаков после запятой):")
print("="*80)

for col in key_columns:
    df_no_var[f'{col}_exact'] = df_no_var[col].round(6)

exact_cols = [f'{col}_exact' for col in key_columns]
exact_groups = df_no_var.groupby(exact_cols)

exact_duplicates = 0
for name, group in exact_groups:
    if len(group) > 1:
        exact_duplicates += 1
        print(f"\nТОЧНЫЙ ДУБЛИКАТ {exact_duplicates}:")
        for i, (idx, row) in enumerate(group.iterrows(), 1):
            print(f"  {i}. {row['strategy']}")

if exact_duplicates == 0:
    print("\n✅ 100% идентичных стратегий не найдено.")
else:
    print(f"\n⚠️ Найдено {exact_duplicates} ТОЧНЫХ дубликатов - это БАГИ!")

print("\n" + "="*80)
print("💡 РЕКОМЕНДАЦИИ:")
print("="*80)

if exact_duplicates > 0:
    print("1. Проверить код стратегий с идентичными результатами")
    print("2. Убедиться что они реально используют разную логику")
    print("3. Если логика одинаковая - удалить дубликат или исправить")
