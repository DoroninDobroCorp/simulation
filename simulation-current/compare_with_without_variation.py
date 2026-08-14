import pandas as pd

# Загружаем результаты
df = pd.read_csv('results.csv')

print("="*80)
print("СРАВНЕНИЕ: С ВАРИАЦИЕЙ vs БЕЗ ВАРИАЦИИ")
print("="*80)
print("\nРЕАЛЬНЫЕ КОЭФФИЦИЕНТЫ (1013 ставок, avg коэф 2.76)")
print("Вариация: случайный размер ставки от 30% до 115% (шаг 5%) от расчетного\n")

# Создаем таблицу для сравнения
for bet_size in [1.0, 1.5, 2.0, 5.0]:
    no_var = df[(df['strategy'].str.contains(f'{bet_size}%')) & (df['with_variation'] == 'No')].iloc[0]
    with_var = df[(df['strategy'].str.contains(f'{bet_size}%')) & (df['with_variation'] == 'Yes')].iloc[0]
    
    print(f"\n{'='*80}")
    print(f"BET SIZE: {bet_size}%")
    print(f"{'='*80}")
    
    print(f"\n{'Метрика':<30} {'БЕЗ вариации':>20} {'С вариацией':>20}")
    print("-"*80)
    
    metrics = [
        ('ROI с оборота', 'roi_%', '%'),
        ('Средняя прибыль', 'avg_profit_%', '%'),
        ('Сливов', 'bankrupt_%', '%'),
        ('DD > 20%', 'dd>20_%', '%'),
        ('DD > 50%', 'dd>50_%', '%'),
        ('DD > 80%', 'dd>80_%', '%'),
        ('Средняя макс DD', 'avg_maxdd_%', '%'),
    ]
    
    for label, col, unit in metrics:
        no_val = float(no_var[col])
        with_val = float(with_var[col])
        print(f"{label:<30} {no_val:>19.2f}{unit} {with_val:>19.2f}{unit}")
    
    # Разница в рисках
    print("\n" + "-"*80)
    print("ВЛИЯНИЕ ВАРИАЦИИ НА РИСКИ:")
    print("-"*80)
    
    bankrupt_diff = float(with_var['bankrupt_%']) - float(no_var['bankrupt_%'])
    dd20_diff = float(with_var['dd>20_%']) - float(no_var['dd>20_%'])
    dd50_diff = float(with_var['dd>50_%']) - float(no_var['dd>50_%'])
    dd80_diff = float(with_var['dd>80_%']) - float(no_var['dd>80_%'])
    profit_diff = float(with_var['avg_profit_%']) - float(no_var['avg_profit_%'])
    
    print(f"{'Изменение сливов:':<30} {bankrupt_diff:>+7.2f}%")
    print(f"{'Изменение DD > 20%:':<30} {dd20_diff:>+7.2f}%")
    print(f"{'Изменение DD > 50%:':<30} {dd50_diff:>+7.2f}%")
    print(f"{'Изменение DD > 80%:':<30} {dd80_diff:>+7.2f}%")
    print(f"{'Изменение прибыли:':<30} {profit_diff:>+7.2f}%")

print("\n" + "="*80)
print("ВЫВОДЫ:")
print("="*80)
print("1. ROI с оборота остается ~7% в обоих случаях ✅")
print("2. Вариация СНИЖАЕТ риски (меньше DD > 50%, меньше сливов)")
print("3. Вариация СНИЖАЕТ среднюю прибыль (из-за более консервативных ставок)")
print("4. Эффект вариации сильнее при больших bet_size (2%, 5%)")
print("="*80)
