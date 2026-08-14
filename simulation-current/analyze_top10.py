"""
ТОП-10 СТРАТЕГИЙ ПО РАЗНЫМ КАТЕГОРИЯМ
"""

import pandas as pd
import json

print("="*120)
print("🏆 ТОП-10 СТРАТЕГИЙ ПО РАЗНЫМ КАТЕГОРИЯМ")
print("="*120)

df = pd.read_csv('combat_results_final.csv')
df = df[df['profit'] > 0]  # Только успешные

# Добавляем k для adaptive
df['k_value'] = df.apply(
    lambda r: json.loads(r['params_json']).get('k', None) if r['strategy'] == 'adaptive_constant_profit' else None,
    axis=1
)

print(f"\nВсего проанализировано: {len(df)} успешных вариантов")

# =============================================================================
# 1. ДЛЯ БОЛЬШИХ БАНКОВ (100k+ €) - БЕЗОПАСНОСТЬ
# =============================================================================
print("\n" + "="*120)
print("1️⃣ ДЛЯ БОЛЬШИХ БАНКОВ (100,000+ €) - БЕЗОПАСНОСТЬ ПРЕВЫШЕ ВСЕГО")
print("="*120)
print("Критерии: 0% сливов + минимальная просадка (Worst DD)")

safe = df[df['bankrupt'] == 0.0].copy()
safe = safe.sort_values('worst_dd', ascending=False)  # Сортируем по лучшему worst_dd

print(f"\n{'#':<4} {'Стратегия':<30} {'Параметры':<25} {'Profit':<12} {'Worst DD':<12} {'DD>50%':<10} {'На 100k €'}")
print("-"*120)

for i, (_, r) in enumerate(safe.head(10).iterrows(), 1):
    profit_100k = 100000 * r['profit'] / 100
    print(f"{i:<4} {r['strategy']:<30} {r['params']:<25} +{r['profit']:<11.0f} {r['worst_dd']:<12.1f} {r['dd50']:<10.1f} +{profit_100k:>10,.0f} €")

if len(safe) > 0:
    best = safe.iloc[0]
    print(f"\n💎 ПОБЕДИТЕЛЬ: {best['strategy']} ({best['params']})")
    print(f"   → Только +{best['profit']:.0f}% прибыль, но worst DD всего {best['worst_dd']:.1f}%")

# =============================================================================
# 2. БАЛАНС ПРИБЫЛЬ/БЕЗОПАСНОСТЬ
# =============================================================================
print("\n" + "="*120)
print("2️⃣ БАЛАНС ПРИБЫЛЬ/БЕЗОПАСНОСТЬ")
print("="*120)
print("Критерии: 0% сливов + Worst DD > -80% + максимальная прибыль")

balanced = df[(df['bankrupt'] == 0.0) & (df['worst_dd'] > -80)].copy()
balanced = balanced.sort_values('profit', ascending=False)

print(f"\n{'#':<4} {'Стратегия':<30} {'Параметры':<25} {'Profit':<12} {'Worst DD':<12} {'DD>80%':<10} {'На 50k €'}")
print("-"*120)

for i, (_, r) in enumerate(balanced.head(10).iterrows(), 1):
    profit_50k = 50000 * r['profit'] / 100
    print(f"{i:<4} {r['strategy']:<30} {r['params']:<25} +{r['profit']:<11.0f} {r['worst_dd']:<12.1f} {r['dd80']:<10.1f} +{profit_50k:>10,.0f} €")

if len(balanced) > 0:
    best = balanced.iloc[0]
    print(f"\n💎 ПОБЕДИТЕЛЬ: {best['strategy']} ({best['params']})")
    print(f"   → +{best['profit']:.0f}% прибыль при worst DD {best['worst_dd']:.1f}%")

# =============================================================================
# 3. МАКСИМУМ ПРИБЫЛИ БЕЗ СЛИВОВ
# =============================================================================
print("\n" + "="*120)
print("3️⃣ МАКСИМУМ ПРИБЫЛИ БЕЗ СЛИВОВ")
print("="*120)
print("Критерии: 0% банкротств + максимальная прибыль (любой Worst DD)")

max_profit_zero = df[df['bankrupt'] == 0.0].copy()
max_profit_zero = max_profit_zero.sort_values('profit', ascending=False)

print(f"\n{'#':<4} {'Стратегия':<30} {'Параметры':<25} {'Profit':<12} {'Worst DD':<12} {'DD>80%':<10} {'На 10k €'}")
print("-"*120)

for i, (_, r) in enumerate(max_profit_zero.head(10).iterrows(), 1):
    profit_10k = 10000 * r['profit'] / 100
    print(f"{i:<4} {r['strategy']:<30} {r['params']:<25} +{r['profit']:<11.0f} {r['worst_dd']:<12.1f} {r['dd80']:<10.1f} +{profit_10k:>10,.0f} €")

if len(max_profit_zero) > 0:
    best = max_profit_zero.iloc[0]
    print(f"\n💎 ПОБЕДИТЕЛЬ: {best['strategy']} ({best['params']})")
    print(f"   → +{best['profit']:.0f}% прибыль (x{best['profit']/100 + 1:.1f}!)")
    print(f"   ⚠️  НО worst DD {best['worst_dd']:.1f}% - может упасть почти до нуля!")

# =============================================================================
# 4. АГРЕССИВНАЯ (до 5% сливов)
# =============================================================================
print("\n" + "="*120)
print("4️⃣ АГРЕССИВНАЯ (до 5% банкротств)")
print("="*120)
print("Критерии: bankrupt 0-5% + максимальная прибыль")

aggressive = df[(df['bankrupt'] > 0) & (df['bankrupt'] <= 5.0)].copy()
aggressive = aggressive.sort_values('profit', ascending=False)

print(f"\n{'#':<4} {'Стратегия':<30} {'Параметры':<25} {'Profit':<12} {'Bankrupt':<12} {'DD>50%':<10} {'На 1k €'}")
print("-"*120)

for i, (_, r) in enumerate(aggressive.head(10).iterrows(), 1):
    profit_1k = 1000 * r['profit'] / 100
    bankrupt_ratio = f"1 из {int(100/r['bankrupt'])}"
    print(f"{i:<4} {r['strategy']:<30} {r['params']:<25} +{r['profit']:<11.0f} {r['bankrupt']:<5.2f}% ({bankrupt_ratio}) {r['dd50']:<10.1f} +{profit_1k:>8,.0f} €")

if len(aggressive) > 0:
    best = aggressive.iloc[0]
    print(f"\n💎 ПОБЕДИТЕЛЬ: {best['strategy']} ({best['params']})")
    print(f"   → +{best['profit']:.0f}% прибыль")
    print(f"   → {best['bankrupt']:.2f}% сливов (1 из {int(100/best['bankrupt'])})")

# =============================================================================
# 5. МЕГА АГРЕССИВНАЯ (5-25% сливов)
# =============================================================================
print("\n" + "="*120)
print("5️⃣ МЕГА АГРЕССИВНАЯ (5-25% банкротств)")
print("="*120)
print("Критерии: bankrupt 5-25% + максимальная прибыль")

mega = df[(df['bankrupt'] > 5.0) & (df['bankrupt'] <= 25.0)].copy()
mega = mega.sort_values('profit', ascending=False)

print(f"\n{'#':<4} {'Стратегия':<30} {'Параметры':<25} {'Profit':<12} {'Bankrupt':<12} {'На 100 €'}")
print("-"*120)

for i, (_, r) in enumerate(mega.head(10).iterrows(), 1):
    profit_100 = 100 * r['profit'] / 100
    bankrupt_ratio = f"1 из {int(100/r['bankrupt'])}"
    print(f"{i:<4} {r['strategy']:<30} {r['params']:<25} +{r['profit']:<11.0f} {r['bankrupt']:<5.2f}% ({bankrupt_ratio}) +{profit_100:>8,.0f} €")

if len(mega) > 0:
    best = mega.iloc[0]
    print(f"\n💎 ПОБЕДИТЕЛЬ: {best['strategy']} ({best['params']})")
    print(f"   → +{best['profit']:.0f}% прибыль (x{best['profit']/100 + 1:.0f}!)")
    print(f"   → {best['bankrupt']:.2f}% сливов (1 из {int(100/best['bankrupt'])})")

# =============================================================================
# 6. МИНИМАЛЬНЫЕ ПРОСАДКИ (DD>50%)
# =============================================================================
print("\n" + "="*120)
print("6️⃣ МИНИМАЛЬНЫЕ ПРОСАДКИ (редкие DD>50%)")
print("="*120)
print("Критерии: 0% сливов + минимальный DD>50% + хорошая прибыль")

low_dd = df[df['bankrupt'] == 0.0].copy()
low_dd = low_dd.sort_values(['dd50', 'profit'], ascending=[True, False])

print(f"\n{'#':<4} {'Стратегия':<30} {'Параметры':<25} {'Profit':<12} {'DD>50%':<10} {'DD>80%':<10} {'Worst DD'}")
print("-"*120)

for i, (_, r) in enumerate(low_dd.head(10).iterrows(), 1):
    print(f"{i:<4} {r['strategy']:<30} {r['params']:<25} +{r['profit']:<11.0f} {r['dd50']:<10.1f} {r['dd80']:<10.1f} {r['worst_dd']:.1f}%")

if len(low_dd) > 0:
    best = low_dd.iloc[0]
    print(f"\n💎 ПОБЕДИТЕЛЬ: {best['strategy']} ({best['params']})")
    print(f"   → DD>50% всего в {best['dd50']:.1f}% случаев!")
    print(f"   → Прибыль +{best['profit']:.0f}%")

# =============================================================================
# 7. ABSOLUTE TOP-10 (по прибыли, любые условия)
# =============================================================================
print("\n" + "="*120)
print("7️⃣ АБСОЛЮТНЫЙ ТОП-10 (максимальная прибыль, любые риски)")
print("="*120)

absolute_top = df.sort_values('profit', ascending=False)

print(f"\n{'#':<4} {'Стратегия':<30} {'Параметры':<25} {'Profit':<12} {'Bankrupt':<10} {'DD>80%'}")
print("-"*120)

for i, (_, r) in enumerate(absolute_top.head(10).iterrows(), 1):
    print(f"{i:<4} {r['strategy']:<30} {r['params']:<25} +{r['profit']:<11.0f} {r['bankrupt']:<10.2f} {r['dd80']:.1f}%")

# =============================================================================
# ИТОГОВАЯ ТАБЛИЦА ПОБЕДИТЕЛЕЙ
# =============================================================================
print("\n" + "="*120)
print("🏆 ПОБЕДИТЕЛИ В КАЖДОЙ КАТЕГОРИИ")
print("="*120)

winners = []

if len(safe) > 0:
    w = safe.iloc[0]
    winners.append({
        'category': '🛡️  Безопасность',
        'strategy': w['strategy'],
        'params': w['params'],
        'profit': w['profit'],
        'bankrupt': w['bankrupt'],
        'worst_dd': w['worst_dd'],
        'for_bank': '100,000 €'
    })

if len(balanced) > 0:
    w = balanced.iloc[0]
    winners.append({
        'category': '⚖️  Баланс',
        'strategy': w['strategy'],
        'params': w['params'],
        'profit': w['profit'],
        'bankrupt': w['bankrupt'],
        'worst_dd': w['worst_dd'],
        'for_bank': '50,000 €'
    })

if len(max_profit_zero) > 0:
    w = max_profit_zero.iloc[0]
    winners.append({
        'category': '💎 Max без сливов',
        'strategy': w['strategy'],
        'params': w['params'],
        'profit': w['profit'],
        'bankrupt': w['bankrupt'],
        'worst_dd': w['worst_dd'],
        'for_bank': '10,000 €'
    })

if len(aggressive) > 0:
    w = aggressive.iloc[0]
    winners.append({
        'category': '⚡ Агрессивная',
        'strategy': w['strategy'],
        'params': w['params'],
        'profit': w['profit'],
        'bankrupt': w['bankrupt'],
        'worst_dd': w['worst_dd'],
        'for_bank': '1,000 €'
    })

if len(mega) > 0:
    w = mega.iloc[0]
    winners.append({
        'category': '🔥 Мега',
        'strategy': w['strategy'],
        'params': w['params'],
        'profit': w['profit'],
        'bankrupt': w['bankrupt'],
        'worst_dd': w['worst_dd'],
        'for_bank': '100 €'
    })

print(f"\n{'Категория':<20} {'Для банка':<15} {'Стратегия':<30} {'Параметры':<20} {'Profit':<10} {'Bankrupt'}")
print("-"*120)

for w in winners:
    print(f"{w['category']:<20} {w['for_bank']:<15} {w['strategy']:<30} {w['params']:<20} +{w['profit']:<9.0f} {w['bankrupt']:.2f}%")

print("\n" + "="*120)
print("✅ АНАЛИЗ ЗАВЕРШЕН!")
print("="*120)
print(f"\n📁 Все данные в: combat_results_final.csv ({len(df)} вариантов)")
print("="*120)
