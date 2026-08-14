"""
СРАВНЕНИЕ: Твоя формула VS Настоящий Kelly
"""

import numpy as np
import math

print("="*100)
print("🔍 АНАЛИЗ ТВОЕЙ ФОРМУЛЫ")
print("="*100)

# Параметры из твоего примера
ROI = 0.06  # 6%
odds = 2.0
bank = 10000
defaultRisk = 15.0
maxBetPercent = 5.0

print(f"\nУсловия:")
print(f"  ROI: {ROI*100}%")
print(f"  Коэффициент: {odds}")
print(f"  Банк: {bank} €")
print(f"  defaultRisk: {defaultRisk}")

# ===== ТВОЯ ФОРМУЛА =====
print("\n" + "="*100)
print("📐 ТВОЯ ФОРМУЛА")
print("="*100)

edge = min(ROI * 100, 8.0)  # Ограничение 8%
edgeDecimal = edge / 100

print(f"\nШаг 1: edge = min({ROI*100}%, 8%) = {edge}%")
print(f"Шаг 2: edgeDecimal = {edge}/100 = {edgeDecimal}")

logFactor = 1 - (1 / (odds / (1 + edgeDecimal)))
print(f"\nШаг 3: logFactor = 1 - (1 / (odds / (1 + edgeDecimal)))")
print(f"       = 1 - (1 / ({odds} / {1 + edgeDecimal}))")
print(f"       = 1 - (1 / {odds / (1 + edgeDecimal):.4f})")
print(f"       = 1 - {1 / (odds / (1 + edgeDecimal)):.4f}")
print(f"       = {logFactor:.4f}")

# Упростим logFactor
simplified = 1 - (1 + edgeDecimal) / odds
print(f"\nУпрощенно: logFactor = 1 - (1 + edgeDecimal) / odds")
print(f"                     = 1 - {(1 + edgeDecimal) / odds:.4f}")
print(f"                     = {simplified:.4f}")

betSizePercent = math.log10(logFactor) / math.log10(10**(-defaultRisk))
print(f"\nШаг 4: betSizePercent = log₁₀({logFactor:.4f}) / log₁₀(10^-{defaultRisk})")
print(f"       = {math.log10(logFactor):.6f} / {math.log10(10**(-defaultRisk)):.1f}")
print(f"       = {math.log10(logFactor):.6f} / {-defaultRisk:.1f}")
print(f"       = {betSizePercent:.6f}")
print(f"       = {betSizePercent * 100:.2f}%")

betSizePercent = min(betSizePercent, maxBetPercent / 100)
print(f"\nШаг 5: betSizePercent = min({betSizePercent*100:.2f}%, {maxBetPercent}%) = {betSizePercent*100:.2f}%")

betSize = betSizePercent * bank
print(f"\nШаг 6: betSize = {betSizePercent:.4f} × {bank} = {betSize:.2f} €")

roundedBetSize = round(betSize / 5) * 5
print(f"Шаг 7: roundedBetSize = {roundedBetSize} €")

print(f"\n✅ ИТОГО ПО ТВОЕЙ ФОРМУЛЕ: {roundedBetSize} € ({betSizePercent*100:.2f}%)")

# ===== НАСТОЯЩИЙ KELLY =====
print("\n" + "="*100)
print("🏆 НАСТОЯЩИЙ KELLY CRITERION")
print("="*100)

win_prob = (1 + ROI) / odds
b = odds - 1
p = win_prob
q = 1 - p

kelly_f = (b * p - q) / b
print(f"\nШаг 1: win_prob = (1 + ROI) / odds")
print(f"       = {1 + ROI} / {odds}")
print(f"       = {win_prob:.4f} ({win_prob*100:.2f}%)")

print(f"\nШаг 2: Kelly formula = (b×p - q) / b")
print(f"       где b = odds - 1 = {b}")
print(f"           p = win_prob = {p:.4f}")
print(f"           q = 1 - p = {q:.4f}")

print(f"\n       Kelly = ({b} × {p:.4f} - {q:.4f}) / {b}")
print(f"             = ({b * p:.4f} - {q:.4f}) / {b}")
print(f"             = {b * p - q:.4f} / {b}")
print(f"             = {kelly_f:.4f}")
print(f"             = {kelly_f * 100:.2f}%")

kelly_bet = kelly_f * bank
print(f"\n✅ ИТОГО ПО KELLY: {kelly_bet:.2f} € ({kelly_f*100:.2f}%)")

# ===== СРАВНЕНИЕ =====
print("\n" + "="*100)
print("⚔️  СРАВНЕНИЕ")
print("="*100)

print(f"\n{'Метод':<30} {'Ставка %':<12} {'Ставка €':<12} {'Разница'}")
print("-"*100)
print(f"{'Твоя формула':<30} {betSizePercent*100:<12.2f} {roundedBetSize:<12.0f} Базовая")
print(f"{'Настоящий Kelly':<30} {kelly_f*100:<12.2f} {kelly_bet:<12.0f} {kelly_f/betSizePercent:.2f}x больше")

fraction = betSizePercent / kelly_f
print(f"\n💡 Твоя формула = Kelly × {fraction:.4f}")
print(f"   Это похоже на fractional Kelly с коэффициентом ~{fraction:.2f}")

# ===== ТЕСТЫ НА РАЗНЫХ КОЭФФИЦИЕНТАХ =====
print("\n" + "="*100)
print("📊 ТЕСТ НА РАЗНЫХ КОЭФФИЦИЕНТАХ (ROI=6%)")
print("="*100)

print(f"\n{'Odds':<8} {'Твоя %':<12} {'Kelly %':<12} {'Твоя/Kelly':<12} {'Примечание'}")
print("-"*100)

test_odds = [1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]

for test_odds_val in test_odds:
    # Твоя формула
    logFactor = 1 - (1 + edgeDecimal) / test_odds_val
    if logFactor > 0:
        your_pct = math.log10(logFactor) / math.log10(10**(-defaultRisk))
        your_pct = min(your_pct, maxBetPercent / 100) * 100
    else:
        your_pct = 0
    
    # Настоящий Kelly
    win_prob = (1 + ROI) / test_odds_val
    b = test_odds_val - 1
    kelly = ((b * win_prob - (1 - win_prob)) / b) * 100 if b > 0 else 0
    kelly = max(0, kelly)
    
    ratio = your_pct / kelly if kelly > 0 else 0
    
    note = ""
    if your_pct >= maxBetPercent:
        note = "⚠️ Ограничено 5%"
    elif ratio < 0.3:
        note = "❌ Очень консервативно"
    elif ratio < 0.5:
        note = "⚠️ Консервативно"
    
    print(f"{test_odds_val:<8.1f} {your_pct:<12.2f} {kelly:<12.2f} {ratio:<12.2f} {note}")

print("\n" + "="*100)
print("💡 ВЫВОД")
print("="*100)
print("""
ТВОЯ ФОРМУЛА ≠ KELLY CRITERION!

Это модифицированная формула с логарифмическим сглаживанием:
1. Использует log₁₀ для уменьшения размера ставок
2. Параметр defaultRisk=15 делает её ОЧЕНЬ консервативной
3. Ограничение maxBetPercent=5% дополнительно урезает

Результат: Ставки в 2.5-3 раза МЕНЬШЕ чем настоящий Kelly!

ПЛЮСЫ:
✅ Очень консервативна (безопасно)
✅ Защита от больших просадок
✅ Хорошо для новичков

МИНУСЫ:
❌ Намного медленнее рост капитала
❌ Не оптимальна математически
❌ Упускаешь потенциал

ЕСЛИ ХОЧЕШЬ НАСТОЯЩИЙ KELLY:
win_prob = (1 + ROI) / odds
kelly = (b × win_prob - (1 - win_prob)) / b

Для консервативности используй fractional Kelly (0.25-0.50)
""")
print("="*100)
EOF
