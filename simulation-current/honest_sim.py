"""
Чистая честная симуляция динамического банкролл-менеджмента (с нуля).

Принципы:
- Дистанция: 1000 ставок (реалистично, как ~1013 реальных в matching_teams_clean.csv).
- Реальные кэфы (семпл с повторениями из реальных, avg ~2.76).
- ROI = +2.5% зафиксирован в КАЖДОЙ симуляции: один базовый набор (кэф, исход)
  с суммарным ROI ровно +2.5%, каждая симуляция — случайная ПЕРЕСТАНОВКА порядка.
  ROI инвариантен к перестановке -> у всех симуляций одинаковый итоговый edge,
  различается только удача в ПОРЯДКЕ (когда выпадают полосы).
- Динамический БМ: ставка = f(стратегия) * ЖИВОЙ банк. Банк не уходит в минус.
- Жёсткое правило: ставка <= 10% живого банка.
- Просадка: классический max drawdown от пика за всю историю.

Стратегии — прозрачные функции доли ставки на шаге.
"""

import numpy as np

INIT = 1000.0
ROI = 0.025
CAP = 0.10


def load_real_odds(path='matching_teams_clean.csv'):
    import pandas as pd
    return pd.read_csv(path)['coefficient'].values.astype(float)


def build_base(num_bets, roi=ROI, seed=42):
    """Базовый набор (odds, win) с суммарным ROI ровно roi (коррекцией флипами)."""
    rng = np.random.default_rng(seed)
    real = load_real_odds()
    odds = rng.choice(real, size=num_bets, replace=True)
    p = np.clip((1.0 + roi) / odds, 0, 1)
    wins = rng.random(num_bets) < p

    target = roi * num_bets  # целевая суммарная прибыль (ед. ставки)

    def profit(w):
        return (w * (odds - 1) - (~w)).sum()

    for _ in range(5000):
        d = profit(wins) - target
        if abs(d) < 0.5:
            break
        if d > 0:  # слишком прибыльно -> убрать победу на низком кэфе (дёшево)
            idx = np.where(wins)[0]
            wins[idx[np.argmin(odds[idx])]] = False
        else:      # мало прибыли -> добавить победу на высоком кэфе
            idx = np.where(~wins)[0]
            wins[idx[np.argmax(odds[idx])]] = True
    return odds, wins, profit(wins) / num_bets * 100


def simulate(frac_func, odds, base_wins, num_sims, seed=0, cap=CAP):
    """Прогон. Каждая симуляция = случайная перестановка (odds, win)."""
    nb = odds.shape[0]
    rng = np.random.default_rng(1000 + seed)
    perm = np.argsort(rng.random((num_sims, nb)), axis=1)
    O = odds[perm]
    W = base_wins[perm]

    bank = np.full(num_sims, INIT)
    hist = np.empty((num_sims, nb + 1))
    hist[:, 0] = bank
    peak = bank.copy()
    win_streak = np.zeros(num_sims)
    loss_streak = np.zeros(num_sims)

    for i in range(nb):
        o = O[:, i]
        f = frac_func(bank, peak, o, win_streak, loss_streak)
        f = np.clip(f, 0.0, cap)
        bet = f * np.maximum(bank, 0.0)
        won = W[:, i]
        bank = np.maximum(bank + np.where(won, bet * (o - 1), -bet), 0.0)
        peak = np.maximum(peak, bank)
        win_streak = np.where(won, win_streak + 1, 0)
        loss_streak = np.where(won, 0, loss_streak + 1)
        hist[:, i + 1] = bank
    return hist


def stats(hist):
    peaks = np.maximum.accumulate(hist, axis=1)
    dd = (hist - peaks) / np.maximum(peaks, 1e-9) * 100
    maxdd = dd.min(axis=1)
    profit = (hist[:, -1] - INIT) / INIT * 100
    return maxdd, profit


# ---------- Стратегии: доля ставки на шаге ----------

def s_flat(pct):
    f = pct / 100.0
    return lambda bank, peak, o, ws, ls: np.full(bank.shape, f)


def s_kelly(fraction, avg_odds=2.76):
    """Kelly доля по edge на кэфе, масштаб fraction (1.0 = полный Kelly)."""
    def f(bank, peak, o, ws, ls):
        p = (1.0 + ROI) / o
        b = o - 1.0
        k = np.maximum((b * p - (1 - p)) / b, 0.0)
        return k * fraction
    return f


def s_dd_guard(pct, cut=25, factor=0.5):
    """Плоский % с защитой: при просадке от пика > cut% режем ставку."""
    base = pct / 100.0
    def f(bank, peak, o, ws, ls):
        dd = (peak - bank) / np.maximum(peak, 1e-9) * 100
        return base * np.where(dd > cut, factor, 1.0)
    return f


def s_const_profit(target_pct, maxf=CAP):
    """Целевая прибыль target_pct% от банка с одной ставки: bet = target/(o-1)."""
    t = target_pct / 100.0
    def f(bank, peak, o, ws, ls):
        return np.minimum(t / (o - 1.0), maxf)
    return f
