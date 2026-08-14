"""
ЧЕСТНЫЙ движок динамического банкролл-менеджмента.

Принципы (согласовано с пользователем):
1. ROI = +2.5% ЖЁСТКО зафиксирован в КАЖДОЙ симуляции.
   Реализация: берём один базовый набор исходов (с нужным числом побед на
   каждый кэф так, что ROI набора = +2.5%), а каждая симуляция — это
   СЛУЧАЙНАЯ ПЕРЕСТАНОВКА порядка этих (кэф, исход) пар. ROI инвариантен к
   перестановке, поэтому итоговый ROI каждой симуляции идентичен. Различается
   только УДАЧА В ПОРЯДКЕ (когда выпадают полосы).
2. Динамический БМ: на каждой ставке размер = pct(стратегия) * ЖИВОЙ банк.
   Банк физически не может уйти в минус (теряем максимум pct% за ставку).
3. Жёсткое правило риска: ставка <= 10% живого банка (кэп).
4. Просадка — классический max drawdown от пика за всю историю.

Стратегии тут — это ФУНКЦИИ, возвращающие долю ставки f в [0..0.10] на каждом
шаге. Реализованы честно и прозрачно (без багов исходного кода).
"""

import numpy as np

INITIAL_BANKROLL = 1000.0
ROI = 0.025
CAP = 0.10


def load_real_odds(filename='matching_teams_clean.csv'):
    import pandas as pd
    return pd.read_csv(filename)['coefficient'].values.astype(float)


def build_fixed_roi_base(num_bets, roi=ROI, seed=42):
    """
    Базовый набор (odds_i, win_i) длиной num_bets, у которого суммарный ROI = roi.
    Победы расставляются так, чтобы на каждый кэф приходилось ~ (1+roi)/odds доля,
    что и даёт нужный ROI. Затем небольшая коррекция, чтобы ROI был ровно roi.
    """
    rng = np.random.default_rng(seed)
    real = load_real_odds()
    odds = rng.choice(real, size=num_bets, replace=True)
    p = np.clip((1.0 + roi) / odds, 0, 1)
    wins = rng.random(num_bets) < p

    # Коррекция числа побед, чтобы ROI был максимально близок к целевому.
    def cur_roi(w):
        return (w * (odds - 1) - (~w)).sum() / num_bets

    target_profit = roi * num_bets  # суммарная прибыль на единичных ставках
    # подгоняем флипом отдельных ставок (грубая, но точная коррекция)
    for _ in range(200):
        diff = cur_roi(wins) * num_bets - target_profit
        if abs(diff) < 0.5:
            break
        if diff > 0:  # слишком прибыльно -> убрать победу (флип win->loss) на низком кэфе
            idx = np.where(wins)[0]
            j = idx[np.argmin(odds[idx])]  # низкий кэф: победа даёт мало, убрать дёшево
            wins[j] = False
        else:  # мало прибыли -> добавить победу на высоком кэфе
            idx = np.where(~wins)[0]
            j = idx[np.argmax(odds[idx])]
            wins[j] = True
    return odds, wins


def simulate(frac_func, odds, base_wins, num_sims, seed=0, cap=CAP):
    """
    Прогоняет num_sims симуляций. Каждая = случайная перестановка (odds,win).
    frac_func(state) -> доля ставки на текущем шаге (вектор по симуляциям).
    Возвращает bankroll_history (num_sims, num_bets+1).
    """
    num_bets = odds.shape[0]
    rng = np.random.default_rng(1000 + seed)

    # перестановки: (num_sims, num_bets)
    perm = np.argsort(rng.random((num_sims, num_bets)), axis=1)
    O = odds[perm]            # кэфы в порядке каждой симуляции
    W = base_wins[perm]       # исходы в порядке каждой симуляции

    bank = np.full(num_sims, INITIAL_BANKROLL)
    hist = np.empty((num_sims, num_bets + 1), dtype=np.float64)
    hist[:, 0] = bank

    # состояние для адаптивных стратегий
    peak = bank.copy()
    win_streak = np.zeros(num_sims, dtype=int)
    loss_streak = np.zeros(num_sims, dtype=int)

    for i in range(num_bets):
        o = O[:, i]
        state = {'bank': bank, 'peak': peak, 'odds': o, 'i': i,
                 'win_streak': win_streak, 'loss_streak': loss_streak,
                 'num_bets': num_bets}
        f = frac_func(state)
        f = np.clip(f, 0.0, cap)
        bet = f * np.maximum(bank, 0.0)
        won = W[:, i]
        bank = bank + np.where(won, bet * (o - 1), -bet)
        bank = np.maximum(bank, 0.0)
        peak = np.maximum(peak, bank)
        win_streak = np.where(won, win_streak + 1, 0)
        loss_streak = np.where(won, 0, loss_streak + 1)
        hist[:, i + 1] = bank
    return hist


def metrics(hist):
    peaks = np.maximum.accumulate(hist, axis=1)
    dd = (hist - peaks) / np.maximum(peaks, 1e-9) * 100
    maxdd = dd.min(axis=1)
    final = hist[:, -1]
    profit = (final - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    return {
        'median_profit': float(np.median(profit)),
        'mean_profit': float(np.mean(profit)),
        'p5_profit': float(np.percentile(profit, 5)),
        'p1_profit': float(np.percentile(profit, 1)),
        'worst_dd': float(maxdd.min()),
        'median_dd': float(np.median(maxdd)),
        'p1_dd': float(np.percentile(maxdd, 1)),
        'p5_dd': float(np.percentile(maxdd, 5)),
        'share_below_66': float((maxdd < -66).mean() * 100),
        'bankrupt': float((final < 1).mean() * 100),
        'maxdd_arr': maxdd,
        'profit_arr': profit,
    }


# ---------- Стратегии (доля ставки на шаге) ----------

def flat_pct(pct):
    """Плоский динамический процент: всегда pct% от живого банка."""
    f = pct / 100.0
    return lambda st: np.full(st['bank'].shape, f)


def roi_per_odds(pct_at_avg, avg_odds=2.76):
    """Ставка пропорциональна edge на данном кэфе: больше на низких кэфах.
    edge_i = (1+ROI) - odds_i*... ; используем долю Kelly-подобную, нормированную."""
    def f(st):
        o = st['odds']
        # Kelly доля при p=(1+ROI)/o: f = (b*p - q)/b, b=o-1
        p = (1.0 + ROI) / o
        b = o - 1.0
        kelly = np.maximum((b * p - (1 - p)) / b, 0.0)
        # масштабируем так, чтобы на среднем кэфе доля = pct_at_avg
        p0 = (1.0 + ROI) / avg_odds
        k0 = max(((avg_odds - 1) * p0 - (1 - p0)) / (avg_odds - 1), 1e-9)
        return kelly * (pct_at_avg / 100.0) / k0
    return f


def drawdown_guard(base_pct, cut_at=20, cut_factor=0.5):
    """Динамический: при просадке от пика > cut_at% режет ставку в cut_factor раз."""
    base = base_pct / 100.0
    def f(st):
        dd = (st['peak'] - st['bank']) / np.maximum(st['peak'], 1e-9) * 100
        mult = np.where(dd > cut_at, cut_factor, 1.0)
        return np.full(st['bank'].shape, base) * mult
    return f
