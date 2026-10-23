"""
Стратегии размера ставки.

Стратегия возвращает ДОЛЮ базы (живого банка или его снимка) на текущую
ставку; движок сам ограничивает её капом и живым банком. Контекст `c`:
  c.odds, c.edge   — кэф и оценка EV ставки (векторы по симуляциям)
  c.bank, c.peak   — живой банк и его пик
  c.base           — база ставки (живой банк или снимок)
  c.bank0, c.step, c.n_bets
После розыгрыша движок вызывает update(c) с c.won, c.stake, c.ret, c.placed.

Все legacy-формулы из simulation-current/run_strategies_real_odds.py
перенесены 1-в-1, кроме исправленных ошибок (отмечены «ИСПРАВЛЕНО»).
"""

from __future__ import annotations

import numpy as np


class Strategy:
    label = 'strategy'

    def reset(self, n_sims: int, bank0: float):
        self.n_sims = n_sims
        self.bank0 = bank0

    def fraction(self, c):
        raise NotImplementedError

    def update(self, c):
        pass

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in vars(self).items()
                           if not k.startswith('_') and k not in ('n_sims', 'bank0', 'label'))
        return f'{type(self).__name__}({params})'


def kelly_fraction(odds, edge):
    """Полный Келли для десятичного кэфа: f* = EV / (кэф - 1), ноль при EV <= 0."""
    return np.maximum(edge, 0.0) / (odds - 1.0)


# --------------------------------------------------------------------------
# Базовые и теоретически обоснованные
# --------------------------------------------------------------------------

class FixedFraction(Strategy):
    """Фиксированный процент от банка."""
    label = 'Фиксированный %'

    def __init__(self, pct=2.0):
        self.pct = pct

    def fraction(self, c):
        return self.pct / 100.0


class FlatStake(Strategy):
    """Фиксированная сумма = pct% от НАЧАЛЬНОГО банка (без компаундинга).
    Именно так фактически считались 23 из 25 стратегий в legacy-коде."""
    label = 'Флэт (от начального банка)'

    def __init__(self, pct=2.0):
        self.pct = pct

    def fraction(self, c):
        return self.pct / 100.0 * c.bank0 / np.maximum(c.base, 1e-12)


class Kelly(Strategy):
    """Дробный Келли: f = mult * EV / (кэф - 1)."""
    label = 'Келли'

    def __init__(self, mult=0.5):
        self.mult = mult

    def fraction(self, c):
        return self.mult * kelly_fraction(c.odds, c.edge)


class LogFormula(Strategy):
    """
    «Логарифмическая» формула из backend/calculator:
        f = log10(1 - (1 + edge) / odds) / log10(10^-risk) = -log10(q) / risk,
    где q = 1 - p — вероятность проигрыша. Это НЕ Келли: размер почти не зависит
    от перевеса (при EV = 0 на кэфе 1.5 и risk=15 ставка всё равно ~3.2%).
    Ставки только при edge > 0 (как в value-беттинге).
    """
    label = 'Лог-формула (backend)'

    def __init__(self, risk=15.0, edge_cap=0.08):
        self.risk = risk
        self.edge_cap = edge_cap

    def fraction(self, c):
        e = np.minimum(c.edge, self.edge_cap)
        q = 1.0 - (1.0 + e) / c.odds
        with np.errstate(divide='ignore', invalid='ignore'):
            f = -np.log10(q) / self.risk
        return np.where((e > 0) & (q > 0), f, 0.0)


class FloorKelly(Strategy):
    """
    Келли с «полом» (Grossman & Zhou, 1993): ставим долю Келли не от всего
    банка, а от подушки над полом floor * пик:
        stake = min(mult * f_kelly, 1) * (bank - floor * peak).
    Это оптимальная по росту политика при ограничении на максимальную просадку:
    при ставке от живого банка просадка от пика не превышает 1 - floor (с точностью
    до округлений). Подушка считается от базы ставки и пика этих баз: если банк
    пересчитывается раз в K ставок, стратегия, как и остальные, знает банк только на
    момент снимка, и гарантия по просадке тогда не действует.
    """
    label = 'Келли с полом просадки'

    def __init__(self, mult=1.0, floor=0.5):
        self.mult = mult
        self.floor = floor

    def reset(self, n_sims, bank0):
        super().reset(n_sims, bank0)
        self._peak_base = np.full(n_sims, float(bank0))

    def fraction(self, c):
        base = np.asarray(c.base, dtype=float)
        self._peak_base = np.maximum(self._peak_base, base)
        cushion = np.maximum(base - self.floor * self._peak_base, 0.0)
        f = np.minimum(self.mult * kelly_fraction(c.odds, c.edge), 1.0)
        return f * cushion / np.maximum(base, 1e-12)


class DrawdownGuard(Strategy):
    """Обёртка: режет ставку базовой стратегии на просадке от пика."""
    label = 'Защита от просадки'

    def __init__(self, inner: Strategy, steps=((0.2, 0.5),)):
        self.inner = inner
        self.steps = tuple(steps)
        self.label = f'{inner.label} + защита'

    def reset(self, n_sims, bank0):
        super().reset(n_sims, bank0)
        self.inner.reset(n_sims, bank0)

    def fraction(self, c):
        dd = (c.peak - c.bank) / np.maximum(c.peak, 1e-12)
        mult = np.ones_like(dd)
        for level, m in sorted(self.steps):
            mult = np.where(dd > level, m, mult)
        return self.inner.fraction(c) * mult

    def update(self, c):
        self.inner.update(c)


class Scaled(Strategy):
    """Обёртка: умножает долю базовой стратегии на scale (для калибровки под риск)."""

    def __init__(self, inner: Strategy, scale: float):
        self.inner = inner
        self.scale = scale
        self.label = inner.label

    def reset(self, n_sims, bank0):
        super().reset(n_sims, bank0)
        self.inner.reset(n_sims, bank0)

    def fraction(self, c):
        return self.scale * np.asarray(self.inner.fraction(c), dtype=float)

    def update(self, c):
        self.inner.update(c)


# --------------------------------------------------------------------------
# Legacy: зависимые от ROI / кэфа (формулы из run_strategies_real_odds.py)
# --------------------------------------------------------------------------

class RoiScaled(Strategy):
    """linear_roi / sqrt_roi / log_roi / exponential_roi: процент ставки как функция ROI."""
    label = 'ROI-масштаб'

    def __init__(self, kind='linear', base_roi=5.0, base_percent=1.0, max_percent=10.0, factor=0.1):
        if kind not in ('linear', 'sqrt', 'log', 'exp'):
            raise ValueError(kind)
        self.kind = kind
        self.base_roi = base_roi
        self.base_percent = base_percent
        self.max_percent = max_percent
        self.factor = factor
        self.label = {'linear': 'Линейно от ROI', 'sqrt': 'Корень от ROI',
                      'log': 'Логарифм от ROI', 'exp': 'Экспонента от ROI'}[kind]

    def fraction(self, c):
        roi = np.asarray(c.edge, dtype=float) * 100.0
        r = np.maximum(roi / self.base_roi, 0.0)
        if self.kind == 'linear':
            pct = self.base_percent * r
        elif self.kind == 'sqrt':
            pct = self.base_percent * np.sqrt(r)
        elif self.kind == 'log':
            pct = self.base_percent * np.log(r + 1.0)
        else:
            pct = self.base_percent * np.exp(self.factor * (roi - self.base_roi))
        pct = np.where(roi > 0, pct, 0.0)
        return np.clip(pct, 0.0, self.max_percent) / 100.0


class ConstantProfit(Strategy):
    """Ставка, которая при выигрыше даёт target_profit_pct% банка: f = target / (кэф - 1)."""
    label = 'Постоянная прибыль'

    def __init__(self, target_profit_pct=1.0, max_percent=10.0):
        self.target_profit_pct = target_profit_pct
        self.max_percent = max_percent

    def fraction(self, c):
        f = self.target_profit_pct / 100.0 / np.maximum(c.odds - 1.0, 1e-9)
        return np.minimum(f, self.max_percent / 100.0)


class AdaptiveConstantProfit(Strategy):
    """Целевая прибыль растёт линейно с ROI от min_target_pct до max_target_pct."""
    label = 'Адаптивная пост. прибыль'

    def __init__(self, min_roi=3.0, max_roi=20.0, min_target_pct=0.5, max_target_pct=3.0,
                 max_bet_percent=15.0):
        self.min_roi, self.max_roi = min_roi, max_roi
        self.min_target_pct, self.max_target_pct = min_target_pct, max_target_pct
        self.max_bet_percent = max_bet_percent

    def fraction(self, c):
        roi = np.asarray(c.edge, dtype=float) * 100.0
        k = np.clip((roi - self.min_roi) / (self.max_roi - self.min_roi), 0, 1) if self.max_roi > self.min_roi else 0.5
        target = self.min_target_pct + (self.max_target_pct - self.min_target_pct) * k
        f = np.where(c.odds > 1.05, target / 100.0 / np.maximum(c.odds - 1.0, 1e-9), 0.01)
        return np.minimum(f, self.max_bet_percent / 100.0)


class LinearScaled(Strategy):
    """ROI в [min_roi, max_roi] линейно отображается в процент [min_percent, max_percent]."""
    label = 'Линейная шкала ROI'

    def __init__(self, min_roi=3.0, max_roi=20.0, min_percent=1.0, max_percent=7.0):
        self.min_roi, self.max_roi = min_roi, max_roi
        self.min_percent, self.max_percent = min_percent, max_percent

    def fraction(self, c):
        roi = np.asarray(c.edge, dtype=float) * 100.0
        k = np.clip((roi - self.min_roi) / (self.max_roi - self.min_roi), 0, 1) if self.max_roi > self.min_roi else 0.5
        return (self.min_percent + (self.max_percent - self.min_percent) * k) / 100.0


class CombinedRoiOdds(Strategy):
    """bet% = base * sqrt(norm_roi) * (1 - 0.5 * norm_odds)."""
    label = 'ROI + кэф'

    def __init__(self, base_percent=1.0, max_percent=10.0, min_roi=3.0, max_roi=15.0,
                 min_odds=1.5, max_odds=5.0):
        self.base_percent, self.max_percent = base_percent, max_percent
        self.min_roi, self.max_roi = min_roi, max_roi
        self.min_odds, self.max_odds = min_odds, max_odds

    def pct(self, c):
        roi = np.asarray(c.edge, dtype=float) * 100.0
        norm_roi = np.clip((roi - self.min_roi) / (self.max_roi - self.min_roi), 0, 1)
        norm_odds = np.clip((c.odds - self.min_odds) / (self.max_odds - self.min_odds), 0, 1)
        return self.base_percent * np.sqrt(norm_roi) * (1 - 0.5 * norm_odds)

    def fraction(self, c):
        return np.clip(self.pct(c), 0, self.max_percent) / 100.0


class Hybrid(Strategy):
    """bet% = base * (w_roi * norm_roi + w_odds * (1 - norm_odds))."""
    label = 'Гибрид ROI/кэф'

    def __init__(self, base_percent=1.0, max_percent=10.0, min_roi=3.0, max_roi=15.0,
                 min_odds=1.5, max_odds=5.0, roi_weight=0.7, odds_weight=0.3):
        self.base_percent, self.max_percent = base_percent, max_percent
        self.min_roi, self.max_roi = min_roi, max_roi
        self.min_odds, self.max_odds = min_odds, max_odds
        self.roi_weight, self.odds_weight = roi_weight, odds_weight

    def fraction(self, c):
        roi = np.asarray(c.edge, dtype=float) * 100.0
        norm_roi = np.clip((roi - self.min_roi) / (self.max_roi - self.min_roi), 0, 1)
        norm_odds = 1 - np.clip((c.odds - self.min_odds) / (self.max_odds - self.min_odds), 0, 1)
        pct = self.base_percent * (self.roi_weight * norm_roi + self.odds_weight * norm_odds)
        return np.clip(pct, 0, self.max_percent) / 100.0


class LinearRoiOdds(Strategy):
    """bet% = base * ROI/base_roi * (1 - penalty * norm_odds)."""
    label = 'Линейно ROI со штрафом кэфа'

    def __init__(self, base_roi=5.0, base_percent=1.0, max_percent=10.0, odds_penalty_factor=0.7,
                 min_odds=1.5, max_odds=5.0):
        self.base_roi, self.base_percent, self.max_percent = base_roi, base_percent, max_percent
        self.odds_penalty_factor = odds_penalty_factor
        self.min_odds, self.max_odds = min_odds, max_odds

    def fraction(self, c):
        roi = np.asarray(c.edge, dtype=float) * 100.0
        norm_odds = np.clip((c.odds - self.min_odds) / (self.max_odds - self.min_odds), 0, 1)
        pct = self.base_percent * (roi / self.base_roi) * (1 - self.odds_penalty_factor * norm_odds)
        return np.clip(pct, 0, self.max_percent) / 100.0


class PortfolioTheory(Strategy):
    """Веса по корзинам кэфов (<2, <3.5, остальные) = 0.4 / 0.4 / 0.2, нормировка на 0.3."""
    label = 'Корзины кэфов'

    def __init__(self, base_percent=2.0):
        self.base_percent = base_percent

    def fraction(self, c):
        w = np.where(c.odds < 2.0, 0.4, np.where(c.odds < 3.5, 0.4, 0.2))
        return self.base_percent * w / 0.3 / 100.0


# --------------------------------------------------------------------------
# Legacy: Келли-подобные (ИСПРАВЛЕНО: доля больше не делится на 100 второй раз)
# --------------------------------------------------------------------------

class ProportionalKelly(Strategy):
    """Келли / risk * confidence. ИСПРАВЛЕНО: в legacy доля делилась на 100 дважды."""
    label = 'Пропорциональный Келли'

    def __init__(self, risk=2.0, confidence=0.7, max_percent=10.0):
        self.risk, self.confidence, self.max_percent = risk, confidence, max_percent

    def fraction(self, c):
        f = kelly_fraction(c.odds, c.edge) / self.risk * self.confidence
        return np.minimum(f, self.max_percent / 100.0)


class DynamicKelly(Strategy):
    """Келли / risk * доля, растущая с ROI. ИСПРАВЛЕНО: двойное деление на 100."""
    label = 'Динамический Келли'

    def __init__(self, risk=2.0, min_fraction=0.1, max_fraction=0.5, min_roi=3.0, max_roi=15.0):
        self.risk = risk
        self.min_fraction, self.max_fraction = min_fraction, max_fraction
        self.min_roi, self.max_roi = min_roi, max_roi

    def fraction(self, c):
        roi = np.asarray(c.edge, dtype=float) * 100.0
        k = np.clip((roi - self.min_roi) / (self.max_roi - self.min_roi), 0, 1)
        dyn = self.min_fraction + (self.max_fraction - self.min_fraction) * k
        return np.minimum(kelly_fraction(c.odds, c.edge) / self.risk * dyn, 0.10)


class BayesianKelly(Strategy):
    """
    Келли по консервативной вероятности p - prior_std.
    ИСПРАВЛЕНО: в legacy p был константой prior_mean для ЛЮБОГО кэфа (на кэфе 5
    стратегия «видела» перевес 60%) и доля делилась на 100 дважды.
    """
    label = 'Байесовский Келли'

    def __init__(self, prior_std=0.02, risk_factor=2.0, max_percent=10.0):
        self.prior_std, self.risk_factor, self.max_percent = prior_std, risk_factor, max_percent

    def fraction(self, c):
        p = np.clip((1.0 + c.edge) / c.odds - self.prior_std, 0.01, 0.99)
        b = c.odds - 1.0
        f = np.maximum((b * p - (1 - p)) / b, 0.0) / self.risk_factor
        return np.minimum(f, self.max_percent / 100.0)


# --------------------------------------------------------------------------
# Legacy: зависящие от состояния банка или истории
# --------------------------------------------------------------------------

class DynamicPercentage(Strategy):
    """pct x1.2 при банке > 120% начального, x0.7 при < 80%, x0.5 при < 60%."""
    label = 'Динамический %'

    def __init__(self, bet_size_pct=2.0):
        self.bet_size_pct = bet_size_pct

    def fraction(self, c):
        r = c.bank / c.bank0
        m = np.where(r > 1.2, 1.2, 1.0)
        m = np.where(r < 0.8, 0.7, m)
        m = np.where(r < 0.6, 0.5, m)
        return self.bet_size_pct * m / 100.0


class Adaptive(CombinedRoiOdds):
    """ROI + кэф, урезание на просадке от пика и потолок при просадке от старта."""
    label = 'Адаптивная'

    def fraction(self, c):
        pct = self.pct(c)
        dd_peak = (c.peak - c.bank) / np.maximum(c.peak, 1e-12)
        pct = pct * np.where(dd_peak > 0.40, 0.5, np.where(dd_peak > 0.20, 0.75, 1.0))
        dd_init = (c.bank0 - c.bank) / c.bank0
        cap = np.where(dd_init > 0.40, self.max_percent * 0.25,
                       np.where(dd_init > 0.30, self.max_percent * 0.5, self.max_percent))
        return np.clip(np.minimum(pct, cap), 0, None) / 100.0


class TargetBased(Strategy):
    """aggressive_pct пока банк < цели, потом conservative_pct."""
    label = 'До цели'

    def __init__(self, target_bankroll_percent=200.0, aggressive_pct=3.0, conservative_pct=1.0):
        self.target_bankroll_percent = target_bankroll_percent
        self.aggressive_pct, self.conservative_pct = aggressive_pct, conservative_pct

    def fraction(self, c):
        target = c.bank0 * self.target_bankroll_percent / 100.0
        return np.where(c.bank < target, self.aggressive_pct, self.conservative_pct) / 100.0


class AntiMartingale(Strategy):
    """После выигрыша процент x multiplier (пока серия выигрышей короче max_streak, то есть
    не больше max_streak - 1 увеличений подряд, как в legacy), после проигрыша — база."""
    label = 'Анти-мартингейл'

    def __init__(self, base_percent=1.0, multiplier=1.5, max_percent=10.0, max_streak=5):
        self.base_percent, self.multiplier = base_percent, multiplier
        self.max_percent, self.max_streak = max_percent, max_streak

    def reset(self, n_sims, bank0):
        super().reset(n_sims, bank0)
        self._pct = np.full(n_sims, float(self.base_percent))
        self._streak = np.zeros(n_sims, dtype=int)

    def fraction(self, c):
        return self._pct / 100.0

    def update(self, c):
        won = c.won
        self._streak = np.where(won, self._streak + 1, 0)
        grow = won & (self._streak < self.max_streak)
        self._pct = np.where(grow, np.minimum(self._pct * self.multiplier, self.max_percent),
                             np.where(won, self._pct, self.base_percent))


class StreakAware(Strategy):
    """Серия выигрышей: x win_mult; серия проигрышей: / loss_div; смена направления — база."""
    label = 'По сериям'

    def __init__(self, base_percent=2.0, win_streak_multiplier=1.2, loss_streak_divider=1.3,
                 max_multiplier=3.0):
        self.base_percent = base_percent
        self.win_streak_multiplier, self.loss_streak_divider = win_streak_multiplier, loss_streak_divider
        self.max_multiplier = max_multiplier

    def reset(self, n_sims, bank0):
        super().reset(n_sims, bank0)
        self._pct = np.full(n_sims, float(self.base_percent))
        self._last = np.full(n_sims, -1)

    def fraction(self, c):
        return self._pct / 100.0

    def update(self, c):
        won = c.won.astype(int)
        same = self._last == won
        changed = (self._last != -1) & ~same
        b = self.base_percent
        self._pct = np.where(same & (won == 1), np.minimum(self._pct * self.win_streak_multiplier,
                                                          b * self.max_multiplier), self._pct)
        self._pct = np.where(same & (won == 0), np.maximum(self._pct / self.loss_streak_divider,
                                                          b / self.max_multiplier), self._pct)
        self._pct = np.where(changed, b, self._pct)
        self._last = won


class _History(Strategy):
    """Хранит последние `lookback` доходностей и банков (кольцевой буфер)."""

    def __init__(self, lookback):
        self.lookback = lookback

    def reset(self, n_sims, bank0):
        super().reset(n_sims, bank0)
        self._rets = np.zeros((n_sims, self.lookback))
        self._banks = np.full((n_sims, self.lookback + 1), float(bank0))
        self._t = 0

    def update(self, c):
        self._rets = np.roll(self._rets, -1, axis=1)
        self._rets[:, -1] = c.ret
        self._banks = np.roll(self._banks, -1, axis=1)
        self._banks[:, -1] = c.bank
        self._t += 1


class VolatilityAdjusted(_History):
    """bet% = base / (1 + factor * std последних доходностей в %)."""
    label = 'По волатильности'

    def __init__(self, base_percent=2.0, lookback=50, volatility_factor=1.0):
        super().__init__(lookback)
        self.base_percent, self.volatility_factor = base_percent, volatility_factor

    def fraction(self, c):
        if self._t < self.lookback:
            return self.base_percent / 100.0
        vol = np.std(self._rets * 100.0, axis=1)
        return self.base_percent / (1 + self.volatility_factor * vol / 100.0) / 100.0


class SharpeOptimized(_History):
    """bet% = base * (1 + clip(Sharpe последних доходностей, -0.5, 1))."""
    label = 'По Шарпу'

    def __init__(self, base_percent=2.0, lookback=100, risk_free_rate=0.0):
        super().__init__(lookback)
        self.base_percent, self.risk_free_rate = base_percent, risk_free_rate

    def fraction(self, c):
        if self._t < self.lookback:
            return self.base_percent / 100.0
        sharpe = (self._rets.mean(axis=1) - self.risk_free_rate) / np.maximum(self._rets.std(axis=1), 1e-6)
        return self.base_percent * (1 + np.clip(sharpe, -0.5, 1.0)) / 100.0


class MultiObjective(_History):
    """Скоринг прибыли/просадки/волатильности за окно -> множитель ставки."""
    label = 'Мультикритериальная'

    def __init__(self, base_percent=2.0, w_profit=0.5, w_drawdown=0.3, w_volatility=0.2, lookback=50):
        super().__init__(lookback)
        self.base_percent = base_percent
        self.w_profit, self.w_drawdown, self.w_volatility = w_profit, w_drawdown, w_volatility

    def fraction(self, c):
        if self._t < self.lookback:
            return self.base_percent / 100.0
        banks = np.maximum(self._banks, 1e-12)
        profit = (banks[:, -1] - banks[:, 0]) / banks[:, 0] * 100
        run_max = np.maximum.accumulate(banks, axis=1)
        max_dd = np.max((run_max - banks) / run_max * 100, axis=1)
        vol = np.std(np.diff(banks, axis=1) / banks[:, :-1], axis=1) * 100
        score = (self.w_profit * np.clip(profit / 50, 0, 2) - self.w_drawdown * np.clip(max_dd / 30, 0, 2)
                 - self.w_volatility * np.clip(vol / 5, 0, 2))
        return self.base_percent * (1 + np.clip(score, -0.5, 1.0)) / 100.0


class MLAdaptive(Strategy):
    """
    Процент дрейфует к фактической проходимости за окно.
    ИСПРАВЛЕНО: ожидаемая проходимость = среднее (1+EV)/кэф по окну, а не
    ROI + 1/средний_кэф (это разные величины, и вторая не имеет смысла).
    """
    label = 'ML-адаптивная'

    def __init__(self, base_percent=2.0, learning_rate=0.01, lookback=30):
        self.base_percent, self.learning_rate, self.lookback = base_percent, learning_rate, lookback

    def reset(self, n_sims, bank0):
        super().reset(n_sims, bank0)
        self._pct = np.full(n_sims, float(self.base_percent))
        self._won = np.zeros((n_sims, self.lookback))
        self._p = np.zeros((n_sims, self.lookback))
        self._t = 0

    def fraction(self, c):
        self._p_now = np.clip((1.0 + c.edge) / c.odds, 0, 1)
        if self._t >= self.lookback:
            expected = self._p.mean(axis=1)
            perf = (self._won.mean(axis=1) - expected) / np.maximum(expected, 1e-9)
            self._pct = np.clip(self._pct * (1 + self.learning_rate * perf),
                                self.base_percent * 0.5, self.base_percent * 2.0)
        return self._pct / 100.0

    def update(self, c):
        self._won = np.roll(self._won, -1, axis=1)
        self._won[:, -1] = c.won
        self._p = np.roll(self._p, -1, axis=1)
        self._p[:, -1] = self._p_now
        self._t += 1


# --------------------------------------------------------------------------
# Каталог для отчётов и сравнения при равном риске
# --------------------------------------------------------------------------

def catalog():
    """Имя -> (фабрика стратегии с параметрами по умолчанию, описание)."""
    return {
        'fixed': (lambda: FixedFraction(2.0), 'f = 2% банка'),
        'flat': (lambda: FlatStake(2.0), 'ставка = 2% НАЧАЛЬНОГО банка (без компаундинга)'),
        'kelly_half': (lambda: Kelly(0.5), 'половина Келли: f = 0.5·EV/(k-1)'),
        'floor_kelly': (lambda: FloorKelly(1.0, 0.5), 'Келли от подушки над полом 50% пика'),
        'log_formula': (lambda: LogFormula(15.0), 'f = -log10(1-p)/15 (backend)'),
        'linear_roi': (lambda: RoiScaled('linear'), '1% при ROI 5%, линейно'),
        'sqrt_roi': (lambda: RoiScaled('sqrt'), '1% при ROI 5%, корень'),
        'constant_profit': (lambda: ConstantProfit(1.0), 'выигрыш = 1% банка'),
        'adaptive_constant_profit': (lambda: AdaptiveConstantProfit(), 'выигрыш 0.5–3% банка по ROI'),
        'linear_roi_odds': (lambda: LinearRoiOdds(), 'ROI со штрафом за высокий кэф'),
        'dynamic_percentage': (lambda: DynamicPercentage(2.0), '2% с поправкой на банк'),
        'drawdown_guard': (lambda: DrawdownGuard(FixedFraction(2.0), ((0.2, 0.5),)), '2%, x0.5 на просадке > 20%'),
        'anti_martingale': (lambda: AntiMartingale(), 'x1.5 после выигрыша'),
        'portfolio_theory': (lambda: PortfolioTheory(), 'веса по корзинам кэфов'),
    }
