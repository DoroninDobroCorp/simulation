"""
Однопроходный векторизованный Monte Carlo движок банкролл-менеджмента.

Инварианты (их проверяют тесты в tests/):
- размер ставки считается ОДИН раз, от базы на текущем шаге (живой банк или
  снимок банка, если пересчёт идёт раз в K ставок) и сразу же разыгрывается;
- ставка никогда не больше живого банка, банк никогда не уходит в минус;
- исходы ставок (кэф, истинная вероятность, выигрыш) берутся из отдельного
  генератора, поэтому разные стратегии с одним seed видят одни и те же
  ставки и исходы (common random numbers) — сравнение честное;
- просадка — классический максимум (peak - bank) / peak по каждой ставке.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional, Sequence, Union

import numpy as np

from .data import BetPool

QUANTILES = (5, 25, 50, 75, 95)


# --------------------------------------------------------------------------
# Модель рынка
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Market:
    """Что видит стратегия (edge_est) и что происходит на самом деле (edge_true)."""
    odds: np.ndarray
    edge_est: np.ndarray
    edge_true: np.ndarray
    noise: float = 0.0          # σ ошибки модели: истинный EV ставки ~ N(edge_true, σ)
    description: str = ''

    @property
    def mean_true_roi(self) -> float:
        return float(np.mean(self.edge_true))

    @property
    def p_true(self) -> np.ndarray:
        return np.clip((1.0 + self.edge_true) / self.odds, 0.0, 1.0)


def make_market(pool: BetPool, roi: Optional[float] = 0.025, ev_model: str = 'flat',
                est_cap: float = 0.08, calibrated: bool = False, noise: float = 0.0) -> Market:
    """
    ev_model:
      'flat'    — у каждой ставки истинный EV = roi, стратегия тоже видит roi
                  (так считались все прогоны в simulation-current);
      'scaled'  — оценка EV = min(ev из данных, est_cap); истинный EV = k * оценка,
                  k подобран так, чтобы средний истинный ROI = roi. Модель игрока
                  оптимистична в 1/k раз. calibrated=True — стратегия знает k и
                  видит уже исправленную оценку;
      'trusted' — истинный EV = оценка = min(ev, est_cap); roi игнорируется.
    """
    odds = pool.odds.astype(float)
    if ev_model == 'flat':
        if roi is None:
            raise ValueError("для ev_model='flat' нужен roi")
        est = np.full_like(odds, roi)
        true = est.copy()
        desc = f'все ставки с EV {roi:+.2%}'
    elif ev_model in ('scaled', 'trusted'):
        est = np.minimum(pool.ev.astype(float), est_cap)
        if ev_model == 'trusted':
            true = est.copy()
            desc = f'EV из данных (кап {est_cap:.0%}), модель точна'
        else:
            if roi is None:
                raise ValueError("для ev_model='scaled' нужен roi")
            mean_est = float(np.mean(est))
            if mean_est <= 0:
                raise ValueError('средняя оценка EV должна быть > 0')
            k = roi / mean_est
            true = est * k
            if calibrated:
                est = true.copy()
            desc = (f'EV из данных (кап {est_cap:.0%}) x {k:.2f} -> средний ROI {roi:+.2%}'
                    + (', оценка откалибрована' if calibrated else f', модель завышает EV в {1 / k:.2f} раза'))
    else:
        raise ValueError(f'неизвестная ev_model: {ev_model}')
    if noise:
        desc += f', шум модели σ={noise:.1%}'
    return Market(odds, est, true, noise, desc)


# --------------------------------------------------------------------------
# Сценарии: откуда берутся (кэф, оценка EV, исход) на каждом шаге
# --------------------------------------------------------------------------

class MarketScenario:
    """
    Генерирует ставки из рынка.
    sampling:
      'bootstrap' — каждая ставка каждой симуляции независимо выбирается из пула;
      'shuffle'   — каждая симуляция идёт по своей случайной перестановке пула;
      'sequence'  — все симуляции идут по пулу в исходном порядке (как legacy).
    """

    def __init__(self, market: Market, n_sims: int, seed: int = 0, sampling: str = 'bootstrap'):
        self.m = market
        self.n_sims = n_sims
        self.sampling = sampling
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self._n = len(market.odds)
        self._perm = None
        if sampling not in ('bootstrap', 'shuffle', 'sequence'):
            raise ValueError(f'неизвестный sampling: {sampling}')

    def _indices(self, t: int) -> np.ndarray:
        if self.sampling == 'bootstrap':
            return self.rng.integers(0, self._n, self.n_sims)
        if self.sampling == 'sequence':
            return np.full(self.n_sims, t % self._n)
        if self._perm is None or t % self._n == 0:
            self._perm = np.argsort(self.rng.random((self.n_sims, self._n)), axis=1)
        return self._perm[:, t % self._n]

    def step(self, t: int):
        idx = self._indices(t)
        odds = self.m.odds[idx]
        edge_true = self.m.edge_true[idx]
        if self.m.noise:
            edge_true = edge_true + self.m.noise * self.rng.standard_normal(self.n_sims)
        p = np.clip((1.0 + edge_true) / odds, 0.0, 1.0)
        won = self.rng.random(self.n_sims) < p
        return odds, self.m.edge_est[idx], won


class FixedScenario:
    """Заранее заданные исходы (n_sims, n_bets) и кэфы (n_bets,) или (n_sims, n_bets)."""

    def __init__(self, outcomes: np.ndarray, odds: np.ndarray, edge: Union[float, np.ndarray]):
        self.outcomes = np.asarray(outcomes, dtype=bool)
        self.n_sims, self.n_bets = self.outcomes.shape
        self.odds = np.asarray(odds, dtype=float)
        self.edge = edge

    def _col(self, a, t):
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            return np.full(self.n_sims, float(a))
        if a.ndim == 1:
            return np.full(self.n_sims, a[t])
        return a[:, t]

    def step(self, t: int):
        return self._col(self.odds, t), self._col(self.edge, t), self.outcomes[:, t]


# --------------------------------------------------------------------------
# Конфигурация и результат
# --------------------------------------------------------------------------

@dataclass
class SimConfig:
    n_sims: int = 2000
    n_bets: int = 1000
    bank0: float = 1000.0
    cap: float = 0.10                 # максимум доли базы на одну ставку
    recalc_every: Union[int, Sequence[int]] = 1   # 1 = живой банк; K или (min, max) = снимок банка
    variation: Optional[tuple] = None  # ('uniform', lo, hi) | ('choice', values) — множитель к ставке
    round_to: float = 0.0             # округление ставки (например, до 5)
    min_stake: float = 0.0            # ставки меньше не делаются
    ruin_level: float = 0.01          # банк < ruin_level * bank0 -> игра окончена
    seed: int = 0
    quantile_every: int = 10          # шаг записи квантилей траектории
    keep_paths: int = 0               # сколько траекторий сохранить (с шагом quantile_every)
    record_full: bool = False         # полная история банка (n_sims, n_bets + 1)


@dataclass
class SimResult:
    config: SimConfig
    strategy: str
    final: np.ndarray
    max_dd: np.ndarray                # 0..1
    min_bank: np.ndarray
    turnover: np.ndarray
    n_placed: np.ndarray
    max_loss_streak: np.ndarray
    ruined: np.ndarray
    stake_pct_sum: np.ndarray         # сумма (ставка / банк) по сделанным ставкам
    steps: np.ndarray
    q_paths: np.ndarray               # (len(QUANTILES), len(steps))
    mean_path: np.ndarray
    sample_paths: np.ndarray
    history: Optional[np.ndarray] = None
    stakes: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Движок
# --------------------------------------------------------------------------

def _recalc_periods(cfg: SimConfig, rng: np.random.Generator, n_sims: int):
    r = cfg.recalc_every
    if isinstance(r, (int, np.integer)):
        return None if r <= 1 else np.full(n_sims, int(r))
    lo, hi = int(r[0]), int(r[1])
    return rng.integers(lo, hi + 1, n_sims)


def simulate(strategy, scenario, cfg: SimConfig) -> SimResult:
    n, T = scenario.n_sims, cfg.n_bets
    if isinstance(scenario, FixedScenario) and scenario.n_bets < T:
        raise ValueError('в FixedScenario меньше ставок, чем cfg.n_bets')

    stake_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 2]))
    bank0 = float(cfg.bank0)
    bank = np.full(n, bank0)
    peak = bank.copy()
    min_bank = bank.copy()
    max_dd = np.zeros(n)
    turnover = np.zeros(n)
    n_placed = np.zeros(n, dtype=np.int64)
    loss_streak = np.zeros(n, dtype=np.int64)
    max_loss_streak = np.zeros(n, dtype=np.int64)
    stake_pct_sum = np.zeros(n)
    ruined = np.zeros(n, dtype=bool)
    ruin_abs = cfg.ruin_level * bank0

    periods = _recalc_periods(cfg, stake_rng, n)
    snapshot = bank.copy()
    since_recalc = np.zeros(n, dtype=np.int64)

    steps = np.arange(0, T + 1, max(1, cfg.quantile_every))
    if steps[-1] != T:
        steps = np.append(steps, T)
    step_set = {int(s): k for k, s in enumerate(steps)}
    q_paths = np.empty((len(QUANTILES), len(steps)))
    mean_path = np.empty(len(steps))
    k_paths = min(cfg.keep_paths, n)
    sample_paths = np.empty((k_paths, len(steps)))
    history = np.empty((n, T + 1)) if cfg.record_full else None
    stakes_hist = np.empty((n, T)) if cfg.record_full else None

    def record(t):
        k = step_set.get(t)
        if k is not None:
            q_paths[:, k] = np.percentile(bank, QUANTILES)
            mean_path[k] = bank.mean()
            if k_paths:
                sample_paths[:, k] = bank[:k_paths]
        if history is not None:
            history[:, t] = bank

    record(0)
    strategy.reset(n, bank0)
    ctx = SimpleNamespace(bank0=bank0, n_bets=T, n_sims=n)

    for t in range(T):
        odds, edge, won = scenario.step(t)

        if periods is not None:
            refresh = since_recalc >= periods
            snapshot = np.where(refresh, bank, snapshot)
            since_recalc = np.where(refresh, 0, since_recalc)
            base = snapshot
        else:
            base = bank

        ctx.step, ctx.odds, ctx.edge = t, odds, edge
        ctx.bank, ctx.peak, ctx.base = bank, peak, base
        f = np.broadcast_to(np.asarray(strategy.fraction(ctx), dtype=float), (n,))
        f = np.clip(np.nan_to_num(f, nan=0.0, posinf=cfg.cap, neginf=0.0), 0.0, cfg.cap)
        stake = f * base

        if cfg.variation is not None:
            kind = cfg.variation[0]
            if kind == 'uniform':
                stake = stake * stake_rng.uniform(cfg.variation[1], cfg.variation[2], n)
            elif kind == 'choice':
                stake = stake * stake_rng.choice(np.asarray(cfg.variation[1], dtype=float), n)
            else:
                raise ValueError(f'неизвестная вариация: {kind}')
        if cfg.round_to:
            stake = np.round(stake / cfg.round_to) * cfg.round_to
        if cfg.min_stake:
            stake = np.where(stake < cfg.min_stake, 0.0, stake)
        stake = np.where(ruined, 0.0, np.minimum(stake, bank))

        placed = stake > 0
        bank_before = bank
        bank = bank + np.where(won, stake * (odds - 1.0), -stake)
        bank = np.maximum(bank, 0.0)   # страховка от ошибок округления

        if stakes_hist is not None:
            stakes_hist[:, t] = stake
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = np.where(bank_before > 0, (bank - bank_before) / bank_before, 0.0)
            stake_pct_sum += np.where(placed, stake / np.maximum(bank_before, 1e-12), 0.0)
        turnover += stake
        n_placed += placed
        lost = placed & ~won
        loss_streak = np.where(lost, loss_streak + 1, np.where(placed, 0, loss_streak))
        np.maximum(max_loss_streak, loss_streak, out=max_loss_streak)

        peak = np.maximum(peak, bank)
        np.maximum(max_dd, (peak - bank) / peak, out=max_dd)
        np.minimum(min_bank, bank, out=min_bank)
        ruined |= bank < ruin_abs
        if periods is not None:
            since_recalc += 1

        ctx.won, ctx.stake, ctx.ret, ctx.placed = won, stake, ret, placed
        ctx.bank, ctx.peak = bank, peak
        strategy.update(ctx)
        record(t + 1)

    return SimResult(cfg, getattr(strategy, 'label', type(strategy).__name__), bank, max_dd, min_bank,
                     turnover, n_placed, max_loss_streak, ruined, stake_pct_sum, steps, q_paths,
                     mean_path, sample_paths, history, stakes_hist)


def run(strategy, market: Market, cfg: SimConfig, sampling: str = 'bootstrap') -> SimResult:
    """Удобная обёртка: сценарий из рынка с тем же seed, что и в конфиге."""
    return simulate(strategy, MarketScenario(market, cfg.n_sims, cfg.seed, sampling), cfg)
