"""
Калибровка стратегий под риск-профиль и сравнение при РАВНОМ риске.

Сравнивать стратегии с произвольными параметрами бессмысленно: та, что ставит
больше, всегда «прибыльнее» и «рискованнее». Правильный вопрос — какая
стратегия даёт больший рост при одинаковом ограничении на риск. Для этого
размер ставки каждой стратегии масштабируется множителем scale, который
подбирается бисекцией до границы профиля. Все прогоны идут с одним seed
(common random numbers), поэтому метрика монотонна по scale, а разница между
стратегиями не тонет в шуме.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .engine import Market, SimConfig, run
from .metrics import summarize
from .strategies import Scaled

# Профили из simulation-original/tz.md: (метрика, максимально допустимый % симуляций)
RISK_PROFILES = {
    'conservative': ('p_dd50_pct', 0.0),
    'cautious': ('p_dd50_pct', 3.5),
    'balanced': ('p_dd50_pct', 10.0),
    'risky': ('p_dd80_pct', 5.0),
    'extreme': ('p_dd80_pct', 50.0),
}


def evaluate(make: Callable, scale: float, market: Market, cfg: SimConfig, sampling='bootstrap') -> dict:
    return summarize(run(Scaled(make(), scale), market, cfg, sampling))


def calibrate(make: Callable, market: Market, cfg: SimConfig, metric: str, limit: float,
              scale_hi: float = 64.0, iters: int = 14, sampling: str = 'bootstrap'):
    """
    Наибольший scale, при котором metric <= limit. Возвращает (scale, метрики)
    или (0, None), если даже минимальная ставка нарушает ограничение.
    """
    lo, hi = 0.0, 1.0
    m_hi = evaluate(make, hi, market, cfg, sampling)
    best = None
    if m_hi[metric] <= limit:
        best = (hi, m_hi)
        while hi < scale_hi:
            lo, hi = hi, hi * 2
            m_hi = evaluate(make, hi, market, cfg, sampling)
            if m_hi[metric] > limit:
                break
            best = (hi, m_hi)
        else:
            return best
    for _ in range(iters):
        mid = (lo + hi) / 2
        m = evaluate(make, mid, market, cfg, sampling)
        if m[metric] <= limit:
            lo, best = mid, (mid, m)
        else:
            hi = mid
    return best if best else (0.0, None)


def leaderboard(strategies: dict, market: Market, cfg: SimConfig, profile: str = 'cautious',
                sampling: str = 'bootstrap', verbose: bool = True):
    """Для каждой стратегии — scale на границе профиля и метрики при нём."""
    metric, limit = RISK_PROFILES[profile]
    rows = []
    for key, make in strategies.items():
        scale, m = calibrate(make, market, replace(cfg), metric, limit, sampling=sampling)
        if m is None:
            continue
        m = dict(m, key=key, scale=scale)
        rows.append(m)
        if verbose:
            print(f"  {key:26s} scale={scale:6.3f}  медиана {m['median_profit_pct']:8.1f}%  "
                  f"{metric}={m[metric]:.2f}%  ставка {m['avg_stake_pct']:.2f}%")
    rows.sort(key=lambda r: r['median_profit_pct'], reverse=True)
    return rows
