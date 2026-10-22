"""
Калибровка стратегий под риск-профиль и сравнение при РАВНОМ риске.

Сравнивать стратегии с произвольными параметрами бессмысленно: та, что ставит
больше, всегда «прибыльнее» в среднем и «рискованнее». Правильный вопрос —
какая стратегия даёт больший МЕДИАННЫЙ рост при одном и том же ограничении
на риск. У каждой стратегии есть ручка агрессивности scale (множитель размера
ставки). Калибровка:
  1) бисекцией ищется граница s_max, где стратегия упирается в лимит профиля;
  2) внутри допустимой области [0, s_max] берётся scale с лучшей медианой —
     иногда это не граница: за полным Келли рост падает, даже если лимит
     по просадке ещё не нарушен.
Все прогоны идут с одним seed (common random numbers), поэтому метрика
монотонна по scale, а разница между стратегиями не тонет в шуме.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from .engine import Market, SimConfig, run
from .metrics import summarize
from . import strategies as S

# Профили из simulation-original/tz.md: (метрика, максимально допустимый % симуляций)
RISK_PROFILES = {
    'conservative': ('p_dd50_pct', 0.0),
    'cautious': ('p_dd50_pct', 3.5),
    'balanced': ('p_dd50_pct', 10.0),
    'risky': ('p_dd80_pct', 5.0),
    'extreme': ('p_dd80_pct', 50.0),
}

Build = Callable[[float], 'S.Strategy']


def evaluate(strategy, market: Market, cfg: SimConfig, sampling='bootstrap') -> dict:
    return summarize(run(strategy, market, cfg, sampling))


def calibrate(build: Build, market: Market, cfg: SimConfig, metric: str, limit: float,
              scale_hi: float = 64.0, iters: int = 12, refine=(0.4, 0.55, 0.7, 0.85),
              sampling: str = 'bootstrap'):
    """
    Возвращает (scale, метрики) с лучшей медианой среди scale, при которых
    metric <= limit, или (0, None), если лимит нарушает даже минимальная ставка.
    """
    def ev(s):
        return evaluate(build(s), market, cfg, sampling)

    feasible = {}
    lo, hi = 0.0, min(1.0, scale_hi)
    m = ev(hi)
    if m[metric] <= limit:
        feasible[hi] = m
        while hi < scale_hi:
            lo, hi = hi, min(hi * 2, scale_hi)
            m = ev(hi)
            if m[metric] > limit:
                break
            feasible[hi] = m
        else:
            lo = hi                       # лимит не достигнут до scale_hi
    if lo != hi:
        for _ in range(iters):
            mid = (lo + hi) / 2
            m = ev(mid)
            if m[metric] <= limit:
                lo, feasible[mid] = mid, m
            else:
                hi = mid
    if not feasible:
        return 0.0, None
    s_max = max(feasible)
    for k in refine:
        s = s_max * k
        if s not in feasible:
            m = ev(s)
            if m[metric] <= limit:
                feasible[s] = m
    best = max(feasible, key=lambda s: feasible[s]['median_profit_pct'])
    return best, dict(feasible[best], scale_max=s_max)


def risk_builders(profile: str) -> Dict[str, Tuple[Build, float, str]]:
    """Ручка агрессивности для каждой стратегии каталога: key -> (build(scale), scale_hi, описание)."""
    metric, _ = RISK_PROFILES[profile]
    floor = 1.0 - int(metric[len('p_dd'):-len('_pct')]) / 100.0
    out = {}
    for key, (make, desc) in S.catalog().items():
        if key == 'floor_kelly':
            # риск задаёт пол (просадка не глубже 1 - floor), ручка — доля Келли от подушки
            out[key] = (lambda s, f=floor: S.FloorKelly(mult=s, floor=f), 1.5,
                        f'Келли от подушки над полом {floor:.0%} пика')
        else:
            out[key] = (lambda s, mk=make: S.Scaled(mk(), s), 64.0, desc)
    return out


def leaderboard(market: Market, cfg: SimConfig, profile: str = 'cautious', keys=None,
                sampling: str = 'bootstrap', verbose: bool = True):
    """Для каждой стратегии — лучший scale в рамках профиля и метрики при нём."""
    metric, limit = RISK_PROFILES[profile]
    rows = []
    for key, (build, scale_hi, desc) in risk_builders(profile).items():
        if keys is not None and key not in keys:
            continue
        scale, m = calibrate(build, market, cfg, metric, limit, scale_hi=scale_hi, sampling=sampling)
        if m is None:
            continue
        rows.append(dict(m, key=key, scale=scale, label=desc))
        if verbose:
            print(f"  {key:26s} scale={scale:6.3f} (граница {m['scale_max']:6.3f})  медиана {m['median_profit_pct']:7.1f}%  "
                  f"{metric}={m[metric]:.2f}%  ставка {m['avg_stake_pct']:.2f}%")
    rows.sort(key=lambda r: r['median_profit_pct'], reverse=True)
    return rows
