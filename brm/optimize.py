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
     по просадке ещё не нарушен. Тогда проходим геометрическую сетку от s_max
     вниз (×0.8 до ~3% от s_max) и уточняем максимум золотым сечением между
     соседними точками.
Все прогоны идут с одним seed (common random numbers), поэтому метрика
монотонна по scale, а разница между стратегиями не тонет в шуме.
"""

from __future__ import annotations

import math
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
              sampling: str = 'bootstrap', golden: int = 8):
    """
    Возвращает (scale, метрики) с лучшей медианой среди scale, при которых
    metric <= limit, или (0, None), если лимит нарушает даже минимальная ставка.
    metric может быть списком пар (метрика, лимит) — тогда должны выполняться все.
    """
    rules = list(metric) if isinstance(metric, (list, tuple)) else [(metric, limit)]

    def ev(s):
        return evaluate(build(s), market, cfg, sampling)

    def ok(m):
        return all(m[k] <= v for k, v in rules)

    feasible = {}
    lo, hi = 0.0, min(1.0, scale_hi)
    m = ev(hi)
    if ok(m):
        feasible[hi] = m
        while hi < scale_hi:
            lo, hi = hi, min(hi * 2, scale_hi)
            m = ev(hi)
            if not ok(m):
                break
            feasible[hi] = m
        else:
            lo = hi                       # лимит не достигнут до scale_hi
    if lo != hi:
        for _ in range(iters):
            mid = (lo + hi) / 2
            m = ev(mid)
            if ok(m):
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
            if ok(m):
                feasible[s] = m

    def argmax():
        return max(feasible, key=lambda s: feasible[s]['median_profit_pct'])

    best = argmax()
    if best < s_max and golden:
        # лимит не держит: максимум медианы внутри области, и точки выше могли его проскочить —
        # при слабом лимите s_max бывает в разы больше оптимума, а у части стратегий кривая
        # медианы не унимодальна. Поэтому сначала сетка s_max·0.8^k до ~3% от s_max, затем
        # золотое сечение между соседями лучшей точки (если она самая нижняя — между 0 и соседом)
        for k in range(1, 16):
            s = s_max * 0.8 ** k
            if s not in feasible:
                m = ev(s)
                if ok(m):
                    feasible[s] = m
        best = argmax()
        xs = sorted(feasible)
        i = xs.index(best)
        a, b = (xs[i - 1] if i else 0.0), xs[i + 1]
        r = (math.sqrt(5) - 1) / 2
        bad = set()

        def med(s):
            if s not in feasible and s not in bad:
                m = ev(s)
                if ok(m):
                    feasible[s] = m
                else:
                    bad.add(s)
            return feasible[s]['median_profit_pct'] if s in feasible else -math.inf

        x1, x2 = b - r * (b - a), a + r * (b - a)
        for _ in range(golden):
            if med(x1) >= med(x2):
                b, x2 = x2, x1
                x1 = b - r * (b - a)
            else:
                a, x1 = x1, x2
                x2 = a + r * (b - a)
        best = argmax()
    return best, dict(feasible[best], scale_max=s_max)


FLOOR_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def calibrate_floor_kelly(market: Market, cfg: SimConfig, metric, limit, floors=FLOOR_GRID,
                          mult_hi: float = 2.0, sampling: str = 'bootstrap', **kw):
    """
    Келли с полом: у стратегии две ручки — пол (доля пика, ниже которой банк не опускается)
    и доля Келли от подушки над полом. Для каждого пола из сетки подбирается доля, затем
    пол уточняется на ±0.05 вокруг лучшего. Возвращает (доля, метрики с полем floor).
    """
    def run_floor(fl):
        scale, m = calibrate(lambda s: S.FloorKelly(s, fl), market, cfg, metric, limit,
                             scale_hi=mult_hi, sampling=sampling, **kw)
        return None if m is None else (scale, dict(m, floor=round(fl, 3)))

    results = [r for r in (run_floor(fl) for fl in floors) if r]
    if not results:
        return 0.0, None
    best = max(results, key=lambda r: r[1]['median_profit_pct'])
    for fl in (round(best[1]['floor'] - 0.05, 3), round(best[1]['floor'] + 0.05, 3)):   # без round 0.9 + 0.05 > 0.95
        if 0.05 <= fl <= 0.95:
            r = run_floor(fl)
            if r and r[1]['median_profit_pct'] > best[1]['median_profit_pct']:
                best = r
    return best


def risk_builders(profile: str) -> Dict[str, Tuple[Build, float, str]]:
    """Ручка агрессивности для каждой стратегии каталога: key -> (build(scale), scale_hi, описание).
    Профиль на ручки не влияет (оставлен в сигнатуре для совместимости)."""
    out = {}
    for key, (make, desc) in S.catalog().items():
        if key == 'floor_kelly':
            # у этой стратегии две ручки (пол и доля); leaderboard подбирает обе через calibrate_floor_kelly
            out[key] = (None, 2.0, 'Келли от подушки над полом (пол подбирается)')
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
        if key == 'floor_kelly':
            scale, m = calibrate_floor_kelly(market, cfg, metric, limit, mult_hi=scale_hi, sampling=sampling)
            if m is not None:
                desc = f"Келли x{scale:.2f} от подушки над полом {m['floor']:.0%} пика"
        else:
            scale, m = calibrate(build, market, cfg, metric, limit, scale_hi=scale_hi, sampling=sampling)
        if m is None:
            continue
        rows.append(dict(m, key=key, scale=scale, label=desc))
        if verbose:
            print(f"  {key:26s} scale={scale:6.3f} (граница {m['scale_max']:6.3f})  медиана {m['median_profit_pct']:7.1f}%  "
                  f"{metric}={m[metric]:.2f}%  ставка {m['avg_stake_pct']:.2f}%")
    rows.sort(key=lambda r: r['median_profit_pct'], reverse=True)
    return rows
