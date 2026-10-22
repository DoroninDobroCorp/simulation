"""Сводные метрики по результату симуляции."""

from __future__ import annotations

import numpy as np

from .engine import SimResult

DD_LEVELS = (20, 30, 50, 66, 80)


def summarize(res: SimResult) -> dict:
    b0 = res.config.bank0
    T = res.config.n_bets
    profit = (res.final / b0 - 1.0) * 100.0
    dd = res.max_dd * 100.0
    with np.errstate(divide='ignore'):
        log_growth = np.log(np.maximum(res.final, 1e-300) / b0) / T
    placed = np.maximum(res.n_placed, 1)
    out = {
        'strategy': res.strategy,
        'n_sims': len(res.final),
        'n_bets': T,
        'mean_profit_pct': float(profit.mean()),
        'median_profit_pct': float(np.median(profit)),
        'p1_profit_pct': float(np.percentile(profit, 1)),
        'p5_profit_pct': float(np.percentile(profit, 5)),
        'p25_profit_pct': float(np.percentile(profit, 25)),
        'p75_profit_pct': float(np.percentile(profit, 75)),
        'p95_profit_pct': float(np.percentile(profit, 95)),
        'p_loss_pct': float((profit < 0).mean() * 100),
        'median_max_dd_pct': float(np.median(dd)),
        'mean_max_dd_pct': float(dd.mean()),
        'p95_max_dd_pct': float(np.percentile(dd, 95)),
        'worst_max_dd_pct': float(dd.max()),
        'p_below_half_start_pct': float((res.min_bank < 0.5 * b0).mean() * 100),
        'p_ruin_pct': float(res.ruined.mean() * 100),
        'median_growth_per_bet_pct': float(np.median(log_growth) * 100),
        # ROI с оборота: суммарная прибыль / суммарный оборот (не среднее отношений)
        'roi_turnover_pct': float((res.final - b0).sum() / max(res.turnover.sum(), 1e-12) * 100),
        'avg_stake_pct': float((res.stake_pct_sum / placed).mean() * 100),
        'median_max_loss_streak': float(np.median(res.max_loss_streak)),
        'max_loss_streak': int(res.max_loss_streak.max()),
    }
    for lvl in DD_LEVELS:
        out[f'p_dd{lvl}_pct'] = float((dd >= lvl).mean() * 100)
    return out


def fmt_row(m: dict) -> str:
    return (f"{m['strategy'][:34]:34s} медиана {m['median_profit_pct']:8.1f}%  среднее {m['mean_profit_pct']:9.1f}%  "
            f"P(минус) {m['p_loss_pct']:5.1f}%  медиана DD {m['median_max_dd_pct']:5.1f}%  "
            f"P(DD≥50) {m['p_dd50_pct']:5.1f}%  P(DD≥80) {m['p_dd80_pct']:5.1f}%  ставка {m['avg_stake_pct']:4.2f}%")
