"""
Источники ставок: реальные ставки из CSV и синтетический генератор по ТЗ.

Ставка описывается парой (кэф, оценка EV). Истинный EV ставки задаёт модель
рынка в `brm.engine.Market`, здесь только «сырьё».
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_BETS_CSV = os.path.join(REPO_ROOT, 'simulation-current', 'matching_teams_clean.csv')


@dataclass(frozen=True)
class BetPool:
    """Пул ставок, из которого семплируются симуляции."""
    odds: np.ndarray       # десятичные кэфы, > 1
    ev: np.ndarray         # оценка EV модели, доля (0.05 = +5%)
    name: str = 'pool'

    def __post_init__(self):
        if self.odds.shape != self.ev.shape:
            raise ValueError('odds и ev должны быть одной длины')
        if np.any(self.odds <= 1.0):
            raise ValueError('все кэфы должны быть > 1')

    def __len__(self):
        return len(self.odds)


def load_real_bets(path: str = REAL_BETS_CSV) -> BetPool:
    """Реальные ставки: coefficient -> кэф, ev_profit/stake -> оценка EV."""
    import csv
    odds, ev = [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            odds.append(float(row['coefficient']))
            ev.append(float(row['ev_profit']) / float(row['stake']))
    return BetPool(np.asarray(odds), np.asarray(ev), name='real')


def beta_params_for_mean(mean01: float, concentration: float = 4.0) -> tuple[float, float]:
    """Параметры Beta(a, b) с заданным средним a/(a+b) и суммой a+b."""
    if not 0.0 < mean01 < 1.0:
        raise ValueError('среднее Beta должно лежать в (0, 1)')
    return mean01 * concentration, (1.0 - mean01) * concentration


def synthetic_pool(size: int = 20000,
                   odds_min: float = 1.5, odds_max: float = 3.5, odds_mean: float = 2.8,
                   roi_ranges=((0.03, 0.06), (0.06, 0.12), (0.12, 0.20)),
                   roi_weights=(0.85, 0.10, 0.05),
                   concentration: float = 4.0,
                   seed: int = 7) -> BetPool:
    """
    Генератор из ТЗ (simulation-original/tz.md):
    кэф в [1.5, 3.5] со смещением вправо и средним 2.8, ROI из смеси трёх
    равномерных диапазонов 3–6% / 6–12% / 12–20% с весами 85/10/5.

    Исправление относительно distribution_generator.py: Beta(2, 2.2) давала
    средний кэф 2.45; здесь параметры Beta выводятся из требуемого среднего.
    """
    rng = np.random.default_rng(seed)
    a, b = beta_params_for_mean((odds_mean - odds_min) / (odds_max - odds_min), concentration)
    odds = odds_min + (odds_max - odds_min) * rng.beta(a, b, size)
    cat = rng.choice(len(roi_ranges), size=size, p=np.asarray(roi_weights) / np.sum(roi_weights))
    lo = np.array([r[0] for r in roi_ranges])[cat]
    hi = np.array([r[1] for r in roi_ranges])[cat]
    ev = rng.uniform(lo, hi)
    return BetPool(odds, ev, name='synthetic')
