"""
Единый управляемый движок прогона стратегий.

Цель: при ИЗВЕСТНОМ ROI (по умолчанию 2.5%) подобрать допустимый размер
ставки и оценить ХУДШИЕ случаи (просадки, банкротство).

Ключевые решения:
- ROI задаётся параметром и корректно прокидывается и в генерацию исходов,
  и в модуль стратегий (через подмену run_strategies_real_odds.TARGET_ROI),
  потому что стратегии читают TARGET_ROI как глобал своего модуля.
- 10000 коэффициентов семплируются С ПОВТОРЕНИЯМИ из реальных 1013.
- В кэш пишем НЕ полную историю банкролла (физически ~30GB на все прогоны),
  а распределения (финальные банкроллы + max-DD по симуляциям) и метрики.
  Этого достаточно для всех графиков дашборда и любых пересчётов.
"""

import os
import json
import time
import numpy as np

import config

# Подменяем ROI ДО любых расчётов: и для генерации, и для стратегий.
# Это единственный корректный способ — иначе модули используют зашитый 0.07.
DEFAULT_ROI = 0.025

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_data')
INITIAL_BANKROLL = config.INITIAL_BANKROLL


def set_roi(roi):
    """Прокидывает ROI во все модули, которые читают TARGET_ROI как глобал."""
    config.TARGET_ROI = roi
    import run_strategies_real_odds as rs
    rs.TARGET_ROI = roi
    return rs


def load_real_odds(filename='matching_teams_clean.csv'):
    import pandas as pd
    df = pd.read_csv(filename)
    return df['coefficient'].values.astype(float)


def build_dataset(num_sims, num_bets, roi, seed=42):
    """
    Семплирует num_bets коэффициентов с повторениями из реальных,
    генерирует исходы под заданный ROI.
    Возвращает (outcomes[bool], odds[float]).
    """
    real_odds = load_real_odds()
    rng = np.random.default_rng(seed)
    odds = rng.choice(real_odds, size=num_bets, replace=True)

    win_probabilities = np.clip((1.0 + roi) / odds, 0.0, 1.0)

    # Генерация исходов. Используем отдельный seeded генератор для воспроизводимости.
    gen = np.random.default_rng(seed + 1)
    outcomes = gen.random((num_sims, num_bets)) < win_probabilities[None, :]
    return outcomes, odds


def actual_roi_from_outcomes(outcomes, odds):
    """Фактический ROI с оборота по сгенерированным данным (sanity-check)."""
    wins_per_bet = outcomes.sum(axis=0)
    total = outcomes.shape[0]
    profit = (wins_per_bet * (odds - 1) - (total - wins_per_bet)).sum()
    turnover = total * outcomes.shape[1]
    return profit / turnover * 100.0


def compute_metrics(bankroll_history, bet_history):
    """Метрики риска + распределения для дашборда."""
    num_sims = bankroll_history.shape[0]

    bankrupt_mask = np.any(bankroll_history < 1.0, axis=1)
    bankrupt_pct = bankrupt_mask.mean() * 100

    peaks = np.maximum.accumulate(bankroll_history, axis=1)
    drawdowns = (bankroll_history - peaks) / peaks * 100  # <=0
    max_dd_per_sim = drawdowns.min(axis=1)  # самый глубокий DD каждой симуляции (<=0)

    dd20 = np.any(drawdowns <= -20, axis=1).mean() * 100
    dd50 = np.any(drawdowns <= -50, axis=1).mean() * 100
    dd80 = np.any(drawdowns <= -80, axis=1).mean() * 100

    final = bankroll_history[:, -1]
    profit_pct = (final - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100

    turnover = bet_history.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        roi_turnover = np.where(turnover > 0, (final - INITIAL_BANKROLL) / turnover * 100, 0.0)

    metrics = {
        'avg_profit_pct': float(np.mean(profit_pct)),
        'median_profit_pct': float(np.median(profit_pct)),
        'p1_profit_pct': float(np.percentile(profit_pct, 1)),
        'p5_profit_pct': float(np.percentile(profit_pct, 5)),
        'p95_profit_pct': float(np.percentile(profit_pct, 95)),
        'min_profit_pct': float(np.min(profit_pct)),
        'max_profit_pct': float(np.max(profit_pct)),
        'bankrupt_pct': float(bankrupt_pct),
        'dd20_pct': float(dd20),
        'dd50_pct': float(dd50),
        'dd80_pct': float(dd80),
        'avg_maxdd_pct': float(np.mean(max_dd_per_sim)),
        'worst_dd_pct': float(np.min(max_dd_per_sim)),
        'p95_maxdd_pct': float(np.percentile(max_dd_per_sim, 5)),   # 5-й перцентиль = глубокая просадка
        'avg_roi_turnover_pct': float(np.mean(roi_turnover)),
    }
    # Распределения (лёгкие массивы для графиков/пересчёта)
    dist = {
        'final_bankroll': final.astype(np.float32),
        'profit_pct': profit_pct.astype(np.float32),
        'max_dd_pct': max_dd_per_sim.astype(np.float32),
    }
    return metrics, dist


def recover_fraction_and_resim(bankroll_strat, bet_history_strat, outcomes, odds, cap=0.10):
    """
    Берёт результат стратегии (её собственный bankroll + bet_history),
    восстанавливает выбранную долю ставки на каждом шаге
        frac[i] = bet_history[i] / bankroll_before_bet[i]
    и ЗАНОВО симулирует банкролл, применяя ЖЁСТКИЙ лимит:
        ставка <= cap * текущий_банк  (и <= текущий банк).

    Это гарантирует:
      - максимальный проигрыш за ставку = cap (10%) текущего банка
      - банк никогда не уходит в минус (просадка физически <= 100%)
    frac воспроизводит логику стратегии в каждой точке; кэп только УМЕНЬШАЕТ ставку.
    """
    num_sims, num_bets = outcomes.shape

    before = bankroll_strat[:, :num_bets]  # банк перед каждой ставкой
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(before > 0, bet_history_strat / before, 0.0)
    frac = np.clip(frac, 0.0, cap)  # доля не больше кэпа

    bankroll = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    bet_history = np.zeros((num_sims, num_bets), dtype=float)
    for i in range(num_bets):
        bank = bankroll[:, i]
        bet = frac[:, i] * bank
        bet = np.minimum(bet, bank * cap)
        bet = np.minimum(bet, bank)
        bet = np.where(bank > 0, bet, 0.0)
        bet_history[:, i] = bet
        o = odds[i]
        bankroll[:, i + 1] = bank + np.where(outcomes[:, i], bet * (o - 1), -bet)
    return bankroll, bet_history


def strategy_registry(rs):
    """
    Все стратегии: имя -> (функция, параметры).
    Параметры — дефолтные из сигнатур; ROI прокидывается через глобал модуля.
    bayesian_kelly требует prior_mean, привязанный к ROI и средним коэффициентам,
    он выставляется динамически в run_all (зависит от odds).
    """
    return {
        'dynamic_percentage_2pct': (rs.dynamic_percentage_strategy_with_real_odds, {'bet_size_pct': 2.0}),
        'kelly_criterion': (rs.kelly_criterion_strategy_with_real_odds, {'risk': 2.0, 'kelly_fraction': 1.0}),
        'linear_roi': (rs.linear_roi_strategy_with_real_odds, {}),
        'sqrt_roi': (rs.sqrt_roi_strategy_with_real_odds, {}),
        'log_roi': (rs.log_roi_strategy_with_real_odds, {}),
        'constant_profit': (rs.constant_profit_strategy_with_real_odds, {'target_profit_pct': 1.0}),
        'combined_roi_odds': (rs.combined_roi_odds_strategy_with_real_odds, {}),
        'adaptive': (rs.adaptive_strategy_with_real_odds, {}),
        'dynamic_kelly': (rs.dynamic_kelly_strategy_with_real_odds, {}),
        'exponential_roi': (rs.exponential_roi_strategy_with_real_odds, {}),
        'hybrid': (rs.hybrid_strategy_with_real_odds, {}),
        'linear_scaled': (rs.linear_scaled_strategy_with_real_odds, {}),
        'linear_roi_odds': (rs.linear_roi_odds_strategy_with_real_odds, {}),
        'adaptive_constant_profit': (rs.adaptive_constant_profit_strategy_with_real_odds, {}),
        'fixed_fraction_2pct': (rs.fixed_fraction_strategy_with_real_odds, {'fixed_percent': 2.0}),
        'proportional_kelly': (rs.proportional_kelly_strategy_with_real_odds, {'risk': 2.0, 'confidence': 0.7}),
        'target_based': (rs.target_based_strategy_with_real_odds, {}),
        'anti_martingale': (rs.anti_martingale_strategy_with_real_odds, {}),
        'volatility_adjusted': (rs.volatility_adjusted_strategy_with_real_odds, {}),
        'streak_aware': (rs.streak_aware_strategy_with_real_odds, {}),
        'sharpe_optimized': (rs.sharpe_optimized_strategy_with_real_odds, {}),
        'bayesian_kelly': (rs.bayesian_kelly_strategy_with_real_odds, {}),
        'multi_objective': (rs.multi_objective_strategy_with_real_odds, {}),
        'portfolio_theory': (rs.portfolio_theory_strategy_with_real_odds, {}),
        'ml_adaptive': (rs.ml_adaptive_strategy_with_real_odds, {}),
    }


def run_single(key, variation, num_sims, num_bets, roi=DEFAULT_ROI, seed=42, cap=0.10,
               override_params=None):
    """Прогоняет одну стратегию и возвращает (bankroll_history, bet_history, odds) c кэпом."""
    rs = set_roi(roi)
    outcomes, odds = build_dataset(num_sims, num_bets, roi, seed=seed)
    registry = strategy_registry(rs)
    bayes_prior = float(np.clip((1.0 + roi) / odds.mean(), 0.01, 0.99))
    registry['bayesian_kelly'] = (rs.bayesian_kelly_strategy_with_real_odds,
                                  {'prior_mean': bayes_prior, 'prior_std': 0.05})
    func, params = registry[key]
    params = dict(params)
    if override_params:
        params.update(override_params)
    bankroll, bet_history, *_ = func(outcomes, odds, apply_variation=variation, **params)
    if cap is not None:
        bankroll, bet_history = recover_fraction_and_resim(bankroll, bet_history, outcomes, odds, cap=cap)
    return bankroll, bet_history, odds


def run_all(num_sims, num_bets, roi=DEFAULT_ROI, seed=42, cache_dir=CACHE_DIR,
            verbose=True, cap=None):
    """
    Прогоняет все стратегии в обоих режимах (с вариацией и без),
    сохраняет распределения и метрики в cache_dir/<key>.npz и summary.json.

    cap: если задан (напр. 0.10), ко всем стратегиям единообразно применяется
         жёсткий лимит "ставка <= cap * текущий банк" (динамический БМ).
    """
    os.makedirs(cache_dir, exist_ok=True)
    rs = set_roi(roi)

    if verbose:
        print(f"Генерация данных: {num_sims} sims x {num_bets} bets, ROI={roi*100:.2f}%")
    t0 = time.time()
    outcomes, odds = build_dataset(num_sims, num_bets, roi, seed=seed)
    gen_dt = time.time() - t0
    actual_roi = actual_roi_from_outcomes(outcomes, odds)
    if verbose:
        print(f"  Готово за {gen_dt:.1f}s. Avg odds={odds.mean():.3f}. "
              f"Фактический ROI оборота={actual_roi:.3f}% (целевой {roi*100:.2f}%)")

    registry = strategy_registry(rs)
    # prior_mean для bayesian_kelly зависит от данных
    bayes_prior = float(np.clip((1.0 + roi) / odds.mean(), 0.01, 0.99))
    registry['bayesian_kelly'] = (rs.bayesian_kelly_strategy_with_real_odds,
                                  {'prior_mean': bayes_prior, 'prior_std': 0.05})

    summary = {
        'roi_pct': roi * 100,
        'actual_roi_turnover_pct': actual_roi,
        'num_sims': num_sims,
        'num_bets': num_bets,
        'seed': seed,
        'avg_odds': float(odds.mean()),
        'initial_bankroll': INITIAL_BANKROLL,
        'strategies': [],
    }

    total = len(registry) * 2
    done = 0
    for key, (func, params) in registry.items():
        for variation in (False, True):
            done += 1
            t = time.time()
            bankroll, bet_history, min_b, max_b, avg_b = func(
                outcomes, odds, apply_variation=variation, **params)
            if cap is not None:
                bankroll, bet_history = recover_fraction_and_resim(
                    bankroll, bet_history, outcomes, odds, cap=cap)
                # пересчёт фактических процентов ставок после кэпа
                before = bankroll[:, :num_bets]
                with np.errstate(divide='ignore', invalid='ignore'):
                    pct = np.where(before > 0, bet_history / before * 100, 0.0)
                nz = pct[pct > 0]
                min_b, max_b, avg_b = (float(nz.min()), float(nz.max()), float(nz.mean())) if nz.size else (0.0, 0.0, 0.0)
            metrics, dist = compute_metrics(bankroll, bet_history)
            dt = time.time() - t

            row = {
                'key': key,
                'with_variation': variation,
                'avg_bet_pct': float(avg_b),
                'min_bet_pct': float(min_b),
                'max_bet_pct': float(max_b),
                **metrics,
            }
            summary['strategies'].append(row)

            cache_file = os.path.join(cache_dir, f"{key}__{'var' if variation else 'novar'}.npz")
            np.savez_compressed(cache_file, **dist)

            if verbose:
                print(f"[{done}/{total}] {key} {'var' if variation else 'novar':5s} "
                      f"| profit avg={metrics['avg_profit_pct']:8.1f}% med={metrics['median_profit_pct']:8.1f}% "
                      f"| bankrupt={metrics['bankrupt_pct']:5.2f}% dd50={metrics['dd50_pct']:5.1f}% "
                      f"worstDD={metrics['worst_dd_pct']:6.1f}% | {dt:.1f}s")

            # Освобождаем память перед следующим прогоном
            del bankroll, bet_history, dist

    with open(os.path.join(cache_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nГотово. Сводка: {os.path.join(cache_dir, 'summary.json')}")
        print(f"Распределения: {cache_dir}/<strategy>__<var|novar>.npz")
    return summary
