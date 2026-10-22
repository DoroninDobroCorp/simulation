"""
=============================================================================
СИСТЕМА ВАРИАЦИИ РАЗМЕРА СТАВОК
=============================================================================

ВАЖНО ДЛЯ ПОНИМАНИЯ:

1. РАСЧЕТНАЯ СТАВКА (Ideal Bet):
   - Каждая стратегия рассчитывает "идеальный" размер ставки по своей формуле
   - Например, Kelly рекомендует 2.5% от банка, dynamic_percentage - 1.5%
   
2. РЕАЛЬНАЯ СТАВКА (Actual Bet):
   - В реальности не всегда получается поставить точно расчетную сумму
   - Причины: лимиты букмекера, доступные суммы, округления и т.д.
   
3. ВАРИАЦИЯ (apply_variation=True):
   - Имитирует реальность: случайный коэффициент от 35% до 115%
   - STAKE_VARIATIONS = [0.35, 0.40, 0.45, ..., 1.10, 1.15]
   - Реальная ставка = Расчетная × случайный_коэффициент
   
4. ПРИМЕР:
   Стратегия рассчитала: 100 единиц
   - БЕЗ вариации: ставим ровно 100
   - С вариацией: ставим случайно от 30 до 115 (например, 85 или 105)

5. КАК ПРОГОНЯТЬ СТРАТЕГИИ:
   
   # Без вариации (идеальные условия):
   result = run_strategy_with_real_odds(
       'kelly_criterion', outcomes, odds, 
       risk=2.0, kelly_fraction=0.5,
       apply_variation=False
   )
   
   # С вариацией (реалистичные условия):
   result = run_strategy_with_real_odds(
       'kelly_criterion', outcomes, odds,
       risk=2.0, kelly_fraction=0.5, 
       apply_variation=True
   )

6. ЭФФЕКТ ВАРИАЦИИ — ОСТОРОЖНО С ВЫВОДАМИ:
   - средний множитель STAKE_VARIATIONS = 0.75, то есть вариация в среднем
     уменьшает ставку на 25%. «Меньше риска и меньше прибыли» объясняется
     в основном этим, а не самим разбросом. Чтобы оценить влияние разброса,
     сравнивайте с вариантом без вариации и ставкой 0.75 × расчётная.
   - ROI с оборота от вариации не зависит (это свойство ставок, а не стаканов).
   - Стратегии ниже — адаптеры на корректный движок brm, см. ../AUDIT.md.

=============================================================================
"""

import numpy as np
import csv
import os
import sys
from datetime import datetime
from generate_real_odds_simulations import load_real_odds_outcomes
from config import INITIAL_BANKROLL, RANDOM_SEED, TARGET_ROI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brm.engine import FixedScenario, SimConfig, simulate  # noqa: E402
from brm import strategies as S  # noqa: E402

# Вариация размера ставок: от 35% до 115% с шагом 5% (РЕАЛЬНЫЕ УСЛОВИЯ РАБОТЫ)
STAKE_VARIATIONS = np.arange(0.35, 1.16, 0.05)  # [0.35, 0.40, 0.45, ... 1.10, 1.15]

def apply_realistic_stake_variation(bet_amounts, seed_offset=0):
    """
    Применяет реалистичную вариацию к размерам ставок.
    
    Args:
        bet_amounts: numpy array (num_sims, num_bets) с расчетными размерами ставок
        seed_offset: смещение для генератора случайных чисел
    
    Returns:
        numpy array с реальными размерами ставок после вариации
    """
    np.random.seed(RANDOM_SEED + seed_offset)
    num_sims, num_bets = bet_amounts.shape
    
    # Для каждой ставки случайно выбираем коэффициент вариации
    variation_factors = np.random.choice(STAKE_VARIATIONS, size=(num_sims, num_bets))
    
    # Применяем вариацию
    real_bet_amounts = bet_amounts * variation_factors
    
    return real_bet_amounts, variation_factors

def calculate_metrics_with_odds(bankroll_history, bet_history, odds_array):
    """
    Рассчитывает метрики с учетом переменных коэффициентов.
    
    Args:
        bankroll_history: numpy array (num_sims, num_bets + 1)
        bet_history: numpy array (num_sims, num_bets)
        odds_array: numpy array (num_bets,) с коэффициентами
    
    Returns:
        dict с метриками
    """
    num_sims = bankroll_history.shape[0]
    
    # Банкротство (ИСПРАВЛЕНО: bank < 1 = фактическое банкротство, работать нельзя)
    bankrupt_count = np.sum(np.any(bankroll_history < 1.0, axis=1))
    bankrupt_pct = bankrupt_count / num_sims * 100
    
    # Просадки от пика
    peaks = np.maximum.accumulate(bankroll_history, axis=1)
    drawdowns_pct = (bankroll_history - peaks) / peaks * 100
    
    max_drawdown_per_sim = np.min(drawdowns_pct, axis=1)
    
    threshold_20_count = np.sum(np.any(drawdowns_pct <= -20, axis=1))
    threshold_50_count = np.sum(np.any(drawdowns_pct <= -50, axis=1))
    threshold_80_count = np.sum(np.any(drawdowns_pct <= -80, axis=1))
    
    # Прибыль
    final_bankrolls = bankroll_history[:, -1]
    final_profits = final_bankrolls - INITIAL_BANKROLL
    final_profits_pct = final_profits / INITIAL_BANKROLL * 100
    
    # ROI с оборота. Средний = суммарная прибыль / суммарный оборот по всем симуляциям:
    # среднее отношений по симуляциям при компаундинге смещено вниз (7% -> ~5%).
    total_turnover = np.sum(bet_history, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        roi_from_turnover = np.where(total_turnover > 0, final_profits / total_turnover * 100, 0.0)
    pooled_roi = np.sum(final_profits) / max(np.sum(total_turnover), 1e-12) * 100
    
    return {
        'bankrupt_pct': bankrupt_pct,
        'drawdown_20_pct': threshold_20_count / num_sims * 100,
        'drawdown_50_pct': threshold_50_count / num_sims * 100,
        'drawdown_80_pct': threshold_80_count / num_sims * 100,
        'avg_profit_pct': np.mean(final_profits_pct),
        'min_profit_pct': np.min(final_profits_pct),
        'max_profit_pct': np.max(final_profits_pct),
        'avg_max_drawdown_pct': np.mean(max_drawdown_per_sim),
        'worst_drawdown_pct': np.min(max_drawdown_per_sim),
        'avg_roi_from_turnover': pooled_roi,
        'min_roi_from_turnover': np.min(roi_from_turnover),
        'max_roi_from_turnover': np.max(roi_from_turnover)
    }

# =============================================================================
# СТРАТЕГИИ — адаптеры на корректный движок brm (../brm)
# =============================================================================
#
# Раньше здесь было 25 копий одного двухпроходного шаблона, и в 22 из них
# размер ставки считался от НЕОБНОВЛЯЕМОГО массива банка (всегда 1000), то есть
# фактически были флэт-ставки, а не % от текущего банка; во втором проходе у
# части функций не было защиты от ухода банка в минус. Подробности: ../AUDIT.md.
#
# Теперь каждая функция — тонкая обёртка: та же сигнатура, те же формулы
# (они живут в brm/strategies.py), но расчёт однопроходный: ставка считается
# от живого банка и сразу разыгрывается, банк не уходит в минус.
# Возвращаемое значение прежнее: (bankroll_history, bet_history, min%, max%, avg%).

def _run_legacy(strategy, outcomes, odds_array, apply_variation, cap):
    num_sims, num_bets = outcomes.shape
    cfg = SimConfig(
        n_sims=num_sims, n_bets=num_bets, bank0=INITIAL_BANKROLL, cap=cap,
        variation=('choice', STAKE_VARIATIONS) if apply_variation else None,
        ruin_level=1.0 / INITIAL_BANKROLL,      # банк < 1 = банкрот (как в calculate_metrics_with_odds)
        seed=RANDOM_SEED, quantile_every=num_bets, record_full=True)
    # TARGET_ROI читается в момент вызова: sim_engine.set_roi подменяет его в этом модуле
    res = simulate(strategy, FixedScenario(outcomes, odds_array, TARGET_ROI), cfg)
    bankroll, bet_history = res.history, res.stakes
    before = bankroll[:, :num_bets]
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(before > 0, bet_history / before * 100, 0.0)
    valid = pct[pct > 0]
    if valid.size:
        return bankroll, bet_history, float(valid.min()), float(valid.max()), float(valid.mean())
    return bankroll, bet_history, 0.0, 0.0, 0.0


CAP_10 = 0.10   # у этих стратегий во втором проходе был жёсткий лимит 10% текущего банка
NO_CAP = 1.0


def kelly_criterion_strategy_with_real_odds(outcomes, odds_array, risk=2.0, kelly_fraction=1.0, apply_variation=False):
    """Kelly: f = (b*p - q) / b * kelly_fraction / risk, p = (1 + ROI) / odds, max 10%."""
    return _run_legacy(S.Kelly(mult=kelly_fraction / risk), outcomes, odds_array, apply_variation, CAP_10)


def linear_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, max_percent=10.0, apply_variation=False):
    """bet% = base_percent * ROI / base_roi, не более max_percent."""
    return _run_legacy(S.RoiScaled('linear', base_roi, base_percent, max_percent), outcomes, odds_array,
                       apply_variation, CAP_10)


def sqrt_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, max_percent=10.0, apply_variation=False):
    """bet% = base_percent * sqrt(ROI / base_roi), не более max_percent."""
    return _run_legacy(S.RoiScaled('sqrt', base_roi, base_percent, max_percent), outcomes, odds_array,
                       apply_variation, CAP_10)


def log_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0, max_percent=10.0, apply_variation=False):
    """bet% = base_percent * ln(ROI / base_roi + 1), не более max_percent."""
    return _run_legacy(S.RoiScaled('log', base_roi, base_percent, max_percent), outcomes, odds_array,
                       apply_variation, CAP_10)


def constant_profit_strategy_with_real_odds(outcomes, odds_array, target_profit_pct=1.0, max_percent=10.0, apply_variation=False):
    """Ставка = target_profit_pct% банка / (odds - 1), не более max_percent."""
    return _run_legacy(S.ConstantProfit(target_profit_pct, max_percent), outcomes, odds_array,
                       apply_variation, CAP_10)


def combined_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, max_percent=10.0,
                                              min_roi=3.0, max_roi=15.0, min_odds=1.5, max_odds=5.0,
                                              apply_variation=False):
    """bet% = base * sqrt(norm_roi) * (1 - 0.5 * norm_odds)."""
    return _run_legacy(S.CombinedRoiOdds(base_percent, max_percent, min_roi, max_roi, min_odds, max_odds),
                       outcomes, odds_array, apply_variation, CAP_10)


def adaptive_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, max_percent=10.0,
                                     min_roi=3.0, max_roi=15.0, min_odds=1.5, max_odds=5.0,
                                     apply_variation=False):
    """Combined ROI-Odds + x0.75/x0.5 на просадке от пика > 20/40%, потолок при просадке от старта."""
    return _run_legacy(S.Adaptive(base_percent, max_percent, min_roi, max_roi, min_odds, max_odds),
                       outcomes, odds_array, apply_variation, NO_CAP)


def dynamic_kelly_strategy_with_real_odds(outcomes, odds_array, risk=2.0, min_fraction=0.1, max_fraction=0.5,
                                          min_roi=3.0, max_roi=15.0, apply_variation=False):
    """Kelly / risk * fraction(ROI). ИСПРАВЛЕНО: доля больше не делится на 100 второй раз."""
    return _run_legacy(S.DynamicKelly(risk, min_fraction, max_fraction, min_roi, max_roi),
                       outcomes, odds_array, apply_variation, CAP_10)


def exponential_roi_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0,
                                            factor=0.1, max_percent=10.0, apply_variation=False):
    """bet% = base_percent * exp(factor * (ROI - base_roi)), не более max_percent."""
    return _run_legacy(S.RoiScaled('exp', base_roi, base_percent, max_percent, factor), outcomes, odds_array,
                       apply_variation, CAP_10)


def hybrid_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, max_percent=10.0,
                                   min_roi=3.0, max_roi=15.0, min_odds=1.5, max_odds=5.0,
                                   roi_weight=0.7, odds_weight=0.3, apply_variation=False):
    """bet% = base * (roi_weight * norm_roi + odds_weight * (1 - norm_odds))."""
    return _run_legacy(S.Hybrid(base_percent, max_percent, min_roi, max_roi, min_odds, max_odds,
                                roi_weight, odds_weight), outcomes, odds_array, apply_variation, CAP_10)


def linear_scaled_strategy_with_real_odds(outcomes, odds_array, min_roi=3.0, max_roi=20.0,
                                          min_percent=1.0, max_percent=7.0, apply_variation=False):
    """ROI [min_roi, max_roi] -> bet [min_percent, max_percent]% линейно."""
    return _run_legacy(S.LinearScaled(min_roi, max_roi, min_percent, max_percent), outcomes, odds_array,
                       apply_variation, CAP_10)


def linear_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_roi=5.0, base_percent=1.0,
                                            max_percent=10.0, odds_penalty_factor=0.7,
                                            min_odds=1.5, max_odds=5.0, apply_variation=False):
    """bet% = base * ROI / base_roi * (1 - penalty * norm_odds)."""
    return _run_legacy(S.LinearRoiOdds(base_roi, base_percent, max_percent, odds_penalty_factor, min_odds, max_odds),
                       outcomes, odds_array, apply_variation, CAP_10)


def adaptive_constant_profit_strategy_with_real_odds(outcomes, odds_array, min_roi=3.0, max_roi=20.0,
                                                     min_target_pct=0.5, max_target_pct=3.0,
                                                     max_bet_percent=15.0, apply_variation=False):
    """Целевая прибыль [min_target, max_target]% по ROI, ставка = target / (odds - 1)."""
    return _run_legacy(S.AdaptiveConstantProfit(min_roi, max_roi, min_target_pct, max_target_pct, max_bet_percent),
                       outcomes, odds_array, apply_variation, CAP_10)


def fixed_fraction_strategy_with_real_odds(outcomes, odds_array, fixed_percent=2.0, apply_variation=False):
    """fixed_percent% от текущего банка."""
    return _run_legacy(S.FixedFraction(fixed_percent), outcomes, odds_array, apply_variation, NO_CAP)


def proportional_kelly_strategy_with_real_odds(outcomes, odds_array, risk=2.0, confidence=0.7, max_percent=10.0, apply_variation=False):
    """Kelly / risk * confidence. ИСПРАВЛЕНО: доля больше не делится на 100 второй раз."""
    return _run_legacy(S.ProportionalKelly(risk, confidence, max_percent), outcomes, odds_array,
                       apply_variation, NO_CAP)


def target_based_strategy_with_real_odds(outcomes, odds_array, target_bankroll_percent=200.0,
                                         aggressive_pct=3.0, conservative_pct=1.0, apply_variation=False):
    """aggressive_pct пока банк < цели, потом conservative_pct."""
    return _run_legacy(S.TargetBased(target_bankroll_percent, aggressive_pct, conservative_pct),
                       outcomes, odds_array, apply_variation, NO_CAP)


def anti_martingale_strategy_with_real_odds(outcomes, odds_array, base_percent=1.0, multiplier=1.5,
                                            max_percent=10.0, max_streak=5, apply_variation=False):
    """После выигрыша x multiplier (до max_streak раз), после проигрыша — base_percent."""
    return _run_legacy(S.AntiMartingale(base_percent, multiplier, max_percent, max_streak),
                       outcomes, odds_array, apply_variation, NO_CAP)


def volatility_adjusted_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                                lookback=50, volatility_factor=1.0, apply_variation=False):
    """bet% = base / (1 + factor * std доходностей за lookback)."""
    return _run_legacy(S.VolatilityAdjusted(base_percent, lookback, volatility_factor),
                       outcomes, odds_array, apply_variation, NO_CAP)


def streak_aware_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                         win_streak_multiplier=1.2, loss_streak_divider=1.3,
                                         max_multiplier=3.0, apply_variation=False):
    """Серии выигрышей увеличивают ставку, серии проигрышей уменьшают."""
    return _run_legacy(S.StreakAware(base_percent, win_streak_multiplier, loss_streak_divider, max_multiplier),
                       outcomes, odds_array, apply_variation, NO_CAP)


def sharpe_optimized_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                             lookback=100, risk_free_rate=0.0, apply_variation=False):
    """bet% = base * (1 + clip(Sharpe за lookback, -0.5, 1))."""
    return _run_legacy(S.SharpeOptimized(base_percent, lookback, risk_free_rate),
                       outcomes, odds_array, apply_variation, NO_CAP)


def bayesian_kelly_strategy_with_real_odds(outcomes, odds_array, prior_mean=0.378, prior_std=0.05,
                                           risk_factor=2.0, max_percent=10.0, apply_variation=False):
    """
    Kelly по консервативной вероятности (1 + ROI) / odds - prior_std.
    ИСПРАВЛЕНО: prior_mean больше не используется как вероятность для ЛЮБОГО кэфа,
    доля не делится на 100 второй раз. Параметр оставлен для совместимости.
    """
    return _run_legacy(S.BayesianKelly(prior_std, risk_factor, max_percent), outcomes, odds_array,
                       apply_variation, NO_CAP)


def multi_objective_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                            w_profit=0.5, w_drawdown=0.3, w_volatility=0.2,
                                            lookback=50, apply_variation=False):
    """Скоринг прибыли / просадки / волатильности за окно -> множитель ставки."""
    return _run_legacy(S.MultiObjective(base_percent, w_profit, w_drawdown, w_volatility, lookback),
                       outcomes, odds_array, apply_variation, NO_CAP)


def portfolio_theory_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                             rebalance_frequency=100, apply_variation=False):
    """Веса по корзинам кэфов 0.4 / 0.4 / 0.2 (rebalance_frequency не использовался и в оригинале)."""
    return _run_legacy(S.PortfolioTheory(base_percent), outcomes, odds_array, apply_variation, NO_CAP)


def ml_adaptive_strategy_with_real_odds(outcomes, odds_array, base_percent=2.0,
                                        learning_rate=0.01, lookback=30, apply_variation=False):
    """Процент дрейфует к фактической проходимости. ИСПРАВЛЕНО: ожидаемая проходимость."""
    return _run_legacy(S.MLAdaptive(base_percent, learning_rate, lookback), outcomes, odds_array,
                       apply_variation, NO_CAP)


def dynamic_percentage_strategy_with_real_odds(outcomes, odds_array, bet_size_pct, apply_variation=False):
    """bet_size_pct x1.2 при банке > 120% начального, x0.7 при < 80%, x0.5 при < 60%.
    ИСПРАВЛЕНО: с вариацией банк больше не уходит в минус (раньше до -1.28M)."""
    return _run_legacy(S.DynamicPercentage(bet_size_pct), outcomes, odds_array, apply_variation, NO_CAP)


def run_strategy_with_real_odds(strategy_name, outcomes, odds_array, apply_variation=False, **strategy_params):
    """Прогоняет стратегию с реальными коэффициентами."""
    variation_suffix = "_with_variation" if apply_variation else ""
    
    # Формируем имя стратегии
    if strategy_name == 'kelly_criterion':
        risk = strategy_params.get('risk', 2.0)
        kelly_fraction = strategy_params.get('kelly_fraction', 1.0)
        unique_name = f"{strategy_name}_r{risk}_f{kelly_fraction}{variation_suffix}"
        param_desc = f"risk={risk}, fraction={kelly_fraction}"
    elif strategy_name == 'dynamic_percentage':
        bet_size_pct = strategy_params.get('bet_size_pct', 1.5)
        unique_name = f"{strategy_name}_{bet_size_pct}%{variation_suffix}"
        param_desc = f"bet_size={bet_size_pct}%"
    elif strategy_name in ['linear_roi', 'sqrt_roi', 'log_roi']:
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_br{base_roi}_bp{base_percent}_max{max_percent}{variation_suffix}"
        param_desc = f"base_roi={base_roi}, base_pct={base_percent}, max={max_percent}"
    elif strategy_name == 'constant_profit':
        target_profit_pct = strategy_params.get('target_profit_pct', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_tp{target_profit_pct}_max{max_percent}{variation_suffix}"
        param_desc = f"target_profit={target_profit_pct}%, max={max_percent}"
    elif strategy_name in ['combined_roi_odds', 'adaptive']:
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        unique_name = f"{strategy_name}_bp{base_percent}_max{max_percent}_roi{min_roi}-{max_roi}{variation_suffix}"
        param_desc = f"base={base_percent}%, max={max_percent}%, roi_range={min_roi}-{max_roi}"
    elif strategy_name == 'dynamic_kelly':
        risk = strategy_params.get('risk', 2.0)
        min_fraction = strategy_params.get('min_fraction', 0.1)
        max_fraction = strategy_params.get('max_fraction', 0.5)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        unique_name = f"{strategy_name}_r{risk}_f{min_fraction}-{max_fraction}_roi{min_roi}-{max_roi}{variation_suffix}"
        param_desc = f"risk={risk}, frac={min_fraction}-{max_fraction}, roi={min_roi}-{max_roi}"
    elif strategy_name == 'exponential_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        factor = strategy_params.get('factor', 0.1)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_br{base_roi}_bp{base_percent}_f{factor}_max{max_percent}{variation_suffix}"
        param_desc = f"base_roi={base_roi}, base_pct={base_percent}, factor={factor}, max={max_percent}"
    elif strategy_name == 'hybrid':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        roi_weight = strategy_params.get('roi_weight', 0.7)
        odds_weight = strategy_params.get('odds_weight', 0.3)
        unique_name = f"{strategy_name}_bp{base_percent}_max{max_percent}_w{roi_weight}-{odds_weight}{variation_suffix}"
        param_desc = f"base={base_percent}%, max={max_percent}%, weights={roi_weight}/{odds_weight}"
    elif strategy_name == 'linear_scaled':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_percent = strategy_params.get('min_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 7.0)
        unique_name = f"{strategy_name}_roi{min_roi}-{max_roi}_bet{min_percent}-{max_percent}{variation_suffix}"
        param_desc = f"roi=[{min_roi}-{max_roi}], bet=[{min_percent}-{max_percent}]%"
    elif strategy_name == 'linear_roi_odds':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        odds_penalty_factor = strategy_params.get('odds_penalty_factor', 0.7)
        max_percent = strategy_params.get('max_percent', 10.0)
        unique_name = f"{strategy_name}_br{base_roi}_bp{base_percent}_pen{odds_penalty_factor}_max{max_percent}{variation_suffix}"
        param_desc = f"base_roi={base_roi}, base%={base_percent}, penalty={odds_penalty_factor}, max={max_percent}"
    elif strategy_name == 'adaptive_constant_profit':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_target_pct = strategy_params.get('min_target_pct', 0.5)
        max_target_pct = strategy_params.get('max_target_pct', 3.0)
        max_bet_percent = strategy_params.get('max_bet_percent', 15.0)
        unique_name = f"{strategy_name}_roi{min_roi}-{max_roi}_t{min_target_pct}-{max_target_pct}_max{max_bet_percent}{variation_suffix}"
        param_desc = f"roi=[{min_roi}-{max_roi}], target=[{min_target_pct}-{max_target_pct}]%, max={max_bet_percent}"
    else:
        unique_name = f"{strategy_name}{variation_suffix}"
        param_desc = str(strategy_params)
    
    print(f"\n{'='*70}")
    print(f"СТРАТЕГИЯ: {unique_name}")
    print(f"Параметры: {param_desc}, Вариация: {'Да (30%-115%)' if apply_variation else 'Нет'}")
    print(f"{'='*70}")
    
    if strategy_name == 'dynamic_percentage':
        bet_size_pct = strategy_params.get('bet_size_pct', 1.5)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            dynamic_percentage_strategy_with_real_odds(outcomes, odds_array, bet_size_pct, apply_variation)
    elif strategy_name == 'kelly_criterion':
        risk = strategy_params.get('risk', 2.0)
        kelly_fraction = strategy_params.get('kelly_fraction', 1.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            kelly_criterion_strategy_with_real_odds(outcomes, odds_array, risk, kelly_fraction, apply_variation)
    elif strategy_name == 'linear_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            linear_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, apply_variation)
    elif strategy_name == 'sqrt_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            sqrt_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, apply_variation)
    elif strategy_name == 'log_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            log_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, apply_variation)
    elif strategy_name == 'constant_profit':
        target_profit_pct = strategy_params.get('target_profit_pct', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            constant_profit_strategy_with_real_odds(outcomes, odds_array, target_profit_pct, max_percent, apply_variation)
    elif strategy_name == 'combined_roi_odds':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            combined_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_percent, max_percent, 
                                                       min_roi, max_roi, min_odds, max_odds, apply_variation)
    elif strategy_name == 'adaptive':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            adaptive_strategy_with_real_odds(outcomes, odds_array, base_percent, max_percent, 
                                             min_roi, max_roi, min_odds, max_odds, apply_variation)
    elif strategy_name == 'dynamic_kelly':
        risk = strategy_params.get('risk', 2.0)
        min_fraction = strategy_params.get('min_fraction', 0.1)
        max_fraction = strategy_params.get('max_fraction', 0.5)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            dynamic_kelly_strategy_with_real_odds(outcomes, odds_array, risk, min_fraction, max_fraction,
                                                  min_roi, max_roi, apply_variation)
    elif strategy_name == 'exponential_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        factor = strategy_params.get('factor', 0.1)
        max_percent = strategy_params.get('max_percent', 10.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            exponential_roi_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, factor, max_percent, apply_variation)
    elif strategy_name == 'hybrid':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        roi_weight = strategy_params.get('roi_weight', 0.7)
        odds_weight = strategy_params.get('odds_weight', 0.3)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            hybrid_strategy_with_real_odds(outcomes, odds_array, base_percent, max_percent,
                                          min_roi, max_roi, min_odds, max_odds, roi_weight, odds_weight, apply_variation)
    elif strategy_name == 'linear_scaled':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_percent = strategy_params.get('min_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 7.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            linear_scaled_strategy_with_real_odds(outcomes, odds_array, min_roi, max_roi, min_percent, max_percent, apply_variation)
    elif strategy_name == 'linear_roi_odds':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        odds_penalty_factor = strategy_params.get('odds_penalty_factor', 0.7)
        min_odds = strategy_params.get('min_odds', 1.5)
        max_odds = strategy_params.get('max_odds', 5.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            linear_roi_odds_strategy_with_real_odds(outcomes, odds_array, base_roi, base_percent, max_percent, 
                                                     odds_penalty_factor, min_odds, max_odds, apply_variation)
    elif strategy_name == 'adaptive_constant_profit':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_target_pct = strategy_params.get('min_target_pct', 0.5)
        max_target_pct = strategy_params.get('max_target_pct', 3.0)
        max_bet_percent = strategy_params.get('max_bet_percent', 15.0)
        bankroll_history, bet_history, min_bet_pct, max_bet_pct, avg_bet_pct = \
            adaptive_constant_profit_strategy_with_real_odds(outcomes, odds_array, min_roi, max_roi,
                                                             min_target_pct, max_target_pct, max_bet_percent, apply_variation)
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy_name}")
    
    metrics = calculate_metrics_with_odds(bankroll_history, bet_history, odds_array)
    
    print(f"\n📊 SIZING:")
    print(f"  Средняя ставка: {avg_bet_pct:.2f}% от текущего банка")
    print(f"  Min: {min_bet_pct:.2f}%, Max: {max_bet_pct:.2f}%")
    
    print(f"\n💰 РЕЗУЛЬТАТЫ:")
    print(f"  ROI с оборота: {metrics['avg_roi_from_turnover']:>7.2f}%")
    print(f"  Средняя прибыль: {metrics['avg_profit_pct']:>8.2f}%")
    
    print(f"\n⚠️  РИСКИ:")
    print(f"  Слито: {metrics['bankrupt_pct']:.2f}%")
    print(f"  DD>20%: {metrics['drawdown_20_pct']:.2f}%")
    print(f"  DD>50%: {metrics['drawdown_50_pct']:.2f}%")
    print(f"  DD>80%: {metrics['drawdown_80_pct']:.2f}%")
    
    # Описание стратегии
    if strategy_name == 'kelly_criterion':
        risk = strategy_params.get('risk', 2.0)
        kelly_fraction = strategy_params.get('kelly_fraction', 1.0)
        description = (f"Kelly Criterion: risk={risk}, fraction={kelly_fraction}. "
                      f"Формула: f = (b×p - q) / b / risk × fraction, где p из ROI={TARGET_ROI*100:.1f}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией размера ставки 30%-115% от расчетного.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'dynamic_percentage':
        bet_size_pct = strategy_params.get('bet_size_pct', 1.5)
        description = (f"Dynamic Percentage: {bet_size_pct}% от текущего банка. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией размера ставки 30%-115% от расчетного.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'linear_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Linear ROI: bet% = {base_percent} × (ROI/{base_roi}), max {max_percent}%. "
                      f"ROI={TARGET_ROI*100:.1f}%. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'sqrt_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Square Root ROI: bet% = {base_percent} × √(ROI/{base_roi}), max {max_percent}%. "
                      f"ROI={TARGET_ROI*100:.1f}%. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'log_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Logarithmic ROI: bet% = {base_percent} × log(ROI/{base_roi} + 1), max {max_percent}%. "
                      f"ROI={TARGET_ROI*100:.1f}%. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'constant_profit':
        target_profit_pct = strategy_params.get('target_profit_pct', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Constant Profit: ставка = {target_profit_pct}% / (odds-1), max {max_percent}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'combined_roi_odds':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        description = (f"Combined ROI-Odds: bet% = {base_percent} × √(norm_ROI) × (1-0.5×norm_odds), max {max_percent}%. "
                      f"ROI range [{min_roi}-{max_roi}]. Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'adaptive':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Adaptive: Combined с адаптивными модификаторами при просадках. "
                      f"Base {base_percent}%, max {max_percent}%. DD>20%: ×0.75, DD>40%: ×0.5. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'dynamic_kelly':
        risk = strategy_params.get('risk', 2.0)
        min_fraction = strategy_params.get('min_fraction', 0.1)
        max_fraction = strategy_params.get('max_fraction', 0.5)
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 15.0)
        description = (f"Dynamic Kelly: Kelly с динамической фракцией [{min_fraction}-{max_fraction}] по ROI[{min_roi}-{max_roi}]. "
                      f"Risk={risk}. При высоком ROI агрессивнее. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'exponential_roi':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        factor = strategy_params.get('factor', 0.1)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Exponential ROI: bet% = {base_percent} × exp({factor} × (ROI-{base_roi})), max {max_percent}%. "
                      f"Очень агрессивная при высоких ROI. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'hybrid':
        base_percent = strategy_params.get('base_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 10.0)
        roi_weight = strategy_params.get('roi_weight', 0.7)
        odds_weight = strategy_params.get('odds_weight', 0.3)
        description = (f"Hybrid: взвешенная комбинация norm_ROI и norm_odds, веса {roi_weight}/{odds_weight}. "
                      f"Base {base_percent}%, max {max_percent}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'linear_scaled':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_percent = strategy_params.get('min_percent', 1.0)
        max_percent = strategy_params.get('max_percent', 7.0)
        description = (f"Linear Scaled: прямая интерполяция ROI[{min_roi}-{max_roi}] → bet[{min_percent}-{max_percent}]%. "
                      f"Простейший линейный mapping. ROI={TARGET_ROI*100:.1f}%. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'linear_roi_odds':
        base_roi = strategy_params.get('base_roi', 5.0)
        base_percent = strategy_params.get('base_percent', 1.0)
        odds_penalty_factor = strategy_params.get('odds_penalty_factor', 0.7)
        max_percent = strategy_params.get('max_percent', 10.0)
        description = (f"Linear ROI-Odds: bet = {base_percent} × (ROI/{base_roi}) × (1 - {odds_penalty_factor}×norm_odds), max {max_percent}%. "
                      f"Линейная по ROI с коррекцией на odds. "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    elif strategy_name == 'adaptive_constant_profit':
        min_roi = strategy_params.get('min_roi', 3.0)
        max_roi = strategy_params.get('max_roi', 20.0)
        min_target_pct = strategy_params.get('min_target_pct', 0.5)
        max_target_pct = strategy_params.get('max_target_pct', 3.0)
        max_bet_percent = strategy_params.get('max_bet_percent', 15.0)
        description = (f"Adaptive Constant Profit: целевая прибыль [{min_target_pct}-{max_target_pct}]% по ROI[{min_roi}-{max_roi}], max_bet {max_bet_percent}%. "
                      f"bet = target / (odds-1). "
                      f"Реальные коэффициенты (avg {odds_array.mean():.2f}). "
                      f"{'С вариацией 30%-115%.' if apply_variation else 'Без вариации.'}")
    else:
        description = f"{strategy_name} with params {strategy_params}"
    
    result = {
        'strategy_name': unique_name,
        'base_strategy': strategy_name,
        'strategy_params': strategy_params,
        'with_variation': apply_variation,
        'description': description,
        'avg_bet_pct': avg_bet_pct,
        'min_bet_pct': min_bet_pct,
        'max_bet_pct': max_bet_pct,
        **metrics
    }
    
    return result

def save_results_to_csv(result, filename='results.csv'):
    """Сохраняет результат в CSV (append режим)."""
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'strategy', 'with_variation', 'avg_bet_%', 'min_bet_%', 'max_bet_%',
                'roi_%', 'avg_profit_%', 'min_profit_%', 'max_profit_%',
                'bankrupt_%', 'dd>20_%', 'dd>50_%', 'dd>80_%',
                'avg_maxdd_%', 'worst_dd_%',
                'timestamp', 'description'
            ])
        
        # Правильная обработка with_variation (может быть bool или string)
        with_var = result['with_variation']
        if isinstance(with_var, str):
            with_var_str = with_var  # Уже строка 'Yes' или 'No'
        else:
            with_var_str = 'Yes' if with_var else 'No'  # Преобразуем bool
        
        writer.writerow([
            result['strategy_name'],
            with_var_str,
            f"{result['avg_bet_pct']:.2f}",
            f"{result['min_bet_pct']:.2f}",
            f"{result['max_bet_pct']:.2f}",
            f"{result['avg_roi_from_turnover']:.2f}",
            f"{result['avg_profit_pct']:.2f}",
            f"{result['min_profit_pct']:.2f}",
            f"{result['max_profit_pct']:.2f}",
            f"{result['bankrupt_pct']:.2f}",
            f"{result['drawdown_20_pct']:.2f}",
            f"{result['drawdown_50_pct']:.2f}",
            f"{result['drawdown_80_pct']:.2f}",
            f"{result['avg_max_drawdown_pct']:.2f}",
            f"{result['worst_drawdown_pct']:.2f}",
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            result['description']
        ])
    
    print(f"✅ Результаты добавлены в {filename}")

if __name__ == '__main__':
    print("Загрузка симуляций с реальными коэффициентами...")
    outcomes, odds_array = load_real_odds_outcomes()
    
    print(f"\nЗагружено {outcomes.shape[0]} симуляций × {outcomes.shape[1]} ставок")
    print(f"Диапазон коэффициентов: {odds_array.min():.2f} - {odds_array.max():.2f}")
    
    # Тестируем dynamic_percentage 1.5% с и без вариации
    strategies_to_test = [
        ('dynamic_percentage', 1.5, False),
        ('dynamic_percentage', 1.5, True),
    ]
    
    for strategy_name, bet_size, apply_variation in strategies_to_test:
        result = run_strategy_with_real_odds(strategy_name, outcomes, odds_array, bet_size, apply_variation)
        save_results_to_csv(result)
    
    print("\n✅ Готово! Результаты в results_real_odds.csv")
