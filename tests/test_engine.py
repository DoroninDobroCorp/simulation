"""
Проверки движка против аналитических формул.

Одна ставка на кэфе o с истинным EV e (p = (1+e)/o), доля f от живого банка:
  E[банк_N] = B0 * (1 + f*e)^N                       (независимость шагов)
  g(f) = p*ln(1 + f*(o-1)) + (1-p)*ln(1 - f)         (рост логарифма за ставку)
  argmax g = e / (o - 1)                              (Келли)
Флэт-ставка s: E[прибыль] = N * s * e.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brm  # noqa: E402
from brm import strategies as S  # noqa: E402
from brm.optimize import calibrate  # noqa: E402

ODDS, EDGE = 2.0, 0.05
P = (1 + EDGE) / ODDS


def single_odds_market(odds=ODDS, edge=EDGE):
    return brm.make_market(brm.BetPool(np.array([odds]), np.array([edge])), roi=edge, ev_model='flat')


def g(f, odds=ODDS, p=P):
    return p * math.log(1 + f * (odds - 1)) + (1 - p) * math.log(1 - f)


def test_fixed_fraction_expected_bank():
    f, N, n = 0.02, 300, 40000
    res = brm.run(S.FixedFraction(f * 100), single_odds_market(), brm.SimConfig(n_sims=n, n_bets=N, cap=1.0))
    expected = 1000 * (1 + f * EDGE) ** N
    se = res.final.std() / math.sqrt(n)
    assert abs(res.final.mean() - expected) < 4 * se


def test_fixed_fraction_log_growth():
    f, N, n = 0.05, 400, 20000
    res = brm.run(S.FixedFraction(f * 100), single_odds_market(), brm.SimConfig(n_sims=n, n_bets=N, cap=1.0))
    lg = np.log(res.final / 1000) / N
    assert abs(lg.mean() - g(f)) < 4 * lg.std() / math.sqrt(n)


def test_flat_stake_expected_profit():
    N, n = 500, 20000
    res = brm.run(S.FlatStake(1.0), single_odds_market(), brm.SimConfig(n_sims=n, n_bets=N, cap=1.0))
    profit = res.final - 1000
    assert abs(profit.mean() - N * 10 * EDGE) < 4 * profit.std() / math.sqrt(n)


def test_kelly_fraction_formula_and_optimality():
    f_star = EDGE / (ODDS - 1)
    c = type('C', (), {'odds': np.array([ODDS]), 'edge': np.array([EDGE])})
    assert S.Kelly(1.0).fraction(c)[0] == pytest.approx(f_star)
    assert g(f_star) > g(0.5 * f_star) and g(f_star) > g(2 * f_star)
    assert abs(g(2 * f_star)) < 1e-4          # двойной Келли ~ нулевой рост


def test_kelly_beats_overbetting_in_median():
    cfg = brm.SimConfig(n_sims=4000, n_bets=2000, cap=1.0, seed=3)
    med = {k: np.median(brm.run(S.Kelly(k), single_odds_market(), cfg).final) for k in (0.5, 1.0, 2.5)}
    assert med[1.0] > med[2.5]
    assert med[1.0] > med[0.5]


@pytest.mark.parametrize('variation', [None, ('uniform', 0.3, 1.6), ('choice', [0.35, 0.75, 1.15])])
@pytest.mark.parametrize('recalc', [1, 30, (30, 70)])
def test_bank_never_negative_and_dd_bounded(variation, recalc):
    pool = brm.load_real_bets()
    market = brm.make_market(pool, roi=0.025, ev_model='scaled', noise=0.05)
    cfg = brm.SimConfig(n_sims=500, n_bets=800, cap=0.5, recalc_every=recalc, variation=variation,
                        round_to=5, min_stake=10, record_full=True)
    res = brm.run(S.FixedFraction(25), market, cfg)
    assert (res.history >= 0).all()
    assert (res.stakes <= res.history[:, :-1] + 1e-9).all()
    assert ((res.max_dd >= 0) & (res.max_dd <= 1)).all()


def test_stakes_follow_live_bank():
    """Ставка = f * банк ПЕРЕД ставкой (то, что legacy-код делал неправильно)."""
    pool = brm.load_real_bets()
    cfg = brm.SimConfig(n_sims=300, n_bets=500, record_full=True)
    res = brm.run(S.FixedFraction(3), brm.make_market(pool, roi=0.025), cfg)
    np.testing.assert_allclose(res.stakes, 0.03 * res.history[:, :-1])


def test_snapshot_recalc_uses_snapshot():
    cfg = brm.SimConfig(n_sims=200, n_bets=120, recalc_every=30, record_full=True)
    res = brm.run(S.FixedFraction(2), single_odds_market(), cfg)
    for t in range(cfg.n_bets):
        snap = res.history[:, (t // 30) * 30]
        np.testing.assert_allclose(res.stakes[:, t], np.minimum(0.02 * snap, res.history[:, t]))


def test_roi_on_turnover_matches_true_edge():
    pool = brm.load_real_bets()
    res = brm.run(S.FlatStake(1.0), brm.make_market(pool, roi=0.025), brm.SimConfig(n_sims=20000, n_bets=500))
    m = brm.summarize(res)
    assert abs(m['roi_turnover_pct'] - 2.5) < 0.25


def test_scaled_market_mean_roi():
    pool = brm.load_real_bets()
    mk = brm.make_market(pool, roi=0.025, ev_model='scaled')
    assert mk.mean_true_roi == pytest.approx(0.025)
    assert (mk.edge_est >= mk.edge_true - 1e-12)[mk.edge_est > 0].all()   # оценка оптимистична


def test_synthetic_pool_matches_spec():
    pool = brm.synthetic_pool(200000)
    assert abs(pool.odds.mean() - 2.8) < 0.01
    assert pool.odds.min() >= 1.5 and pool.odds.max() <= 3.5
    expected_ev = 0.85 * 0.045 + 0.10 * 0.09 + 0.05 * 0.16
    assert abs(pool.ev.mean() - expected_ev) < 0.001


def test_floor_kelly_caps_drawdown():
    pool = brm.load_real_bets()
    market = brm.make_market(pool, roi=0.025, ev_model='scaled')
    res = brm.run(S.FloorKelly(mult=1.0, floor=0.6), market, brm.SimConfig(n_sims=3000, n_bets=1500, cap=1.0))
    assert res.max_dd.max() <= 0.4 + 1e-9


def test_common_random_numbers_and_determinism():
    pool = brm.load_real_bets()
    market = brm.make_market(pool, roi=0.025)
    cfg = brm.SimConfig(n_sims=300, n_bets=300, seed=11, cap=1.0)
    a = brm.run(S.FixedFraction(2), market, cfg)
    b = brm.run(S.FixedFraction(2), market, cfg)
    np.testing.assert_array_equal(a.final, b.final)
    # другая стратегия видит те же ставки и исходы: удвоенный флэт = ровно удвоенная прибыль
    c = brm.run(S.FlatStake(0.5), market, cfg)
    d = brm.run(S.FlatStake(1.0), market, cfg)
    np.testing.assert_allclose(d.final - 1000, 2 * (c.final - 1000), atol=1e-6)


def test_calibration_hits_profile_boundary():
    pool = brm.load_real_bets()
    market = brm.make_market(pool, roi=0.025)
    cfg = brm.SimConfig(n_sims=600, n_bets=600, seed=5)
    build = lambda s: S.Scaled(S.FixedFraction(1.0), s)  # noqa: E731
    scale, m = calibrate(build, market, cfg, 'p_dd50_pct', 5.0, iters=10, refine=())
    assert m['p_dd50_pct'] <= 5.0 and scale == m['scale_max']
    from brm.optimize import evaluate
    assert evaluate(build(scale * 1.15), market, cfg)['p_dd50_pct'] > 5.0


def test_calibration_with_several_constraints():
    pool = brm.load_real_bets()
    market = brm.make_market(pool, roi=0.05)
    cfg = brm.SimConfig(n_sims=600, n_bets=600, seed=7)
    rules = [('p_dd20_pct', 5.0), ('p_profit_below_m10_pct', 5.0)]
    scale, m = calibrate(lambda s: S.Kelly(s), market, cfg, rules, None, scale_hi=4.0, iters=8)
    assert all(m[k] <= v for k, v in rules)
    single, _ = calibrate(lambda s: S.Kelly(s), market, cfg, 'p_dd20_pct', 5.0, scale_hi=4.0, iters=8, refine=())
    assert scale <= single + 1e-9


def test_beta_odds_pool_matches_request():
    pool = brm.data.beta_odds_pool(1.5, 4.0, 2.75, roi=0.05, concentration=3.0, size=200000)
    assert abs(pool.odds.mean() - 2.75) < 0.01
    assert pool.odds.min() >= 1.5 and pool.odds.max() <= 4.0
    assert (pool.ev == 0.05).all()


def test_random_recalc_period_is_redrawn_each_time():
    """Снимок банка раз в 30–70 ставок: период тянется заново после каждого снимка."""
    cfg = brm.SimConfig(n_sims=50, n_bets=2000, recalc_every=(30, 70), cap=1.0, record_full=True, seed=4)
    res = brm.run(S.FixedFraction(1), single_odds_market(odds=2.37), cfg)
    for s in range(5):
        base = res.stakes[s] / 0.01
        # база ставки (снимок банка) меняется только в момент пересчёта
        refresh = np.flatnonzero(~np.isclose(base[1:], base[:-1])) + 1
        gaps = np.diff(np.concatenate([[0], refresh]))
        assert gaps.min() >= 30 and gaps.max() <= 70
        assert len(set(gaps.tolist())) > 3


def test_no_stake_when_bank_below_min_stake():
    pool = brm.BetPool(np.array([2.0]), np.array([0.0]))
    market = brm.make_market(pool, roi=0.0)
    cfg = brm.SimConfig(n_sims=400, n_bets=400, bank0=100.0, cap=1.0, min_stake=10.0, record_full=True,
                        recalc_every=50, seed=3)
    res = brm.run(S.FixedFraction(30), market, cfg)
    placed = res.stakes > 0
    assert (res.stakes[placed] >= 10.0 - 1e-9).all()
    assert (res.stakes <= res.history[:, :-1] + 1e-9).all()


def test_floor_kelly_calibration_tunes_floor():
    from brm.optimize import calibrate_floor_kelly
    pool = brm.load_real_bets()
    market = brm.make_market(pool, roi=0.05)
    cfg = brm.SimConfig(n_sims=400, n_bets=500, seed=9)
    scale, m = calibrate_floor_kelly(market, cfg, 'p_dd30_pct', 10.0, iters=6)
    assert m is not None and m['p_dd30_pct'] <= 10.0
    assert 0.05 <= m['floor'] <= 0.95


def test_floor_kelly_refine_reaches_edge_floor(monkeypatch):
    """Если лучший пол сетки 0.9, уточнение пробует и 0.95 (0.9 + 0.05 в float больше 0.95)."""
    from brm import optimize
    tried = []

    def fake_calibrate(build, market, cfg, metric, limit, **kw):
        fl = build(1.0).floor
        tried.append(round(fl, 3))
        return 1.0, {'median_profit_pct': fl}   # чем выше пол, тем лучше

    monkeypatch.setattr(optimize, 'calibrate', fake_calibrate)
    _, m = optimize.calibrate_floor_kelly(None, None, 'p_dd30_pct', 10.0)
    assert 0.95 in tried and m['floor'] == 0.95


def test_floor_kelly_uses_snapshot_when_bank_is_recalculated():
    """При снимке банка раз в K ставок подушка считается от снимка и пика снимков, а не от живого банка."""
    cfg = brm.SimConfig(n_sims=200, n_bets=300, recalc_every=30, cap=1.0, record_full=True, seed=8)
    res = brm.run(S.FloorKelly(mult=1.0, floor=0.5), single_odds_market(), cfg)
    f_star = EDGE / (ODDS - 1)
    for s in range(5):
        snaps = res.history[s, [(t // 30) * 30 for t in range(cfg.n_bets)]]
        peak_snap = np.maximum.accumulate(snaps)
        expected = np.minimum(f_star * np.maximum(snaps - 0.5 * peak_snap, 0.0), res.history[s, :-1])
        np.testing.assert_allclose(res.stakes[s], expected, rtol=1e-9, atol=1e-9)


def test_live_cap_limits_stake_after_variation():
    cfg = brm.SimConfig(n_sims=300, n_bets=300, cap=0.10, live_cap=0.10, variation=('uniform', 1.0, 1.5),
                        record_full=True, seed=6)
    res = brm.run(S.FixedFraction(10), single_odds_market(), cfg)
    assert (res.stakes <= 0.10 * res.history[:, :-1] + 1e-9).all()
    loose = brm.run(S.FixedFraction(10), single_odds_market(), brm.SimConfig(
        n_sims=300, n_bets=300, cap=0.10, variation=('uniform', 1.0, 1.5), record_full=True, seed=6))
    assert (loose.stakes > 0.10 * loose.history[:, :-1] + 1e-9).any()
