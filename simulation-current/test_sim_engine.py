#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest smoke test for sim_engine.run_all integration path.

Verifies:
- run_all returns a summary dict with expected keys
- summary.json is written to the cache directory
- Runs with small deterministic inputs (no network dependencies)
"""

import os
import json
import tempfile

import sim_engine


def test_run_all_returns_summary_and_writes_summary_json():
    """Direct call to run_all with small inputs; asserts summary return and file write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Small deterministic inputs for fast smoke test
        summary = sim_engine.run_all(
            num_sims=10,
            num_bets=50,
            roi=0.025,
            seed=42,
            cache_dir=tmpdir,
            verbose=False,
        )

        # --- API contract: run_all must return a summary dict ---
        assert isinstance(summary, dict), "run_all must return a dict"

        # --- Required top-level keys ---
        required_keys = {
            'roi_pct', 'actual_roi_turnover_pct', 'num_sims', 'num_bets',
            'seed', 'avg_odds', 'initial_bankroll', 'strategies',
        }
        assert required_keys.issubset(summary.keys()), (
            f"Summary missing keys: {required_keys - summary.keys()}"
        )

        # --- strategies list must be non-empty ---
        assert isinstance(summary['strategies'], list), "strategies must be a list"
        assert len(summary['strategies']) > 0, "strategies list is empty"

        # --- each strategy row must have expected fields ---
        expected_row_keys = {
            'key', 'with_variation', 'avg_bet_pct', 'min_bet_pct', 'max_bet_pct',
            'avg_profit_pct', 'median_profit_pct', 'p1_profit_pct', 'p5_profit_pct',
            'p95_profit_pct', 'min_profit_pct', 'max_profit_pct',
            'bankrupt_pct', 'dd20_pct', 'dd50_pct', 'dd80_pct',
            'avg_maxdd_pct', 'worst_dd_pct', 'p95_maxdd_pct', 'avg_roi_turnover_pct',
        }
        first_row = summary['strategies'][0]
        assert expected_row_keys.issubset(first_row.keys()), (
            f"Strategy row missing keys: {expected_row_keys - first_row.keys()}"
        )

        # --- summary.json must be written ---
        summary_path = os.path.join(tmpdir, 'summary.json')
        assert os.path.exists(summary_path), (
            f"summary.json not written to {tmpdir}"
        )

        # --- summary.json content must match returned summary ---
        with open(summary_path, encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == summary, "summary.json content differs from returned summary"

        # --- distribution .npz files must be written (one per strategy/variation) ---
        npz_count = len([f for f in os.listdir(tmpdir) if f.endswith('.npz')])
        assert npz_count == len(summary['strategies']), (
            f"Expected {len(summary['strategies'])} .npz files, got {npz_count}"
        )