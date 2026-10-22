# Architecture

The previous version of this file described a physics engine (fixed timestep, collision detection, Web Workers) and had nothing to do with this project.

## Modules

- `distribution_generator.py` — odds from a scaled Beta distribution on [1.5, 3.5] with the requested mean (2.8), ROI from a 3-range mixture (85/10/5%).
- `bet_strategies.py` — stake-sizing functions `calculate_*_bet(odds, roi, bank, ...)` returning an amount.
- `bet_simulator.py` — one series of bets: draws odds and ROI, win probability `(1 + ROI) / odds`, stake capped by the live bank, drawdown tracking, optional early stop below 20% of peak. Strategies that take `initial_bank` / `max_bank` receive the live values.
- `strategy_optimizer.py` — random search of strategy parameters per risk profile; candidates share one seed (common random numbers), top-5 are re-checked on an independent seed, ranking by median growth, strict profile limits.
- `visualization.py`, `gui.py`, `main.py` — matplotlib charts and the PyQt5 GUI.
- `sim.html`, `simulation.html`, `supersim.html` — standalone browser simulators of the adaptive constant-profit strategy.

The corrected, tested engine used for current research lives in [`../brm`](../brm).
