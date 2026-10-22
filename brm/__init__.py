"""brm — корректный Monte Carlo симулятор банкролл-менеджмента."""

from .data import BetPool, load_real_bets, synthetic_pool
from .engine import FixedScenario, Market, MarketScenario, SimConfig, SimResult, make_market, run, simulate
from .metrics import summarize
from . import strategies

__all__ = ['BetPool', 'load_real_bets', 'synthetic_pool', 'FixedScenario', 'Market', 'MarketScenario',
           'SimConfig', 'SimResult', 'make_market', 'run', 'simulate', 'summarize', 'strategies']
