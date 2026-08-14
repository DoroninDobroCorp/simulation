"""Полный прогон всех стратегий с жёстким кэпом 10% от текущего банка."""
import sys
import sim_engine

ROI = 0.025
CAP = 0.10
OUT = sim_engine.CACHE_DIR + '_capped'

if __name__ == '__main__':
    smoke = len(sys.argv) > 1 and sys.argv[1] == 'smoke'
    if smoke:
        sim_engine.run_all(num_sims=500, num_bets=2000, roi=ROI,
                           cache_dir=OUT + '_smoke', cap=CAP)
    else:
        s = sim_engine.run_all(num_sims=10000, num_bets=10000, roi=ROI,
                               cache_dir=OUT, cap=CAP)
        print(f"\nФактический ROI оборота: {s['actual_roi_turnover_pct']:.3f}%")
        print(f"Кэш: {OUT}/")
