"""Monte Carlo dispersion simulator for the Kelly-style staking strategy.

We act as the house: every bet has a fixed, trusted ROI (edge) of 2.5%.
True win probability of a bet at coefficient `odds` is therefore (1+ROI)/odds.
Goal: find a `risk` value such that across ALL runs the bank never drops
below 34% of the start (-66%), and study how often it dips to -50% etc.
"""
import numpy as np

# ---- fixed inputs (agreed with user) ----
N_RUNS = 10_000
N_BETS = 10_000
BANK0 = 10_500.0
ROI = 0.025                 # trusted real ROI -> drives true win prob
EDGE = min(ROI, 0.08)       # edge used in Kelly sizing (capped at 8%)
MAX_BET_PCT = 0.10          # f capped at 10% of bank
MIN_BET = 10.0              # stakes below this are not placed
RECALC_EVERY = 30           # bank snapshot for sizing refreshes every 30 bets
SEED = 12345                # common random numbers across the risk sweep

# odds model: clip(lognormal, 1.05, 8.0) -> mean ~2.7, bulk in 1.4-4.0
ODDS_MU, ODDS_SIGMA, ODDS_LO, ODDS_HI = 0.92, 0.40, 1.05, 8.0


def simulate(risk, store_every=RECALC_EVERY):
    """Vectorized over runs; one Python step per bet. Common RNG via fixed seed."""
    rng = np.random.default_rng(SEED)
    bank = np.full(N_RUNS, BANK0)
    snapshot = bank.copy()
    min_bank = bank.copy()
    streak = np.zeros(N_RUNS, dtype=np.int32)
    max_streak = np.zeros(N_RUNS, dtype=np.int32)
    traj = [bank.copy()]

    for t in range(N_BETS):
        if t % RECALC_EVERY == 0:
            snapshot = bank.copy()
        odds = np.clip(rng.lognormal(ODDS_MU, ODDS_SIGMA, N_RUNS), ODDS_LO, ODDS_HI)
        f = np.log10(1.0 - (1.0 + EDGE) / odds) / (-risk)
        f = np.minimum(f, MAX_BET_PCT)
        stake = np.round(f * snapshot / 5.0) * 5.0
        stake[stake < MIN_BET] = 0.0

        win = rng.random(N_RUNS) < (1.0 + ROI) / odds
        bank += np.where(win, stake * (odds - 1.0), -stake)

        lost = (~win) & (stake > 0)
        streak = np.where(lost, streak + 1, 0)
        np.maximum(max_streak, streak, out=max_streak)
        np.minimum(min_bank, bank, out=min_bank)
        if (t + 1) % store_every == 0:
            traj.append(bank.copy())

    return {
        "final": bank,
        "min_frac": min_bank / BANK0,
        "max_streak": max_streak,
        "traj": np.array(traj),
    }


def metrics(risk, r):
    mf, fin = r["min_frac"], r["final"] / BANK0
    return {
        "risk": risk,
        "global_min_%": 100 * mf.min(),
        "p1_min_%": 100 * np.percentile(mf, 1),
        "pct_below_66": 100 * np.mean(mf <= 0.34),
        "pct_below_50": 100 * np.mean(mf <= 0.50),
        "pct_below_40": 100 * np.mean(mf <= 0.60),  # dip to -40%
        "median_profit_%": 100 * (np.median(fin) - 1),
        "mean_profit_%": 100 * (fin.mean() - 1),
        "p5_final_%": 100 * np.percentile(fin, 5),
        "max_lose_streak": int(r["max_streak"].max()),
    }


def stake_table(risks, odds=(1.05, 1.4, 2.0, 2.7, 4.0, 8.0)):
    """Stake as % of current bank (f after the 10% cap) for odds x risk."""
    print("stake % of current bank (f):")
    print("risk \\ odds | " + " | ".join(f"{o:>7}" for o in odds))
    for risk in risks:
        f = np.minimum(np.log10(1 - (1 + EDGE) / np.array(odds)) / (-risk), MAX_BET_PCT)
        print(f"{risk:>10} | " + " | ".join(f"{100*v:>6.2f}%" for v in f))


def main():
    risks = [15, 30, 50, 55, 60, 65, 70, 75]
    rows, results = [], {}
    for risk in risks:
        results[risk] = simulate(risk)
        rows.append(metrics(risk, results[risk]))

    cols = ["risk", "global_min_%", "p1_min_%", "pct_below_66", "pct_below_50",
            "pct_below_40", "median_profit_%", "mean_profit_%", "p5_final_%",
            "max_lose_streak"]
    hdr = " | ".join(f"{c:>15}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(" | ".join(f"{row[c]:>15.2f}" if isinstance(row[c], float)
                          else f"{row[c]:>15}" for c in cols))

    safe = [row["risk"] for row in rows if row["global_min_%"] >= 34.0]
    chosen = 70 if 70 in risks else (min(safe) if safe else max(risks))
    print(f"\nSmallest risk keeping bank >= 34% on this seed: "
          f"{min(safe) if safe else 'none in sweep'} | "
          f"seed-robust recommendation plotted: {chosen}")
    _plots(risks, rows, results, chosen)


def _plots(risks, rows, results, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    gmin = [row["global_min_%"] for row in rows]
    p1 = [row["p1_min_%"] for row in rows]
    ax[0].plot(risks, gmin, "o-", label="global min (worst run)")
    ax[0].plot(risks, p1, "s--", label="1st pct of min")
    ax[0].axhline(34, color="r", ls=":", label="34% floor")
    ax[0].set_xlabel("risk"); ax[0].set_ylabel("min bank, % of start")
    ax[0].set_title("Worst drawdown vs risk"); ax[0].legend()

    tr = results[chosen]["traj"]            # (snap, runs), % of start
    pct = 100 * np.percentile(tr / BANK0, [1, 5, 50, 95, 99], axis=1)
    x = np.arange(tr.shape[0]) * RECALC_EVERY
    ax[1].fill_between(x, pct[0], pct[4], alpha=.15, label="1-99%")
    ax[1].fill_between(x, pct[1], pct[3], alpha=.30, label="5-95%")
    ax[1].plot(x, pct[2], "k", label="median")
    ax[1].axhline(34, color="r", ls=":")
    ax[1].set_xlabel("bet #"); ax[1].set_ylabel("bank, % of start")
    ax[1].set_title(f"Bank trajectory bands (risk={chosen})"); ax[1].legend()

    ax[2].hist(100 * results[chosen]["final"] / BANK0, bins=80)
    ax[2].set_xlabel("final bank, % of start"); ax[2].set_ylabel("runs")
    ax[2].set_title(f"Final bank distribution (risk={chosen})")

    fig.tight_layout()
    fig.savefig("dispersion_report.png", dpi=110)
    print("Saved plots -> dispersion_report.png")


if __name__ == "__main__":
    main()
