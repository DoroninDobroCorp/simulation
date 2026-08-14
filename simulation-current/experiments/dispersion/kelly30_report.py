"""Kelly risk=30, ROI 2.5% — 1000 прогонов x 10000 ставок. Отчёт в HTML."""
import io, base64, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

N, NB, BANK0, ROI, EDGE, RISK = 1000, 10000, 10500.0, 0.025, 0.025, 30
MAXP, MINBET, STEP = 0.10, 10.0, 30
MU, SIG, LO, HI = 0.92, 0.40, 1.05, 8.0

rng = np.random.default_rng(12345)
bank = np.full(N, BANK0); snap = bank.copy(); peak = bank.copy()
minf = bank.copy(); maxdd = np.zeros(N); streak = np.zeros(N, int); maxstreak = np.zeros(N, int)
traj = [bank.copy()]
for t in range(NB):
    if t % STEP == 0: snap = bank.copy()
    odds = np.clip(rng.lognormal(MU, SIG, N), LO, HI)
    f = np.minimum(np.log10(1 - (1 + EDGE) / odds) / (-RISK), MAXP)
    stake = np.round(f * snap / 5) * 5; stake[stake < MINBET] = 0
    win = rng.random(N) < (1 + ROI) / odds
    bank += np.where(win, stake * (odds - 1), -stake)
    np.maximum(peak, bank, out=peak); np.maximum(maxdd, 1 - bank / peak, out=maxdd)
    np.minimum(minf, bank, out=minf)
    lost = (~win) & (stake > 0); streak = np.where(lost, streak + 1, 0)
    np.maximum(maxstreak, streak, out=maxstreak)
    if (t + 1) % STEP == 0: traj.append(bank.copy())
traj = np.array(traj); fin = bank / BANK0; mn = minf / BANK0

def png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=100, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()

x = np.arange(traj.shape[0]) * STEP
f1, a = plt.subplots(figsize=(9, 5))
a.plot(x, 100 * traj[:, :40] / BANK0, lw=.6, alpha=.6)
a.axhline(100, color="k", ls=":"); a.axhline(34, color="r", ls="--", label="34% (−66%)")
a.set_yscale("log"); a.set_title("40 примеров кривых банка"); a.set_xlabel("ставка"); a.set_ylabel("% старта (log)"); a.legend()
img1 = png(f1)

pc = 100 * np.percentile(traj / BANK0, [1, 5, 50, 95, 99], axis=1)
f2, a = plt.subplots(figsize=(9, 5))
a.fill_between(x, pc[0], pc[4], alpha=.15, label="1–99%"); a.fill_between(x, pc[1], pc[3], alpha=.3, label="5–95%")
a.plot(x, pc[2], "k", label="медиана"); a.axhline(34, color="r", ls="--")
a.set_yscale("log"); a.set_title("Коридор перцентилей банка"); a.set_xlabel("ставка"); a.set_ylabel("% старта (log)"); a.legend()
img2 = png(f2)

f3, a = plt.subplots(figsize=(9, 5))
a.hist(np.clip(fin, 0, np.percentile(fin, 99)), bins=60)
a.axvline(1, color="k", ls=":"); a.set_title("Финальный банк (× от старта, обрезано на 99%)"); a.set_xlabel("× старта"); a.set_ylabel("прогонов")
img3 = png(f3)

f4, a = plt.subplots(figsize=(9, 5))
a.hist(100 * maxdd, bins=60, color="#c0504d"); a.set_title("Макс. просадка от пика"); a.set_xlabel("%"); a.set_ylabel("прогонов")
img4 = png(f4)

def pct(a): return f"{a:.2f}%"
rows = [
    ("Медианная прибыль", pct(100 * (np.median(fin) - 1))),
    ("Средняя прибыль", pct(100 * (fin.mean() - 1))),
    ("Финал: 5% / 25% / 50% / 75% / 95% (× старта)",
     " / ".join(f"{v:.2f}" for v in np.percentile(fin, [5, 25, 50, 75, 95]))),
    ("Глобальный минимум банка (худший прогон)", pct(100 * mn.min())),
    ("P(банк падал ниже 34% старта)", pct(100 * np.mean(mn <= 0.34))),
    ("P(банк падал ниже 50% старта)", pct(100 * np.mean(mn <= 0.50))),
    ("Медианная макс. просадка от пика", pct(100 * np.median(maxdd))),
    ("95% макс. просадка от пика", pct(100 * np.percentile(maxdd, 95))),
    ("Макс. серия проигрышей", str(int(maxstreak.max()))),
]
odds_demo = [1.4, 2.0, 2.7, 4.0, 8.0]
fdemo = np.minimum(np.log10(1 - (1 + EDGE) / np.array(odds_demo)) / (-RISK), MAXP)
stake_rows = "".join(f"<tr><td>{o}</td><td>{100*fv:.2f}%</td><td>{round(fv*BANK0/5)*5:.0f} €</td></tr>"
                     for o, fv in zip(odds_demo, fdemo))
stat_rows = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows)

html = f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Kelly risk=30 — отчёт</title>
<style>body{{font-family:system-ui,Arial;margin:24px;max-width:1000px}}
h1{{font-size:22px}}table{{border-collapse:collapse;margin:8px 0 24px}}
td,th{{border:1px solid #ccc;padding:6px 12px;text-align:left}}img{{max-width:100%;margin:8px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}small{{color:#666}}</style>
<h1>Симулятор дисперсии — Kelly risk=30</h1>
<p><b>Параметры:</b> ROI 2.5% (фикс), кэфы 1.05–8.0 (средн. 2.7), risk=30,
{N} прогонов × {NB} ставок, банк {BANK0:.0f} €, пересчёт каждые {STEP} ставок,
ставка кратна 5 €, мин. 10 €, потолок 10% банка.</p>
<h2>Ключевые метрики</h2><table>{stat_rows}</table>
<h2>Размер ставки по кэфам (% банка)</h2>
<table><tr><th>кэф</th><th>% банка</th><th>€ на банке {BANK0:.0f}</th></tr>{stake_rows}</table>
<small>Просадка «от старта» считается относительно стартового банка; «от пика» — относительно текущего максимума.</small>
<div class=grid>
<div><img src="data:image/png;base64,{img1}"></div>
<div><img src="data:image/png;base64,{img2}"></div>
<div><img src="data:image/png;base64,{img3}"></div>
<div><img src="data:image/png;base64,{img4}"></div>
</div></html>"""
open("kelly30_report.html", "w").write(html)
print("median x", round(np.median(fin), 2), "| P<34%", round(100*np.mean(mn<=.34), 2),
      "| global min %", round(100*mn.min(), 1), "| max dd median %", round(100*np.median(maxdd), 1))
print("HTML written: kelly30_report.html")
