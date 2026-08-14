"""
Дашборд результатов прогона стратегий (localhost).

Читает ТОЛЬКО кэш из dashboard_data/ (summary.json + распределения .npz),
поэтому работает мгновенно, без повторного прогона симуляций.

Запуск:
  python3 dashboard.py            # использует dashboard_data/
  python3 dashboard.py smoke      # использует dashboard_data_smoke/

Открыть: http://127.0.0.1:5000
"""

import os
import sys
import json
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, 'dashboard_data_smoke' if (len(sys.argv) > 1 and sys.argv[1] == 'smoke')
                         else 'dashboard_data')
TOP_DIR = os.path.join(BASE, 'top_data')

app = Flask(__name__)


def load_summary():
    path = os.path.join(CACHE_DIR, 'summary.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def histogram(values, bins=40, clip_lo=None, clip_hi=None):
    v = np.asarray(values, dtype=float)
    if clip_lo is not None or clip_hi is not None:
        lo = clip_lo if clip_lo is not None else np.percentile(v, 0.5)
        hi = clip_hi if clip_hi is not None else np.percentile(v, 99.5)
        v = np.clip(v, lo, hi)
    counts, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {'x': [round(float(c), 2) for c in centers],
            'y': [int(c) for c in counts]}


@app.route('/api/summary')
def api_summary():
    s = load_summary()
    if s is None:
        return jsonify({'error': f'Нет данных. Сначала запусти прогон. Ожидался {CACHE_DIR}/summary.json'}), 404
    return jsonify(s)


@app.route('/api/dist')
def api_dist():
    key = request.args.get('key')
    variation = request.args.get('var', 'novar')
    fname = os.path.join(CACHE_DIR, f"{key}__{variation}.npz")
    if not os.path.exists(fname):
        return jsonify({'error': f'Нет файла {fname}'}), 404
    data = np.load(fname)
    profit = data['profit_pct']
    maxdd = data['max_dd_pct']
    return jsonify({
        'key': key,
        'variation': variation,
        'profit_hist': histogram(profit, bins=50, clip_lo=float(np.percentile(profit, 0.5)),
                                 clip_hi=float(np.percentile(profit, 99.5))),
        'maxdd_hist': histogram(maxdd, bins=50),
        'profit_p1': float(np.percentile(profit, 1)),
        'profit_p5': float(np.percentile(profit, 5)),
        'profit_median': float(np.median(profit)),
        'maxdd_p5': float(np.percentile(maxdd, 5)),
        'maxdd_worst': float(np.min(maxdd)),
    })


INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Стратегии — риск-дашборд (ROI {{ roi }}%)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#0f1115; color:#e6e6e6; }
  header { padding: 16px 24px; background:#171a21; border-bottom:1px solid #262b35; }
  h1 { font-size: 18px; margin:0; }
  .meta { color:#8a93a2; font-size:13px; margin-top:6px; }
  .wrap { padding: 16px 24px; }
  table { border-collapse: collapse; width:100%; font-size:13px; }
  th, td { padding:7px 9px; border-bottom:1px solid #232833; text-align:right; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; position:sticky; left:0; background:#0f1115; }
  th { cursor:pointer; user-select:none; background:#171a21; position:sticky; top:0; }
  th:hover { color:#fff; }
  tr:hover td { background:#161a22; }
  tr.sel td { background:#1d2738 !important; }
  .neg { color:#ff6b6b; } .pos { color:#51cf66; }
  .badge { font-size:11px; padding:2px 6px; border-radius:4px; background:#2a3340; color:#9fb3c8; }
  .charts { display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px; }
  .card { background:#171a21; border:1px solid #232833; border-radius:8px; padding:14px; }
  .card h3 { margin:0 0 10px; font-size:14px; color:#cdd6e0; }
  .hint { color:#8a93a2; font-size:12px; }
  .stat { display:inline-block; margin-right:18px; font-size:13px; }
  .stat b { color:#fff; }
  select { background:#0f1115; color:#e6e6e6; border:1px solid #2a3340; padding:4px 8px; border-radius:5px; }
</style>
</head>
<body>
<header>
  <h1>Риск-дашборд стратегий банкролл-менеджмента</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="wrap">
  <div class="hint">Клик по строке — графики распределения для стратегии. Клик по заголовку — сортировка. Главные риск-метрики: bankrupt%, DD&gt;50%, worst DD, P1/P5 прибыли (худшие случаи).</div>
  <div style="overflow:auto; max-height:55vh; margin-top:10px;">
    <table id="tbl"><thead></thead><tbody></tbody></table>
  </div>

  <div class="charts">
    <div class="card">
      <h3>Распределение итоговой прибыли (%) <span id="sel1" class="badge"></span></h3>
      <div id="profitStats" class="meta"></div>
      <canvas id="profitChart" height="150"></canvas>
    </div>
    <div class="card">
      <h3>Распределение макс. просадки (%) <span id="sel2" class="badge"></span></h3>
      <div id="ddStats" class="meta"></div>
      <canvas id="ddChart" height="150"></canvas>
    </div>
  </div>
</div>

<script>
const COLS = [
  {k:'key', t:'Стратегия', fmt:v=>v},
  {k:'with_variation', t:'Вариация', fmt:v=>v?'да':'нет'},
  {k:'avg_bet_pct', t:'Ср.ставка%', fmt:f1},
  {k:'avg_profit_pct', t:'Ср.приб%', fmt:f1, color:true},
  {k:'median_profit_pct', t:'Медиана%', fmt:f1, color:true},
  {k:'p5_profit_pct', t:'P5 приб%', fmt:f1, color:true},
  {k:'p1_profit_pct', t:'P1 приб%', fmt:f1, color:true},
  {k:'bankrupt_pct', t:'Слив%', fmt:f2, risk:true},
  {k:'dd20_pct', t:'DD>20%', fmt:f1, risk:true},
  {k:'dd50_pct', t:'DD>50%', fmt:f1, risk:true},
  {k:'dd80_pct', t:'DD>80%', fmt:f1, risk:true},
  {k:'avg_maxdd_pct', t:'Ср.maxDD%', fmt:f1, color:true},
  {k:'worst_dd_pct', t:'Худш.DD%', fmt:f1, color:true},
  {k:'avg_roi_turnover_pct', t:'ROI оборота%', fmt:f2},
];
function f1(v){return (v==null)?'':Number(v).toFixed(1);}
function f2(v){return (v==null)?'':Number(v).toFixed(2);}

let DATA=[], sortKey='worst_dd_pct', sortAsc=true, selectedRow=null;
let profitChart=null, ddChart=null;

fetch('/api/summary').then(r=>r.json()).then(s=>{
  if(s.error){document.getElementById('meta').textContent=s.error;return;}
  document.getElementById('meta').innerHTML =
    `ROI=<b>${s.roi_pct.toFixed(2)}%</b> (факт оборота ${s.actual_roi_turnover_pct.toFixed(2)}%) · `+
    `${s.num_sims.toLocaleString()} симуляций × ${s.num_bets.toLocaleString()} ставок · `+
    `ср.коэф ${s.avg_odds.toFixed(2)} · старт.банк ${s.initial_bankroll}`;
  DATA = s.strategies;
  renderHead(); renderBody();
  // авто-выбор первой строки
  if(DATA.length){ selectRow(DATA[sorted()[0]._i]); }
});

function renderHead(){
  const tr = COLS.map(c=>`<th onclick="setSort('${c.k}')">${c.t}</th>`).join('');
  document.querySelector('#tbl thead').innerHTML = `<tr>${tr}</tr>`;
}
function sorted(){
  const arr = DATA.map((d,i)=>({...d,_i:i}));
  arr.sort((a,b)=>{
    let x=a[sortKey], y=b[sortKey];
    if(typeof x==='boolean'){x=x?1:0;y=y?1:0;}
    return sortAsc ? (x-y) : (y-x);
  });
  return arr;
}
function renderBody(){
  const rows = sorted().map(d=>{
    const tds = COLS.map(c=>{
      let v=d[c.k]; let cls='';
      if(c.color && typeof v==='number') cls = v<0?'neg':'pos';
      if(c.risk && typeof v==='number' && v>0) cls='neg';
      return `<td class="${cls}">${c.fmt(v)}</td>`;
    }).join('');
    const selCls = (selectedRow && selectedRow.key===d.key && selectedRow.with_variation===d.with_variation)?'sel':'';
    return `<tr class="${selCls}" onclick='selectByKey("${d.key}", ${d.with_variation})'>${tds}</tr>`;
  }).join('');
  document.querySelector('#tbl tbody').innerHTML = rows;
}
function setSort(k){ if(sortKey===k) sortAsc=!sortAsc; else {sortKey=k; sortAsc=true;} renderBody(); }
function selectByKey(key, varb){
  const row = DATA.find(d=>d.key===key && d.with_variation===varb);
  selectRow(row);
}
function selectRow(row){
  selectedRow=row; renderBody();
  const variation = row.with_variation?'var':'novar';
  const label = `${row.key} · ${row.with_variation?'с вариацией':'без вариации'}`;
  document.getElementById('sel1').textContent=label;
  document.getElementById('sel2').textContent=label;
  fetch(`/api/dist?key=${row.key}&var=${variation}`).then(r=>r.json()).then(drawCharts);
}
function drawCharts(d){
  if(d.error){return;}
  document.getElementById('profitStats').innerHTML =
    `<span class="stat">Медиана: <b>${d.profit_median.toFixed(1)}%</b></span>`+
    `<span class="stat">P5: <b>${d.profit_p5.toFixed(1)}%</b></span>`+
    `<span class="stat">P1 (худшее): <b>${d.profit_p1.toFixed(1)}%</b></span>`;
  document.getElementById('ddStats').innerHTML =
    `<span class="stat">P5 (глубокая): <b>${d.maxdd_p5.toFixed(1)}%</b></span>`+
    `<span class="stat">Худшая просадка: <b>${d.maxdd_worst.toFixed(1)}%</b></span>`;
  profitChart = bar(profitChart,'profitChart',d.profit_hist,'#4c9aff');
  ddChart = bar(ddChart,'ddChart',d.maxdd_hist,'#ff6b6b');
}
function bar(chart, id, hist, color){
  if(chart) chart.destroy();
  return new Chart(document.getElementById(id),{
    type:'bar',
    data:{labels:hist.x, datasets:[{data:hist.y, backgroundColor:color, borderWidth:0}]},
    options:{plugins:{legend:{display:false}},
      scales:{x:{ticks:{color:'#8a93a2',maxTicksLimit:12},grid:{color:'#1e242e'}},
              y:{ticks:{color:'#8a93a2'},grid:{color:'#1e242e'}}}}
  });
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    s = load_summary()
    roi = f"{s['roi_pct']:.2f}" if s else "?"
    return render_template_string(INDEX_HTML, roi=roi)


@app.route('/api/top')
def api_top():
    path = os.path.join(TOP_DIR, 'top_summary.json')
    if not os.path.exists(path):
        return jsonify({'error': 'Нет top_data. Запусти: python3 top_strategies.py'}), 404
    with open(path, encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/top_png/<path:fname>')
def top_png(fname):
    return send_from_directory(TOP_DIR, fname)


TOP_HTML = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>ТОП безопасных стратегий</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body { font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0f1115; color:#e6e6e6; }
  header { padding:16px 24px; background:#171a21; border-bottom:1px solid #262b35; }
  h1 { font-size:18px; margin:0; } a { color:#4c9aff; }
  .meta { color:#8a93a2; font-size:13px; margin-top:6px; }
  .wrap { padding:16px 24px; }
  .crit { background:#15233a; border:1px solid #244; border-radius:8px; padding:12px 14px; margin-bottom:16px; font-size:14px; }
  table { border-collapse:collapse; width:100%; font-size:13px; margin-bottom:8px; }
  th,td { padding:8px 10px; border-bottom:1px solid #232833; text-align:right; }
  th:first-child,td:first-child { text-align:left; }
  th { background:#171a21; }
  .pos{color:#51cf66;} .neg{color:#ff6b6b;}
  .card { background:#171a21; border:1px solid #232833; border-radius:8px; padding:14px; margin:16px 0; }
  .card h3 { margin:0 0 4px; font-size:15px; }
  .ok { color:#51cf66; } .bad { color:#ff6b6b; }
  .checks { font-size:12px; color:#9fb3c8; margin:6px 0 10px; }
  .checks span { margin-right:14px; }
  img { width:100%; border-radius:6px; background:#fff; }
  .grid2 { display:grid; grid-template-columns:1.2fr .8fr; gap:16px; align-items:start; }
  canvas { background:#0f1115; border-radius:6px; }
  .small { font-size:12px; color:#8a93a2; }
</style>
</head>
<body>
<header>
  <h1>ТОП безопасных стратегий — никогда ниже 34% от пика</h1>
  <div class="meta" id="meta"></div>
  <div class="meta"><a href="/">← все 50 стратегий (общая таблица)</a></div>
</header>
<div class="wrap">
  <div class="crit" id="crit"></div>

  <div class="card">
    <h3>Сводка: сколько из 50 прогонов не пробивают порог просадки</h3>
    <div class="small">Жёсткий критерий по ВСЕМ 10000 симуляциям. Чем строже порог, тем меньше стратегий проходят.</div>
    <canvas id="thrChart" height="90"></canvas>
  </div>

  <div class="card">
    <h3>ТОП-3 (безопасны И доходны)</h3>
    <table id="topTbl"><thead></thead><tbody></tbody></table>
    <div class="small">worstDD = самая глубокая просадка среди всех 10000 симуляций. medDD = типичная (медианная) макс-просадка. Все три: 0% банкротств, ставка ≤ 10% банка, никогда ниже 34% от пика.</div>
  </div>

  <div id="cards"></div>
</div>

<script>
fetch('/api/top').then(r=>r.json()).then(d=>{
  if(d.error){document.getElementById('meta').textContent=d.error;return;}
  document.getElementById('meta').innerHTML =
    `ROI=<b>${d.roi_pct.toFixed(2)}%</b> (факт ${d.actual_roi_turnover_pct.toFixed(2)}%) · `+
    `${d.num_sims.toLocaleString()}×${d.num_bets.toLocaleString()} · кэп ставки ${d.cap_pct}% от банка`;
  document.getElementById('crit').innerHTML =
    `<b>Критерий отбора:</b> ни одна из ${d.num_sims.toLocaleString()} симуляций не опускается ниже `+
    `<b>34% от своего пика</b> (макс. просадка ≤ ${-d.floor_dd_pct}%). Плюс правило: ставка никогда не больше ${d.cap_pct}% текущего банка.`;

  // сводка порогов
  const th=d.threshold_summary;
  new Chart(document.getElementById('thrChart'),{type:'bar',
    data:{labels:th.map(t=>`не ниже ${100-t.threshold_pct}% от пика`),
      datasets:[{data:th.map(t=>t.safe_count),backgroundColor:'#4c9aff'}]},
    options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.raw} из 50 стратегий`}}},
      scales:{x:{ticks:{color:'#8a93a2'},grid:{color:'#1e242e'}},
              y:{ticks:{color:'#8a93a2'},grid:{color:'#1e242e'},title:{display:true,text:'стратегий из 50',color:'#8a93a2'}}}}
  });

  // таблица топ
  const cols=[['label','Стратегия'],['median_profit_pct','Медиана приб%'],['p5_profit_pct','P5 приб%'],
    ['p1_profit_pct','P1 приб%'],['worst_dd_pct','Худш.DD%'],['median_maxdd_pct','Тип.DD%'],
    ['avg_bet_pct','Ср.ставка%'],['bankrupt_pct','Слив%']];
  document.querySelector('#topTbl thead').innerHTML='<tr>'+cols.map(c=>`<th>${c[1]}</th>`).join('')+'</tr>';
  document.querySelector('#topTbl tbody').innerHTML=d.strategies.map(s=>'<tr>'+cols.map(c=>{
    let v=s[c[0]]; if(typeof v==='number'){const cls=(c[0].includes('profit')&&v<0)||c[0].includes('dd')||c[0]==='bankrupt_pct'?(v<0||c[0]==='bankrupt_pct'&&v>0?'neg':'pos'):'';return `<td class="${v<0?'neg':'pos'}">${v.toFixed(c[0]==='avg_bet_pct'||c[0]==='bankrupt_pct'?2:1)}</td>`;}
    return `<td>${v}</td>`;}).join('')+'</tr>').join('');

  // карточки с графиками
  document.getElementById('cards').innerHTML=d.strategies.map((s,i)=>{
    const c=s.checks;
    const chk=(k,t)=>`<span class="${c[k]?'ok':'bad'}">${c[k]?'✓':'✗'} ${t}</span>`;
    const rob=(s.robustness||[]).map(r=>`<span class="${r.safe?'ok':'bad'}">seed ${r.seed}: ${r.worst_dd_pct}%${r.safe?' ✓':' ✗'}</span>`).join('');
    return `<div class="card">
      <h3>${i+1}. ${s.label}</h3>
      <div class="checks">
        ${chk('no_negative_bank','банк не отрицателен')}
        ${chk('bet_le_10pct','ставка ≤10%')}
        ${chk('passes_floor_-66','никогда ниже 34% пика')}
        ${chk('balance_invariant_ok','баланс сходится')}
        <span>worstDD=${c.worst_dd.toFixed(1)}% · max ставка=${c.max_bet_pct.toFixed(3)}%</span>
      </div>
      <div class="checks">Устойчивость на разных наборах исходов: ${rob}</div>
      <div class="grid2">
        <img src="/top_png/${s.png}" alt="${s.label}">
        <div><canvas id="dd${i}" height="200"></canvas>
          <div class="small">Распределение макс-просадки по 10000 симуляциям. Левый край = ${s.worst_dd_pct.toFixed(1)}% (худшая).</div>
        </div>
      </div>
    </div>`;
  }).join('');

  d.strategies.forEach((s,i)=>{
    new Chart(document.getElementById('dd'+i),{type:'bar',
      data:{labels:s.maxdd_hist.x,datasets:[{data:s.maxdd_hist.y,backgroundColor:'#ff6b6b'}]},
      options:{plugins:{legend:{display:false}},
        scales:{x:{ticks:{color:'#8a93a2',maxTicksLimit:10},grid:{color:'#1e242e'},title:{display:true,text:'макс просадка от пика, %',color:'#8a93a2'}},
                y:{ticks:{color:'#8a93a2'},grid:{color:'#1e242e'}}}}
    });
  });
});
</script>
</body>
</html>
"""


@app.route('/top')
def top_page():
    return render_template_string(TOP_HTML)


if __name__ == '__main__':
    print(f"Данные из: {CACHE_DIR}")
    print("ТОП-3 безопасных стратегий: http://127.0.0.1:5000/top")
    print("Все 50 стратегий:           http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', port=5000, debug=False)
