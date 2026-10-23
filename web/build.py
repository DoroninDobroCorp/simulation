"""
Собирает web/index.html: подставляет в web/app.html движок (web/engine.js),
реальные ставки (только кэф и EV), предрасчёты reports/results.json и
reports/month_roi5.json и сравнение reports/legacy_before_after.json.

  python web/build.py                 # web/index.html — самостоятельная страница
  python web/build.py --fragment OUT  # фрагмент без <html>/<body> для публикации
"""

import argparse
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_bets():
    path = os.path.join(ROOT, 'simulation-current', 'matching_teams_clean.csv')
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {'odds': [float(r['coefficient']) for r in rows],
            'ev': [round(float(r['ev_profit']) / float(r['stake']), 5) for r in rows]}


def slim_month(res):
    """Из reports/month_roi5.json — только то, что показывает вкладка «Месяц»."""
    keep_f = ('param', 'median_profit_pct', 'p5_profit_pct', 'median_max_dd_pct', 'p_dd50_pct', 'p_loss_pct', 'avg_stake_pct')
    task = lambda r: {k: r[k] for k in ('key', 'param', 'scale', 'eval', 'floor') if k in r}  # noqa: E731
    rec = lambda r: dict(task(r), stakes=r['stakes'], robustness=r['robustness'],  # noqa: E731
                         alternatives=[dict(task(x), stakes=x['stakes'], robustness=x['robustness']) for x in r.get('alternatives', [])])
    return {'settings': res['settings'], 'ceiling': res['ceiling'],
            'frontier': {k: [{f: p[f] for f in keep_f} for p in v] for k, v in res['frontier'].items()},
            'tasks': {t: [task(r) for r in rows] for t, rows in res['tasks'].items()},
            'recommended': {t: rec(r) for t, r in res['recommended'].items()}}


def compact(obj):
    # в JSON символ < встречается только внутри строк, поэтому его можно заменить на \u003c: так никакие
    # данные не закроют <script> и не откроют комментарий
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')


def build():
    with open(os.path.join(ROOT, 'web', 'app.html'), encoding='utf-8') as f:
        page = f.read()
    with open(os.path.join(ROOT, 'web', 'engine.js'), encoding='utf-8') as f:
        engine = f.read()
    with open(os.path.join(ROOT, 'reports', 'results.json'), encoding='utf-8') as f:
        results = json.load(f)
    with open(os.path.join(ROOT, 'reports', 'legacy_before_after.json'), encoding='utf-8') as f:
        before_after = json.load(f)
    with open(os.path.join(ROOT, 'reports', 'month_roi5.json'), encoding='utf-8') as f:
        month = slim_month(json.load(f))
    if '</script' in engine.lower():
        raise ValueError('engine.js не должен содержать </script')
    for key, value in (('__ENGINE__', engine), ('__REAL_BETS__', compact(load_bets())),
                       ('__RESULTS__', compact(results)), ('__BEFORE_AFTER__', compact(before_after)),
                       ('__MONTH__', compact(month))):
        if page.count(key) != 1:
            raise ValueError(f'метка {key} должна встречаться ровно один раз')
        page = page.replace(key, value)
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fragment', help='куда записать фрагмент для публикации')
    args = ap.parse_args()
    page = build()
    if args.fragment:
        with open(args.fragment, 'w', encoding='utf-8') as f:
            f.write(page)
        print('фрагмент:', args.fragment)
    out = os.path.join(ROOT, 'web', 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    print('страница:', out, f'{os.path.getsize(out) / 1024:.0f} КБ')


if __name__ == '__main__':
    main()
