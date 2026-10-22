"""
Собирает web/index.html: подставляет в web/app.html движок (web/engine.js),
реальные ставки (только кэф и EV), предрасчёт reports/results.json и сравнение
reports/legacy_before_after.json.

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


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')


def build():
    with open(os.path.join(ROOT, 'web', 'app.html'), encoding='utf-8') as f:
        page = f.read()
    with open(os.path.join(ROOT, 'web', 'engine.js'), encoding='utf-8') as f:
        engine = f.read()
    with open(os.path.join(ROOT, 'reports', 'results.json'), encoding='utf-8') as f:
        results = json.load(f)
    with open(os.path.join(ROOT, 'reports', 'legacy_before_after.json'), encoding='utf-8') as f:
        before_after = json.load(f)
    if '</script' in engine:
        raise ValueError('engine.js не должен содержать </script')
    for key, value in (('__ENGINE__', engine), ('__REAL_BETS__', compact(load_bets())),
                       ('__RESULTS__', compact(results)), ('__BEFORE_AFTER__', compact(before_after))):
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
