"""
ТЕСТ ВСЕХ СТРАТЕГИЙ с реалистичным пересчетом банка (раз в 30-70 ставок)

Протестируем все стратегии с агрессивными параметрами для мелких инвесторов.
"""

import numpy as np
from config import INITIAL_BANKROLL, TARGET_ROI
from generate_real_odds_simulations import load_real_odds_outcomes


def calculate_metrics_quick(bankroll_history):
    """Быстрый расчет метрик"""
    num_sims = bankroll_history.shape[0]
    
    # Банкротство
    bankrupt_count = np.sum(np.any(bankroll_history < 1.0, axis=1))
    bankrupt_pct = bankrupt_count / num_sims * 100
    
    # Просадки
    peaks = np.maximum.accumulate(bankroll_history, axis=1)
    drawdowns_pct = (bankroll_history - peaks) / peaks * 100
    
    dd50_count = np.sum(np.any(drawdowns_pct <= -50, axis=1))
    dd80_count = np.sum(np.any(drawdowns_pct <= -80, axis=1))
    
    # Прибыль
    final_bankrolls = bankroll_history[:, -1]
    profit_pcts = (final_bankrolls - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100
    
    return {
        'profit': np.mean(profit_pcts),
        'bankrupt': bankrupt_pct,
        'dd50': dd50_count / num_sims * 100,
        'dd80': dd80_count / num_sims * 100,
        'worst_dd': np.min(np.min(drawdowns_pct, axis=1))
    }


def test_kelly_realistic(outcomes, odds_array, risk=2.0, kelly_fraction=1.0, 
                        recalc_min=30, recalc_max=70):
    """Kelly Criterion с реалистичным пересчетом"""
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        
        np.random.seed(sim_idx + 100)
        
        while current_pos < num_bets:
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                odds = odds_array[i]
                win_prob = 1.0 / odds  # Упрощение
                
                # Kelly formula: f = (b*p - q) / b, где b=odds-1, p=win_prob, q=1-p
                b = odds - 1
                p = win_prob
                q = 1 - p
                kelly_f = (b * p - q) / b if b > 0 else 0
                kelly_f = max(0, kelly_f)
                
                bet_pct = kelly_f * kelly_fraction * risk
                bet_pct = min(bet_pct, 10.0)  # Max 10%
                
                bet_size = base_bank * bet_pct / 100
                bet_size = min(bet_size, current_bank, current_bank * 0.10)
                
                if outcomes[sim_idx, i]:
                    current_bank += bet_size * (odds - 1)
                else:
                    current_bank -= bet_size
                
                bankroll_history[sim_idx, i + 1] = current_bank
            
            if current_bank < 1:
                break
            
            base_bank = current_bank
            current_pos = period_end
    
    return bankroll_history


def test_sqrt_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=1.0, 
                            max_percent=10.0, recalc_min=30, recalc_max=70):
    """Sqrt ROI с реалистичным пересчетом"""
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        
        np.random.seed(sim_idx + 200)
        
        while current_pos < num_bets:
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                odds = odds_array[i]
                
                # Sqrt scaling
                bet_pct = base_percent * np.sqrt(roi_pct / base_roi)
                bet_pct = min(bet_pct, max_percent)
                
                bet_size = base_bank * bet_pct / 100
                bet_size = min(bet_size, current_bank, current_bank * 0.10)
                
                if outcomes[sim_idx, i]:
                    current_bank += bet_size * (odds - 1)
                else:
                    current_bank -= bet_size
                
                bankroll_history[sim_idx, i + 1] = current_bank
            
            if current_bank < 1:
                break
            
            base_bank = current_bank
            current_pos = period_end
    
    return bankroll_history


def test_exponential_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=1.0,
                                   exponent=1.5, max_percent=10.0, recalc_min=30, recalc_max=70):
    """Exponential ROI с реалистичным пересчетом"""
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        
        np.random.seed(sim_idx + 300)
        
        while current_pos < num_bets:
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                odds = odds_array[i]
                
                # Exponential scaling
                bet_pct = base_percent * ((roi_pct / base_roi) ** exponent)
                bet_pct = min(bet_pct, max_percent)
                
                bet_size = base_bank * bet_pct / 100
                bet_size = min(bet_size, current_bank, current_bank * 0.10)
                
                if outcomes[sim_idx, i]:
                    current_bank += bet_size * (odds - 1)
                else:
                    current_bank -= bet_size
                
                bankroll_history[sim_idx, i + 1] = current_bank
            
            if current_bank < 1:
                break
            
            base_bank = current_bank
            current_pos = period_end
    
    return bankroll_history


def test_anti_martingale_realistic(outcomes, odds_array, base_percent=1.0, multiplier=1.5,
                                   max_percent=10.0, recalc_min=30, recalc_max=70):
    """Anti-Martingale (увеличение после выигрышей) с реалистичным пересчетом"""
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        win_streak = 0
        
        np.random.seed(sim_idx + 400)
        
        while current_pos < num_bets:
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                odds = odds_array[i]
                
                # Увеличиваем ставку после выигрышей
                bet_pct = base_percent * (multiplier ** min(win_streak, 3))  # Max 3 streak
                bet_pct = min(bet_pct, max_percent)
                
                bet_size = base_bank * bet_pct / 100
                bet_size = min(bet_size, current_bank, current_bank * 0.10)
                
                won = outcomes[sim_idx, i]
                
                if won:
                    current_bank += bet_size * (odds - 1)
                    win_streak += 1
                else:
                    current_bank -= bet_size
                    win_streak = 0
                
                bankroll_history[sim_idx, i + 1] = current_bank
            
            if current_bank < 1:
                break
            
            base_bank = current_bank
            current_pos = period_end
    
    return bankroll_history


def test_linear_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=1.0,
                              max_percent=10.0, recalc_min=30, recalc_max=70):
    """Linear ROI с реалистичным пересчетом"""
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    roi_pct = TARGET_ROI * 100
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        
        np.random.seed(sim_idx + 500)
        
        while current_pos < num_bets:
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                # Linear scaling
                bet_pct = base_percent * (roi_pct / base_roi)
                bet_pct = min(bet_pct, max_percent)
                
                bet_size = base_bank * bet_pct / 100
                bet_size = min(bet_size, current_bank, current_bank * 0.10)
                
                if outcomes[sim_idx, i]:
                    current_bank += bet_size * (odds_array[i] - 1)
                else:
                    current_bank -= bet_size
                
                bankroll_history[sim_idx, i + 1] = current_bank
            
            if current_bank < 1:
                break
            
            base_bank = current_bank
            current_pos = period_end
    
    return bankroll_history


def test_fixed_fraction_realistic(outcomes, odds_array, fixed_percent=2.0,
                                  recalc_min=30, recalc_max=70):
    """Fixed Fraction с реалистичным пересчетом"""
    num_sims, num_bets = outcomes.shape
    bankroll_history = np.full((num_sims, num_bets + 1), INITIAL_BANKROLL, dtype=float)
    
    for sim_idx in range(num_sims):
        current_bank = INITIAL_BANKROLL
        current_pos = 0
        
        np.random.seed(sim_idx + 600)
        
        while current_pos < num_bets:
            period = np.random.randint(recalc_min, recalc_max + 1)
            period_end = min(current_pos + period, num_bets)
            base_bank = current_bank
            
            for i in range(current_pos, period_end):
                if current_bank < 1:
                    break
                
                bet_size = base_bank * fixed_percent / 100
                bet_size = min(bet_size, current_bank, current_bank * 0.10)
                
                if outcomes[sim_idx, i]:
                    current_bank += bet_size * (odds_array[i] - 1)
                else:
                    current_bank -= bet_size
                
                bankroll_history[sim_idx, i + 1] = current_bank
            
            if current_bank < 1:
                break
            
            base_bank = current_bank
            current_pos = period_end
    
    return bankroll_history


# ============================================================================
# ОСНОВНОЙ ТЕСТ
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("🔬 ТЕСТ ВСЕХ СТРАТЕГИЙ (реалистичный пересчет банка раз в 30-70 ставок)")
    print("="*100)
    
    # Загружаем только 1000 симуляций для скорости
    outcomes_full, odds_array = load_real_odds_outcomes()
    outcomes = outcomes_full[:1000]  # Только 1000 для теста
    
    print(f"\nДанные: {outcomes.shape[0]} симуляций, {outcomes.shape[1]} ставок")
    print(f"Средний коэффициент: {np.mean(odds_array):.2f}")
    
    results = []
    
    print("\n" + "="*100)
    print("📊 РЕЗУЛЬТАТЫ (сортировка по прибыли)")
    print("="*100)
    print(f"\n{'Стратегия':<40} {'Params':<20} {'Profit':<10} {'Bankrupt':<10} {'DD>50%':<10} {'Worst DD'}")
    print("-"*100)
    
    # 1. adaptive_constant_profit (уже знаем что работает)
    from realistic_simulation import adaptive_constant_profit_realistic
    for k in [1.0, 1.5, 2.0]:
        br = adaptive_constant_profit_realistic(
            outcomes, odds_array,
            min_roi=4.733, max_roi=23.005,
            min_target_pct=3.982 * k, max_target_pct=13.078 * k,
            max_bet_percent=20.0 * k, apply_variation=False,
            recalc_min=30, recalc_max=70
        )[0]
        m = calculate_metrics_quick(br)
        results.append(('adaptive_constant_profit', f'k={k}', m))
        print(f"{'adaptive_constant_profit':<40} {f'k={k}':<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:<10.1f} {m['worst_dd']:.1f}%")
    
    # 2. Kelly Criterion
    for risk in [1.0, 2.0, 3.0]:
        for frac in [0.5, 1.0]:
            br = test_kelly_realistic(outcomes, odds_array, risk=risk, kelly_fraction=frac)
            m = calculate_metrics_quick(br)
            results.append(('kelly_criterion', f'r={risk},f={frac}', m))
            print(f"{'kelly_criterion':<40} {f'r={risk},f={frac}':<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:<10.1f} {m['worst_dd']:.1f}%")
    
    # 3. Linear ROI
    for base_pct in [1.0, 2.0, 3.0, 5.0]:
        br = test_linear_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=base_pct, max_percent=10.0)
        m = calculate_metrics_quick(br)
        results.append(('linear_roi', f'base={base_pct}%', m))
        print(f"{'linear_roi':<40} {f'base={base_pct}%':<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:<10.1f} {m['worst_dd']:.1f}%")
    
    # 4. Sqrt ROI
    for base_pct in [1.0, 2.0, 3.0, 5.0]:
        br = test_sqrt_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=base_pct, max_percent=10.0)
        m = calculate_metrics_quick(br)
        results.append(('sqrt_roi', f'base={base_pct}%', m))
        print(f"{'sqrt_roi':<40} {f'base={base_pct}%':<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:<10.1f} {m['worst_dd']:.1f}%")
    
    # 5. Exponential ROI
    for base_pct in [1.0, 2.0, 3.0]:
        for exp in [1.5, 2.0]:
            br = test_exponential_roi_realistic(outcomes, odds_array, base_roi=5.0, base_percent=base_pct, exponent=exp, max_percent=10.0)
            m = calculate_metrics_quick(br)
            results.append(('exponential_roi', f'base={base_pct}%,e={exp}', m))
            print(f"{'exponential_roi':<40} {f'base={base_pct}%,e={exp}':<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:<10.1f} {m['worst_dd']:.1f}%")
    
    # 6. Anti-Martingale
    for base_pct in [1.0, 2.0, 3.0]:
        for mult in [1.3, 1.5, 2.0]:
            br = test_anti_martingale_realistic(outcomes, odds_array, base_percent=base_pct, multiplier=mult, max_percent=10.0)
            m = calculate_metrics_quick(br)
            results.append(('anti_martingale', f'base={base_pct}%,m={mult}', m))
            print(f"{'anti_martingale':<40} {f'base={base_pct}%,m={mult}':<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:<10.1f} {m['worst_dd']:.1f}%")
    
    # 7. Fixed Fraction
    for pct in [2.0, 3.0, 4.0, 5.0]:
        br = test_fixed_fraction_realistic(outcomes, odds_array, fixed_percent=pct)
        m = calculate_metrics_quick(br)
        results.append(('fixed_fraction', f'{pct}%', m))
        print(f"{'fixed_fraction':<40} {f'{pct}%':<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:<10.1f} {m['worst_dd']:.1f}%")
    
    # Сортируем по прибыли
    results.sort(key=lambda x: x[2]['profit'], reverse=True)
    
    print("\n" + "="*100)
    print("🏆 ТОП-10 ПО ПРИБЫЛИ")
    print("="*100)
    print(f"\n{'#':<4} {'Стратегия':<40} {'Params':<20} {'Profit':<10} {'Bankrupt':<10} {'DD>50%'}")
    print("-"*100)
    
    for i, (name, params, m) in enumerate(results[:10], 1):
        print(f"{i:<4} {name:<40} {params:<20} +{m['profit']:<9.0f} {m['bankrupt']:<10.2f} {m['dd50']:.1f}%")
    
    print("\n" + "="*100)
    print("🛡️ ЛУЧШИЕ ДЛЯ АГРЕССИВНОЙ ИГРЫ (profit > 200%, bankrupt < 5%)")
    print("="*100)
    
    aggressive = [(name, params, m) for name, params, m in results 
                  if m['profit'] > 200 and m['bankrupt'] < 5.0]
    
    if aggressive:
        print(f"\n✅ Найдено {len(aggressive)} вариантов:\n")
        for i, (name, params, m) in enumerate(aggressive[:10], 1):
            print(f"{i}. {name} ({params})")
            print(f"   💰 Profit: +{m['profit']:.0f}%")
            print(f"   ⚠️  Bankrupt: {m['bankrupt']:.2f}%")
            print(f"   ⚠️  DD>50%: {m['dd50']:.1f}%, DD>80%: {m['dd80']:.1f}%")
            print(f"   📊 Worst DD: {m['worst_dd']:.1f}%")
            print()
    else:
        print("\n❌ Нет стратегий с такими критериями")
    
    print("="*100)
    print("✅ ТЕСТ ЗАВЕРШЕН!")
    print("="*100)
