# performance.py

import numpy as np


def compute_and_print_performance_metrics(portfolios, labels=None, price_data=None, risk_free_rate=0.02, title="RECAP PERFORMANCE"):
    """
    Calcule et affiche les métriques de performance pour une liste de portfolios.
    portfolios: liste de DataFrames
    labels: liste de noms pour les stratégies
    price_data : DataFrame avec colonne 'Close' pour calculer Buy & Hold (optionnel)
    risk_free_rate: taux sans risque annuel (défaut 2%)
    Retourne une liste de dicts de métriques.
    """
    if labels is None:
        labels = [f"Stratégie {i+1}" for i in range(len(portfolios))]

    # === CALCUL BUY & HOLD ===
    buy_hold_return = float('nan')
    if price_data is not None and 'Close' in price_data.columns and len(portfolios) > 0:
        first_portfolio = portfolios[0]
        start_date = first_portfolio.index[0]
        end_date = first_portfolio.index[-1]
        
        price_period = price_data.loc[start_date:end_date, 'Close'].dropna()
        
        if len(price_period) > 1:
            initial_price = price_period.iloc[0]
            final_price = price_period.iloc[-1]
            buy_hold_return = (final_price / initial_price - 1) * 100

    all_metrics = []

    for pf, label in zip(portfolios, labels):
        # === RENDEMENT TOTAL ET ANNUALISÉ ===
        if 'total' in pf.columns and len(pf['total'].dropna()) > 1:
            equity_curve = pf['total'].dropna()
            initial_value = equity_curve.iloc[0]
            final_value = equity_curve.iloc[-1]
            
            total_return = (final_value / initial_value - 1) * 100
            
            n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
            n_years = max(n_days / 365.25, 1/252)  
            
            if final_value > 0 and initial_value > 0 and n_years > 0:
                try:
                    annualized_return = ((final_value / initial_value) ** (1 / n_years) - 1) * 100
                except (ValueError, ZeroDivisionError, OverflowError):
                    annualized_return = float('nan')
            else:
                annualized_return = float('nan')
        else:
            total_return = annualized_return = float('nan')

        # === MAX DRAWDOWN ===
        if 'total' in pf.columns and len(pf['total'].dropna()) > 1:
            equity_curve = pf['total'].dropna()
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = float('nan')

        # === VOLATILITÉ ANNUALISÉE ===
        if 'returns' in pf.columns and len(pf['returns'].dropna()) > 1:
            returns_clean = pf['returns'].dropna()
            annualized_vol = returns_clean.std() * np.sqrt(252) * 100
        else:
            annualized_vol = float('nan')

        # === SHARPE RATIO ===
        if 'returns' in pf.columns and len(pf['returns'].dropna()) > 1:
            returns_clean = pf['returns'].dropna()
            if returns_clean.std() > 0:
                daily_rf = risk_free_rate / 252
                excess_returns = returns_clean - daily_rf
                sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            else:
                sharpe = float('nan')
        else:
            sharpe = float('nan')

        # === NOMBRE DE TRADES ===
        if 'positions' in pf.columns:
            position_changes = pf['positions'].diff().fillna(0) # A corriger avec les signaux
            num_trades = (position_changes.abs() > 0).sum()
        else:
            num_trades = 0

        # === WIN RATE ===
        if 'returns' in pf.columns:
            positive_returns = (pf['returns'] > 0).sum()
            total_periods = len(pf['returns'].dropna())
            win_rate = (positive_returns / total_periods * 100) if total_periods > 0 else float('nan')
        else:
            win_rate = float('nan')

        metrics = {
            "Label": label,
            "Total Return (%)": round(total_return, 2) if not np.isnan(total_return) else 'NaN',
            "Buy & Hold Return": round(buy_hold_return, 2) if not np.isnan(buy_hold_return) else 'NaN',
            "Annualized Return (%)": round(annualized_return, 2) if not np.isnan(annualized_return) else 'NaN',
            "Annualized Volatility (%)": round(annualized_vol, 2) if not np.isnan(annualized_vol) else 'NaN',
            "Sharpe Ratio": round(sharpe, 2) if not np.isnan(sharpe) else 'NaN',
            "Max Drawdown (%)": round(max_drawdown, 2) if not np.isnan(max_drawdown) else 'NaN',
            "Win Rate (%)": round(win_rate, 2) if not np.isnan(win_rate) else 'NaN',
            "Number of Trades": int(num_trades)
        }
        all_metrics.append(metrics)

    # === AFFICHAGE EN TABLEAU COMPARATIF ===
    keys = ["Label", "Total Return (%)", "Buy & Hold Return", 
            "Annualized Return (%)", "Annualized Volatility (%)",
            "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)", "Number of Trades"]
    
    col_widths = [max(len(str(m[k])) for m in all_metrics + [{k: k}]) + 2 for k in keys]
    sep = "+" + "+".join("-"*w for w in col_widths) + "+"

    print("=" * (sum(col_widths) + len(col_widths) + 1))
    print(f"{title:^{sum(col_widths) + len(col_widths) + 1}}")
    print("=" * (sum(col_widths) + len(col_widths) + 1))
    print(sep)
    print("|" + "|".join(k.center(w) for k, w in zip(keys, col_widths)) + "|")
    print(sep)
    for m in all_metrics:
        print("|" + "|".join(str(m[k]).center(w) for k, w in zip(keys, col_widths)) + "|")
    print(sep)

    return all_metrics