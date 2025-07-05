# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_equity_curves(portfolios, labels, title="Comparaison des courbes d'équité"):
    plt.figure(figsize=(12, 6))
    for pf, label in zip(portfolios, labels):
        # Calculer la performance cumulative depuis le début
        cumulative_returns = (1 + pf['returns']).cumprod()
        performance_pct = (cumulative_returns - 1) * 100
        
        plt.plot(pf.index, performance_pct, label=label, linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Performance cumulative (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)  # Ligne de référence à 0%
    plt.tight_layout()
    plt.show()

def plot_price_with_signals_and_volume(data, signals, portfolio=None, title="Price and Trading Signals"):
    """
    Affiche l'equity curve, le prix/spread avec signaux, et le volume.
    portfolio : DataFrame du backtest (optionnel, pour l'equity curve)
    """

    # Créer 3 sous-graphiques avec ratios ajustés
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                        gridspec_kw={'height_ratios': [0.5, 1.2, 0.2]}, 
                                        sharex=True)

    # === GRAPHIQUE 1 : EQUITY CURVE ===
    if portfolio is not None:
        # Calculer la performance cumulative
        cumulative_returns = (1 + portfolio['returns']).cumprod()
        performance_pct = (cumulative_returns - 1) * 100

        peak_idx = performance_pct.idxmax()
        peak_value = performance_pct.max()

        cumulative_returns = (1 + portfolio['returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd_value = drawdown.min() * 100
        max_dd_idx = drawdown.idxmin()
        performance_at_max_dd = performance_pct.loc[max_dd_idx]
        
        ax1.plot(portfolio.index, performance_pct, color='cadetblue', linewidth=2, label='Strategy Performance')
        ax1.plot(peak_idx, peak_value, 'o', color='gold', markersize=8, 
             label=f'Peak ({peak_value:.2f}%)', zorder=5)
        ax1.plot(max_dd_idx, performance_at_max_dd, 'o', color='darkred', markersize=8, 
             label=f'Max Drawdown ({max_dd_value:.2f}%)', zorder=5)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Performance (%)')
        ax1.set_title(f'{title} - Strategy Performance', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No Portfolio Data Available', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_ylabel('Performance (%)')
        ax1.set_title(f'{title} - Strategy Performance', fontsize=14)

    # === GRAPHIQUE 2 : PRIX + SIGNAUX ===
    
    # Détection du prix à tracer
    if 'spread' in signals.columns:
        plot_data = signals['spread']
        label = 'Spread'
    elif 'crack_spread' in signals.columns:
        plot_data = signals['crack_spread']
        label = 'Crack Spread'
    elif 'Close' in data.columns:
        plot_data = data['Close']
        label = 'Price'
    else:
        raise ValueError("Aucune colonne 'Close', 'spread' ou 'crack_spread' détectée pour affichage.")

    # Tracé du prix
    ax2.plot(plot_data.index, plot_data, label=label, color='slategrey', linewidth=1)

    # MAs standards (si présents)
    if 'short_ma' in signals.columns:
        ax2.plot(signals.index, signals['short_ma'], label='Short MA', linestyle='--', color='blue')
    if 'long_ma' in signals.columns:
        ax2.plot(signals.index, signals['long_ma'], label='Long MA', linestyle='--', color='red')
    if 'spread_ma' in signals.columns:
        ax2.plot(signals.index, signals['spread_ma'], label='Spread MA', linestyle='--', color='orange')
    if 'ma' in signals.columns:
        ax2.plot(signals.index, signals['ma'], label='MA', linestyle='--', color='darkgoldenrod', linewidth = 2)

    # Signaux d'achat / vente
    if 'signal' in signals.columns:
        buy_signals = signals[signals['signal'] > 0]
        sell_signals = signals[signals['signal'] < 0]

        ax2.plot(buy_signals.index, plot_data.loc[buy_signals.index],
                '^', markersize=8, color='green', label='Buy Signal')
        ax2.plot(sell_signals.index, plot_data.loc[sell_signals.index],
                'v', markersize=8, color='red', label='Sell Signal')

    ax2.set_ylabel(label)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # === GRAPHIQUE 3 : VOLUME (réduit) ===
    
    volume_col = None
    for col in ['Volume', 'volume', 'Vol']:
        if col in data.columns:
            volume_col = col
            break
    
    if volume_col is not None:
        colors = []
        for i in range(len(data)):
            if i == 0:
                colors.append('gray')
            else:
                if data['Close'].iloc[i] >= data['Close'].iloc[i-1]:
                    colors.append('green')
                else:
                    colors.append('red')
        
        ax3.bar(data.index, data[volume_col], color=colors, alpha=0.7, width=0.7)
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        max_vol = data[volume_col].max()
        if max_vol > 1e6:
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        elif max_vol > 1e3:
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
    else:
        ax3.text(0.5, 0.5, 'No Volume Data Available', 
                transform=ax3.transAxes, ha='center', va='center')
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Date')

    plt.tight_layout()
    plt.show()