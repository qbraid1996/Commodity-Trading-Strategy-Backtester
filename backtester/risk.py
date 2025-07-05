# risk_analysis.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_risk_analysis(portfolio, figsize=(12, 6), show_normal=True):
    """
    Affiche la distribution des returns avec VaR/CVaR et optionnellement la distribution normale.
    
    Parameters:
    -----------
    portfolio : DataFrame
        Portfolio avec colonne 'returns'
    label : str
        Nom de la stratégie pour le titre
    figsize : tuple
        Taille de la figure
    show_normal : bool
        Afficher la distribution normale de référence
    """
    returns = portfolio['returns'].dropna()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogramme
    ax.hist(returns, bins=50, density=True, alpha=0.7, color='steelblue', 
            edgecolor='black', label='Returns observés')
    
    # VaR/CVaR calculations
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # VaR/CVaR lines
    ax.axvline(var_95, color='red', linestyle='--', linewidth=2, 
               label=f'VaR 95%: {var_95*100:.2f}%')
    ax.axvline(cvar_95, color='darkred', linestyle='--', linewidth=2, 
               label=f'CVaR 95%: {cvar_95*100:.2f}%')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # Distribution normale de référence (optionnel)
    if show_normal:
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x, returns.mean(), returns.std())
        ax.plot(x, normal_dist, 'orange', linewidth=2, alpha=0.8, label='Distribution normale')
    
    ax.set_title(f'Returns distribution & VaR', fontsize=14)
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_risk_metrics(portfolios, labels=None, title="RISK ANALYSIS COMPARISON"):
    """
    Compare les métriques de risque entre plusieurs portfolios avec affichage tableau.
    
    Parameters:
    -----------
    portfolios : list
        Liste de DataFrames portfolios
    labels : list
        Liste des noms des stratégies
    title : str
        Titre du tableau
    """
    if labels is None:
        labels = [f"Strategy {i+1}" for i in range(len(portfolios))]

    all_metrics = []

    for portfolio, label in zip(portfolios, labels):
        returns = portfolio['returns'].dropna()
        
        # VaR/CVaR calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Statistiques de distribution
        mean_ret = returns.mean()
        std_ret = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # CVaR/VaR Ratio
        cvar_var_ratio = abs(cvar_95/var_95) if var_95 != 0 else float('nan')
        
        metrics = {
            "Label": label,
            "Mean (%)": round(mean_ret * 100, 3) if not np.isnan(mean_ret) else 'NaN',
            "Std (%)": round(std_ret * 100, 2) if not np.isnan(std_ret) else 'NaN',
            "Skewness": round(skewness, 3) if not np.isnan(skewness) else 'NaN',
            "Kurtosis": round(kurtosis, 3) if not np.isnan(kurtosis) else 'NaN',
            "VaR 95% (%)": round(var_95 * 100, 2) if not np.isnan(var_95) else 'NaN',
            "CVaR 95% (%)": round(cvar_95 * 100, 2) if not np.isnan(cvar_95) else 'NaN',
            "CVaR/VaR Ratio": round(cvar_var_ratio, 2) if not np.isnan(cvar_var_ratio) else 'NaN',
        }
        all_metrics.append(metrics)

    # === AFFICHAGE EN TABLEAU COMPARATIF ===
    keys = ["Label", "Mean (%)", "Std (%)", "Skewness", "Kurtosis", 
            "VaR 95% (%)", "CVaR 95% (%)", "CVaR/VaR Ratio"]
    
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

    # return all_metrics