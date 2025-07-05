import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import scipy.stats as stats

# Ajouter le répertoire parent au path pour importer les modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports des modules locaux
from utils.data_loader import load_commodity_data, download_data
from strategies.moving_average import MovingAverageCrossover
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.calendar_lag import CalendarLagStrategy
from strategies.LogMomentum import LogMomentumStrategy
from backtester.engine_v2 import BacktestEngineV2
from utils.tickers import *
from strategies.OU import OrnsteinUhlenbeckStrategy

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Backtester Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            min-width: 330px;
            width: 330px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Set Up")
        
        # Instrument selection
        instruments = {
            'Brent Crude Oil': 'BRENT',
            'WTI Crude Oil': 'WTI',
            'Gasoline (RBOB)': 'GASOLINE',
            'Heating Oil': 'HEATING_OIL',
            'Crack Spread US': CRACK_US,
            'Crack Spread EU': CRACK_EU
        }
        selected_instrument = st.selectbox(
            "Choose the commodity",
            list(instruments.keys()),
            index=0
        )
        
        # Period selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value='2021-01-01',
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value='2024-07-01',
                max_value=datetime.now()
            )
        
        # Strategy selection avec restrictions
        st.subheader("Strategy & Tuning")

        # Get available strategies based on instrument
        available_strategies = get_available_strategies(selected_instrument)

        selected_strategy = st.selectbox(
            "Choose the strategy",
            list(available_strategies.keys()),
            index=0
        )

        # Strategy parameters
        strategy_params = get_strategy_params(available_strategies[selected_strategy], selected_instrument)

        current_params = {
            'instrument': selected_instrument,
            'start_date': start_date,
            'end_date': end_date,
            'strategy': selected_strategy,
            'params': str(strategy_params)  # Convertir en string pour comparaison
        }
        
        # Initialiser session state si nécessaire
        if 'last_params' not in st.session_state:
            st.session_state.last_params = None
            st.session_state.should_run = True
        
        # Vérifier si les paramètres ont changé
        if st.session_state.last_params != current_params:
            st.session_state.last_params = current_params
            st.session_state.should_run = True    

    # SORTIE DE LA SIDEBAR - CODE PRINCIPAL
    if st.session_state.should_run:
        st.session_state.should_run = False  # Reset flag
        
        with st.spinner("Loading data and computing signals..."):
            try:
                # Data loading
                instrument_code = instruments[selected_instrument]
                data = load_commodity_data(
                    instrument_code,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if data.empty:
                    st.error("No data available for this period")
                    return
                
                # Signal generation
                strategy_obj = create_strategy(available_strategies[selected_strategy], strategy_params)
                signals = strategy_obj.generate_signals(data)
                
                # Backtest
                engine = BacktestEngineV2(data, signals)
                portfolio = engine.run()
                
                # Calculate all metrics
                total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0] - 1) * 100
                cumulative_returns = (1 + portfolio['returns']).cumprod()
                running_max = cumulative_returns.cummax()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min() * 100
                
                returns_std = portfolio['returns'].std() * np.sqrt(252)
                returns_mean = portfolio['returns'].mean() * 252
                sharpe_ratio = returns_mean / returns_std if returns_std != 0 else 0
                annualized_volatility = returns_std * 100
                
                buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                num_trades = (signals['signal'] != 0).sum()
                
                # Calculate win rate
                win_rate = calculate_win_rate(portfolio)

                risk_metrics = calculate_risk_metrics(portfolio)
                
                # === PERFORMANCE METRICS IN SIDEBAR ===
                with st.sidebar:
                    st.markdown("-----")
                    st.markdown("### Performance Metrics")
                    
                    st.markdown(f"""
                    <div style="font-family: 'Monaco', 'Courier New', monospace; font-size: 0.95rem; white-space: pre; line-height: 1.4; margin: -1rem 0;">
                Return [%]                {total_return:6.2f}
                Buy&Hold Return [%]       {buy_hold_return:6.2f}
                Max Drawdown [%]          {max_drawdown:6.2f}
                Sharpe Ratio              {sharpe_ratio:6.2f}
                Vol. (Ann.) [%]           {annualized_volatility:6.2f}
                Win Rate [%]              {win_rate:6.1f}
                # of Trades               {num_trades:6d}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### Risk Analysis")
                    st.markdown(f"""
                    <div style="font-family: 'Monaco', 'Courier New', monospace; font-size: 0.95rem; white-space: pre; line-height: 1.4; margin-top: -1rem;">
                Return Mean [%]          {risk_metrics['mean']:7.2f}
                Return Std [%]           {risk_metrics['std']:7.2f}
                Skewness                 {risk_metrics['skewness']:7.2f}
                VaR 95% [%]              {risk_metrics['var_95']:7.2f}
                CVaR 95% [%]             {risk_metrics['cvar_95']:7.2f}
                CVaR/VaR Ratio           {risk_metrics['cvar_var_ratio']:7.2f}
                    </div>
                    """, unsafe_allow_html=True)
   
                # === CHART IN MAIN AREA ===
                fig = create_main_chart(data, signals, portfolio, selected_instrument, selected_strategy)
                st.plotly_chart(fig, use_container_width=True)

                # === RISK ANALYSIS CHART (Left half only) ===

                col1, col2 = st.columns([1, 1])  # Split in half

                with col1:
                    risk_fig = create_risk_analysis_chart(portfolio)
                    if risk_fig is not None:
                        st.plotly_chart(risk_fig, use_container_width=True)
                    else:
                        st.info("No data available for risk analysis")

                with col2:
                    # Create comparison chart (now includes main portfolio)
                    comparison_fig = create_comparison_chart(
                        strategy_obj, 
                        strategy_params, 
                        selected_instrument, 
                        start_date, 
                        end_date,
                        portfolio  # Pass the main portfolio as reference
                    )
                
                    if comparison_fig is not None:
                        st.plotly_chart(comparison_fig, use_container_width=True)
                    else:
                        st.info("No comparison data available")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)

def calculate_win_rate(portfolio):
    """Calculate win rate from portfolio data"""
    try:
        # Get trade returns by looking at portfolio returns when positions change
        trade_returns = []
        current_pos = 0
        entry_price = None
        
        for i in range(len(portfolio)):
            pos = portfolio['positions'].iloc[i]
            price = portfolio['price'].iloc[i]
            
            if pos != current_pos:  # Position change
                if current_pos != 0 and entry_price is not None:  # Closing a position
                    trade_return = (price - entry_price) / entry_price * current_pos
                    trade_returns.append(trade_return)
                
                if pos != 0:  # Opening a new position
                    entry_price = price
                
                current_pos = pos
        
        # Close final position if still open
        if current_pos != 0 and entry_price is not None:
            final_price = portfolio['price'].iloc[-1]
            trade_return = (final_price - entry_price) / entry_price * current_pos
            trade_returns.append(trade_return)
        
        # Calculate win rate
        if len(trade_returns) > 0:
            winning_trades = sum(1 for ret in trade_returns if ret > 0)
            win_rate = (winning_trades / len(trade_returns)) * 100
        else:
            win_rate = 0
            
        return win_rate
    except:
        return 0

def calculate_risk_metrics(portfolio):
    """Calculate risk metrics from portfolio returns"""
    returns = portfolio['returns'].dropna()
    
    if len(returns) == 0:
        return {
            'mean': 0, 'std': 0, 'skewness': 0, 
            'var_95': 0, 'cvar_95': 0, 'cvar_var_ratio': 0
        }
    
    # Basic statistics
    mean_ret = returns.mean() * 100  # Convert to percentage
    std_ret = returns.std() * 100    # Convert to percentage
    skewness = stats.skew(returns)
    
    # VaR/CVaR calculations
    var_95 = np.percentile(returns, 5) * 100  # 95% VaR in percentage
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100  # CVaR in percentage
    
    # CVaR/VaR Ratio
    cvar_var_ratio = abs(cvar_95/var_95) if var_95 != 0 else 0
    
    return {
        'mean': mean_ret,
        'std': std_ret, 
        'skewness': skewness,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'cvar_var_ratio': cvar_var_ratio
    }

def get_strategy_params(strategy_type, selected_instrument):
    """Interface to configure parameters by strategy with instrument restrictions"""
    params = {}
   
    if strategy_type == 'ma_crossover':
        col1, col2 = st.columns(2)
        with col1:
            params['short_window'] = st.number_input("Short MA", 1, 100, 10, 10)
        with col2:
            params['long_window'] = st.number_input("Long MA", 1, 200, 30, 10)

    elif strategy_type == 'mean_reversion':
        col1, col2 = st.columns(2)
        with col1:
            params['ma_window'] = st.number_input("MA window", 1, 100, 30, 5)
            params['exit_factor'] = st.slider("Exit Factor (%)", 0 , 70 , 10 ,10)
        with col2:
            params['threshold'] = st.number_input("Threshold (%)", 0.5, 10.0, 5.0, 0.5)
      
    elif strategy_type == 'momentum':
        col1, col2 = st.columns(2)
        with col1:
            params['lookback'] = st.number_input("Lookback period", 1, 100, 30, 5)
            params['exit'] = st.slider("Exit Factor (%)", 0 , 70 , 30 ,10)
        with col2:
            params['threshold'] = st.number_input("Threshold (%)", 2, 15, 6, 1)   
    
    elif strategy_type == 'calendar_spread':
        col1, col2 = st.columns(2)
        with col1:
            params['lag_days'] = st.number_input("Lag days", 1, 100, 30, 5)
            params['exit'] = st.slider("Exit Factor (%)", 0 , 70 , 30 ,10)
        with col2:
            params['threshold'] = st.number_input("Threshold", 1.0 , 10.0 , 2.0 , 0.5)
            
    elif strategy_type == 'log_momentum':
        col1, col2 = st.columns(2)
        with col1:
            params['window'] = st.number_input("Window", 1, 50, 20,5)
            params['exit'] = st.slider("Exit Factor (%)", 0 , 70 , 10 ,10)
        with col2:
            params['threshold'] = st.number_input("Threshold", 0.01, 0.10, 0.04, 0.01)
    
    elif strategy_type == 'ornstein_uhlenbeck':
        col1, col2 = st.columns(2)
        with col1:
            params['z_entry'] = st.number_input("Z-Score Entry", 0.5, 5.0, 3.0, 0.1)
            params['z_stop'] = st.number_input("Z-Score Stop", 2.0, 10.0, 6.0, 0.5)
        with col2:
            params['z_exit'] = st.number_input("Z-Score Exit", 0.5, 2.0, 0.8, 0.1)
            params['z_reset'] = st.number_input("Z-Score Reset",  0.5, 5.0, 1.5, 0.5)
    
    return params

def get_available_strategies(selected_instrument):
    """Get available strategies based on selected instrument"""
    
    # Strategies for CRACK spreads only
    crack_strategies = {
        'Mean Reversion': 'mean_reversion',
        'Ornstein-Uhlenbeck': 'ornstein_uhlenbeck'
    }
    
    # All strategies for other instruments
    all_strategies = {
        'Moving Average Crossover': 'ma_crossover',
        'Mean Reversion': 'mean_reversion',
        'Log Momentum': 'log_momentum',
        'Momentum': 'momentum',
        'Calendar Lag': 'calendar_spread'
    }
    
    # Check if it's a crack spread
    if selected_instrument in ['Crack Spread US', 'Crack Spread EU']:
        return crack_strategies
    else:
        return all_strategies

def create_strategy(strategy_type, params):
    """Factory to create strategy objects"""
    if strategy_type == 'ma_crossover':
        return MovingAverageCrossover(**params)
    elif strategy_type == 'momentum':
        return MomentumStrategy(**params)
    elif strategy_type == 'mean_reversion':
        return MeanReversionStrategy(**params)
    elif strategy_type == 'calendar_spread':
        return CalendarLagStrategy(**params)
    elif strategy_type == 'log_momentum':
        return LogMomentumStrategy(**params)
    elif strategy_type == 'ornstein_uhlenbeck':
        return OrnsteinUhlenbeckStrategy(**params)

def create_main_chart(data, signals, portfolio, instrument_name, strategy_name):
    """Create main chart with separate legends for each subplot"""
    
    # Create subplots with reduced spacing
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.18, 0.6, 0.1],
        vertical_spacing=0.04,
        shared_xaxes=True
    )
    
    # ========== CHART 1: EQUITY CURVE ==========
    if portfolio is not None:
        cumulative_returns = (1 + portfolio['returns']).cumprod()
        performance_pct = (cumulative_returns - 1) * 100
        
        # Peak and max drawdown
        peak_idx = performance_pct.idxmax()
        peak_value = performance_pct.max()
        
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min() * 100
        performance_at_max_dd = performance_pct.loc[max_dd_idx]
        
        # Performance line
        fig.add_trace(
            go.Scatter(
                x=portfolio.index,
                y=performance_pct,
                mode='lines',
                name='Performance',
                line=dict(color="#055c50", width=3),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Peak point
        fig.add_trace(
            go.Scatter(
                x=[peak_idx],
                y=[peak_value],
                mode='markers',
                name=f'Peak ({peak_value:.1f}%)',
                marker=dict(color="#9c860a", size=12, symbol='circle'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Max drawdown point
        fig.add_trace(
            go.Scatter(
                x=[max_dd_idx],
                y=[performance_at_max_dd],
                mode='markers',
                name=f'Max DD ({max_dd_value:.1f}%)',
                marker=dict(color="#991304", size=12, symbol='circle'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Reference line at 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # ========== CHART 2: PRICE + SIGNALS ==========
    
    # Detect price to plot
    if 'spread' in signals.columns:
        plot_data = signals['spread']
        label = 'Spread'
    elif 'crack_spread' in signals.columns:
        plot_data = signals['crack_spread']
        label = 'Crack Spread'
    else:
        plot_data = data['Close']
        label = 'Price'
    
    # Main price line
    fig.add_trace(
        go.Scatter(
            x=plot_data.index,
            y=plot_data,
            mode='lines',
            name='Price',
            line=dict(color="#246960", width=1.2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Moving averages
    if 'short_ma' in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['short_ma'],
                mode='lines',
                name='Short MA',
                line=dict(color="#1a37b9", dash='dash', width=2.5),
                showlegend=False
            ),
            row=2, col=1
        )
    
    if 'long_ma' in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['long_ma'],
                mode='lines',
                name='Long MA',
                line=dict(color='#e74c3c', dash='dash', width=2.5),
                showlegend=False
            ),
            row=2, col=1
        )
    
    if 'ma' in signals.columns:
        fig.add_trace(
            go.Scatter(
                x=signals.index,
                y=signals['ma'],
                mode='lines',
                name='MA',
                line=dict(color='#f39c12', width=2.5, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

    # Buy/Sell signals avec distinction exit_on_weakness
    if 'signal' in signals.columns:
        
        # === SIGNAUX NORMAUX (triangles) ===
        if 'exit_on_weakness' in signals.columns:
            # Filtrer les signaux normaux (pas exit_on_weakness)
            normal_signals = signals[signals['exit_on_weakness'] == False]
            buy_signals_normal = normal_signals[normal_signals['signal'] > 0]
            sell_signals_normal = normal_signals[normal_signals['signal'] < 0]
            
            # Signaux exit_on_weakness (croix)
            weakness_signals = signals[signals['exit_on_weakness'] == True]
            buy_signals_weakness = weakness_signals[weakness_signals['signal'] > 0]
            sell_signals_weakness = weakness_signals[weakness_signals['signal'] < 0]
        else:
            # Fallback si pas de colonne exit_on_weakness
            buy_signals_normal = signals[signals['signal'] > 0]
            sell_signals_normal = signals[signals['signal'] < 0]
            buy_signals_weakness = pd.DataFrame()
            sell_signals_weakness = pd.DataFrame()
        
        # TRIANGLES pour signaux normaux
        if not buy_signals_normal.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals_normal.index,
                    y=plot_data.loc[buy_signals_normal.index],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color="#08883d", size=15, symbol='triangle-up'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        if not sell_signals_normal.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals_normal.index,
                    y=plot_data.loc[sell_signals_normal.index],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color="#cc2311", size=15, symbol='triangle-down'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # CROIX pour exit_on_weakness
        if not buy_signals_weakness.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals_weakness.index,
                    y=plot_data.loc[buy_signals_weakness.index],
                    mode='markers',
                    name='Exit on Weakness (Buy)',
                    marker=dict(color="#08883d", size=15, symbol='x'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        if not sell_signals_weakness.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals_weakness.index,
                    y=plot_data.loc[sell_signals_weakness.index],
                    mode='markers',
                    name='Exit on Weakness (Sell)',
                    marker=dict(color="#cc2311", size=15, symbol='x'),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # ========== CHART 3: VOLUME ==========
    
    volume_col = None
    for col in ['Volume', 'volume', 'Vol']:
        if col in data.columns:
            volume_col = col
            break
    
    if volume_col is not None and data[volume_col].sum() > 0:
        colors = []
        for i in range(len(data)):
            if i == 0:
                colors.append("#687475")
            else:
                if data['Close'].iloc[i] >= data['Close'].iloc[i-1]:
                    colors.append("#1bc261")
                else:
                    colors.append("#be2d1d")
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data[volume_col],
                name='Volume',
                marker_color=colors,
                opacity=1,
                showlegend=False
            ),
            row=3, col=1
        )

    # LAYOUT
    fig.update_layout(
        height=650,  # Increased height since no metrics on top
        title={
            'text': f'<b>Backtesting: {strategy_name} on {instrument_name}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 25, 'color': "#252525", 'family' : "Courier New, monospace"}
        },
        showlegend=False,
        template='plotly_white',
        plot_bgcolor='white',
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    # Add black borders to subplots
    for i in range(1, 4):
        fig.update_xaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            mirror=True,
            row=i, col=1
        )
        fig.update_yaxes(
            showline=True, 
            linewidth=1, 
            linecolor='black', 
            mirror=True,
            row=i, col=1
        )
    
    # Axis formatting
    fig.update_yaxes(title_text="Performance (%)", title_font=dict(size=14, color='#2c3e50', family='Arial Black'), row=1, col=1)
    fig.update_yaxes(title_text=label, title_font=dict(size=14, color='#2c3e50', family='Arial Black'), row=2, col=1)
    fig.update_yaxes(title_text="Volume", title_font=dict(size=14, color='#2c3e50', family='Arial Black'), row=3, col=1)
    
    # ADD CUSTOM ANNOTATIONS AS LEGENDS
    
    # Legend 1: Performance
    legend1_text = f"<b>Performance Legend:</b><br>"
    legend1_text += f"<span style='color:#055c50'>■</span> Performance<br>"
    legend1_text += f"<span style='color:#9c860a'>●</span> Peak ({peak_value:.1f}%)<br>"
    legend1_text += f"<span style='color:#991304'>●</span> Max DD ({max_dd_value:.1f}%)"

    fig.add_annotation(
        text=legend1_text,
        xref="paper", yref="paper",
        x=-0.002, y=1.005,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=13)
    )

    # Legend 2: Trading (mise à jour avec croix)
    legend2_text = "<b>Legend:              </b><br>"
    legend2_text += f"<span style='color:#246960'>■</span> Price<br>"

    if 'ma' in signals.columns:
        legend2_text += f"<span style='color:#f39c12'>--</span> MA<br>"
    if 'short_ma' in signals.columns:
        legend2_text += f"<span style='color:#3498db'>--</span> Short MA<br>"
    if 'long_ma' in signals.columns:
        legend2_text += f"<span style='color:#e74c3c'>--</span> Long MA<br>"

    if 'positions' in signals.columns:
        legend2_text += f"<span style='color:#08883d'>▲</span> Buy Signal<br>"
        legend2_text += f"<span style='color:#cc2311'>▼</span> Sell Signal<br>"
        
        # Ajouter les croix à la légende
        if 'exit_on_weakness' in signals.columns:
            legend2_text += f"<span style='color:#08883d'>✕</span> Exit (Buy)<br>"
            legend2_text += f"<span style='color:#cc2311'>✕</span> Exit (Sell)"

    fig.add_annotation(
        text=legend2_text,
        xref="paper", yref="paper",
        x=-0.002, y=0.78,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=13)
    )
    
    return fig

def create_risk_analysis_chart(portfolio):
    """Create risk analysis chart with VaR/CVaR using Plotly"""
    
    returns = portfolio['returns'].dropna()
    
    if len(returns) == 0:
        return None
    
    # Calculate VaR/CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Create figure
    fig = go.Figure()
    
    # Histogram of returns
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        histnorm='probability density',
        name='Returns Distribution',
        marker_color='teal',
        marker_line_color='black',
        marker_line_width=0.5,
        opacity=0.85
    ))
    
    # Add normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_dist,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='dodgerblue', width=2.5),
        opacity=0.95
    ))
    
    # Add VaR line - NO ANNOTATION, JUST LEGEND
    fig.add_trace(go.Scatter(
        x=[var_95, var_95],
        y=[0, max(normal_dist)],
        mode='lines',
        name=f'VaR 95% ({var_95*100:.2f}%)',
        line=dict(color='red', width=3, dash='dot'),
        showlegend=True
    ))
    
    # Add CVaR line - NO ANNOTATION, JUST LEGEND
    fig.add_trace(go.Scatter(
        x=[cvar_95, cvar_95],
        y=[0, max(normal_dist)],
        mode='lines',
        name=f'CVaR 95% ({cvar_95*100:.2f}%)',
        line=dict(color='darkred', width=3, dash='dot'),
        showlegend=True
    ))
    
    # Add zero line
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, max(normal_dist)],
        mode='lines',
        name='Zero Line',
        line=dict(color='black', width=1),
        opacity=0.3,
        showlegend=False
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Returns Distribution & Value at Risk',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Daily Returns",
        yaxis_title="Density",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
        ),
        template='plotly_white',
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    # Add grid - NO VERTICAL GRID LINES
    fig.update_xaxes(showgrid=False)  # Pas de lignes verticales
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')  # Seulement horizontales
    
    return fig

def get_comparison_commodities(selected_instrument):
    """Get list of commodities to compare based on selected instrument"""
    
    # Physical commodities
    physical_commodities = {
        'WTI Crude Oil': ['BRENT', 'GASOLINE', 'HEATING_OIL','WTI'],
        'Brent Crude Oil': ['WTI', 'GASOLINE', 'HEATING_OIL','BRENT'], 
        'Gasoline (RBOB)': ['WTI', 'BRENT', 'HEATING_OIL','Gasoline (RBOB)'],
        'Heating Oil': ['WTI', 'BRENT', 'GASOLINE','Heating Oil'],
    }
    
    # Crack spreads - only compare with each other
    crack_spreads = {
        'Crack Spread US': ['CRACK_EU'],
        'Crack Spread EU': ['CRACK_US']
    }
    
    if selected_instrument in physical_commodities:
        return physical_commodities[selected_instrument]
    elif selected_instrument in crack_spreads:
        return crack_spreads[selected_instrument]
    else:
        return []


    """Create comparison chart with equity curves for similar commodities"""
    
    comparison_commodities = get_comparison_commodities(selected_instrument)
    
    if not comparison_commodities:
        return None
    
    fig = go.Figure()
    
    # Colors for different commodities
    colors = ['#e74c3c', '#3498db', '#f39c12', '#27ae60', '#9b59b6']
    
    for i, commodity_code in enumerate(comparison_commodities):
        try:
            # Load data for comparison commodity
            comp_data = load_commodity_data(
                commodity_code,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if comp_data.empty:
                continue
                
            # Generate signals with same strategy
            comp_signals = strategy_obj.generate_signals(comp_data)
            
            # Run backtest
            comp_engine = BacktestEngineV2(comp_data, comp_signals)
            comp_portfolio = comp_engine.run()
            
            # Calculate equity curve
            cumulative_returns = (1 + comp_portfolio['returns']).cumprod()
            performance_pct = (cumulative_returns - 1) * 100
            
            # Get commodity name for legend
            commodity_names = {
                'WTI': 'WTI Crude',
                'BRENT': 'Brent Crude', 
                'GASOLINE': 'Gasoline',
                'HEATING_OIL': 'Heating Oil',
                'CRACK_US': 'Crack Spread US',
                'CRACK_EU': 'Crack Spread EU'
            }
            
            commodity_name = commodity_names.get(commodity_code, commodity_code)
            
            # Add to chart
            fig.add_trace(go.Scatter(
                x=comp_portfolio.index,
                y=performance_pct,
                mode='lines',
                name=commodity_name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'{commodity_name}<br>Performance: %{{y:.2f}}%<extra></extra>'
            ))
            
        except Exception as e:
            print(f"Error processing {commodity_code}: {e}")
            continue
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Strategy Performance Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': '#2c3e50'}
        },
        xaxis_title="Date",
        yaxis_title="Performance (%)",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def create_comparison_chart(strategy_obj, strategy_params, selected_instrument, start_date, end_date, main_portfolio):
    """Create comparison chart with equity curves for similar commodities including selected one"""
    
    comparison_commodities = get_comparison_commodities(selected_instrument)
    
    fig = go.Figure()
    
    # Colors for different commodities (excluding the main one)
    colors = ["#3f9abe", "#64a761", "#a7792f"]
    
    # === ADD MAIN SELECTED COMMODITY FIRST (as reference) ===
    cumulative_returns = (1 + main_portfolio['returns']).cumprod()
    performance_pct = (cumulative_returns - 1) * 100
    
    # Get main commodity name
    instruments = {
        'WTI Crude Oil': 'WTI',
        'Brent Crude Oil': 'BRENT', 
        'Gasoline (RBOB)': 'GASOLINE',
        'Heating Oil': 'HEATING_OIL',
        'Natural Gas': 'NATURAL_GAS',
        'Crack Spread US': 'CRACK_US',
        'Crack Spread EU': 'CRACK_EU'
    }
    
    commodity_names = {
        'WTI': 'WTI Crude',
        'BRENT': 'Brent Crude', 
        'GASOLINE': 'Gasoline',
        'HEATING_OIL': 'Heating Oil',
        'NATURAL_GAS': 'Natural Gas',
        'CRACK_US': 'Crack Spread US',
        'CRACK_EU': 'Crack Spread EU'
    }
    
    main_commodity_code = instruments.get(selected_instrument, selected_instrument)
    main_commodity_name = commodity_names.get(main_commodity_code, selected_instrument)
    
    # Add main commodity with special color
    fig.add_trace(go.Scatter(
        x=main_portfolio.index,
        y=performance_pct,
        mode='lines',
        name=f'{main_commodity_name} (Selected)',
        line=dict(color='#055c50', width=4),  # Thicker line for main commodity
        hovertemplate=f'{main_commodity_name}<br>Performance: %{{y:.2f}}%<extra></extra>'
    ))
    
    # === ADD COMPARISON COMMODITIES ===
    if comparison_commodities:
        for i, commodity_code in enumerate(comparison_commodities):
            try:
                # Load data for comparison commodity
                comp_data = load_commodity_data(
                    commodity_code,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if comp_data.empty:
                    continue
                    
                # Generate signals with same strategy
                comp_signals = strategy_obj.generate_signals(comp_data)
                
                # Run backtest
                comp_engine = BacktestEngineV2(comp_data, comp_signals)
                comp_portfolio = comp_engine.run()
                
                # Calculate equity curve
                comp_cumulative_returns = (1 + comp_portfolio['returns']).cumprod()
                comp_performance_pct = (comp_cumulative_returns - 1) * 100
                
                commodity_name = commodity_names.get(commodity_code, commodity_code)
                
                # Add to chart
                fig.add_trace(go.Scatter(
                    x=comp_portfolio.index,
                    y=comp_performance_pct,
                    mode='lines',
                    name=commodity_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'{commodity_name}<br>Performance: %{{y:.2f}}%<extra></extra>'
                ))
                
            except Exception as e:
                print(f"Error processing {commodity_code}: {e}")
                continue
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Strategy Performance Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title="Date",
        yaxis_title="Performance (%)",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        height=400,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

if __name__ == "__main__":
    main()