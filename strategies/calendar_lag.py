# strategies/calendar_lag.py

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class CalendarLagStrategy(BaseStrategy):
    def __init__(self, lag_days=30, threshold=2.0, exit = 50):
        """
        Stratégie de calendar spread avec mean reversion.
        
        Parameters:
        -----------
        lag_days : int
            Nombre de jours de décalage pour le spread
        threshold : float
            Seuil de déclenchement en % de la MA du spread (ex: 2.0 = ±2% de la MA)
        """
        self.lag_days = lag_days
        self.threshold = threshold
        self.exit_factor = exit

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close'] 

        signals['spread'] = data['Close'] - data['Close'].shift(self.lag_days)
        signals['spread_ma'] = signals['spread'].rolling(window=self.lag_days).mean()

        spread_std = signals['spread'].rolling(window=self.lag_days).std()
        signals['spread_zscore'] = (signals['spread'] - signals['spread_ma']) / spread_std
        
        threshold_zscore = self.threshold  

        signals['signal'] = 0.0
        signals['positions'] = 0.0
        signals['exit_on_weakness'] = False
        
        current_position = 0  
        signal_count = 0
        mean_reversion_exits = 0
        
        exit_f = self.exit_factor/100
        exit_threshold = threshold_zscore * exit_f
        
        for i in range(self.lag_days * 2, len(signals)):  
            spread_z = signals['spread_zscore'].iloc[i]
            
            if np.isnan(spread_z):
                signals.iloc[i, signals.columns.get_loc('positions')] = current_position
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = False
                continue
            
            signal = 0
            is_mean_reversion_exit = False
            
            if current_position == 0:  # Position plate
                if spread_z > threshold_zscore:  # Spread trop élevé -> short
                    signal = -1
                    current_position = -1
                    signal_count += 1
                elif spread_z < -threshold_zscore:  # Spread trop bas -> long
                    signal = 1
                    current_position = 1
                    signal_count += 1
            
            elif current_position == 1:  # Long spread
                if spread_z > threshold_zscore:  # Retournement
                    signal = -2  # Long->Short
                    current_position = -1
                    signal_count += 1
                elif abs(spread_z) < exit_threshold:  # Retour vers moyenne
                    signal = -1  # Fermer long
                    current_position = 0
                    is_mean_reversion_exit = True
                    signal_count += 1
                    mean_reversion_exits += 1
            
            elif current_position == -1:  # Short spread
                if spread_z < -threshold_zscore:  # Retournement
                    signal = 2  # Short->Long
                    current_position = 1
                    signal_count += 1
                elif abs(spread_z) < exit_threshold:  # Retour vers moyenne
                    signal = 1  # Fermer short
                    current_position = 0
                    is_mean_reversion_exit = True
                    signal_count += 1
                    mean_reversion_exits += 1
            
            signals.iloc[i, signals.columns.get_loc('signal')] = signal
            signals.iloc[i, signals.columns.get_loc('positions')] = current_position
            signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = is_mean_reversion_exit
        
        return signals[['Close', 'positions', 'signal', 'spread', 'spread_ma', 'spread_zscore', 'exit_on_weakness']]