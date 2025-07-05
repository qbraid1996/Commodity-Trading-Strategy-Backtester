# strategies/mean_reversion.py

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, ma_window=20, threshold=2.0, exit_factor=50):
        """
        Stratégie de mean reversion générique.
        
        Parameters:
        -----------
        ma_window : int
            Fenêtre pour la moyenne mobile
        threshold : float
            Seuil de déclenchement en % (ex: 2.0 = ±2% de la MA)
        """
        self.ma_window = ma_window
        self.threshold = threshold
        self.exit_factor = exit_factor

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close']
        
        signals['ma'] = data['Close'].rolling(window=self.ma_window).mean()
        
        threshold_factor = self.threshold / 100
        exit_factor_p = self.exit_factor/100
        signals['upper_threshold'] = signals['ma'] * (1 + threshold_factor)
        signals['lower_threshold'] = signals['ma'] * (1 - threshold_factor)
        
        signals['signal'] = 0.0
        signals['positions'] = 0.0
        signals['exit_on_weakness'] = False 
        
        current_position = 0  
        signal_count = 0
        mean_reversion_exits = 0  

        exit_trigger = threshold_factor * exit_factor_p  
        
        for i in range(self.ma_window, len(signals)):
            price = signals['Close'].iloc[i]
            ma = signals['ma'].iloc[i]
            upper_thresh = signals['upper_threshold'].iloc[i]
            lower_thresh = signals['lower_threshold'].iloc[i]
            
            if np.isnan(ma) or np.isnan(price):
                signals.iloc[i, signals.columns.get_loc('positions')] = current_position
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = False
                continue
            
            signal = 0
            is_mean_reversion_exit = False
            
            if current_position == 0:  # Position plate
                if price < lower_thresh:  # Prix trop bas -> acheter (mean reversion)
                    signal = 1
                    current_position = 1
                    signal_count += 1
                elif price > upper_thresh:  # Prix trop haut -> vendre (mean reversion)
                    signal = -1
                    current_position = -1
                    signal_count += 1
            
            elif current_position == 1:  # Position longue
                if price > upper_thresh:  # Retournement : prix devient trop haut
                    signal = -2  # Retournement Long->Short
                    current_position = -1
                    signal_count += 1
                elif abs(price - ma) < abs(ma * exit_trigger):  # Retour vers moyenne (sortie)
                    signal = -1  # Fermer long
                    current_position = 0
                    is_mean_reversion_exit = True
                    signal_count += 1
                    mean_reversion_exits += 1
            
            elif current_position == -1:  # Position courte
                if price < lower_thresh:  # Retournement : prix devient trop bas
                    signal = 2  # Retournement Short->Long
                    current_position = 1
                    signal_count += 1
                elif abs(price - ma) < abs(ma * exit_trigger):  # Retour vers moyenne (sortie)
                    signal = 1  # Fermer short
                    current_position = 0
                    is_mean_reversion_exit = True
                    signal_count += 1
                    mean_reversion_exits += 1
            
            signals.iloc[i, signals.columns.get_loc('signal')] = signal
            signals.iloc[i, signals.columns.get_loc('positions')] = current_position
            signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = is_mean_reversion_exit
        
        return signals[['Close', 'positions', 'signal', 'ma', 'upper_threshold', 'lower_threshold', 'exit_on_weakness']]