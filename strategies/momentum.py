# strategies/momentum.py

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback=30, threshold=4, exit = 50):
        """
        Stratégie Momentum : prend position long/short si la variation du prix dépasse un seuil
        :param lookback: période de calcul du momentum (en jours)
        :param threshold: seuil (décimal, ex: 0.04 = 4%) pour déclencher une position
        """
        self.lookback = lookback
        self.threshold = threshold  
        self.exit = exit

    def generate_signals(self, data):
        """
        Génère des signaux basés sur le momentum avec tracking des sorties sur affaiblissement
        """
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close']
        
        signals['momentum'] = data['Close'].pct_change(periods=self.lookback)
        
        signals['signal'] = 0.0
        signals['positions'] = 0.0
        signals['exit_on_weakness'] = False 
        
        current_position = 0 
        signal_count = 0
        weakness_exits = 0 
        
        threshold_factor =  self.threshold/100
        exit_factor = self.exit/100

        long_weakness_threshold = threshold_factor * exit_factor  
        short_weakness_threshold = -threshold_factor * exit_factor
        
        for i in range(self.lookback, len(signals)):
            momentum = signals['momentum'].iloc[i]
            
            if np.isnan(momentum):
                signals.iloc[i, signals.columns.get_loc('positions')] = current_position
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = False
                continue
            
            signal = 0
            is_weakness_exit = False
            
            if current_position == 0:  # Position plate
                if momentum > threshold_factor:  # Momentum haussier fort
                    signal = 1
                    current_position = 1
                    signal_count += 1
                elif momentum < -threshold_factor:  # Momentum baissier fort
                    signal = -1
                    current_position = -1
                    signal_count += 1
            
            elif current_position == 1:  # Position longue
                if momentum < -threshold_factor:  # Retournement baissier
                    signal = -2
                    current_position = -1
                    signal_count += 1
                elif momentum < long_weakness_threshold:  # Sortie sur affaiblissement
                    signal = -1
                    current_position = 0
                    is_weakness_exit = True
                    signal_count += 1
                    weakness_exits += 1
            
            elif current_position == -1:  # Position courte
                if momentum > threshold_factor:  # Retournement haussier
                    signal = 2
                    current_position = 1
                    signal_count += 1
                elif momentum > short_weakness_threshold:  # Sortie sur affaiblissement
                    signal = 1
                    current_position = 0
                    is_weakness_exit = True
                    signal_count += 1
                    weakness_exits += 1
            
            signals.iloc[i, signals.columns.get_loc('signal')] = signal
            signals.iloc[i, signals.columns.get_loc('positions')] = current_position
            signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = is_weakness_exit
        
        return signals[['Close', 'positions', 'signal', 'momentum', 'exit_on_weakness']]