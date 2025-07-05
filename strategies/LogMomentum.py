import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class LogMomentumStrategy(BaseStrategy):
    def __init__(self, window=10, threshold=0.02, exit = 50):
        """
        Stratégie basée sur le log-momentum :
        log_momentum = log(price_t / price_{t-window})

        Un signal long est déclenché si le log-momentum > seuil,
        un signal short si log-momentum < -seuil.
        """
        self.window = window
        self.threshold = threshold
        self.exit = exit

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data['Close']

        price = data['Close']
        shifted_price = price.shift(self.window)
        log_momentum = pd.Series(np.nan, index=data.index)
        valid = (price > 0) & (shifted_price > 0)
        log_momentum[valid] = np.log(price[valid] / shifted_price[valid])
        signals['log_momentum'] = log_momentum
        signals['momentum_ma'] = log_momentum.rolling(window=self.window).mean()

        signals['signal'] = 0.0
        signals['positions'] = 0.0
        signals['exit_on_weakness'] = False  

        current_position = 0
        signal_count = 0
        weakness_exits = 0 
        exit_factor = self.exit/100
        exit_threshold = self.threshold * exit_factor 

        for i in range(self.window, len(signals)):
            lm = signals['log_momentum'].iloc[i]

            if np.isnan(lm):
                signals.iloc[i, signals.columns.get_loc('positions')] = current_position
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = False
                continue

            signal = 0
            is_weakness_exit = False
            
            if current_position == 0:  # Pas de position
                if lm > self.threshold:  # Signal d'achat
                    signal = 1
                    current_position = 1
                    signal_count += 1
                elif lm < -self.threshold:  # Signal de vente
                    signal = -1
                    current_position = -1
                    signal_count += 1
            
            elif current_position == 1:  # Position longue
                if lm < -self.threshold:  # Retournement vers short
                    signal = -2
                    current_position = -1
                    signal_count += 1
                elif lm < exit_threshold:  # Sortie sur affaiblissement
                    signal = -1
                    current_position = 0
                    is_weakness_exit = True
                    signal_count += 1
                    weakness_exits += 1
            
            elif current_position == -1:  # Position courte
                if lm > self.threshold:  # Retournement vers long
                    signal = 2
                    current_position = 1
                    signal_count += 1
                elif lm > -exit_threshold:  # Sortie sur affaiblissement
                    signal = 1
                    current_position = 0
                    is_weakness_exit = True
                    signal_count += 1
                    weakness_exits += 1
            
            signals.iloc[i, signals.columns.get_loc('signal')] = signal
            signals.iloc[i, signals.columns.get_loc('positions')] = current_position
            signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = is_weakness_exit
        
        return signals[['Close', 'positions', 'signal', 'log_momentum', 'momentum_ma', 'exit_on_weakness']]