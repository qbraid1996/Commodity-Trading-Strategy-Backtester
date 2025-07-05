# strategies/moving_average.py

import pandas as pd
from .base_strategy import BaseStrategy
import numpy as np

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['short_ma'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_ma'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()

        signals['desired_position'] = 0.0
        signals.loc[signals['short_ma'] > signals['long_ma'], 'desired_position'] = 1.0
        signals.loc[signals['short_ma'] < signals['long_ma'], 'desired_position'] = -1.0

        signals['signal'] = signals['desired_position'].diff().fillna(0)
        
        signals['positions'] = signals['desired_position']

        return signals