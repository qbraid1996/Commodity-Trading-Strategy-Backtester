import pandas as pd

class BacktestEngineV2:
    def __init__(self, data, signals, initial_capital=10000, capital_usage=0.95, transaction_cost=0.001):
        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.capital_usage = capital_usage
        self.transaction_cost = transaction_cost
        self.unit_per_trade = None
    
    def run(self):
        df = self.data.copy()
        signals = self.signals.copy()

        if 'positions' in signals.columns:
            df['target_positions'] = signals['positions']
        else:
            df['target_positions'] = signals['signal']  
            
        df['signal'] = signals['signal'] if 'signal' in signals.columns else 0
        df['price'] = df['Close']
        df['positions'] = 0.0
        df['returns'] = 0.0
        df['trades'] = 0  

        current_position = 0.0
        
        for i in range(len(df)):
            if df['target_positions'].iloc[i] != 0:
                price = df['price'].iloc[i]
                usable_capital = self.initial_capital * self.capital_usage
                self.unit_per_trade = int(usable_capital // (price * (1 + self.transaction_cost)))
                break
                
        if self.unit_per_trade is None or self.unit_per_trade == 0:
            raise ValueError("Impossible de déterminer unit_per_trade, capital trop faible ou pas de signal.")

        for i in range(len(df)):
            target_position = df['target_positions'].iloc[i]
            signal = df['signal'].iloc[i]
            
            if target_position != current_position:
                df.iloc[i, df.columns.get_loc('trades')] = 1
                current_position = target_position
            
            df.iloc[i, df.columns.get_loc('positions')] = current_position

        df['position_change'] = df['positions'].diff().fillna(0)
        df['holdings'] = df['positions'] * df['price'] * self.unit_per_trade
        
        transaction_costs = df['position_change'].abs() * df['price'] * self.unit_per_trade * self.transaction_cost
        position_cash_flow = df['position_change'] * df['price'] * self.unit_per_trade
        
        df['cash'] = self.initial_capital - (transaction_costs + position_cash_flow).cumsum()
        df['total'] = df['cash'] + df['holdings']
        df['returns'] = df['total'].pct_change().fillna(0)

        return df[['price', 'signal', 'positions', 'holdings', 'cash', 'total', 'returns', 'trades']].copy()