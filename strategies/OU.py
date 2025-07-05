# strategies/OU.py

import pandas as pd
import numpy as np
from scipy import stats
from .base_strategy import BaseStrategy

class OUModel:
    def __init__(self):
        self.theta = None
        self.mu = None
        self.sigma = None
        self.fitted = False
    
    def fit(self, series, dt=1.0):
        """Estime les paramètres OU avec gestion d'erreurs robuste"""
        X = series.dropna().values
        
        if len(X) < 10:
            raise ValueError(f"Pas assez de données pour estimer OU: {len(X)} < 10")
        
        if np.any(np.isinf(X)):
            raise ValueError("Données contiennent des valeurs infinies")
        
        if np.std(X) == 0:
            raise ValueError("Série constante, pas de variation")
        
        dX = np.diff(X)
        X_lag = X[:-1]
        

        if np.var(dX) == 0:
            raise ValueError("Variance des différences nulle")
        
        try:
            A = np.vstack([np.ones(len(X_lag)), X_lag]).T
            coeffs, residuals, rank, s = np.linalg.lstsq(A, dX, rcond=None)
            
            if rank < 2:
                raise ValueError("Matrice singulière dans la régression")
            
            a, b = coeffs
            
            # Extraction des paramètres OU
            self.theta = -b / dt
            self.mu = a / (self.theta * dt) if abs(self.theta) > 1e-10 else np.mean(X)
            
            # Estimation de sigma
            residual_var = np.var(dX - (a + b * X_lag))
            if residual_var <= 0:
                raise ValueError("Variance résiduelle négative ou nulle")
            
            self.sigma = np.sqrt(residual_var / dt)
            
            if self.sigma <= 0:
                raise ValueError(f"Sigma invalide: {self.sigma}")
            
            if not np.isfinite(self.theta) or not np.isfinite(self.mu) or not np.isfinite(self.sigma):
                raise ValueError("Paramètres non finis")
            
            self.fitted = True
            return self
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Erreur algèbre linéaire: {e}")
    
    def fit_rolling(self, series, window=252, dt=1.0):
        """Estimation rolling avec gestion d'erreurs"""
        results = []
        
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            
            try:
                temp_model = OUModel()
                temp_model.fit(window_data, dt)
                
                results.append({
                    'date': series.index[i],
                    'theta': temp_model.theta,
                    'mu': temp_model.mu,
                    'sigma': temp_model.sigma
                })
            except Exception as e:
                if results:
                    last_params = results[-1].copy()
                    last_params['date'] = series.index[i]
                    results.append(last_params)
                else:
                    continue
        
        return pd.DataFrame(results).set_index('date')
    
    def calculate_zscore(self, value):
        if not self.fitted:
            raise ValueError("Modèle non ajusté")
        if self.sigma <= 0:
            raise ValueError(f"Sigma invalide: {self.sigma}")
        return (value - self.mu) / self.sigma
    
    def half_life(self):
        if not self.fitted or self.theta <= 0:
            return np.inf
        return np.log(2) / self.theta


class OrnsteinUhlenbeckStrategy(BaseStrategy):
    def __init__(self, 
                 z_entry=2.0,
                 z_exit=0.5,
                 z_stop=4.0,
                 z_reset=2,
                 rolling_window=50,
                 min_half_life=1,
                 max_half_life=252):
        
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.z_reset = z_reset
        self.rolling_window = rolling_window
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
        self.ou_model = OUModel()
        self.rolling_params = None
    
    def generate_signals(self, data):
        """Version avec tracking des sorties sur z-stop"""
        if 'Close' not in data.columns:
            raise ValueError("Colonne 'Close' manquante")
        
        price_series = data['Close']
        
        # Initialisation
        signals = pd.DataFrame(index=data.index)
        signals['Close'] = price_series
        signals['signal'] = 0.0
        signals['positions'] = 0.0
        signals['z_score'] = np.nan
        signals['theta'] = np.nan
        signals['mu'] = np.nan
        signals['sigma'] = np.nan
        signals['half_life'] = np.nan
        signals['z_stop_triggered'] = False
        signals['in_cooldown'] = False
        signals['exit_on_weakness'] = False  
        
        # === ESTIMATION DES PARAMÈTRES ===
        try:
            if self.rolling_window is None:

                self.ou_model.fit(price_series.dropna())
                half_life = self.ou_model.half_life()

                for i in range(len(signals)):
                    if not np.isnan(price_series.iloc[i]):
                        z_score = self.ou_model.calculate_zscore(price_series.iloc[i])
                        signals.iloc[i, signals.columns.get_loc('z_score')] = z_score
                        signals.iloc[i, signals.columns.get_loc('theta')] = self.ou_model.theta
                        signals.iloc[i, signals.columns.get_loc('mu')] = self.ou_model.mu
                        signals.iloc[i, signals.columns.get_loc('sigma')] = self.ou_model.sigma
                        signals.iloc[i, signals.columns.get_loc('half_life')] = half_life
            
            else:
                rolling_params = self.ou_model.fit_rolling(price_series, window=self.rolling_window)
                
                if rolling_params.empty:
                    raise ValueError("Aucun paramètre rolling calculé")
                
                self.rolling_params = rolling_params
                
                for date in rolling_params.index:
                    if date in signals.index:
                        idx = signals.index.get_loc(date)
                        
                        theta = rolling_params.loc[date, 'theta']
                        mu = rolling_params.loc[date, 'mu']
                        sigma = rolling_params.loc[date, 'sigma']
                        half_life = np.log(2) / theta if theta > 0 else np.inf
                        
                        if (self.min_half_life <= half_life <= self.max_half_life and 
                            not np.isnan(price_series.iloc[idx]) and
                            sigma > 0):
                            
                            z_score = (price_series.iloc[idx] - mu) / sigma
                            
                            signals.iloc[idx, signals.columns.get_loc('z_score')] = z_score
                            signals.iloc[idx, signals.columns.get_loc('theta')] = theta
                            signals.iloc[idx, signals.columns.get_loc('mu')] = mu
                            signals.iloc[idx, signals.columns.get_loc('sigma')] = sigma
                            signals.iloc[idx, signals.columns.get_loc('half_life')] = half_life
        
        except Exception as e:
            return signals
        
        valid_z_scores = signals['z_score'].notna().sum()
        if valid_z_scores == 0:
            return signals
        
        # === GÉNÉRATION DES SIGNAUX ET POSITIONS ===
        current_position = 0
        in_cooldown = False
        signal_count = 0
        z_stop_exits = 0  
        
        for i in range(len(signals)):
            z_score = signals['z_score'].iloc[i]
            
            if np.isnan(z_score):
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                signals.iloc[i, signals.columns.get_loc('positions')] = current_position
                signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = False
                continue
            
            signal = 0
            is_z_stop_exit = False
            
            # === Z-SCORE STOP : FERMER LA POSITION ===
            if abs(z_score) > self.z_stop:
                signals.iloc[i, signals.columns.get_loc('z_stop_triggered')] = True
                
                if current_position != 0:  # Si on a une position ouverte
                    # Fermer la position (aller à 0)
                    signal = -current_position  # Si long(1) -> signal(-1), si short(-1) -> signal(1)
                    current_position = 0  # Position fermée
                    in_cooldown = True
                    is_z_stop_exit = True
                    signal_count += 1
                    z_stop_exits += 1
            
            else:
                # Sortie de cooldown
                if in_cooldown and abs(z_score) < self.z_reset:
                    in_cooldown = False
                
                signals.iloc[i, signals.columns.get_loc('in_cooldown')] = in_cooldown
                
                # === LOGIQUE NORMALE DE TRADING ===
                should_buy = z_score < -self.z_entry
                should_sell = z_score > self.z_entry
                should_exit = abs(z_score) < self.z_exit
                
                if current_position == 0 and not in_cooldown:
                    if should_buy:
                        signal = 1  # Signal d'achat
                        current_position = 1  # Position longue
                        signal_count += 1
                    elif should_sell:
                        signal = -1  # Signal de vente
                        current_position = -1  # Position courte
                        signal_count += 1
                
                elif current_position == 1:  # Position longue
                    if should_exit:
                        signal = -1  # Signal de vente pour fermer
                        current_position = 0  # Position fermée
                        signal_count += 1
                    elif should_sell:
                        signal = -2  # Signal fort pour passer de long à short
                        current_position = -1  # Position courte
                        signal_count += 1
                
                elif current_position == -1:  # Position courte
                    if should_exit:
                        signal = 1  # Signal d'achat pour fermer
                        current_position = 0  # Position fermée
                        signal_count += 1
                    elif should_buy:
                        signal = 2  # Signal fort pour passer de short à long
                        current_position = 1  # Position longue
                        signal_count += 1
            
            signals.iloc[i, signals.columns.get_loc('signal')] = signal
            signals.iloc[i, signals.columns.get_loc('positions')] = current_position
            signals.iloc[i, signals.columns.get_loc('exit_on_weakness')] = is_z_stop_exit
        
        return signals[['Close', 'positions', 'signal', 'z_score', 'mu', 'theta', 'sigma', 'half_life', 'z_stop_triggered', 'in_cooldown', 'exit_on_weakness']]