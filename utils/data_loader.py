# data_loader.py

import yfinance as yf
import pandas as pd
from typing import Union, List, Dict, Optional
from utils.tickers import *

def download_data(tickers: Union[str, List[str]], start: str, end: str, 
                 columns: str = 'Close') -> pd.DataFrame:
    """
    Fonction unifiée pour télécharger des données depuis Yahoo Finance.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    df = yf.download(tickers, start=start, end=end, auto_adjust=False)
    
    # Corriger MultiIndex 
    if isinstance(df.columns, pd.MultiIndex):
        if len(tickers) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            if columns == 'Close':
                df = df['Close']
            elif columns == 'OHLCV':
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    if columns == 'Close' and len(tickers) == 1:
        df = df[['Close']]
    elif columns == 'OHLCV' and len(tickers) == 1:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    df = df.ffill().dropna()
    
    return df


def calculate_crack_spread(data: pd.DataFrame, ratio: str = 'US') -> pd.DataFrame:
    """
    Calcule le crack spread selon le ratio spécifié.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame avec les colonnes nécessaires selon le ratio
    ratio : str
        'US' pour 3:2:1 (WTI) ou 'EU' pour 2:1:1 (BRENT)
    """
    df = data.copy()
    
    if ratio.upper() == 'US':
        # Crack spread US : nécessite WTI, RBOB, HO
        required_cols = ['WTI', 'RBOB', 'HO']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes pour crack spread US: {missing_cols}")
        
        # Conversion et calcul
        df['RBOB_bbl'] = df['RBOB'] * 42
        df['HO_bbl'] = df['HO'] * 42
        df['crack_spread'] = 2 * df['RBOB_bbl'] + df['HO_bbl'] - 3 * df['WTI']
        
    elif ratio.upper() == 'EU':
        # Crack spread EU : nécessite BRENT, RBOB, HO
        required_cols = ['BRENT', 'RBOB', 'HO']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes pour crack spread EU: {missing_cols}")
        
        # Conversion et calcul
        df['RBOB_bbl'] = df['RBOB'] * 42
        df['HO_bbl'] = df['HO'] * 42
        df['crack_spread'] = 1 * df['RBOB_bbl'] + df['HO_bbl'] - 2 * df['BRENT']
        
    else:
        raise ValueError(f"Ratio '{ratio}' non supporté. Utilisez 'US' ou 'EU'.")
    
    return df


def load_commodity_data(instrument: str, start: str, end: str, **kwargs) -> pd.DataFrame:
    """
    Fonction principale pour charger différents types de données commodités.
    """
    
    # Tickers virtuels : CRACK SPREADS
    if instrument == CRACK_US:
        tickers = [WTI, RBOB, HO]
        data = download_data(tickers, start, end, columns='Close')
        data.columns = ['WTI', 'RBOB', 'HO']
        data = calculate_crack_spread(data, ratio='US')
        
        data['Open'] = data['crack_spread']
        data['High'] = data['crack_spread']
        data['Low'] = data['crack_spread']
        data['Close'] = data['crack_spread']
        data['Volume'] = 0
        
        return data

    elif instrument == CRACK_EU:
        tickers = [BRENT, RBOB, HO]
        data = download_data(tickers, start, end, columns='Close')
        data.columns = ['BRENT', 'RBOB', 'HO']
        data = calculate_crack_spread(data, ratio='EU')
        
        data['Open'] = data['crack_spread']
        data['High'] = data['crack_spread']
        data['Low'] = data['crack_spread']
        data['Close'] = data['crack_spread']
        data['Volume'] = 0
        
        return data
        
    else:
        ticker_map = {
            'WTI': WTI, 'BRENT': BRENT, 'GASOLINE': GASOLINE, 'RBOB': RBOB,
            'HEATING_OIL': HO, 'HO': HO, 'NATURAL_GAS': NG, 'NG': NG,
            'GOLD': GOLD, 'SILVER': SILVER
        }
        
        actual_ticker = ticker_map.get(instrument.upper(), instrument)
        columns = kwargs.get('columns', 'OHLCV')
        
        return download_data(actual_ticker, start, end, columns=columns)