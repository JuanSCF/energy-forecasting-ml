import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class LagNaive(BaseEstimator, RegressorMixin):
    """
    Naive baseline usando el valor inmediatamente anterior (t-1).
    Para series diarias, lag_24 = mismo momento ayer.
    Para series semanales, lag_7 = mismo día semana pasada.

    Parameters
    ----------
    lag_column : str, default='y_lag_1'
        Nombre de la columna con el lag a usar
    """
    def __init__(self, lag_column="y_lag_1"):
        self.lag_column = lag_column
    
    def fit(self, X, y):
        if self.lag_column not in X.columns:
            raise ValueError(f"Columna '{self.lag_column}' no encontrada en X")
        return self
    
    def predict(self, X):
        if self.lag_column not in X.columns:
            raise ValueError(f"Columna '{self.lag_column}' no encontrada en X")
        return X[self.lag_column].values


class MovingAverageNaive(BaseEstimator, RegressorMixin):
    """
    Moving Average baseline: promedio de las últimas N observaciones.
    En la práctica no es tan útil porque se necesita crear muchas columnas lag_

    Parameters
    ----------
    window_columns : list of str
        Columnas de lags para promediar (ej: ['lag_1', 'lag_2', ..., 'lag_24'])
    
    Examples
    --------
    >>> window_cols = [f"lag_{i}" for i in range(1, 25)]
    >>> model = MovingAverageNaive(window_columns=window_cols)
    """
    def __init__(self, window_columns):
        self.window_columns = window_columns
    
    def fit(self, X, y):
        missing = [col for col in self.window_columns if col not in X.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        return self
    
    def predict(self, X):
        missing = [col for col in self.window_columns if col not in X.columns]
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        return X[self.window_columns].mean(axis=1).valuesy
    


class RollingMeanNaive(BaseEstimator, RegressorMixin):
    '''
    Usada para mostrar el promedio de las últimas 24 y 168 horas
    '''
    def __init__(self, rolling_column):
        self.rolling_column = rolling_column
    
    def fit(self, X, y):
        if self.rolling_column not in X.columns:
            raise ValueError(f"{self.rolling_column} no encontrada")
        return self
    
    def predict(self, X):
        if self.rolling_column not in X.columns:
            raise ValueError(f"{self.rolling_column} no encontrada")
        return X[self.rolling_column].values
