import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class NaiveForecaster(BaseEstimator, RegressorMixin):
    """
    Baseline más simple: predice el último valor observado.
    """
    def __init__(self, lag=1):
        self.lag = lag
        self.last_value_ = None
    
    def fit(self, X, y):
        """Guarda el último valor del training set"""
        self.last_value_ = y.iloc[-self.lag]
        return self
    
    def predict(self, X):
        """Predice el último valor para todas las observaciones"""
        return np.full(len(X), self.last_value_)


class MovingAverageForecaster(BaseEstimator, RegressorMixin):
    """
    Promedio móvil: predice el promedio de las últimas N observaciones.
    """
    def __init__(self, window=24):  # 24 horas por defecto
        self.window = window
        self.moving_avg_ = None
    
    def fit(self, X, y):
        """Calcula el promedio móvil del training set"""
        self.moving_avg_ = y.iloc[-self.window:].mean()
        return self
    
    def predict(self, X):
        """Predice el promedio móvil para todas las observaciones"""
        return np.full(len(X), self.moving_avg_)
    
