'''
Modelos avanzados para series temporales: SARIMAX y Prophet
'''

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from prophet import Prophet


# warnings.filterwarnings('ignore', category=ConvergenceWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)

class SARIMAXWrapper:
    '''
    SARIMAX = Seasonal AutoRegressive Integrated Moving Average with eXogenous variables

    Parameters
    ----------
    order: tuple (p,d,q)
        autoregressive, differencing, moving average
        default: (1,1,1)
    seasonal_order : tuple (P,D,Q,s)
        seasonal AR, seasonal diff, seasonal mov avg, seasonality
        s = 24 estacionalidad diaria
        s = 168 estacionalidad semanal
        default: (1,1,1,24)
    exog_cols : list of str
        nombres de columnas exógenas, variables externas que influyen
    '''

    def __init__(
        self,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        exog_cols=None,
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = exog_cols
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model = None
        self.results = None
        self.fitted = False
        self.train_index = None

    # -------------------------------------------------------------------------

    def _validate_index(self, y):
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("y debe tener DatetimeIndex")
        if y.index.freq is None:
            print("Advertencia: índice sin frecuencia explícita.")

    # -------------------------------------------------------------------------

    def _prepare_exog(self, X):
        if self.exog_cols is None:
            return None

        if X is None:
            raise ValueError("Se requieren variables exógenas")

        missing = [c for c in self.exog_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Columnas exógenas faltantes: {missing}")

        return X[self.exog_cols]

    # -------------------------------------------------------------------------

    def fit(self, y_train, X_train=None, maxiter=200, disp=False):
        print(f"Entrenando SARIMAX{self.order}x{self.seasonal_order}...")

        self._validate_index(y_train)
        exog_data = self._prepare_exog(X_train)

        self.model = SARIMAX(
            y_train,
            exog=exog_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        self.results = self.model.fit(maxiter=maxiter, disp=disp)
        self.fitted = True
        self.train_index = y_train.index

        print(f"Modelo entrenado | AIC: {self.results.aic:.2f}")
        return self

    # -------------------------------------------------------------------------
    # In-sample prediction

    def predict_in_sample(self):
        if not self.fitted:
            raise ValueError("Modelo no entrenado")

        fitted_vals = self.results.fittedvalues
        return fitted_vals

    # -------------------------------------------------------------------------
    # Forecast out-of-sample

    def forecast(self, steps, X_future=None):
        if not self.fitted:
            raise ValueError("Modelo no entrenado")

        exog_future = self._prepare_exog(X_future)

        forecast = self.results.forecast(steps=steps, exog=exog_future)
        return forecast

    # -------------------------------------------------------------------------

    def summary(self):
        if not self.fitted:
            raise ValueError("Modelo no entrenado")
        return self.results.summary()

    # -------------------------------------------------------------------------

    def save(self, filepath):
        if not self.fitted:
            raise ValueError("Modelo no entrenado")

        save_dict = {
            "results": self.results,
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "exog_cols": self.exog_cols,
            "trend": self.trend,
        }

        joblib.dump(save_dict, filepath)
        print(f"Modelo guardado en: {filepath}")

    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, filepath):
        save_dict = joblib.load(filepath)

        model = cls(
            order=save_dict["order"],
            seasonal_order=save_dict["seasonal_order"],
            exog_cols=save_dict["exog_cols"],
            trend=save_dict["trend"],
        )

        model.results = save_dict["results"]
        model.fitted = True

        print(f"Modelo cargado desde: {filepath}")
        return model


# =============================================================================
# PROPHET WRAPPER
# =============================================================================

class ProphetWrapper:
    """
    Wrapper limpio para Prophet.
    """

    def __init__(
        self,
        seasonality_mode="additive",
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    ):

        if Prophet is None:
            raise ImportError("Prophet no instalado. Ejecuta: pip install prophet")

        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale

        self.model = None
        self.regressors = []
        self.fitted = False

    # -------------------------------------------------------------------------

    def fit(self, df_train):
        if "ds" not in df_train.columns or "y" not in df_train.columns:
            raise ValueError("df_train debe tener columnas 'ds' y 'y'")

        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )

        # Detectar regresores adicionales
        extra_cols = [c for c in df_train.columns if c not in ["ds", "y"]]

        for col in extra_cols:
            self.model.add_regressor(col)
            self.regressors.append(col)

        self.model.fit(df_train)
        self.fitted = True

        print("Prophet entrenado")
        return self

    # -------------------------------------------------------------------------

    def predict(self, df_future):
        if not self.fitted:
            raise ValueError("Modelo no entrenado")

        forecast = self.model.predict(df_future)
        return forecast["yhat"]

    # -------------------------------------------------------------------------

    def predict_full(self, df_future):
        if not self.fitted:
            raise ValueError("Modelo no entrenado")

        return self.model.predict(df_future)

    # -------------------------------------------------------------------------

    def save(self, filepath):
        if not self.fitted:
            raise ValueError("Modelo no entrenado")

        save_dict = {
            "model": self.model,
            "seasonality_mode": self.seasonality_mode,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "regressors": self.regressors,
        }

        joblib.dump(save_dict, filepath)
        print(f"Modelo Prophet guardado en: {filepath}")

    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, filepath):
        save_dict = joblib.load(filepath)

        wrapper = cls(
            seasonality_mode=save_dict["seasonality_mode"],
            yearly_seasonality=save_dict["yearly_seasonality"],
            weekly_seasonality=save_dict["weekly_seasonality"],
            daily_seasonality=save_dict["daily_seasonality"],
        )

        wrapper.model = save_dict["model"]
        wrapper.regressors = save_dict["regressors"]
        wrapper.fitted = True

        print(f"Modelo Prophet cargado desde: {filepath}")
        return wrapper