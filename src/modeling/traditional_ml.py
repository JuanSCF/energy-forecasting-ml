import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import joblib
from pathlib import Path

def train_random_forest(X_train, y_train, X_val, y_val, save_path=None):
    """
    Random Forest con hiperparámetros razonables para forecasting.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Entrenando Random Forest...")
    model.fit(X_train, y_train)
    
    # Predicciones
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"Modelo guardado en: {save_path}")
    
    return model, train_pred, val_pred


def train_xgboost(X_train, y_train, X_val, y_val, save_path=None):
    """
    XGBoost optimizado para series temporales.
    """
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    
    print("Entrenando XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )
    
    print(f"Best iteration: {model.best_iteration}")
    
    train_pred = model.predict(X_train, iteration_range=(0, model.best_iteration))
    val_pred = model.predict(X_val, iteration_range=(0, model.best_iteration))
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"Modelo guardado en: {save_path}")
    
    return model, train_pred, val_pred


def train_lightgbm(X_train, y_train, X_val, y_val, save_path=None):
    """
    LightGBM - generalmente más rápido que XGBoost.
    """
    model = LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    print("Entrenando LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    print(f"Best iteration: {model.best_iteration}")
    
    train_pred = model.predict(X_train, iteration_range=(0, model.best_iteration))
    val_pred = model.predict(X_val, iteration_range=(0, model.best_iteration))
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"Modelo guardado en: {save_path}")
    
    return model, train_pred, val_pred