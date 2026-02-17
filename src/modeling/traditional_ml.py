import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgbm
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
    
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse',
        early_stopping_rounds=30
    )
    
    print("Entrenando XGBoost...")
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"Modelo guardado en: {save_path}")
    
    return model, train_pred, val_pred

def train_lightgbm(X_train, y_train, X_val, y_val, save_path=None):
    """
    LightGBM - generalmente más rápido que XGBoost.
    """
    model = lgbm.LGBMRegressor(
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
        callbacks=[lgbm.early_stopping(30), lgbm.log_evaluation(100)]
    )

    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"Modelo guardado en: {save_path}")
    
    return model, train_pred, val_pred