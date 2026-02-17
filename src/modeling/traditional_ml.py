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
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=22,
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
        min_child_weight=3,       # evita splits en nodos pequeños
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        gamma=0.1,                # pérdida mínima requerida para split
        random_state=22,
        n_jobs=-1,
        eval_metric='rmse',
        early_stopping_rounds=30 
    )
    
    print("Entrenando XGBoost...")
    
    model.fit(
        X_train,
        y_train,
        #eval_set=[(X_val, y_val)]
        eval_set=[(X_train, y_train), (X_val, y_val)]
        , verbose=60
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
        num_leaves=31,           # parámetro principal (default, funciona bien)
        max_depth=8,            # complementario a num_leaves
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,    # mínimo de muestras por hoja
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        random_state=22,
        n_jobs=-1,
        verbosity=1
    )
    
    print("Entrenando LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[lgbm.early_stopping(30), lgbm.log_evaluation(60)]
    )
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"Modelo guardado en: {save_path}")
    
    return model, train_pred, val_pred



import matplotlib.pyplot as plt
def plot_learning_curve(model, model_name='XGBoost'):
    
    if model_name == 'XGBoost':
        results = model.evals_result()
        metric = list(results['validation_0'].keys())[0]
        train_scores = results['validation_0'][metric]  # eval_set[0] → train
        val_scores   = results['validation_1'][metric]  # eval_set[1] → val

    elif model_name == 'LightGBM':
        results = model.evals_result_
        keys = list(results.keys())  # ['training', 'valid_1']
        metric = list(results[keys[0]].keys())[0]  # 'l2'
        train_scores = results[keys[0]][metric]   # 'training'
        val_scores   = results[keys[1]][metric]   # 'valid_1'
    # Si la métrica es l2 (MSE), convertir a RMSE para que sea comparable con XGBoost
    if metric == 'l2':
        train_scores = [np.sqrt(v) for v in train_scores]
        val_scores   = [np.sqrt(v) for v in val_scores]
        metric = 'rmse'

    best_iter = int(model.best_iteration_) if model_name == 'LightGBM' \
                else int(model.best_iteration)

    plt.figure(figsize=(10, 5))
    plt.plot(train_scores, label='Train', linewidth=2)
    plt.plot(val_scores,   label='Validation', linewidth=2)
    plt.axvline(x=best_iter, color='red', linestyle='--', 
                label=f'Mejor iteración ({best_iter})')
    plt.xlabel('Número de árboles')
    plt.ylabel(metric.upper())
    plt.title(f'Curva de aprendizaje - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()