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
def plot_learning_curve(model, model_name='Model'):
    """
    Grafica la curva de aprendizaje (train vs validation) durante el entrenamiento.
    
    Compatible con XGBoost y LightGBM.
    """
    
    # Detectar tipo de modelo
    model_type = type(model).__name__
    
    if 'XGB' in model_type:
        # XGBoost
        if not hasattr(model, 'evals_result'):
            print(f"{model_name}: No hay eval_results. Asegúrate de entrenar con eval_set.")
            return
        
        results = model.evals_result()
        
        # Detectar métrica (rmse, mae, etc.)
        # results tiene estructura: {'validation_0': {'rmse': [...]}, 'validation_1': {'rmse': [...]}}
        keys = list(results.keys())
        if len(keys) == 0:
            print(f"{model_name}: No hay métricas disponibles.")
            return
        
        # Tomar la primera métrica disponible
        first_key = keys[0]
        available_metrics = list(results[first_key].keys())
        
        if len(available_metrics) == 0:
            print(f"{model_name}: No hay métricas en eval_result.")
            return
        
        metric = available_metrics[0]  # ✅ Definir metric ANTES de usarla
        
        # Extraer scores
        if len(keys) >= 2:
            train_scores = results[keys[0]][metric]  # 'validation_0' (train)
            val_scores   = results[keys[1]][metric]  # 'validation_1' (val)
        else:
            # Solo hay un set (probablemente solo validation)
            train_scores = []
            val_scores   = results[keys[0]][metric]
        
        metric_label = metric.upper()
        
    elif 'LGBM' in model_type:
        # LightGBM
        if not hasattr(model, 'evals_result_'):
            print(f"{model_name}: No hay evals_result_. Asegúrate de entrenar con eval_set.")
            return
        
        results = model.evals_result_
        
        # results tiene estructura: {'training': {'l2': [...]}, 'valid_1': {'l2': [...]}}
        keys = list(results.keys())
        if len(keys) == 0:
            print(f"{model_name}: No hay métricas disponibles.")
            return
        
        # Detectar métrica
        first_key = keys[0]
        available_metrics = list(results[first_key].keys())
        
        if len(available_metrics) == 0:
            print(f"{model_name}: No hay métricas en evals_result_.")
            return
        
        metric = available_metrics[0]  # Definir metric ANTES de usarla
        
        # Extraer scores
        if len(keys) >= 2:
            train_scores = results[keys[0]][metric]  # 'training'
            val_scores   = results[keys[1]][metric]  # 'valid_1'
        else:
            train_scores = []
            val_scores   = results[keys[0]][metric]
        
        # Convertir l2 (MSE) a RMSE para comparabilidad
        if metric == 'l2':
            train_scores = [np.sqrt(v) for v in train_scores] if train_scores else []
            val_scores   = [np.sqrt(v) for v in val_scores]
            metric_label = 'RMSE'
        else:
            metric_label = metric.upper()
    
    else:
        print(f"{model_name}: Tipo de modelo no soportado para learning curves.")
        return
    
    # ==================
    # GRAFICAR
    # ==================
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(val_scores) + 1)
    
    if train_scores:
        plt.plot(epochs, train_scores, 'b-', label='Train', linewidth=2, alpha=0.8)
    plt.plot(epochs, val_scores, color='orange', linestyle='-', label='Validation', linewidth=2, alpha=0.8)
    
    # Marcar best iteration si existe
    if hasattr(model, 'best_iteration') and model.best_iteration > 0:
        best_iter = model.best_iteration
        best_score = val_scores[best_iter] if best_iter < len(val_scores) else val_scores[-1]
        plt.axvline(best_iter, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best Iteration ({best_iter})')
        plt.scatter([best_iter], [best_score], color='red', s=100, zorder=5)
    
    elif hasattr(model, 'best_iteration_') and model.best_iteration_ > 0:
        best_iter = model.best_iteration_
        best_score = val_scores[best_iter] if best_iter < len(val_scores) else val_scores[-1]
        plt.axvline(best_iter, color='red', linestyle='--', alpha=0.7,
                   label=f'Best Iteration ({best_iter})')
        plt.scatter([best_iter], [best_score], color='red', s=100, zorder=5)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(f'{metric_label}', fontsize=12)
    plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar
    from pathlib import Path
    fig_dir = Path('../reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = fig_dir / f'learning_curve_{model_name}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Learning curve guardada: {fig_path}")
    
    plt.show()