fh = 24 # forecast horizon, predice el consumo dentro de fh horas

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import json
sys.path.append('../src')

results = []

# Cargar splits
data_dir = Path('../../data/processed')


train_df = pd.read_csv(data_dir / f'train_{fh}hr.csv', index_col='Datetime', parse_dates=True)
val_df = pd.read_csv(data_dir / f'val_{fh}hr.csv', index_col='Datetime', parse_dates=True)
test_df = pd.read_csv(data_dir / f'test_{fh}hr.csv', index_col='Datetime', parse_dates=True)

# Separar X e y
TARGET_COL = 'PJME_MW'
feature_cols = [col for col in train_df.columns if col != TARGET_COL]

X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}\n")

train_df.info()
# train_df.describe()
# train_df['PJME_MW'].describe()

import optuna
from xgboost import XGBRegressor
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
import numpy as np

# ==================
# XGBOOST OPTIMIZATION
# ==================
def objective_xgb(trial):
    """Función objetivo para Optuna - XGBoost"""
    
    params = {
        'n_estimators': 500,  # fijo, usamos early stopping
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': 22,
        'n_jobs': -1,
        'early_stopping_rounds': 30,
        'eval_metric': 'rmse'
    }
    
    model = XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

# Ejecutar optimización XGBoost
print("Optimizando XGBoost...")
study_xgb = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print("\n" + "="*70)
print("MEJORES PARÁMETROS - XGBOOST")
print("="*70)
print(f"Best RMSE: {study_xgb.best_value:.2f}")
print(f"Best params: {study_xgb.best_params}")

# ==================
# LIGHTGBM OPTIMIZATION
# ==================
def objective_lgbm(trial):
    """Función objetivo para Optuna - LightGBM"""
    
    params = {
        'n_estimators': 500,
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 22,
        'n_jobs': -1,
        'verbosity': -1
    }
    
    model = lgbm.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgbm.early_stopping(30, verbose=False)]
    )
    
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse

# Ejecutar optimización LightGBM
print("\nOptimizando LightGBM...")
study_lgbm = optuna.create_study(direction='minimize', study_name='lightgbm_optimization')
study_lgbm.optimize(objective_lgbm, n_trials=50, show_progress_bar=True)

print("\n" + "="*70)
print("MEJORES PARÁMETROS - LIGHTGBM")
print("="*70)
print(f"Best RMSE: {study_lgbm.best_value:.2f}")
print(f"Best params: {study_lgbm.best_params}")

# ==================
# ENTRENAR CON MEJORES PARÁMETROS
# ==================
print("\n" + "="*70)
print("ENTRENANDO CON MEJORES PARÁMETROS")
print("="*70)

# ==================
# XGBOOST OPTIMIZADO
# ==================
print("\n🔹 Entrenando XGBoost optimizado...")

# COMBINAR best_params con parámetros fijos
best_xgb_params = {
    **study_xgb.best_params,      # Parámetros optimizados
    'n_estimators': 500,           # ← Fijo (faltaba)
    'early_stopping_rounds': 30,   # ← Fijo (faltaba)
    'eval_metric': 'rmse',         # ← Fijo (faltaba)
    'random_state': 22,
    'n_jobs': -1
}

print("Parámetros XGBoost:")
for key, value in best_xgb_params.items():
    print(f"  {key}: {value}")

best_xgb = XGBRegressor(**best_xgb_params)
best_xgb.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)

print(f"XGBoost paró en iteración: {best_xgb.best_iteration}/{best_xgb_params['n_estimators']}")

# ==================
# LIGHTGBM OPTIMIZADO
# ==================
print("\n🔹 Entrenando LightGBM optimizado...")

# COMBINAR best_params con parámetros fijos
best_lgbm_params = {
    **study_lgbm.best_params,      # Parámetros optimizados
    'n_estimators': 500,           # ← Fijo (faltaba)
    'random_state': 22,
    'n_jobs': -1,
    'verbosity': 1
}

print("Parámetros LightGBM:")
for key, value in best_lgbm_params.items():
    print(f"  {key}: {value}")

best_lgbm = lgbm.LGBMRegressor(**best_lgbm_params)
best_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    callbacks=[
        lgbm.early_stopping(30, verbose=True),
        lgbm.log_evaluation(100)
    ]
)

print(f"LightGBM paró en iteración: {best_lgbm.best_iteration_}/{best_lgbm_params['n_estimators']}")

# ==================
# GUARDAR MODELOS
# ==================



models_dir = Path('../../models')
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_xgb, models_dir / f'xgboost_optimized_{fh}hr.pkl')
joblib.dump(best_lgbm, models_dir / f'lightgbm_optimized_{fh}hr.pkl')

print("\nModelos optimizados guardados")

# ==================
# GUARDAR PARÁMETROS
# ==================

optuna_results = {
    'forecast_horizon': fh,
    'n_trials': 50,
    'xgboost': {
        'best_rmse': float(study_xgb.best_value),
        'best_params': study_xgb.best_params,
        'best_iteration': int(best_xgb.best_iteration)
    },
    'lightgbm': {
        'best_rmse': float(study_lgbm.best_value),
        'best_params': study_lgbm.best_params,
        'best_iteration': int(best_lgbm.best_iteration_)
    }
}

with open(models_dir / f'optuna_results_{fh}hr.json', 'w') as f:
    json.dump(optuna_results, f, indent=2)

print("Resultados de optimización guardados")

