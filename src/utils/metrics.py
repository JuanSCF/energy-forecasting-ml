import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, set_name=""):
    """
    Calcula métricas de forecasting.
    
    Métricas incluidas:
    - MAE: Error absoluto medio (en MW)
    - RMSE: Raíz del error cuadrático medio
    - MAPE: Error porcentual absoluto medio
    - R²: Coeficiente de determinación
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'set': set_name
    }
    
    return metrics

def print_metrics(metrics):
    """Imprime métricas de forma legible"""
    print(f"\n{'='*50}")
    print(f"Métricas - {metrics['set']}")
    print(f"{'='*50}")
    print(f"MAE:  {metrics['MAE']:,.2f} MW")
    print(f"RMSE: {metrics['RMSE']:,.2f} MW")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"R²:   {metrics['R2']:.4f}")
    print(f"{'='*50}\n")