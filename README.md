Predicción de Demanda Energética

Descripción
Proyecto de regresión y forecasting para predecir la demanda energética (MWh) en diferentes regiones de Estados Unidos utilizando datos horarios. El proyecto compara múltiples enfoques de modelado, desde baselines estadísticos hasta redes neuronales recurrentes.
Objetivo: Predecir el consumo de energía con alta precisión para optimizar la generación, distribución y compra en mercados spot.


Problema de Negocio

Forecasting a corto plazo: Predecir consumo próximas 24-48 horas
Planificación operativa: Optimizar generación y compra de energía
Reducción de costos: Evitar excesos/déficits en producción
Estabilidad de red: Anticipar picos de demanda


Dataset

Fuente: [Hourly Energy Consumption - Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
Tamaño: ~145,000 registros horarios (2002-2018)
Región: PJME (Pennsylvania-New Jersey-Maryland Interconnection)
Granularidad: Datos por hora
Variable objetivo: Consumo en MWh


Características del dataset:

Multi-región (múltiples estados USA)
Estacionalidad clara (diaria, semanal, anual)
Patrones temporales ricos
in valores faltantes


Características
Modelos Implementados: Modelos Base y Tradicionales

Baselines:

Naive Lag (t-1, t-24, t-168)
Moving Average
Rolling Mean


Machine Learning Tradicional:

Random Forest Regressor
XGBoost
LightGBM



Modelos Avanzados (En desarrollo)

Modelos Estadísticos:

ARIMA con auto-tuning (pmdarima)
SARIMAX (con variables exógenas)
Facebook Prophet


Deep Learning:

LSTM (Long Short-Term Memory)
GRU (Gated Recurrent Unit)
LSTM con Atención



Feature Engineering
El proyecto incluye un pipeline completo de features con +50 variables:

Temporales: hour, day_of_week, month, year, quarter
Cíclicas: Codificación sin/cos para hora, día, mes
Calendarios: is_holiday, near_holiday, is_weekend, is_month_start/end
Lags: lag_24, lag_48, lag_168 (día anterior, 2 días, semana)
Estadísticas rolling: mean, std, min, max (ventanas 24h y 168h)
Diferencias: diff_24h, diff_168h, pct_change
Volatilidad: std, coefficient_variation, range, IQR, cambios
Expanding: expanding_mean (media acumulada)



API para Predicciones (Próximamente)