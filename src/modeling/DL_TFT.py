"""
DL_TFT.py — Temporal Fusion Transformer para consumo energético
===============================================================
Usa pytorch-forecasting para implementar TFT.
Instalación: pip install pytorch-forecasting

Diferencias vs MLP/LSTM/TCN:
- Distingue features de futuro conocido vs pasado desconocido
- Predice múltiples cuantiles (P10, P50, P90)
- Alta interpretabilidad via Variable Selection Network
- API declarativa — el pipeline de datos es diferente

Uso desde notebook:
    from modeling.DL_TFT import prepare_data, build_model,
                                 train_model, evaluate_model,
                                 save_model, load_model
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
sys.path.append('../src')
from utils.metrics import calculate_metrics, print_metrics

# pytorch-forecasting imports
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
SEED = 22

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)
    print(f"Semilla fijada: {seed}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 1. FEATURES POR TIPO
# El TFT necesita saber qué features son conocidas en el futuro
# y cuáles solo están disponibles en el pasado
# ─────────────────────────────────────────────

# Futuro conocido: variables que conoces para t+24 en el momento de predecir
# (calendario, festivos — sabes qué hora/día será mañana)
KNOWN_FUTURE_FEATURES = [
    'hour', 'dayofweek', 'quarter', 'month', 'year',
    'dayofyear', 'weekofyear', 'is_weekend',
    'is_month_start', 'is_month_end',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'is_holiday', 'near_holiday',
]

# Pasado desconocido: lags y rolling features — solo disponibles hasta t
# no puedes saber el lag_24 de mañana sin predecirlo primero
UNKNOWN_PAST_FEATURES = [
    'lag_1', 'lag_2', 'lag_3',
    'lag_24', 'lag_25',
    'lag_168', 'lag_169',
    'rolling_mean_24', 'rolling_std_24',
    'rolling_min_24', 'rolling_max_24',
    'rolling_mean_168', 'rolling_std_168',
    'rolling_min_168', 'rolling_max_168',
    'diff_24h', 'diff_168h',
    'pct_change_24h', 'pct_change_168h',
    'volatility_std_24', 'volatility_std_168',
    'volatility_cv_24', 'volatility_cv_168',
    'volatility_range_24', 'volatility_range_168',
    'change_1h',
    'volatility_changes_24', 'volatility_changes_168',
    'volatility_iqr_24', 'volatility_iqr_168',
    'expanding_mean',
]


# ─────────────────────────────────────────────
# 2. PREPARACIÓN DE DATOS
# pytorch-forecasting necesita formato específico
# ─────────────────────────────────────────────
def prepare_data(train_df, val_df, test_df,
                 target_col='PJME_MW',
                 max_encoder_length=168,
                 max_prediction_length=24,
                 batch_size=64):
    """
    Convierte los DataFrames al formato que espera pytorch-forecasting.

    Cambios respecto a los otros modelos:
    1. Agrega columna 'time_idx' — índice entero incremental
    2. Agrega columna 'group_id' — identifica la serie ('PJME')
    3. Declara explícitamente qué features son futuro conocido vs pasado

    Args:
        max_encoder_length   : horas de historia que ve el modelo (seq_len)
        max_prediction_length: horizonte de predicción (24h)
        batch_size           : tamaño del batch

    Returns:
        train_dataset, val_dataset, test_dataset,
        train_loader, val_loader, test_loader
    """

    # Combinar para crear time_idx consistente
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    # time_idx: entero incremental desde 0
    # es el "reloj" del TFT — debe ser continuo
    all_df = pd.concat([train_df, val_df, test_df])
    all_df = all_df.sort_index()
    all_df['time_idx'] = np.arange(len(all_df))
    all_df['group_id'] = 'PJME'   # una sola serie

    # Recuperar splits con time_idx
    train_idx = all_df.index.isin(train_df.index)
    val_idx   = all_df.index.isin(val_df.index)
    test_idx  = all_df.index.isin(test_df.index)

    train_data = all_df[train_idx]
    val_data   = all_df[val_idx | train_idx]   # val necesita el contexto del train
    test_data  = all_df                         # test necesita contexto de todo

    # Verificar que las features declaradas existen en el DataFrame
    available_cols    = set(all_df.columns)
    known_available   = [f for f in KNOWN_FUTURE_FEATURES   if f in available_cols]
    unknown_available = [f for f in UNKNOWN_PAST_FEATURES   if f in available_cols]

    missing_known   = set(KNOWN_FUTURE_FEATURES)   - set(known_available)
    missing_unknown = set(UNKNOWN_PAST_FEATURES)   - set(unknown_available)
    if missing_known:
        print(f"⚠️  Features futuro no encontradas: {missing_known}")
    if missing_unknown:
        print(f"⚠️  Features pasado no encontradas: {missing_unknown}")

    print(f"Features futuro conocido : {len(known_available)}")
    print(f"Features pasado desconocido: {len(unknown_available)}")
    print(f"max_encoder_length : {max_encoder_length}h")
    print(f"max_prediction_length: {max_prediction_length}h")

    # TimeSeriesDataSet — formato nativo de pytorch-forecasting
    train_dataset = TimeSeriesDataSet(
        train_data,
        time_idx                  = 'time_idx',
        target                    = target_col,
        group_ids                 = ['group_id'],
        min_encoder_length        = max_encoder_length // 2,   # mínimo de historia
        max_encoder_length        = max_encoder_length,
        min_prediction_length     = max_prediction_length,
        max_prediction_length     = max_prediction_length,
        time_varying_known_reals  = known_available,
        time_varying_unknown_reals= [target_col] + unknown_available,
        target_normalizer         = GroupNormalizer(groups=['group_id']),
        add_relative_time_idx     = True,   # agrega índice temporal relativo como feature
        add_target_scales         = True,   # agrega media/std del target como features
        add_encoder_length        = True,   # agrega longitud del encoder como feature
    )

    # Val y test usan los parámetros del train dataset
    val_dataset  = TimeSeriesDataSet.from_dataset(train_dataset, val_data,  predict=False, stop_randomization=True)
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_data, predict=False,  stop_randomization=True)

    # DataLoaders
    train_loader = train_dataset.to_dataloader(train=True,  batch_size=batch_size, num_workers=12, persistent_workers = True)
    val_loader   = val_dataset.to_dataloader(train=False,   batch_size=batch_size, num_workers=12, persistent_workers = True)
    test_loader  = test_dataset.to_dataloader(train=False,  batch_size=batch_size, num_workers=12, persistent_workers = True)

    print(f"\nTrain samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 3. CONSTRUIR MODELO
# ─────────────────────────────────────────────
def build_model(train_dataset,
                hidden_size=64,
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=32,
                learning_rate=1e-3):
    """
    Instancia TFT a partir del train_dataset.
    Los parámetros de input se infieren automáticamente del dataset.

    Args:
        hidden_size            : dimensión del hidden state (LSTM interno + attention)
        attention_head_size    : número de cabezas de atención
        dropout                : tasa de dropout
        hidden_continuous_size : dimensión para procesar variables continuas
        learning_rate          : lr inicial

    Returns:
        model : TFT listo para entrenar
    """
    model = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate          = learning_rate,
        hidden_size            = hidden_size,
        attention_head_size    = attention_head_size,
        dropout                = dropout,
        hidden_continuous_size = hidden_continuous_size,
        loss                   = QuantileLoss(),          # predice P10, P50, P90
        optimizer              = 'adam',
        reduce_on_plateau_patience = 5,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales : {total_params:,}")
    print(f"Hidden size        : {hidden_size}")
    print(f"Attention heads    : {attention_head_size}")
    print(f"Device             : {DEVICE}")

    return model


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO
# pytorch-forecasting usa PyTorch Lightning
# el loop de entrenamiento es diferente a los otros modelos
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader,
                epochs=50, patience=5, min_delta=4.0,
                save_dir='../models'):
    """
    Entrena el TFT usando PyTorch Lightning.

    Diferencia vs otros modelos:
    - No hay loop manual de épocas
    - Lightning maneja early stopping y checkpoints automáticamente
    - El trainer devuelve el mejor modelo directamente

    Returns:
        trainer  : objeto Lightning con historial de entrenamiento
        best_path: ruta al mejor checkpoint
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Callbacks
    early_stop = EarlyStopping(
        monitor  = 'val_loss',
        patience = patience,
        mode     = 'min',
        verbose  = True,
        min_delta=min_delta
    )

    checkpoint = ModelCheckpoint(
        dirpath   = save_dir,
        filename  = 'dl_best_tft',
        monitor   = 'val_loss',
        mode      = 'min',
        save_top_k = 1,
    )

    # Trainer de Lightning
    trainer = pl.Trainer(
        max_epochs        = epochs,
        accelerator       = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices           = 1,
        gradient_clip_val = 0.1,
        callbacks         = [early_stop, checkpoint],
        enable_progress_bar = True,
        logger            = False,   # desactiva TensorBoard para simplicidad
    )

    trainer.fit(
        model,
        train_dataloaders = train_loader,
        val_dataloaders   = val_loader,
    )

    best_path = checkpoint.best_model_path
    print(f"\nMejor checkpoint: {best_path}")
    return trainer, best_path


# ─────────────────────────────────────────────
# 5. EVALUACIÓN
# ─────────────────────────────────────────────
def evaluate_model(model, test_loader, test_dataset, set_name='Test'):
    """
    Genera predicciones y calcula métricas.

    El TFT predice cuantiles [P10, P50, P90].
    Para comparar con otros modelos usamos P50 (mediana) como predicción puntual.

    Returns:
        metrics      : dict con MAE, RMSE, MAPE, R²
        y_true       : valores reales en MW
        y_pred_p50   : predicción puntual (mediana) en MW
        y_pred_p10   : límite inferior intervalo de confianza
        y_pred_p90   : límite superior intervalo de confianza
    """
    model.eval()

    # Predicciones — pytorch-forecasting devuelve dict con cuantiles
    predictions = model.predict(
        test_loader,
        mode           = 'quantiles',
        return_y     = True,
        trainer_kwargs = dict(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    )

    # predictions.output shape: (n_samples, prediction_length, n_quantiles)
    # cuantiles por defecto: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    # índice 3 = P50 (mediana)
    y_pred_all = predictions.output.cpu().numpy()  # (n, 24, 7)
    y_true_all = predictions.y[0].cpu().numpy()    # (n, 24)

    # Aplanar — tomamos el primer paso de predicción (t+1)
    # para comparar consistentemente con otros modelos que predicen t+24
    y_pred_p10 = y_pred_all[:, -1, 1]   # P10 — último paso del horizonte
    y_pred_p50 = y_pred_all[:, -1, 3]   # P50 — mediana
    y_pred_p90 = y_pred_all[:, -1, 5]   # P90
    y_true     = y_true_all[:, -1]

    metrics = calculate_metrics(y_true, y_pred_p50, set_name=set_name)
    print_metrics(metrics)

    print(f"\nIntervalo de confianza promedio (P10-P90):")
    print(f"  Ancho promedio: {(y_pred_p90 - y_pred_p10).mean():.2f} MW")

    return metrics, y_true, y_pred_p50, y_pred_p10, y_pred_p90


# ─────────────────────────────────────────────
# 6. INTERPRETABILIDAD
# Esta es la ventaja principal del TFT sobre LSTM y TCN
# ─────────────────────────────────────────────
def plot_feature_importance(model, test_loader):
    """
    Extrae y visualiza la importancia de variables del TFT.
    El Variable Selection Network produce estos pesos nativamente.
    """
    import matplotlib.pyplot as plt

    # Importancias nativas del TFT
    interpretation = model.interpret_output(
        model.predict(test_loader, mode='raw', return_x=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Encoder (pasado)
    encoder_importance = interpretation['encoder_variables'].mean(0)
    encoder_vars       = pd.Series(
        encoder_importance.numpy(),
        index = model.encoder_variables
    ).sort_values(ascending=True)

    encoder_vars.tail(15).plot(kind='barh', ax=axes[0])
    axes[0].set_title('TFT — Importancia variables pasado (encoder)')
    axes[0].set_xlabel('Importancia promedio')

    # Decoder (futuro conocido)
    decoder_importance = interpretation['decoder_variables'].mean(0)
    decoder_vars       = pd.Series(
        decoder_importance.numpy(),
        index = model.decoder_variables
    ).sort_values(ascending=True)

    decoder_vars.tail(15).plot(kind='barh', ax=axes[1])
    axes[1].set_title('TFT — Importancia variables futuro (decoder)')
    axes[1].set_xlabel('Importancia promedio')

    plt.tight_layout()
    plt.savefig('../reports/figures/dl_tft_feature_importance.png', dpi=150)
    plt.show()

    return encoder_vars, decoder_vars


# ─────────────────────────────────────────────
# 7. SERIALIZACIÓN
# ─────────────────────────────────────────────
def save_model(trainer, best_path, history=None, path='../models/dl_tft_final.pt'):
    """
    Copia el mejor checkpoint de Lightning a la ruta final.
    """
    import shutil
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_path, path)
    print(f"Modelo guardado en {path}")

    if history is not None:
        import json
        history_path = path.replace('.pt', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        print(f"History guardado en {history_path}")


def load_model(path='../models/dl_tft_final.pt'):
    """
    Carga el TFT desde checkpoint de Lightning.
    """
    model = TemporalFusionTransformer.load_from_checkpoint(path)
    model.eval()
    print(f"Modelo cargado desde {path}")
    return model