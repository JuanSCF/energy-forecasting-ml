"""
DL_LSTM_seq.py — LSTM Forecaster para consumo energético
=========================================================
Opción B: Input secuencial (sliding window).
El modelo recibe una ventana de N horas consecutivas como secuencia.
Shape: (batch, seq_len, n_features)

Diferencia vs Opción A:
    - Opción A: seq_len=1, cada fila es un vector de features
    - Opción B: seq_len=N, cada sample es una ventana de N horas

Uso desde notebook:
    from modeling.DL_LSTM_seq import LSTMForecaster, get_dataloaders, build_model,
                                      train_model, evaluate_model, save_model, load_model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
sys.path.append('../src')
from utils.metrics import calculate_metrics, print_metrics


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
SEED = 22

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Semilla fijada: {seed}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 1. DATASET — SLIDING WINDOW
# ─────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    """
    Convierte un DataFrame en pares (ventana, target) usando
    una ventana deslizante de tamaño seq_len.

    Ejemplo con seq_len=168 (1 semana):
        sample 0: filas [0:168]   → predice fila 168
        sample 1: filas [1:169]   → predice fila 169
        sample 2: filas [2:170]   → predice fila 170
        ...

    X.shape = (n_samples, seq_len, n_features)
    y.shape = (n_samples, 1)
    """

    def __init__(self, data: np.ndarray, target_idx: int, seq_len: int):
        # data      : array 2D (n_filas, n_features) — ya escalado
        # target_idx: índice de la columna objetivo en data
        # seq_len   : tamaño de la ventana histórica

        self.seq_len    = seq_len
        self.target_idx = target_idx

        X_list, y_list = [], []

        # Recorre el array creando ventanas
        for i in range(len(data) - seq_len):
            window = data[i : i + seq_len, :]          # (seq_len, n_features)
            target = data[i + seq_len, target_idx]     # escalar — el valor siguiente

            X_list.append(window)
            y_list.append(target)

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)  # (N, seq_len, features)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 2. SCALER + DATALOADERS
# ─────────────────────────────────────────────
def get_dataloaders(train_df, val_df, test_df,
                    target_col='PJME_MW', seq_len=168, batch_size=64):
    """
    Escala los datos y construye los DataLoaders con ventana deslizante.

    Args:
        seq_len : tamaño de la ventana histórica en horas
                  168 = 1 semana (recomendado para consumo energético)
                   24 = 1 día
                   48 = 2 días

    Nota sobre leakage:
        El scaler se fitea solo en train.
        La ventana usa SOLO datos pasados para predecir el siguiente paso —
        no hay leakage porque data[i+seq_len] es el futuro respecto a data[i:i+seq_len].
    """
    feature_cols = [col for col in train_df.columns if col != target_col]
    all_cols     = [target_col] + feature_cols          # target siempre en índice 0
    target_idx   = 0

    # Reordenar columnas para que target quede en índice 0
    train_arr = train_df[all_cols].values
    val_arr   = val_df[all_cols].values
    test_arr  = test_df[all_cols].values

    # Escalar todo junto (fit solo en train)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_arr)
    val_scaled   = scaler.transform(val_arr)
    test_scaled  = scaler.transform(test_arr)

    # Guardar scaler_y por separado para desnormalizar predicciones
    # Es el scaler de la columna 0 (target)
    scaler_y_mean = scaler.mean_[target_idx]
    scaler_y_std  = scaler.scale_[target_idx]

    # Datasets
    train_ds = SlidingWindowDataset(train_scaled, target_idx, seq_len)
    val_ds   = SlidingWindowDataset(val_scaled,   target_idx, seq_len)
    test_ds  = SlidingWindowDataset(test_scaled,  target_idx, seq_len)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"seq_len         : {seq_len} horas")
    print(f"n_features      : {len(all_cols)}")
    print(f"Samples train   : {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"X shape ejemplo : {train_ds[0][0].shape}  → (seq_len, n_features)")

    # Devolver también mean/std para desnormalizar
    return train_loader, val_loader, test_loader, scaler_y_mean, scaler_y_std, feature_cols


# ─────────────────────────────────────────────
# 3. ARQUITECTURA LSTM (idéntica a opción A)
# ─────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """
    LSTM para forecasting secuencial.
    La arquitectura es idéntica a la opción A —
    la diferencia está en el Dataset, no en el modelo.

    Con seq_len > 1 el LSTM ahora SÍ procesa una secuencia real,
    propagando el hidden state a través de N pasos temporales.
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # Solo usamos el último paso — resume toda la secuencia
        last_step = lstm_out[:, -1, :]      # (batch, hidden_size)
        out = self.dropout(last_step)
        out = self.fc(out)                  # (batch, 1)
        return out


def build_model(train_loader, hidden_size=64, num_layers=2, dropout=0.2):
    """
    Instancia LSTMForecaster infiriendo input_size del loader.
    """
    sample_X, _ = next(iter(train_loader))
    input_size  = sample_X.shape[2]         # (batch, seq_len, features) → dim 2

    model = LSTMForecaster(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    seq_len      = sample_X.shape[1]
    print(f"Arquitectura : LSTM(seq={seq_len}, features={input_size} → hidden={hidden_size}, layers={num_layers}) → 1")
    print(f"Parámetros   : {total_params:,}")
    print(f"Device       : {DEVICE}")

    return model


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO (idéntico a opción A)
# ─────────────────────────────────────────────
def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
    return total_loss / len(loader)


def train_model(model, train_loader, val_loader,
                epochs=50, lr=1e-3, patience=10,
                save_path='../models/best_lstm_seq.pt'):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history           = {'train_loss': [], 'val_loss': []}
    best_val_loss     = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion)
        val_loss   = _val_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f} | LR: {current_lr:.2e}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping en época {epoch} (sin mejora por {patience} épocas)")
            break

    print(f"\nMejor Val Loss: {best_val_loss:.4f} → guardado en {save_path}")
    return history


# ─────────────────────────────────────────────
# 5. EVALUACIÓN
# ─────────────────────────────────────────────
def evaluate_model(model, loader, scaler_y_mean, scaler_y_std, set_name='Test'):
    """
    Desnormaliza usando mean y std del target directamente.
    En opción B no hay scaler_y separado — se extrajo del scaler global.
    """
    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(DEVICE)).cpu().numpy()
            preds_list.append(preds)
            trues_list.append(y_batch.numpy())

    y_pred_scaled = np.concatenate(preds_list).flatten()
    y_true_scaled = np.concatenate(trues_list).flatten()

    # Desnormalizar: x_original = x_scaled * std + mean
    y_pred = y_pred_scaled * scaler_y_std + scaler_y_mean
    y_true = y_true_scaled * scaler_y_std + scaler_y_mean

    metrics = calculate_metrics(y_true, y_pred, set_name=set_name)
    print_metrics(metrics)

    return metrics, y_true, y_pred


# ─────────────────────────────────────────────
# 6. SERIALIZACIÓN
# ─────────────────────────────────────────────
def save_model(model, scaler_y_mean, scaler_y_std, feature_cols,
               history=None, path='../models/dl_lstm_seq.pt'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'model_config'     : {
            'input_size'  : model.lstm.input_size,
            'hidden_size' : model.lstm.hidden_size,
            'num_layers'  : model.lstm.num_layers,
            'dropout'     : model.dropout.p,
        },
        'scaler_y_mean' : scaler_y_mean,
        'scaler_y_std'  : scaler_y_std,
        'feature_cols'  : feature_cols,
        'history'       : history,
    }, path)
    print(f"Modelo guardado en {path}")


def load_model(path='../models/dl_lstm_seq.pt'):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint['model_config']

    model = LSTMForecaster(
        input_size  = cfg['input_size'],
        hidden_size = cfg['hidden_size'],
        num_layers  = cfg['num_layers'],
        dropout     = cfg['dropout'],
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo cargado desde {path}")

    return (model,
            checkpoint['scaler_y_mean'],
            checkpoint['scaler_y_std'],
            checkpoint['feature_cols'],
            checkpoint.get('history'))