"""
DL_LSTM.py — LSTM Forecaster para consumo energético
=====================================================
Esquema: Input tabular (seq_len=1), igual que MLP. No procesa orden temporal explícitamente
Cada fila = un vector de 49 features → LSTM lo procesa como secuencia de longitud 1.

Uso desde notebook:
    from modeling.DL_LSTM import LSTMForecaster, get_dataloaders, build_model,
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
# 1. DATASET
# ─────────────────────────────────────────────
class EnergyDataset(Dataset):
    """
    Dataset tabular para LSTM opción A.
    Agrega dimensión de secuencia: (N, features) → (N, 1, features)
    El LSTM espera input de shape (batch, seq_len, input_size).
    Con seq_len=1, cada fila es tratada como una secuencia de un solo paso.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, features)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 2. SCALER + DATALOADERS
# ─────────────────────────────────────────────
def get_dataloaders(train_df, val_df, test_df, target_col='PJME_MW', batch_size=64):
    """
    Idéntica a DL_MLP — el reshape lo maneja EnergyDataset internamente.
    """
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)
    X_test_scaled  = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    train_ds = EnergyDataset(X_train_scaled, y_train_scaled)
    val_ds   = EnergyDataset(X_val_scaled,   y_val_scaled)
    test_ds  = EnergyDataset(X_test_scaled,  y_test_scaled)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Input size (n_features) : {X_train_scaled.shape[1]}")
    print(f"X shape en dataset      : (batch, seq_len=1, {X_train_scaled.shape[1]})")
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader, scaler_y, feature_cols


# ─────────────────────────────────────────────
# 3. ARQUITECTURA LSTM
# ─────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """
    LSTM para forecasting tabular (opción A, seq_len=1).

    Arquitectura:
        Input (batch, 1, input_size)
        → LSTM (num_layers, hidden_size)
        → último hidden state (batch, hidden_size)
        → Dropout
        → Linear → 1 valor

    Args:
        input_size  : número de features (49 en tu caso)
        hidden_size : neuronas en cada celda LSTM (dim del hidden state)
        num_layers  : capas LSTM apiladas
        dropout     : dropout entre capas LSTM (solo activo si num_layers > 1)
                      + dropout antes de la capa final
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
            dropout     = dropout if num_layers > 1 else 0.0,  # dropout solo entre capas
            batch_first = True,   # espera (batch, seq, features)
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
        # x: (batch, seq_len=1, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len=1, hidden_size)
        # Tomamos el último paso de la secuencia
        last_step = lstm_out[:, -1, :]          # (batch, hidden_size)
        out = self.dropout(last_step)
        out = self.fc(out)                       # (batch, 1)
        return out


def build_model(train_loader, hidden_size=64, num_layers=2, dropout=0.2):
    """
    Instancia LSTMForecaster infiriendo input_size del loader.
    """
    sample_X, _ = next(iter(train_loader))
    input_size  = sample_X.shape[2]  # (batch, seq_len, features) → dim 2

    model = LSTMForecaster(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Arquitectura : LSTM({input_size} → hidden={hidden_size}, layers={num_layers}) → 1")
    print(f"Parámetros   : {total_params:,}")
    print(f"Device       : {DEVICE}")

    return model


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO (idéntico a DL_MLP)
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
                save_path='../models/best_lstm.pt'):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history = {'train_loss': [], 'val_loss': []}
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
# 5. EVALUACIÓN (idéntica a DL_MLP)
# ─────────────────────────────────────────────
def evaluate_model(model, loader, scaler_y, set_name='Test'):
    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(DEVICE)).cpu().numpy()
            preds_list.append(preds)
            trues_list.append(y_batch.numpy())

    y_pred_scaled = np.concatenate(preds_list).flatten()
    y_true_scaled = np.concatenate(trues_list).flatten()

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    metrics = calculate_metrics(y_true, y_pred, set_name=set_name)
    print_metrics(metrics)

    return metrics, y_true, y_pred


# ─────────────────────────────────────────────
# 6. SERIALIZACIÓN
# ─────────────────────────────────────────────
def save_model(model, scaler_y, feature_cols, history=None, path='../models/dl_lstm.pt'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'model_config'     : {
            'input_size'  : model.lstm.input_size,
            'hidden_size' : model.lstm.hidden_size,
            'num_layers'  : model.lstm.num_layers,
            'dropout'     : model.dropout.p,
        },
        'scaler_y'     : scaler_y,
        'feature_cols' : feature_cols,
        'history'      : history,
    }, path)
    print(f"Modelo guardado en {path}")


def load_model(path='../models/dl_lstm.pt'):
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
    return model, checkpoint['scaler_y'], checkpoint['feature_cols'], checkpoint.get('history')