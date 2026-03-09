"""
DL_TCN.py — TCN Forecaster para consumo energético
====================================================
Temporal Convolutional Network con convoluciones dilatadas y residual connections.
Input: ventana deslizante (batch, seq_len, n_features)

Uso desde notebook:
    from modeling.DL_TCN import TCNForecaster, get_dataloaders, build_model,
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
    Ventana deslizante idéntica a DL_LSTM_seq.
    X.shape = (n_samples, seq_len, n_features)
    y.shape = (n_samples, 1)
    """
    def __init__(self, data: np.ndarray, target_idx: int, seq_len: int):
        self.seq_len    = seq_len
        self.target_idx = target_idx

        X_list, y_list = [], []
        for i in range(len(data) - seq_len):
            X_list.append(data[i : i + seq_len, :])
            y_list.append(data[i + seq_len, target_idx])

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 2. SCALER + DATALOADERS
# ─────────────────────────────────────────────
def get_dataloaders(train_df, val_df, test_df,
                    target_col='PJME_MW', seq_len=48, batch_size=64):
    """
    seq_len=48 por defecto — mejor tradeoff error/tiempo según experimentos LSTM.
    """
    feature_cols = [col for col in train_df.columns if col != target_col]
    all_cols     = [target_col] + feature_cols
    target_idx   = 0

    train_arr = train_df[all_cols].values
    val_arr   = val_df[all_cols].values
    test_arr  = test_df[all_cols].values

    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(train_arr)
    val_scaled   = scaler.transform(val_arr)
    test_scaled  = scaler.transform(test_arr)

    scaler_y_mean = scaler.mean_[target_idx]
    scaler_y_std  = scaler.scale_[target_idx]

    train_ds = SlidingWindowDataset(train_scaled, target_idx, seq_len)
    val_ds   = SlidingWindowDataset(val_scaled,   target_idx, seq_len)
    test_ds  = SlidingWindowDataset(test_scaled,  target_idx, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"seq_len         : {seq_len} horas")
    print(f"n_features      : {len(all_cols)}")
    print(f"Samples train   : {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"X shape ejemplo : {train_ds[0][0].shape}  → (seq_len, n_features)")

    return train_loader, val_loader, test_loader, scaler_y_mean, scaler_y_std, feature_cols


# ─────────────────────────────────────────────
# 3. ARQUITECTURA TCN
# ─────────────────────────────────────────────
class CausalConv1d(nn.Module):
    """
    Convolución 1D causal con dilatación.

    'Causal' significa que en el paso t solo usa información de t y pasos anteriores — nunca del futuro.
    Se logra agregando padding al inicio de la secuencia y recortando el exceso al final.

    Args:
        in_channels  : canales de entrada (= n_features o n_filters)
        out_channels : canales de salida (= n_filters)
        kernel_size  : tamaño del filtro convolucional
        dilation     : factor de dilatación (1, 2, 4, 8, ...)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        # Padding causal: agrega (kernel_size-1)*dilation ceros al inicio
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size  = kernel_size,
            dilation     = dilation,
            padding      = self.padding,
        )

        # WeightNorm estabiliza el entrenamiento en TCN
        # (alternativa a BatchNorm para secuencias)
        self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Recortar el padding agregado al inicio para mantener seq_len original
        return out[:, :, : -self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    """
    Bloque residual TCN con dos convoluciones causales dilatadas.

    Estructura:
        Input
          │
          ├─ CausalConv1d → ReLU → Dropout
          ├─ CausalConv1d → ReLU → Dropout
          │
          └─ Residual (1x1 Conv si in_channels ≠ out_channels)
          │
        Output = F(x) + x

    La residual connection permite que el gradiente fluya
    directamente desde output a input — evita vanishing gradient.

    Args:
        in_channels  : canales de entrada
        out_channels : canales de salida (= n_filters)
        kernel_size  : tamaño del filtro
        dilation     : factor de dilatación del bloque
        dropout      : tasa de dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        self.conv1   = CausalConv1d(in_channels,  out_channels, kernel_size, dilation)
        self.conv2   = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual: si los canales cambian, necesitamos proyectar x
        # para que tenga la misma dimensión que F(x)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.conv.weight_v, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.conv.weight_v, nonlinearity='relu')

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        # Residual connection
        return self.relu(out + self.residual(x))


class TCNForecaster(nn.Module):
    """
    TCN completo apilando TCNBlocks con dilaciones crecientes.

    Dilaciones: [1, 2, 4, 8, 16, ...] — cada bloque duplica la dilatación.
    Con kernel_size=3 y 5 bloques la ventana receptiva cubre:
        2 * (3-1) * (1+2+4+8+16) = 124 pasos

    Arquitectura:
        Input (batch, seq_len, n_features)
          → Transponer a (batch, n_features, seq_len)   ← Conv1d espera esto
          → TCNBlock(dilation=1)
          → TCNBlock(dilation=2)
          → TCNBlock(dilation=4)
          → ...
          → Último paso temporal [:, :, -1]
          → Linear → 1

    Args:
        input_size   : número de features (n_features)
        n_filters    : canales en cada bloque (ancho de la red)
        kernel_size  : tamaño del filtro convolucional
        n_blocks     : número de bloques TCN (dilaciones = 2^0, 2^1, ..., 2^(n-1))
        dropout      : tasa de dropout
    """
    def __init__(self, input_size: int, n_filters: int = 64,
                 kernel_size: int = 3, n_blocks: int = 5, dropout: float = 0.2):
        super().__init__()

        self.input_size = input_size
        self.n_filters  = n_filters
        self.kernel_size = kernel_size
        self.n_blocks   = n_blocks
        self.dropout_p  = dropout

        blocks     = []
        in_ch      = input_size

        for i in range(n_blocks):
            dilation = 2 ** i          # 1, 2, 4, 8, 16, ...
            blocks.append(TCNBlock(in_ch, n_filters, kernel_size, dilation, dropout))
            in_ch = n_filters          # después del primer bloque siempre n_filters

        self.tcn = nn.Sequential(*blocks)
        self.fc  = nn.Linear(n_filters, 1)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)          # → (batch, input_size, seq_len)
        out = self.tcn(x)              # → (batch, n_filters, seq_len)
        last = out[:, :, -1]           # → (batch, n_filters) — último paso temporal
        return self.fc(last)           # → (batch, 1)


def build_model(train_loader, n_filters=64, kernel_size=3, n_blocks=5, dropout=0.2):
    """
    Instancia TCNForecaster infiriendo input_size del loader.
    Calcula y muestra la ventana receptiva efectiva.
    """
    sample_X, _ = next(iter(train_loader))
    input_size  = sample_X.shape[2]    # (batch, seq_len, features) → dim 2
    seq_len     = sample_X.shape[1]

    # Ventana receptiva = 2 * (kernel_size - 1) * sum(dilations)
    receptive_field = 2 * (kernel_size - 1) * sum(2**i for i in range(n_blocks))

    model = TCNForecaster(
        input_size  = input_size,
        n_filters   = n_filters,
        kernel_size = kernel_size,
        n_blocks    = n_blocks,
        dropout     = dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Arquitectura     : TCN(features={input_size}, filters={n_filters}, kernel={kernel_size}, blocks={n_blocks})")
    print(f"Ventana receptiva: {receptive_field} pasos  (seq_len={seq_len})")
    print(f"Parámetros       : {total_params:,}")
    print(f"Device           : {DEVICE}")

    if receptive_field < seq_len:
        print(f"Ventana receptiva ({receptive_field}) < seq_len ({seq_len}) — considera más bloques")
    else:
        print(f"Ventana receptiva cubre toda la secuencia")

    return model


# ─────────────────────────────────────────────
# 4. ENTRENAMIENTO
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
                save_path='../models/best_tcn.pt'):
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
# 5. EVALUACIÓN (idéntica a DL_LSTM_seq)
# ─────────────────────────────────────────────
def evaluate_model(model, loader, scaler_y_mean, scaler_y_std, set_name='Test'):
    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(DEVICE)).cpu().numpy()
            preds_list.append(preds)
            trues_list.append(y_batch.numpy())

    y_pred_scaled = np.concatenate(preds_list).flatten()
    y_true_scaled = np.concatenate(trues_list).flatten()

    y_pred = y_pred_scaled * scaler_y_std + scaler_y_mean
    y_true = y_true_scaled * scaler_y_std + scaler_y_mean

    metrics = calculate_metrics(y_true, y_pred, set_name=set_name)
    print_metrics(metrics)

    return metrics, y_true, y_pred


# ─────────────────────────────────────────────
# 6. SERIALIZACIÓN
# ─────────────────────────────────────────────
def save_model(model, scaler_y_mean, scaler_y_std, feature_cols,
               history=None, path='../models/dl_tcn.pt'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'model_config'     : {
            'input_size'  : model.input_size,
            'n_filters'   : model.n_filters,
            'kernel_size' : model.kernel_size,
            'n_blocks'    : model.n_blocks,
            'dropout'     : model.dropout_p,
        },
        'scaler_y_mean' : scaler_y_mean,
        'scaler_y_std'  : scaler_y_std,
        'feature_cols'  : feature_cols,
        'history'       : history,
    }, path)
    print(f"Modelo guardado en {path}")


def load_model(path='../models/dl_tcn.pt'):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint['model_config']

    model = TCNForecaster(
        input_size  = cfg['input_size'],
        n_filters   = cfg['n_filters'],
        kernel_size = cfg['kernel_size'],
        n_blocks    = cfg['n_blocks'],
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