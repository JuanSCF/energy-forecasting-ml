"""
DL_MLP.py — MLP Forecaster para consumo energético
framework: PyTorch
====================================================
Uso desde notebook:
    from modeling.DL_MLP import MLPForecaster, get_dataloaders, train_model, evaluate_model, save_model, load_model
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
    Dataset tabular para MLP.
    Cada fila del DataFrame = un sample (X, y).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
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
    Recibe los DataFrames ya spliteados, escala y devuelve:
        - train_loader, val_loader, test_loader
        - scaler (para desnormalizar predicciones después)
        - feature_cols (para referencia)

    El scaler se fitea SOLO en train para evitar data leakage.
    """
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # Escalar features (fit solo en train)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled   = scaler_X.transform(X_val)
    X_test_scaled  = scaler_X.transform(X_test)

    # Escalar target (fit solo en train)
    # Lo guardamos para desnormalizar predicciones
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled   = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Datasets
    train_ds = EnergyDataset(X_train_scaled, y_train_scaled)
    val_ds   = EnergyDataset(X_val_scaled,   y_val_scaled)
    test_ds  = EnergyDataset(X_test_scaled,  y_test_scaled)

    # DataLoaders — shuffle=True solo en train
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Input size (n_features): {X_train_scaled.shape[1]}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, scaler_y, feature_cols


# ─────────────────────────────────────────────
# 3. ARQUITECTURA MLP
# ─────────────────────────────────────────────
class MLPForecaster(nn.Module):
    """
    MLP con BatchNorm + Dropout.
    El input_size se infiere automáticamente del primer batch.

    Args:
        input_size   : número de features (se obtiene de X_train.shape[1])
        hidden_sizes : lista con neuronas por capa, e.g. [128, 64, 32]
        dropout      : tasa de dropout entre capas ocultas
    """
    def __init__(self, input_size: int, hidden_sizes: list = [128, 64, 32], dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers += [
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))  # output: 1 valor (consumo t+24)

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


def build_model(train_loader, hidden_sizes=[128, 64, 32], dropout=0.2):
    """
    Instancia MLPForecaster infiriendo input_size del loader.
    Devuelve el modelo ya movido al device correcto.
    """
    # Inferir input_size del primer batch
    sample_X, _ = next(iter(train_loader))
    input_size = sample_X.shape[1]

    model = MLPForecaster(
        input_size   = input_size,
        hidden_sizes = hidden_sizes,
        dropout      = dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Arquitectura: {input_size} → {hidden_sizes} → 1")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Device: {DEVICE}")

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
                save_path='../models/best_mlp.pt'):
    """
    Entrena el modelo con early stopping y ReduceLROnPlateau.

    Args:
        patience : épocas sin mejora antes de detener
        save_path: dónde guardar el mejor modelo

    Returns:
        history : dict con listas 'train_loss' y 'val_loss'
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5#, verbose=True
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss   = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion)
        val_loss   = _val_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping + guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping en época {epoch} (sin mejora por {patience} épocas)")
            break

    print(f"\nMejor Val Loss: {best_val_loss:.4f} → guardado en {save_path}")
    return history


# ─────────────────────────────────────────────
# 5. EVALUACIÓN
# ─────────────────────────────────────────────
def evaluate_model(model, loader, scaler_y, set_name='Test'):
    """
    Genera predicciones sobre un loader y calcula métricas.
    Desnormaliza usando scaler_y antes de calcular métricas.

    Returns:
        metrics : dict con MAE, RMSE, MAPE, R²
        y_true  : array desnormalizado (MW reales)
        y_pred  : array desnormalizado (MW predichos)
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

    # Desnormalizar
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    metrics = calculate_metrics(y_true, y_pred, set_name=set_name)
    print_metrics(metrics)

    return metrics, y_true, y_pred


# ─────────────────────────────────────────────
# 6. SERIALIZACIÓN
# ─────────────────────────────────────────────
def save_model(model, scaler_y, feature_cols, history=None, path='../models/mlp_final.pt'):
    """Guarda modelo + scaler + feature_cols + history (opcional) en un solo archivo."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict' : model.state_dict(),
        'model_config'     : {
            'input_size'   : model.network[0].in_features,
            'hidden_sizes' : [l.out_features for l in model.network if isinstance(l, nn.Linear)][:-1],
            'dropout'      : next(l.p for l in model.network if isinstance(l, nn.Dropout)),
        },
        'scaler_y'     : scaler_y,
        'feature_cols' : feature_cols,
        'history'      : history,
    }, path)
    print(f"Modelo guardado en {path}")


def load_model(path='../models/dl_mlp.pt'):
    """Carga modelo + scaler + feature_cols + history desde archivo."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = checkpoint['model_config']
    model = MLPForecaster(
        input_size   = cfg['input_size'],
        hidden_sizes = cfg['hidden_sizes'],
        dropout      = cfg['dropout'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo cargado desde {path}")
    return model, checkpoint['scaler_y'], checkpoint['feature_cols'], checkpoint.get('history')