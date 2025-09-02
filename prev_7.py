import os #2.213
import gc
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
class Config:
    DATA_PATH = "/home/sac/67_data/Mihir/Sample_Data_For_LSTM.csv"
    SAVE_DIR = "./lstm_output"
    DEVICE_IDS = [0, 1, 2] if torch.cuda.device_count() >= 3 else list(range(torch.cuda.device_count()))
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 2016    # 7 days of 5-min data
    PRED_LEN = 288    # 1 day of 5-min data
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64  # Reduced complexity
    NUM_LAYERS = 2    # Reduced layers
    DROPOUT = 0.2
    LEARNING_RATE = 0.0001  # Reduced learning rate
    EPOCHS = 50
    PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6
    SEED = 42
    GRADIENT_CLIP = 1.0
    WARMUP_EPOCHS = 3

config = Config()
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

# ==================== HELPERS ====================
def parse_mixed_date(date_str):
    formats = ['%d-%m-%Y %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return pd.NaT

def preprocess_data(df):
    df['Time'] = df['Time'].apply(parse_mixed_date)
    df.dropna(subset=['Time'], inplace=True)
    df.sort_values('Time', inplace=True)
    df.set_index('Time', inplace=True)

    # Interpolate & fill
    df['value'] = df['value'].interpolate(method='time')
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Store original values for later denormalization
    df['original_value'] = df['value'].copy()
    
    # Normalize the target variable
    value_mean = df['value'].mean()
    value_std = df['value'].std()
    df['value'] = (df['value'] - value_mean) / value_std

    # Enhanced Time Features
    df['hour'] = df.index.hour
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['day_of_year'] = df.index.dayofyear
    
    # Cyclical encoding
    df['sin_of_day'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440.0)
    df['cos_of_day'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440.0)
    df['sin_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Progress features
    df['day_progress'] = df['minute_of_day'] / 1440.0  # Normalize to [0, 1]
    df['year_progress'] = df['day_of_year'] / 365.25   # Normalize to [0, 1]
    
    # Moving averages (on normalized values)
    df['ma_1h'] = df['value'].rolling(window=12).mean()
    df['ma_3h'] = df['value'].rolling(window=36).mean()
    df['ma_6h'] = df['value'].rolling(window=72).mean()
    
    # Lag features (on normalized values)
    df['lag_1h'] = df['value'].shift(12)
    df['lag_3h'] = df['value'].shift(36)
    df['lag_6h'] = df['value'].shift(72)
    
    df.reset_index(inplace=True)
    
    return df, value_mean, value_std


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states):
        # hidden_states: [B, seq_len, hidden_size]
        attention_weights = self.attention(hidden_states)  # [B, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # [B, seq_len, 1]
        attended = torch.sum(hidden_states * attention_weights, dim=1)  # [B, hidden_size]
        return attended, attention_weights

# ==================== MODEL ====================
class ImprovedLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len, dropout):
        super().__init__()
        self.pred_len = pred_len
        
        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Batch normalization after LSTM
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Dense layers with batch normalization and skip connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, pred_len)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if len(param.shape) >= 2:
                # Apply Xavier initialization only to weight matrices
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
            else:
                # For 1D parameters (biases and BatchNorm params)
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:  # BatchNorm weights
                    nn.init.constant_(param, 1.0)
                
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Apply batch norm to features
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.input_bn(x)
        x = x.transpose(1, 2)  # [batch, seq_len, features]
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]
        
        # Apply batch norm to LSTM output
        lstm_out_bn = lstm_out.transpose(1, 2)  # [batch, hidden*2, seq_len]
        lstm_out_bn = self.bn1(lstm_out_bn)
        lstm_out_bn = lstm_out_bn.transpose(1, 2)  # [batch, seq_len, hidden*2]
        
        # Apply attention
        context, _ = self.attention(lstm_out_bn)  # [batch, hidden_size*2]
        
        # Dense layers with skip connections
        out = self.fc1(context)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        preds = self.fc3(out)  # [batch, pred_len]
        return preds
# ==================== DATASET ====================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def create_sequences(data, feature_cols, target_col):
    print(f"Creating sequences from {len(data)} data points...")
    X, y = [], []
    values = data[feature_cols].values
    targets = data[target_col].values

    need = config.SEQ_LEN + config.PRED_LEN
    if len(data) < need:
        print(f"Warning: Data length ({len(data)}) is less than required length ({need})")
        return np.array([]), np.array([])

    for i in range(len(data) - config.SEQ_LEN - config.PRED_LEN + 1):
        X.append(values[i:i+config.SEQ_LEN])
        y.append(targets[i+config.SEQ_LEN:i+config.SEQ_LEN+config.PRED_LEN])

    X = np.array(X)
    y = np.array(y)
    print(f"Created {len(X)} sequences of shape X: {X.shape}, y: {y.shape}")
    return X, y

# ==================== TRAINING ====================
def weighted_mse_loss(pred, target):
    day_start = config.PRED_LEN // 4
    day_end = 3 * config.PRED_LEN // 4
    weights = torch.ones_like(target)
    weights[:, day_start:day_end] = 2.0
    return torch.mean(weights * (pred - target) ** 2)

def train(model, train_loader, val_loader, criterion, optimizer, scaler):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR
    )
    
    best_loss = float('inf')
    patience = 0
    train_losses, val_losses = [], []
    
    # Learning rate warmup
    warmup_factor = config.LEARNING_RATE / config.WARMUP_EPOCHS

    for epoch in range(1, config.EPOCHS + 1):
        # Warmup learning rate
        if epoch <= config.WARMUP_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_factor * epoch
        
        model.train()
        total_train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=torch.cuda.is_available()):
                out = model(xb)
                loss = criterion(out, yb)
                
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                with autocast(enabled=torch.cuda.is_available()):
                    out = model(xb)
                    loss = criterion(out, yb)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss - 1e-6:
            best_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(config.SAVE_DIR, 'best_model.pth'))
            print("Best model saved.")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print("Early stopping.")
                break
    
    return train_losses, val_losses

def predict_day(model, history_data, feature_cols, value_mean, value_std):
    """Predict one day using 7 days of history"""
    model.eval()
    with torch.no_grad():
        history = torch.FloatTensor(history_data[feature_cols].values).unsqueeze(0).to(config.DEVICE)
        prediction = model(history)
        # Denormalize the prediction
        prediction = prediction.cpu().numpy().flatten() * value_std + value_mean
        return prediction

def main():
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    print("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    
    # Call preprocess_data only once
    df, value_mean, value_std = preprocess_data(df)
    
    # Save normalization parameters
    np.savez(os.path.join(config.SAVE_DIR, 'norm_params.npz'),
             mean=value_mean, std=value_std)

    print(f"Total data points: {len(df)}")
    print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")

    # Verify all columns exist
    feature_cols = [
        'value',
        'sin_of_day', 'cos_of_day',
        'sin_of_year', 'cos_of_year',
        'day_progress', 'year_progress',
        'ma_1h', 'ma_3h', 'ma_6h',
        'lag_1h', 'lag_3h', 'lag_6h'
    ]

    print("Available columns:", df.columns.tolist())
    
    # Verify all required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Scale features (except 'value' which was already normalized in preprocess_data)
    scaler = StandardScaler()
    for col in feature_cols:
        if col != 'value':
            df[col] = scaler.fit_transform(df[[col]].values)

    # Ensure dates are datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Data splitting
    train_end = pd.Timestamp('2023-06-20')  # Changed from 23 to 20 to get more validation data
    val_end = pd.Timestamp('2023-07-01')

    train_df = df[df['Time'] < train_end]
    val_df = df[(df['Time'] >= train_end) & (df['Time'] < val_end)]

    print(f"Training data points: {len(train_df)}")
    print(f"Validation data points: {len(val_df)}")

    # Create sequences
    X_train, y_train = create_sequences(train_df, feature_cols, 'value')
    X_val, y_val = create_sequences(val_df, feature_cols, 'value')

    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")

    # Ensure minimum validation size
    if len(X_val) < 10:  # If we have very few validation sequences
        # Take some sequences from training for validation
        val_size = min(len(X_train) // 10, 1000)  # 10% or 1000 sequences, whichever is smaller
        indices = np.random.permutation(len(X_train))
        X_train, X_val_extra = X_train[indices[val_size:]], X_train[indices[:val_size]]
        y_train, y_val_extra = y_train[indices[val_size:]], y_train[indices[:val_size]]
        
        # Combine with existing validation data
        X_val = np.concatenate([X_val, X_val_extra]) if len(X_val) > 0 else X_val_extra
        y_val = np.concatenate([y_val, y_val_extra]) if len(y_val) > 0 else y_val_extra
        
        print(f"Updated validation set size: {len(X_val)}")

    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")

    if len(X_train) == 0:
        raise ValueError("No training sequences could be created. Not enough training data.")
    if len(X_val) == 0:
        raise ValueError("No validation sequences could be created. Not enough validation data.")

    val_batch_size = min(config.BATCH_SIZE, len(X_val))
    train_batch_size = config.BATCH_SIZE

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val),
        batch_size=val_batch_size,
        num_workers=0
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    model = ImprovedLSTMPredictor(
        input_size=len(feature_cols),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        pred_len=config.PRED_LEN,
        dropout=config.DROPOUT
    )
    if len(config.DEVICE_IDS) > 1:
        model = nn.DataParallel(model, device_ids=config.DEVICE_IDS)
    model = model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    print("Training model...")
    train_losses, val_losses = train(model, train_loader, val_loader,
                                   weighted_mse_loss, optimizer, scaler)

    # Load best model for predictions
    best_model_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for predictions.")

    # Predict July day by day
    print("Predicting July 2023...")
    july_predictions = []
    july_trues = []

    for day in range(1, 32):
        history_start = pd.Timestamp(f'2023-06-{24+day}')
        history_end = pd.Timestamp(f'2023-07-{day}')

        history_data = df[(df['Time'] >= history_start) &
                         (df['Time'] < history_end)].tail(config.SEQ_LEN)

        if len(history_data) < config.SEQ_LEN:
            print(f"Warning: Insufficient history data for {history_end.date()}")
            continue

        true_data = df[(df['Time'] >= history_end) &
                      (df['Time'] < history_end + pd.Timedelta(days=1))]

        # Predict using denormalization
        pred = predict_day(model, history_data, feature_cols, value_mean, value_std)

        # Store predictions and true values (use original_value for true values)
        july_predictions.append(pred)
        july_trues.append(true_data['original_value'].values[:config.PRED_LEN])

        # Plot daily prediction
        plt.figure(figsize=(12, 6))
        plt.plot(true_data['original_value'].values[:config.PRED_LEN],
                label='True', alpha=0.7)
        plt.plot(pred, label='Predicted', alpha=0.7)
        plt.title(f'Solar Prediction for {history_end.date()}')
        plt.xlabel('Time (5-minute intervals)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.SAVE_DIR, f'prediction_{history_end.date()}.png'))
        plt.close()
        print(f"Completed prediction for {history_end.date()}")

    # Plot full month comparison
    july_predictions = np.concatenate(july_predictions)
    july_trues = np.concatenate(july_trues)

    plt.figure(figsize=(20, 10))
    plt.plot(july_trues, label='True', alpha=0.7)
    plt.plot(july_predictions, label='Predicted', alpha=0.7)
    plt.title('July 2023 - Full Month Prediction vs True Values')
    plt.xlabel('Time (5-minute intervals)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.SAVE_DIR, 'july_2023_full_month.png'))
    plt.close()

    # Calculate and print metrics
    mse = np.mean((july_predictions - july_trues) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(july_predictions - july_trues))
    
    # Calculate R² score
    r2 = 1 - np.sum((july_trues - july_predictions) ** 2) / np.sum((july_trues - np.mean(july_trues)) ** 2)
    
    print(f"July Predictions Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Save metrics to file
    with open(os.path.join(config.SAVE_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Metrics for predictions ({datetime.now()}):\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")

if __name__ == '__main__':
    main()




import os #2.214
import gc 
import time
import warnings
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint

warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
class Config:
    # Current user and timestamp
    CURRENT_USER = "Kenmi1712"
    TIMESTAMP = "2025-09-02 11:01:30"

    DATA_PATH = "/home/sac/data_67/Mihir/Sample_Data_For_LSTM.csv"
    SAVE_DIR = "./lstm_output"
    SCALER_PATH = os.path.join(SAVE_DIR, "scaler.pkl")

    # GPU Configuration
    DEVICE_IDS = [0, 1] if torch.cuda.device_count() >= 3 else list(range(torch.cuda.device_count()))
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model Parameters
    SEQ_LEN = 2016    # 7 days history @ 5-min (288 points/day)
    PRED_LEN = 288    # 1 day forecast
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3

    # Training Parameters
    LEARNING_RATE = 0.001
    EPOCHS = 50
    PATIENCE = 10
    GRADIENT_ACCUMULATION_STEPS = 4
    GRADIENT_CLIP_VALUE = 1.0

    # Performance Optimizations
    USE_CHECKPOINTING = True
    PIN_MEMORY = True
    NUM_WORKERS = 4

    # Random Seed
    SEED = 42

config = Config()
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

# Enable CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==================== HELPERS ====================
def base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def save_state(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(base_model(model).state_dict(), path)

# Add these functions after the preprocess_data function and before create_sequences_for_day

def fit_scaler_on_train(train_df):
    """
    Fit MinMaxScaler on training data
    """
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    arr = train_df['value'].values.reshape(-1, 1)
    scaler.fit(arr)
    return scaler

def apply_scaler_to_df(df, scaler):
    """
    Apply fitted scaler to dataframe
    """
    df = df.copy()
    df['value_scaled'] = scaler.transform(df[['value']])
    return df

def inverse_transform_values(values, scaler):
    """
    Inverse transform scaled values
    """
    if len(values.shape) == 1:
        values = values.reshape(-1, 1)
    return scaler.inverse_transform(values).flatten()

def load_state(model, path, device):
    state = torch.load(path, map_location=device)
    base_model(model).load_state_dict(state)

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)

def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_metadata(path):
    """Save run metadata including user and timestamp"""
    metadata = {
        "user": config.CURRENT_USER,
        "timestamp": config.TIMESTAMP,
        "model_config": {k: v for k, v in vars(config).items() if not k.startswith('_')}
    }
    with open(os.path.join(path, 'metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=4)

# ==================== DATA PROCESSING ====================
def parse_mixed_date(date_str):
    formats = ['%d-%m-%Y %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except Exception:
        return pd.NaT

def preprocess_data(df):
    df = df.copy()
    df['Time'] = df['Time'].apply(parse_mixed_date)
    df.dropna(subset=['Time'], inplace=True)
    df.sort_values('Time', inplace=True)
    df.set_index('Time', inplace=True)

    # Interpolate missing values
    df['value'] = df['value'].interpolate(method='time')
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Time features
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month

    # Cyclical encodings
    df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
    df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)

    df.reset_index(inplace=True)
    return df

def create_training_sequences(df_scaled, feature_cols):
    """Create sequences for training using sliding window approach"""
    sequences = []
    targets = []
    
    # Convert Time column to datetime if it's not already
    df_scaled['Time'] = pd.to_datetime(df_scaled['Time'])
    
    # Print data info for debugging
    print(f"Total data points: {len(df_scaled)}")
    print(f"Date range: {df_scaled['Time'].min()} to {df_scaled['Time'].max()}")
    
    # We need at least SEQ_LEN + PRED_LEN points
    if len(df_scaled) < config.SEQ_LEN + config.PRED_LEN:
        print("Not enough data points for sequence creation")
        return None, None
    
    # Create sequences with sliding window
    for i in range(len(df_scaled) - config.SEQ_LEN - config.PRED_LEN + 1):
        # Get sequence
        seq_df = df_scaled.iloc[i:(i + config.SEQ_LEN)]
        # Get target
        target_df = df_scaled.iloc[(i + config.SEQ_LEN):(i + config.SEQ_LEN + config.PRED_LEN)]
        
        if len(seq_df) == config.SEQ_LEN and len(target_df) == config.PRED_LEN:
            seq = seq_df[feature_cols].values
            target = target_df['value_scaled'].values
            sequences.append(seq)
            targets.append(target)
    
    if not sequences:
        print("No valid sequences created")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

# ==================== DATASET ====================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== MODEL ====================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len, dropout):
        super().__init__()
        self.pred_len = pred_len
        
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper handling of different parameter types"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
            elif 'layer_norm' in name:
                if 'weight' in name:
                    param.data.fill_(1)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, x):
        x = self.layer_norm(x)
        
        if self.training and config.USE_CHECKPOINTING:
            out, _ = torch.utils.checkpoint.checkpoint(self.lstm, x)
        else:
            out, _ = self.lstm(x)
        
        last = out[:, -1, :]
        preds = self.fc(last)
        return preds

# ==================== TRAINING ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler_amp):
    best_loss = float('inf')
    patience = 0
    train_losses, val_losses = [], []
    step = 0

    for epoch in range(1, config.EPOCHS + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)

            with autocast(enabled=torch.cuda.is_available()):
                out = model(xb)
                loss = criterion(out, yb)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS

            scaler_amp.scale(loss).backward()

            if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)

                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            total_train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS

        # Validation phase
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
                with autocast(enabled=torch.cuda.is_available()):
                    out = model(xb)
                    loss = criterion(out, yb)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss - 1e-8:
            best_loss = avg_val_loss
            patience = 0
            save_state(model, os.path.join(config.SAVE_DIR, 'best_model.pth'))
            print(f"Epoch {epoch}: Best model saved (val_loss: {best_loss:.6f})")
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        print(f"Epoch {epoch}/{config.EPOCHS} ({time.time() - start_time:.2f}s)")
        print(f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        print(f"Patience: {patience}/{config.PATIENCE}")

    return train_losses, val_losses

# ==================== PREDICTION ====================
def predict_day(model, input_sequence, scaler):
    """Make prediction for a single day"""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(input_sequence).unsqueeze(0).to(config.DEVICE)
        with autocast(enabled=torch.cuda.is_available()):
            pred = model(x)
        pred = pred.cpu().numpy().flatten()
    return pred

def create_daily_predictions(model, df_scaled, scaler, feature_cols):
    """Generate predictions for all days in July"""
    predictions = {}

    for day in range(1, 32):  # July has 31 days
        date = f"2023-07-{day:02d}"
        X, y = create_sequences_for_day(df_scaled, feature_cols, date)

        if X is not None and y is not None:
            pred = predict_day(model, X, scaler)
            predictions[date] = {
                'prediction': pred,
                'true_values': y,
                'input_sequence': X
            }
            print(f"Generated prediction for {date}")
        else:
            print(f"Insufficient data for {date}")

    return predictions

# ==================== VISUALIZATION ====================
def plot_predictions(predictions, scaler, save_dir):
    """Create plots for all July predictions"""
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = os.path.join(save_dir, 'prediction_metrics.txt')

    with open(metrics_file, 'w') as f:
        f.write(f"Predictions generated by: {config.CURRENT_USER}\n")
        f.write(f"Generated on: {config.TIMESTAMP}\n\n")

    for date, data in predictions.items():
        pred = data['prediction']
        true = data['true_values']

        # Inverse transform
        pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        true_inv = scaler.inverse_transform(true.reshape(-1, 1)).flatten()

        # Create time points
        time_points = pd.date_range(
            start=pd.to_datetime(date),
            periods=len(true_inv),
            freq='5min'
        )

        # Plot
        plt.figure(figsize=(15, 7))
        plt.plot(time_points, true_inv, label='Actual', color='blue', alpha=0.7)
        plt.plot(time_points[:len(pred_inv)], pred_inv, label='Predicted', color='red', alpha=0.7)

        plt.title(f'Prediction vs Actual Values - {date}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f'prediction_{date}.png'))
        plt.close()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_inv, pred_inv[:len(true_inv)]))
        mae = mean_absolute_error(true_inv, pred_inv[:len(true_inv)])

        with open(metrics_file, 'a') as f:
            f.write(f"\nDate: {date}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write("-" * 50 + "\n")

# ==================== MAIN ====================
def main():
    print(f"Starting LSTM Time Series Prediction")
    print(f"User: {config.CURRENT_USER}")
    print(f"Timestamp: 2025-09-02 11:39:25")  # Updated timestamp
    print(f"Device: {config.DEVICE}")

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    save_metadata(config.SAVE_DIR)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = pd.read_csv(config.DATA_PATH)
    df = preprocess_data(df)

    # Print date range for verification
    print(f"Full data range: {df['Time'].min()} to {df['Time'].max()}")

    feature_cols = [
        'value_scaled', 'sin_hour', 'cos_hour',
        'sin_day', 'cos_day', 'sin_month', 'cos_month'
    ]

    # Prepare scaler using all data except the last month
    print("Preparing data scaling...")
    train_df = df[df['Time'] < '2024-07-01'].reset_index(drop=True)
    scaler = fit_scaler_on_train(train_df)
    save_scaler(scaler, config.SCALER_PATH)

    # Scale all data
    df_scaled = apply_scaler_to_df(df, scaler)
    
    # Split data - use data up to June 2024 for training
    print("\nSplitting data into train and validation sets...")
    train_data = df_scaled[df_scaled['Time'] < '2024-06-01'].copy()
    val_data = df_scaled[
        (df_scaled['Time'] >= '2024-06-01') & 
        (df_scaled['Time'] < '2024-07-01')
    ].copy()
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    # Create training sequences
    print("\nCreating training sequences...")
    X_train, y_train = create_training_sequences(train_data, feature_cols)

    if X_train is None or y_train is None:
        print("Error: Failed to create training sequences")
        return

    # Create validation sequences
    print("\nCreating validation sequences...")
    X_val, y_val = create_training_sequences(val_data, feature_cols)

    if X_val is None:
        print("Using part of training data for validation...")
        split_idx = int(len(X_train) * 0.9)
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

    print(f"Final training set shape: {X_train.shape}")
    print(f"Final validation set shape: {X_val.shape}")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Initialize model
    print("\nInitializing model...")
    model = LSTMPredictor(
        input_size=len(feature_cols),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        pred_len=config.PRED_LEN,
        dropout=config.DROPOUT
    )

    if len(config.DEVICE_IDS) > 1:
        print(f"Using DataParallel on devices: {config.DEVICE_IDS}")
        model = nn.DataParallel(model, device_ids=config.DEVICE_IDS)
    model = model.to(config.DEVICE)

    # Initialize training components
    print("\nSetting up training components...")
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    scaler_amp = GradScaler(enabled=torch.cuda.is_available())

    # Train model
    print("\nStarting model training...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        scaler_amp
    )

    # Save training history
    print("\nSaving training history...")
    plot_dir = os.path.join(config.SAVE_DIR, 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss_plot.png'))
    plt.close()

    # Generate predictions for July 2024
    print("\nGenerating predictions for July 2024...")
    predictions = {}
    
    # Modify prediction generation for July 2024
    for day in range(1, 32):
        date = f"2024-07-{day:02d}"
        X, y = create_sequences_for_day(df_scaled, feature_cols, date)
        
        if X is not None and y is not None:
            pred = predict_day(model, X, scaler)
            predictions[date] = {
                'prediction': pred,
                'true_values': y,
                'input_sequence': X
            }
            print(f"Generated prediction for {date}")
        else:
            print(f"Insufficient data for {date}")

    # Create prediction plots
    print("\nCreating prediction plots...")
    plot_dir = os.path.join(config.SAVE_DIR, 'july_2024_predictions')
    plot_predictions(predictions, scaler, plot_dir)

    print(f"\nPrediction plots and metrics saved to: {plot_dir}")
    print("Process completed successfully")

if __name__ == '__main__':
    main()
