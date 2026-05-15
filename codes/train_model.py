from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data_ingestion import fetch_data
from label_generator import generate_labels
from feature_generation import features, ema_20, ema_50
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fetch data and signals
ticker = "NVDA"
df = fetch_data(ticker=ticker)

ema_signal = (ema_20(df) > ema_50(df)).astype(int)
labels = generate_labels(price_series=df["Close"], signal_series=ema_signal)
feat = features(df)

# Align features, labels, EMA signal
idx = feat.index.intersection(labels.index)
feat = feat.loc[idx].replace([np.inf, -np.inf], np.nan).dropna()
labels = labels.loc[feat.index].astype(int)
ema_signal = ema_signal.loc[feat.index]

# Train-test split (80-20)
train_end = int(0.6 * len(feat))
val_end = int(0.8 * len(feat))


X_train_df = feat.iloc[:train_end]
X_val_df = feat.iloc[train_end:val_end]
X_test_df = feat.iloc[val_end:]

y_train_df = labels.iloc[:train_end]
y_val_df = labels.iloc[train_end:val_end]
y_test_df = labels.iloc[val_end:]

ema_val = ema_signal.iloc[train_end:val_end]
ema_test = ema_signal.iloc[val_end:]


# Scale features
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train_df), dtype=torch.float32)
X_val = torch.tensor(scaler.transform(X_val_df), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test_df), dtype=torch.float32)

y_train = torch.tensor(y_train_df.values, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val_df.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_df.values, dtype=torch.float32).unsqueeze(1)

# DataLoader
dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

# MLP model


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer(x)


model = MLP(input_size=X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

# Training with early stopping
best_val_loss = float("inf")
patience = 10
counter = 0

for epoch in range(100):
    model.train()
    epoch_loss = 0.0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        val_loss = loss_fn(val_logits, y_val.to(device))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping at epoch", epoch)
        break
model.load_state_dict(best_model_state)

# Predict ML probabilities

model.eval()
with torch.no_grad():
    raw_logits = model(X_val.to(device)).cpu().numpy().flatten()

print(f"Logit range: {raw_logits.min():.4f} to {raw_logits.max():.4f}")
print(f"Logit std: {raw_logits.std():.4f}")


model.eval()
with torch.no_grad():
    train_prob = torch.sigmoid(
        model(X_train.to(device))).cpu().numpy().flatten()
    val_prob = torch.sigmoid(model(X_val.to(device))).cpu().numpy().flatten()
    test_prob = torch.sigmoid(model(X_test.to(device))).cpu().numpy().flatten()

val_prob_series = pd.Series(val_prob, index=X_val_df.index)
test_prob_series = pd.Series(test_prob, index=X_test_df.index)

# Compute returns
returns = df['Close'].pct_change().fillna(0)


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0


# Threshold optimization
best_threshold, best_sharpe = 0.5, -np.inf

for t in np.arange(0.3, 0.75, 0.005):
    val_filter = (val_prob_series > t).astype(int)
    val_filtered = ema_val * val_filter

    n_trades = val_filtered.sum()

    if n_trades < 10:
        continue

    val_returns = val_filtered * returns.loc[X_val_df.index]
    s = sharpe(val_returns)

    if s > best_sharpe:
        best_sharpe = s
        best_threshold = t

print(f"Best threshold : {best_threshold}, Best Sharpe : {best_sharpe}")

ml_filter_best = (test_prob_series > best_threshold).astype(int)
filtered_signal = ema_test * ml_filter_best

# Strategy returns
raw_returns = pd.Series(
    ema_test * returns.loc[X_test_df.index], index=X_test_df.index)
filtered_returns = pd.Series(
    filtered_signal * returns.loc[X_test_df.index], index=X_test_df.index)


# Drawdown Calculation


def max_drawdown(strategy_returns: pd.Series):
    anchored = pd.concat([pd.Series([0, 0]), strategy_returns])
    equity = (1 + anchored).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


raw_drawdown = max_drawdown(raw_returns)
filtered_drawdown = max_drawdown(filtered_returns)

# Metrics
print("Mean prob:", test_prob.mean())
print("Std prob:", test_prob.std())
print("Min prob:", test_prob.min())
print("Max prob:", test_prob.max())
print("Best Threshold:",  best_threshold)
print("Raw EMA Trades:", ema_test.sum())
print("Filtered Trades:", filtered_signal.sum())
print("Raw Sharpe:", sharpe(raw_returns))
print("Filtered Sharpe:", sharpe(filtered_returns))
print("Raw Drawdown :", raw_drawdown)
print("Max_drawdown :", filtered_drawdown)

# Accuracy Score

train_preds = (train_prob > 0.5).astype(int)
val_preds = (val_prob > 0.5).astype(int)
test_preds = (test_prob > 0.5).astype(int)

train_accuracy = accuracy_score(y_train_df.values, train_preds)
val_accuracy = accuracy_score(y_val_df.values, val_preds)
test_accuracy = accuracy_score(y_test_df.values, test_preds)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy : {val_accuracy:.4f}")
print(f"Test Accuracy:  {test_accuracy:.4f}")

# Cost

for cost in [0.0, 0.001, 0.002]:
    net_returns = filtered_returns - (filtered_signal.diff().abs() * cost/2)
    print(f"Cost : {cost}, Sharpe : {sharpe(net_returns)}")































