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
df = fetch_data()
ema_signal = (ema_20() > ema_50()).astype(int)
labels = generate_labels(price_series=df["Close"], signal_series=ema_signal)
feat = features()

# Align features, labels, EMA signal
idx = feat.index.intersection(labels.index)
feat = feat.loc[idx].replace([np.inf, -np.inf], np.nan).dropna()
labels = labels.loc[feat.index].astype(int)
ema_signal = ema_signal.loc[feat.index]

# Train-test split (80-20)
split = int(0.8 * len(feat))
X_train_df, X_test_df = feat.iloc[:split], feat.iloc[split:]
y_train_df, y_test_df = labels.iloc[:split], labels.iloc[split:]
ema_test = ema_signal.iloc[split:]

# Scale features
scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train_df), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test_df), dtype=torch.float32)
y_train = torch.tensor(y_train_df.values, dtype=torch.float32).unsqueeze(1)
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
best_loss = float("inf")
patience = 5
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
    avg_loss = epoch_loss / len(dataloader)
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping at epoch", epoch)
        break

# Predict ML probabilities
model.eval()
with torch.no_grad():
    train_prob = torch.sigmoid(
        model(X_train.to(device))).cpu().numpy().flatten()
    test_prob = torch.sigmoid(model(X_test.to(device))).cpu().numpy().flatten()

test_prob_series = pd.Series(test_prob, index=X_test_df.index)

# Compute returns
returns = df['Close'].pct_change().fillna(0)

# Threshold optimization


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0


ml_filter_best = (test_prob_series > 0.44000000000000006).astype(int)
filtered_signal = ema_test * ml_filter_best

# Strategy returns
raw_returns = pd.Series(
    ema_test * returns.loc[X_test_df.index], index=X_test_df.index)
filtered_returns = pd.Series(
    filtered_signal * returns.loc[X_test_df.index], index=X_test_df.index)

# Cumulative returns
raw_cum = (raw_returns + 1).cumprod()
filtered_cum = (filtered_returns + 1).cumprod()

# Metrics
print("Mean prob:", test_prob.mean())
print("Std prob:", test_prob.std())
print("Min prob:", test_prob.min())
print("Max prob:", test_prob.max())
print("Best Threshold:",  0.44000000000000006)
print("Raw EMA Trades:", ema_test.sum())
print("Filtered Trades:", filtered_signal.sum())
print("Raw Sharpe:", sharpe(raw_returns))
print("Filtered Sharpe:", sharpe(filtered_returns))

# Accuracy Score

model.eval()
with torch.no_grad():
    # Train predictions
    train_probs = torch.sigmoid(
        model(X_train.to(device))).cpu().numpy().flatten()
    train_preds = (train_probs > 0.5).astype(int)
    train_acc = accuracy_score(y_train.cpu().numpy(), train_preds)

    # Test predictions
    test_probs = torch.sigmoid(
        model(X_test.to(device))).cpu().numpy().flatten()
    test_preds = (test_probs > 0.5).astype(int)
    test_acc = accuracy_score(y_test.cpu().numpy(), test_preds)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

# Plot cumulative returns
plt.plot(raw_cum.index, raw_cum, label="Raw EMA")
plt.plot(filtered_cum.index, filtered_cum, label="ML Filtered EMA")
plt.title("Cumulative Returns: Raw vs ML Filtered EMA")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()
