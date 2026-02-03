# 📘 Machine-Learning Filtered Momentum Strategy  
### A Quantitative Research Study on Signal Selectivity

---

## 1. Research Objective

This project investigates a focused and realistic question in systematic trading:

> **Can machine learning improve a classical momentum strategy by filtering low-quality trades, without directly generating alpha?**

Rather than predicting prices, the model learns **when not to trade**.  
This mirrors how machine learning is actually deployed in professional quantitative research.

---

## 2. Conceptual Framework

### Core Principle

Let:
- \( S_t \in \{0,1\} \): a deterministic momentum signal (EMA crossover)
- \( Y_t \in \{0,1\} \): trade success label
- \( X_t \): feature vector describing market conditions
- \( f(X_t) = P(Y_t = 1 \mid X_t) \): ML-estimated trade quality

A trade is executed **only if**:

\[
S_t = 1 \quad \text{and} \quad f(X_t) > \tau
\]

where \( \tau \) is a confidence threshold.

**Machine learning does not create alpha.  
It filters noise from an existing anomaly.**

---

## 3. Data Description

### Market Data
- OHLCV time-series
- Daily frequency
- Single asset (architecture supports extension)

### Data Integrity Rules
- No forward-looking features
- Rolling statistics are shifted
- Labels are aligned strictly after feature computation
- Time-ordered train/test split

---


---

## 4. File-by-File Explanation

---

## 📂 `data_ingestion.py`

### Purpose
Load and standardize raw OHLCV market data.

### Logic
- Reads price data
- Enforces datetime index
- Sorts chronologically
- No transformations applied

This file defines the **base probability space** for all subsequent analysis.

---

## 📂 `feature_generation.py`

### Purpose
Construct causal, interpretable features describing **trade environment quality**.

### Features and Mathematics

#### 1️⃣ Volatility (20-day)
\[
\sigma_t = \sqrt{\frac{1}{20} \sum_{i=1}^{20} (r_{t-i} - \bar{r})^2}
\]

Measures noise and regime instability.

---

#### 2️⃣ Relative Strength Index (RSI-14)
\[
RSI = 100 - \frac{100}{1 + RS}
\]

Captures momentum exhaustion rather than direction.

---

#### 3️⃣ Average True Range (ATR-14)
\[
ATR_t = EMA_{14}(\max(H-L, |H-C_{prev}|, |L-C_{prev}|))
\]

Measures volatility expansion and stop-loss risk.

---

#### 4️⃣ Volume Z-Score (252-day)
\[
Z_t = \frac{V_t - \mu_{252}}{\sigma_{252}}
\]

Identifies abnormal participation and regime shifts.

---

### Design Constraints
- All features are shifted by one period
- No price-level leakage
- No overlap with label construction

---

## 📂 `label_generator.py`

### Purpose
Define economically meaningful supervision targets.

### Label Definition

A trade entered at \( t+1 \) is successful if:

\[
\frac{P_{t+h+1} - P_{t+1}}{P_{t+1}} > c
\]

Where:
- \( h \): holding horizon
- \( c \): transaction cost

### Conditional Labeling

Labels are generated **only when the momentum signal is active**:

\[
Y_t =
\begin{cases}
1 & \text{if trade succeeds and } S_t = 1 \\
\text{NaN} & \text{otherwise}
\end{cases}
\]

This prevents training on irrelevant market periods.

---

## 📂 `train_model.py`

### Purpose
Train a probabilistic classifier that estimates **trade success probability**.

---

### Model Architecture

A deliberately small multilayer perceptron (MLP):

\[
X_t \rightarrow \text{ReLU} \rightarrow \text{ReLU} \rightarrow \text{Dropout} \rightarrow \hat{p}_t
\]

Output:
\[
\hat{p}_t = P(Y_t = 1 \mid X_t)
\]

---

### Loss Function

Binary Cross-Entropy with Logits:

\[
\mathcal{L} = -[y \log(\sigma(z)) + (1-y)\log(1-\sigma(z))]
\]

Chosen because:
- Proper scoring rule
- Penalizes overconfident errors
- Suitable for threshold-based decision rules

---

### Training Design

- Time-based train/test split
- Feature scaling fit on training data only
- Early stopping on validation loss
- No look-ahead bias

---

### Diagnostics Produced

- Train vs test accuracy
- Probability distributions
- Calibration curves
- Probability time-series

---

## 📂 `ml_filter.py`

### Purpose
Convert model probabilities into a **decision layer**.

### Logic

\[
\text{FilteredSignal}_t = S_t \cdot \mathbb{1}(\hat{p}_t > \tau)
\]

This decouples:
- Prediction from capital allocation
- Learning from execution

---

## 📂 `backtest_ml_strategy.py`

### Purpose
Evaluate the ML-filtered strategy in the **same environment** as the baseline.

### Return Construction

Raw EMA:
\[
r_t = S_t \cdot \frac{P_t - P_{t-1}}{P_{t-1}}
\]

ML-filtered:
\[
r_t^{ML} = S_t \cdot \mathbb{1}(\hat{p}_t > \tau) \cdot r_t
\]

---

### Performance Metrics

- Sharpe Ratio:
\[
\text{Sharpe} = \frac{\mu}{\sigma} \sqrt{252}
\]

- Drawdown:
\[
DD_t = \frac{E_t}{\max(E)} - 1
\]

---

### Interpretation Rule

If:
- Sharpe ≈ unchanged
- Drawdown ↓
- Trade frequency ↓

Then ML improves **capital efficiency**, not raw predictability.

---

## 📂 `figure_plotting.py`

### Purpose
Visualize statistical and economic behavior.

### Plotting Choices

**Matplotlib**
- Cumulative returns
- Drawdowns
- Calibration curves
- Research-grade static plots

**Plotly**
- Threshold sensitivity analysis
- 3D probability–return–frequency surfaces
- Interactive regime diagnostics

---

## 5. Results Summary

| Metric | Raw EMA | ML-Filtered EMA |
|------|--------|----------------|
| Trade Frequency | Higher | Lower |
| Sharpe Ratio | Similar | Similar or slightly higher |
| Drawdown | Larger | Smaller |
| Return Variance | Higher | Lower |

---

## 6. Key Research Takeaways

- Accuracy is a weak objective in trading
- Probability calibration matters more than hit rate
- ML is best used as a **filter**
- Economic labels outperform directional labels

---

## 7. Extensions

- Walk-forward retraining
- Regime-aware models
- Multi-horizon labeling
- Portfolio-level ML filters

---


