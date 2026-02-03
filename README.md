# 📘 Machine-Learning Filtered Momentum Strategy  
### A Research-Grade Quantitative Study on Signal Selectivity

---

## 1. Research Objective

This project investigates a realistic and professional quantitative research question:

> **Can machine learning improve a classical momentum strategy by filtering low-quality trades, without directly generating alpha?**

The machine learning model **does not predict prices**.  
Instead, it estimates the **conditional probability that a momentum trade will succeed**, given the prevailing market state.

This design mirrors how machine learning is used in professional systematic trading research.

---

## 2. Core Conceptual Framework

Define:

- $S_t \in \{0,1\}$ — deterministic momentum signal (EMA crossover)
- $X_t$ — feature vector describing market conditions at time $t$
- $Y_t \in \{0,1\}$ — trade success label
- $\hat{p}_t = P(Y_t = 1 \mid X_t)$ — ML-estimated probability of trade success

A trade is executed **only if**:

$$
S_t = 1 \quad \text{and} \quad \hat{p}_t > \tau
$$

where $\tau$ is a confidence threshold.

> **Machine learning does not create alpha.  
It filters noise from a known anomaly.**

---

## 3. Data Description

### Market Data
- Daily OHLCV time series
- Single-asset research setup (architecture extensible)
- Chronologically ordered

### Data Integrity Constraints
- No forward-looking features
- All rolling statistics are shifted
- Labels are derived strictly from future prices
- Time-based train/test split (no shuffling)

---

## 4. File-by-File Explanation

---

## 📂 `data_ingestion.py`

### Purpose
Load, clean, and standardize raw market data.

### Logic
- Reads OHLCV data
- Enforces datetime index
- Sorts chronologically
- No transformations applied

This file defines the **base probability space** for the research.

---

## 📂 `feature_generation.py`

### Purpose
Encode the **market state at time $t$** without using future information.

---

### Feature Definitions

#### 1️⃣ 20-Day Volatility

$$
\sigma_t = \sqrt{\frac{1}{20} \sum_{i=1}^{20} (r_{t-i} - \bar{r})^2}
$$

Measures regime noise and instability.

---

#### 2️⃣ Relative Strength Index (RSI-14)

$$
RSI_t = 100 - \frac{100}{1 + RS_t}
$$

Captures momentum exhaustion rather than direction.

---

#### 3️⃣ Average True Range (ATR-14)

$$
ATR_t = EMA_{14}\Big(\max(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|)\Big)
$$

Measures volatility expansion and stop-loss risk.

---

#### 4️⃣ Volume Z-Score (252-Day)

$$
Z_t = \frac{V_t - \mu_{252}}{\sigma_{252}}
$$

Detects abnormal participation and regime shifts.

---

### Causality Constraint

All features are shifted:

$$
X_t = \text{information available at } t-1
$$

This enforces strict temporal causality.

---

## 📂 `label_generator.py`

### Purpose
Define economically meaningful supervised learning targets.

---

### Label Construction

For a signal at time $t$:

$$
\text{Entry Price} = P_{t+1}
$$

$$
\text{Exit Price} = P_{t+1+h}
$$

$$
R_t = \frac{P_{t+1+h} - P_{t+1}}{P_{t+1}}
$$

A trade is labeled successful if:

$$
R_t > c
$$

where:
- $h$ = holding horizon
- $c$ = transaction cost

Labels are defined **only when the momentum signal is active**:

$$
Y_t =
\begin{cases}
1, & \text{if } S_t = 1 \text{ and } R_t > c \\
\text{NaN}, & \text{otherwise}
\end{cases}
$$

This prevents training on irrelevant periods.

---

## 📂 `train_model.py`

### Purpose
Train a probabilistic classifier to estimate **trade success likelihood**.

---

### Model Architecture

A compact multilayer perceptron (MLP):

$$
X_t \rightarrow \text{ReLU} \rightarrow \text{ReLU} \rightarrow \text{Dropout} \rightarrow z_t
$$

Output probability:

$$
\hat{p}_t = \sigma(z_t)
$$

---

### Loss Function

Binary Cross-Entropy with Logits:

$$
\mathcal{L} = -\Big[y \log(\sigma(z)) + (1 - y)\log(1 - \sigma(z))\Big]
$$

This loss:
- Is a proper scoring rule
- Penalizes overconfident errors
- Enables threshold-based decisions

---

### Training Design
- Time-based train/test split
- Feature scaling fit on training data only
- Early stopping
- No trading logic inside the model

---

### Filtering Rule

$$
\text{FilteredSignal}_t = S_t \cdot \mathbb{1}(\hat{p}_t > \tau)
$$

The ML model:
- Does **not** size trades
- Does **not** compute PnL
- Only decides whether to trust the signal

---

### Return Construction

Raw EMA strategy:

$$
r_t^{\text{EMA}} = S_t \cdot r_t
$$

ML-filtered strategy:

$$
r_t^{\text{ML}} = S_t \cdot \mathbb{1}(\hat{p}_t > \tau) \cdot r_t
$$

---

### Performance Metrics

**Sharpe Ratio**

$$
\text{Sharpe} = \frac{\mathbb{E}[r]}{\sigma(r)} \sqrt{252}
$$

**Drawdown**

$$
DD_t = \frac{E_t}{\max(E)} - 1
$$

---

## 📂 `figure_plotting.py`

### Purpose
Visual validation and diagnostic analysis.

---

### Matplotlib (Static)
- Cumulative returns
- Drawdowns
- Calibration curves
- Probability distributions

### Plotly (Interactive)
- Threshold × Sharpe × Trade Frequency (3D)
- Confidence vs forward return
- Regime diagnostics

Plotly is used where **interaction reveals structure**.

---

## 5. Results Summary

| Metric | Raw EMA | ML-Filtered EMA |
|------|--------|----------------|
| Trade Frequency | Higher | Lower |
| Sharpe Ratio | Similar | Similar |
| Drawdown | Higher | Lower |
| Return Variance | Higher | Lower |

---

## 6. Key Research Takeaways

- Accuracy is a weak objective in trading
- ML improves **selectivity**, not predictability
- Fewer trades can mean better capital efficiency
- Label quality matters more than model complexity

---

## 7. Extensions

- Walk-forward retraining
- Regime-aware models
- Multi-horizon labeling
- Portfolio-level ML filters

---


