# Strategy Definition

## Objective

Implementing the baseline momentum strategy using EMA (Exponential Moving Average) crossovers (20 Day vs 50 Day) at the baseline signal

Enhance the baseline strategy using an ML filter to seperate "bad" trades and "good" trades

## Components:

### 1. Raw Strategy

Buy Signal : EMA 20 > EMA 50
Sell Signal : EMA 20 < EMA 50
Position : Full allocation of price
Frequency : Daily Close Prices

### 2. ML Filtering Strategy

Use a trained MLP (Multi Layer Perceptron) to assign probability to raw EMA Signals
Apply a threshold (ex 0.5515) to filter trades 
Only exceute the trade if ML model probability exceeds the threshold

## Rationale

Raw EMA strategy may generate false positives
The Machine Learning filter is used to reduce the low confidence trades

