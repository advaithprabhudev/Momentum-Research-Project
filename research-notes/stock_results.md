# Stock Results — ML-Filtered Momentum Strategy

## Metrics Reference

| Metric | Description |
|---|---|
| Raw Trades | Total EMA crossover signals before ML filter |
| Filtered Trades | Signals accepted after ML confidence threshold |
| Threshold | ML probability cutoff selected on validation set |
| Raw Sharpe | Annualised Sharpe on unfiltered EMA returns |
| Filtered Sharpe | Annualised Sharpe on ML-filtered returns |
| Raw Max Drawdown | Worst peak-to-trough on raw strategy |
| Filtered Max Drawdown | Worst peak-to-trough on filtered strategy |
| Train Accuracy | MLP accuracy on training set (threshold 0.5) |
| Val Accuracy | MLP accuracy on validation set (threshold 0.5) |
| Test Accuracy | MLP accuracy on test set (threshold 0.5) |

**Split:** 60% train / 20% validation / 20% test (chronological)
**Data:** 2010-01-01 to present via yfinance
**Model:** MLP 64→32→1, BCEWithLogitsLoss, Adam lr=1e-4, patience=5, seed=42

---

## Results

### AAPL

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 529 | 81 |
| Threshold | — | 0.58 |
| Sharpe Ratio | 1.2254 | 0.4749 |
| Max Drawdown | -22.92% | -8.68% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 56.15% |
| Val | 53.88% |
| Test | 53.88% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.4753 |
| 0.1% | 0.2492 |
| 0.2% | 0.0213 |

**Notes:** Filter cuts 85% of trades (529→81). Drawdown improves dramatically (−22.9%→−8.7%) but Sharpe degrades. Too few trades remain for a stable Sharpe estimate.

---

### GOOGL

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 523 | 478 |
| Threshold | — | 0.51 |
| Sharpe Ratio | 1.8349 | 1.9701 |
| Max Drawdown | -16.53% | -16.53% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 55.51% |
| Val | 55.07% |
| Test | 55.64% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 1.9310 |
| 0.1% | 1.8927 |
| 0.2% | 1.8544 |

**Notes:** Best result in the set. Minimal trade reduction (523→478) with genuine Sharpe improvement (+0.135). Test accuracy above 55%. Drawdown unchanged. Transaction costs have negligible impact. Strong candidate for paper primary example.

---

### NVDA

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 558 | 558 |
| Threshold | — | 0.30 |
| Sharpe Ratio | 1.2795 | 1.2795 |
| Max Drawdown | -35.95% | -35.95% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 59.03% |
| Val | 59.86% |
| Test | 50.36% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 1.2454 |
| 0.1% | 1.2454 |
| 0.2% | 1.2454 |

**Notes:** Filter completely ineffective — threshold swept to floor (0.30) and all 558 signals pass. ML model cannot distinguish signals on NVDA. Largest drawdown in the set (−35.95%).

---

### MSFT

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 537 | 525 |
| Threshold | — | 0.48 |
| Sharpe Ratio | 1.0209 | 1.1102 |
| Max Drawdown | -17.92% | -17.92% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 58.48% |
| Val | 52.89% |
| Test | 51.96% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 1.1087 |
| 0.1% | 1.0887 |
| 0.2% | 1.0688 |

**Notes:** Moderate filtering (537→525) with a meaningful Sharpe improvement (+0.089). Drawdown unchanged. Robust to transaction costs. Consistent with the strategy's intended behaviour — good secondary paper example.

---

### JPM

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 520 | 268 |
| Threshold | — | 0.52 |
| Sharpe Ratio | 1.2598 | 1.2270 |
| Max Drawdown | -17.05% | -13.02% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 54.56% |
| Val | 52.88% |
| Test | 61.15% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 1.2282 |
| 0.1% | 1.1076 |
| 0.2% | 0.9871 |

**Notes:** Highest test accuracy in the set (61.15%). Filter halves trade count (520→268) and reduces drawdown by 4 pp at a small Sharpe cost (−0.033). Clear risk-reduction use case.

---

### JNJ

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 472 | 13 |
| Threshold | — | 0.55 |
| Sharpe Ratio | 0.9194 | -0.8347 |
| Max Drawdown | -13.46% | -9.54% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 53.75% |
| Val | 51.17% |
| Test | 50.64% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | -0.8356 |
| 0.1% | -0.8968 |
| 0.2% | -0.9572 |

**Notes:** Over-filtering failure case. Extreme trade reduction (472→13) concentrates the strategy in a handful of poor-performing signals. Negative filtered Sharpe. Illustrates the over-filtering failure mode.

---

### XOM

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 407 | 298 |
| Threshold | — | 0.48 |
| Sharpe Ratio | 0.4366 | 0.5854 |
| Max Drawdown | -24.70% | -21.12% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 55.94% |
| Val | 44.72% |
| Test | 46.93% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.5811 |
| 0.1% | 0.4601 |
| 0.2% | 0.3388 |

**Notes:** Positive result — filter improves Sharpe (+0.149) and drawdown (−24.7%→−21.1%) despite sub-50% classification accuracy. Supports the calibration framing: the model suppresses losing trades without needing clean classification signal.

---

### WMT

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 522 | 507 |
| Threshold | — | 0.47 |
| Sharpe Ratio | 1.7737 | 1.6297 |
| Max Drawdown | -17.39% | -19.73% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 52.52% |
| Val | 46.55% |
| Test | 48.28% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 1.6106 |
| 0.1% | 1.5869 |
| 0.2% | 1.5631 |

**Notes:** Negative result. Filter adds noise — Sharpe drops (−0.144) and drawdown worsens. The raw EMA strategy already performs well on WMT without ML filtering.

---

### HD

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 527 | 231 |
| Threshold | — | 0.59 |
| Sharpe Ratio | -0.0069 | 0.4998 |
| Max Drawdown | -20.95% | -14.28% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 57.69% |
| Val | 57.12% |
| Test | 49.15% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.4662 |
| 0.1% | 0.2726 |
| 0.2% | 0.0792 |

**Notes:** Strongest turnaround in the set — raw EMA is near-zero Sharpe and the ML filter rescues it to 0.50. Drawdown also improves (−21.0%→−14.3%). Demonstrates ML filter's ability to salvage a weak baseline. Transaction costs erode filtered Sharpe significantly at 0.2%.

---

### UNH

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 537 | 70 |
| Threshold | — | 0.58 |
| Sharpe Ratio | 0.0642 | 0.8155 |
| Max Drawdown | -38.92% | -7.27% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 57.89% |
| Val | 51.96% |
| Test | 50.28% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.7918 |
| 0.1% | 0.6327 |
| 0.2% | 0.4720 |

**Notes:** Most dramatic risk-reduction result. Filter cuts trades by 87% (537→70) and reduces max drawdown from −38.9% to −7.3% — an 81% improvement. Sharpe improves from near-zero to 0.82. UNH's sharp regime changes make signal selectivity especially valuable.

---

### GS

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 477 | 40 |
| Threshold | — | 0.54 |
| Sharpe Ratio | 1.7158 | 0.3224 |
| Max Drawdown | -16.74% | -6.11% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 54.34% |
| Val | 49.79% |
| Test | 48.43% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.3227 |
| 0.1% | 0.1328 |
| 0.2% | -0.0578 |

**Notes:** Over-filtering on a strong baseline. Raw Sharpe 1.72 collapses to 0.32 with only 40 trades remaining. Drawdown improvement (−16.7%→−6.1%) is real but not worth the return sacrifice. Turns negative under 0.2% transaction costs.

---

### BA

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 447 | 0 |
| Threshold | — | 0.74 |
| Sharpe Ratio | 0.2744 | 0.0 |
| Max Drawdown | -33.42% | 0.0% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 53.10% |
| Val | 42.60% |
| Test | 46.31% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.0 |
| 0.1% | 0.0 |
| 0.2% | 0.0 |

**Notes:** Complete filter failure — all 447 trades rejected (0 filtered trades). Threshold sweeps to 0.74 on validation; no test probability exceeds it. BA's erratic history (737 MAX, COVID, defence contract volatility) creates label noise the model cannot handle. Reported 0.0 Sharpe and 0.0% drawdown reflect an empty strategy, not performance. Exclude from paper results.

---

### AMZN

| Metric | Raw EMA | ML Filtered |/co
|---|---|---|
| Trades | 528 | 1 |
| Threshold | — | 0.59 |
| Sharpe Ratio | 0.9468 | 0.6908 |
| Max Drawdown | -20.70% | 0.0% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 56.95% |
| Val | 49.81% |
| Test | 52.27% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.6915 |
| 0.1% | 0.5812 |
| 0.2% | 0.4219 |

**Notes:** Near-total filtering (528→1 trade). With only 1 accepted trade, filtered metrics are not statistically meaningful. AMZN's non-normal return distribution and 2022 regime break likely prevent generalisation. Exclude from paper results.

---

### SPY

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 607 | 1 |
| Threshold | — | 0.66 |
| Sharpe Ratio | 1.5467 | 0.6443 |
| Max Drawdown | -8.75% | 0.0% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 57.36% |
| Val | 57.00% |
| Test | 58.65% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.6449 |
| 0.1% | 0.6068 |
| 0.2% | 0.5621 |

**Notes:** High test accuracy (58.65%) but near-total trade rejection (607→1). EMA momentum already works well on broad-market ETFs — ML filter adds no value here. SPY has the lowest raw drawdown of all tickers (−8.75%). High accuracy without usable filtered trades highlights the accuracy-vs-quality disconnect.

---

### QQQ

| Metric | Raw EMA | ML Filtered |
|---|---|---|
| Trades | 591 | 534 |
| Threshold | — | 0.53 |
| Sharpe Ratio | 1.4049 | 0.9393 |
| Max Drawdown | -15.05% | -17.77% |

**Accuracy**

| Split | Accuracy |
|---|---|
| Train | 56.83% |
| Val | 57.87% |
| Test | 58.04% |

**Transaction Cost Sensitivity**

| Cost (round-trip) | Filtered Sharpe |
|---|---|
| 0.0% | 0.9278 |
| 0.1% | 0.8466 |
| 0.2% | 0.7654 |

**Notes:** Negative result despite high accuracy (58%). Minimal trade reduction (591→534), Sharpe declines (1.40→0.94), drawdown worsens. High classification accuracy does not guarantee better signal filtering — strong evidence for the paper's framing that accuracy is a poor proxy for strategy quality.

---

## Cross-Ticker Summary

| Ticker | Raw Trades | Filtered Trades | Raw Sharpe | Filtered Sharpe | Δ Sharpe | Raw Drawdown | Filtered Drawdown | Test Acc |
|---|---|---|---|---|---|---|---|---|
| AAPL | 529 | 81 | 1.2254 | 0.4749 | −0.750 | −22.92% | −8.68% | 53.88% |
| GOOGL | 523 | 478 | 1.8349 | **1.9701** | **+0.135** | −16.53% | −16.53% | 55.64% |
| NVDA | 558 | 558 | 1.2795 | 1.2795 | 0.000 | −35.95% | −35.95% | 50.36% |
| MSFT | 537 | 525 | 1.0209 | 1.1102 | **+0.089** | −17.92% | −17.92% | 51.96% |
| JPM | 520 | 268 | 1.2598 | 1.2270 | −0.033 | −17.05% | −13.02% | **61.15%** |
| JNJ | 472 | 13 | 0.9194 | −0.8347 | −1.754 | −13.46% | −9.54% | 50.64% |
| XOM | 407 | 298 | 0.4366 | 0.5854 | **+0.149** | −24.70% | −21.12% | 46.93% |
| WMT | 522 | 507 | 1.7737 | 1.6297 | −0.144 | −17.39% | −19.73% | 48.28% |
| HD | 527 | 231 | −0.0069 | 0.4998 | **+0.507** | −20.95% | −14.28% | 49.15% |
| UNH | 537 | 70 | 0.0642 | 0.8155 | **+0.751** | **−38.92%** | **−7.27%** | 50.28% |
| GS | 477 | 40 | 1.7158 | 0.3224 | −1.393 | −16.74% | −6.11% | 48.43% |
| BA | 447 | 0 | 0.2744 | — | — | −33.42% | — | 46.31% |
| AMZN | 528 | 1 | 0.9468 | — | — | −20.70% | — | 52.27% |
| SPY | 607 | 1 | 1.5467 | — | — | −8.75% | — | 58.65% |
| QQQ | 591 | 534 | 1.4049 | 0.9393 | −0.466 | −15.05% | −17.77% | 58.04% |

### Outcome Breakdown (excluding degenerate cases: BA, AMZN, SPY)

| Outcome | Tickers |
|---|---|
| Filter improves Sharpe | GOOGL, MSFT, XOM, HD, UNH (5/12) |
| Filter improves drawdown | AAPL, JPM, JNJ, XOM, HD, UNH, GS (7/12) |
| Filter neutral | NVDA |
| Filter hurts Sharpe | AAPL, JPM, JNJ, WMT, GS, QQQ (6/12) |
| Filter hurts drawdown | WMT, QQQ (2/12) |

---

*All runs: 2010-01-01 to 2025 · 60/20/20 chronological split · MLP (64→32→1) · BCEWithLogitsLoss · Adam lr=1e-4 · Early stopping patience=5 · torch.manual_seed(42) · np.random.seed(42)*

---

## Overall Summary (Means)

Degenerate cases (BA, AMZN, SPY) are excluded from filtered metrics due to 0 or 1 accepted trades. All 15 tickers are included in raw and accuracy means.

### Raw Strategy (all 15 tickers)

| Metric | Mean |
|---|---|
| Raw Trades | 519 |
| Raw Sharpe | 1.046 |
| Raw Max Drawdown | −21.36% |
| Train Accuracy | 56.01% |
| Test Accuracy | 52.13% |

### ML-Filtered Strategy (12 valid tickers)

| Metric | Mean |
|---|---|
| Filtered Trades | 300 |
| Filtered Sharpe | 0.835 |
| Filtered Max Drawdown | −15.66% |
| Delta Sharpe (filtered − raw) | −0.242 |
| Drawdown Improvement | +5.81 pp |

### Interpretation

On average across the 12 valid tickers, the ML filter reduces trade count by roughly 42% and improves maximum drawdown by 5.8 percentage points. Mean Sharpe declines by 0.242, largely driven by outlier over-filtering cases (JNJ, GS). The filter is more consistent as a risk-reduction tool than a return-enhancement tool. Mean test accuracy of 52.13% is marginally above chance, reinforcing the calibration framing used throughout the paper.
