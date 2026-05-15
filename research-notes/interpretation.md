# Interpretation of the results

Raw EMA Momentum Strategy does result in more trades, slighlt lower Sharpe Ratio due to capturing all profitable movements (mainly the small ones)
ML filters the trades, focuses on high confidence signals to cross the threshold, slightly increases Sharpe
Model validation shows similar accuracy on train and test data -> minimal overfitting
Overall, ML is feasible for trading, and shows incremental improvement without unrealistic claims