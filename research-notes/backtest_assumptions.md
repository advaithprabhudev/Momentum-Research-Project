# Backtest Assumptions

No transaction costs 
Daily rebalancing of positions
Positions invested per signal
Features are lagged a bit to prevent the look-ahead bias
ML Filter is static and doesn't change (this part can be optimized)
Missing data handled by dropna()