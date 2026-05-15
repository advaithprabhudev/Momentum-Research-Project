# Success Criteria

## Sharpe Ratio for Portfolio report

Sharpe ratio measures risk adjusted return, using a mathematical formula.

Single-trade sharpe ratio is redundant, hence, I use portfolio sharpe ratio that is locked because:

* Ignore volatility when returns are good
* Ignore volatility when returns are bad

## Max Drawdown

This measures the worst case scenarios, where there is capital risk involved.

If drawdown increases, the strategy completely fails, even if Sharpe Ratio is good.

## Win rate

The win rate calculation is pretty simple:

`Win Rate = Winning Trades / Total Trades`

Higher is not always better, but the ML Model should filter the bad trades.

## Comparision

* Buy & Hold strategy - Baseline
* Momentum Strategy
* Momentum + ML Strategy

