# What each trade is classified under?

## What exactly is the model learning?

Trade = 1 -> Succesful trade  (profitable)
Trade = 0 -> Unsuccesful trade (not profitable)

## Prediction Horizon

### How long should I wait to judge this model?

So, currently, my holding period is around 5 days, due to the momentum strategy.

Hence, this avoids most cases like:

* Look ahead bias
* Different exit days
* Non reproducible outcomes

Every trade:

* One entry time : t
* One exit time : t + 5
* One label (1 = profitable, 0 = not)

## Train-Validation-Test split

Train - 2010-2017
Validation - 2018-2020
Test - 2021-2025

## Constraints:

* No future features
* Strict labels
* Fixed holding period (5 days)
