# pump.fun Top-5 Forecast — make.fun entry

Forecasts the 5 largest new pump.fun coins by market cap at each of the
seven answer reads, for the `pumpfun-top5-forecast-01` bounty on make.fun.

## How it works

`main.py` (Python 3.13, stdlib only) reads the house's 10-minute snapshots
(`/input/history.ndjson`) and task description (`/input/task.json`), then
writes `/output/forecast.json`:

- **Per-coin prediction.** Each coin's future log10 market cap is predicted
  from its current cap, shrunk short-term momentum (10-minute momentum is
  mean-reverting in the practice data, corr -0.14), a baseline decay that
  depends on venue (pool vs bonding curve), and a small boost for coins
  that graduated to their PumpSwap pool within the last hour.
- **Place probabilities.** A Gaussian kernel over predicted log-caps gives
  each coin a probability at each place 1-5. A large share of every place
  is left unnamed for coins born after the history cut: in the practice,
  3 of the 5 coins in the last read's top 5 never appeared in the history.
- **Sizes.** Each place's market-cap distribution is a mixture of an
  incumbent normal (centred on the place's predicted cap) and a newcomer
  normal (centred above the top incumbent), widened with the forecast
  horizon.
- **Horizon-adaptive.** The horizon (history cut to mean answer read) is
  read from `task.json`, so the same code adapts to the 1-hour practice
  and the 10-hour final: newcomer share, kernel width and size spread all
  grow with horizon.

## Local backtests (practice data)

| Backtest | This model | House baseline (starter) |
|---|---|---|
| Predict 20:50 mark from marks to 20:20 | -46.70 | -52.72 |
| Predict 20:50 mark from marks to 20:30 | -33.04 | -37.31 |

## Run locally

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
