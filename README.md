# Ornstein-Uhlenbeck Mean Reversion on Volatility Markets

This repository contains the code developed for my IE University Final Degree Project.

The project studies whether volatility markets contain exploitable mean-reverting structure and whether that structure can be traded systematically using an Ornstein-Uhlenbeck (OU) framework. The core benchmark models volatility as a mean-reverting process, generates trading signals from deviations from equilibrium, and trades `VXX` as the tradable volatility proxy.

In addition to the OU benchmark, the repository includes recurrent neural network baselines (`LSTM` and `GRU`) to test whether more flexible sequence models can outperform the parametric OU approach.

## Research Question

The main question is:

Can a rolling Ornstein-Uhlenbeck framework extract a useful equilibrium signal from volatility data and generate trading performance that is competitive with standard benchmarks such as the S&P 500?

A secondary question is:

Can recurrent neural networks improve on the OU benchmark, or is the simpler and more interpretable OU model more robust?

## What the Repository Implements

The repository includes:

- rolling OU parameter estimation
- a systematic mean-reversion trading strategy
- historical backtesting on volatility-linked assets
- benchmark comparison against `VXX` buy-and-hold and the `S&P 500`
- robustness tests across parameter changes, transaction costs, split dates, and RNN seeds
- recurrent neural network baselines (`LSTM` and `GRU`)

## Current Strategy Setup

The current OU benchmark uses:

- historical `VIX` data as the volatility signal source
- `VXX` as the traded asset
- a rolling OU calibration
- a signal based on the ratio `log(VIX / VXX)`
- a post-COVID tradable comparison window, starting in May 2020
- out-of-sample comparisons for RNNs starting in May 2022

This design was chosen because it aligns the signal with the traded volatility instrument and produces a cleaner benchmark comparison after the COVID shock regime.

## Ornstein-Uhlenbeck Model

The continuous-time OU process is

\[
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
\]

where:

- `mu` is the long-run equilibrium level
- `theta` is the speed of mean reversion
- `sigma` is the diffusion volatility
- `W_t` is a Brownian motion

The OU process implies that deviations from equilibrium decay over time. In expectation,

\[
\mathbb{E}[X_t] = \mu + (X_0 - \mu)e^{-\theta t}
\]

For estimation, the process is written in discrete AR(1)-style form and estimated on rolling windows.

## Signal Construction

The benchmark strategy estimates OU parameters on a rolling basis and computes a standardized deviation from the estimated equilibrium:

\[
z_t = \frac{X_t - \mu_t}{\sigma_t}
\]

In the current implementation:

- `X_t` is the signal `log(VIX / VXX)`
- `mu_t` is the rolling OU equilibrium estimate
- `sigma_t` is the rolling OU volatility estimate

Positions are then mapped from the sign and magnitude of the deviation, with position sizing, leverage caps, transaction costs, and cash allocation handled in the backtest layer.

## Data

The project uses:

- historical `VIX` data from the included CSV file
- market data downloaded for `VXX` and benchmark indices such as `^GSPC`

Main assets used:

- `VIX`: volatility signal source
- `VXX`: traded volatility proxy
- `^GSPC`: benchmark equity index

## Methodology

The repository follows this pipeline:

1. Load historical data
2. Construct the volatility signal
3. Estimate rolling OU parameters
4. Generate trading positions from equilibrium deviations
5. Backtest the strategy with costs and cash allocation
6. Compare against benchmarks
7. Run robustness checks
8. Compare against RNN baselines

A major focus of the implementation is avoiding forward-looking bias. Signals are formed using information available at time `t` and applied to the next trading period.

## Neural Network Extension

The repository also includes `LSTM` and `GRU` baselines.

These models:

- use lagged volatility and traded-asset features
- are trained on pre-2022 data
- are evaluated on a common out-of-sample window
- are compared directly against the OU strategy and the S&P 500

The RNN section is included as an extension to test whether flexible sequence models can outperform the interpretable OU benchmark.

## Main Findings

The main conclusions from the current experiments are:

- the OU strategy is more stable and interpretable than the RNN alternatives
- the OU benchmark is competitive with standard benchmarks on some samples, but does not consistently dominate the S&P 500
- RNNs can occasionally perform well, but their performance is much more sensitive to random seed and split choice
- the OU strategy appears to be the more robust benchmark for this thesis

This makes the project academically useful even when the OU strategy does not outperform the S&P on every metric: the contribution is not only performance, but also robustness, interpretability, and disciplined out-of-sample evaluation.

## Repository Structure

```text
src/
  backtest/
    backtest_ou.py            # OU trading/backtest logic
  config/
    baseline_config.py        # baseline configuration values
  data/
    market_loader.py          # Yahoo Finance loaders
    vix_loader.py             # VIX CSV loader
  models/
    ou_estimation.py          # rolling OU parameter estimation
    lstm_vxx.py               # LSTM/GRU dataset and model utilities
  strategies/
    ou_threshold.py           # signal-to-position mapping
  plot_compare_results.py     # benchmark and comparison plots
  plot_ou_results.py          # OU diagnostics and plots
  run_ou_vix.py               # main OU experiment

run_compare_benchmarks.py     # OU vs benchmark comparison
run_robustness_tests.py       # OU and RNN robustness checks
run_lstm_benchmark.py         # LSTM/GRU vs OU vs S&P benchmark
data/
  VIX_History.csv             # historical VIX dataset
figures/
  ...                         # generated figures

