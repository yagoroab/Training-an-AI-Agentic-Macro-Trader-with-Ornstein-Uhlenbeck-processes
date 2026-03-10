Ornstein–Uhlenbeck Mean Reversion Strategy on the VIX

This repository contains the code developed for my IE University Final Degree Project.

The project investigates whether the VIX volatility index can be modeled as a mean-reverting stochastic process and traded systematically using an Ornstein–Uhlenbeck (OU) framework.

The repository implements

• rolling estimation of OU parameters
• a mean-reversion trading strategy derived from OU dynamics
• a full historical backtest
• performance diagnostics and robustness checks

The objective is to evaluate whether OU-driven equilibrium signals contain exploitable structure in volatility markets.

Research Contribution

This project builds a systematic trading pipeline that models the VIX as a mean-reverting Ornstein–Uhlenbeck process and tests a trading strategy derived from deviations from the estimated equilibrium level.

The implementation combines

• stochastic process modelling
• rolling statistical calibration
• signal generation from equilibrium deviations
• historical backtesting with risk metrics

Ornstein–Uhlenbeck Model

The VIX is modeled as a continuous-time Ornstein–Uhlenbeck process

dX_t = θ(μ − X_t) dt + σ dW_t

Where

Parameter	Meaning
μ	Long-run equilibrium level
θ	Speed of mean reversion
σ	Volatility parameter
W_t	Brownian motion

The model implies that deviations from equilibrium decay exponentially over time.

Mean Reversion Property

The expected value of the process evolves as

E[X_t] = μ + (X_0 − μ) e^(−θt)

This means that any deviation from equilibrium gradually disappears at rate θ.

Discrete Representation

For estimation purposes, the OU process can be written as a discrete AR(1) process

X_{t+1} = a + b X_t + ε_t

From this representation the OU parameters can be recovered as

θ = −ln(b) / Δt
μ = a / (1 − b)

This approach allows the parameters to be estimated using standard regression methods.

Trading Signal

The trading signal is based on the standardized deviation from the OU equilibrium.

z_t = (X_t − μ_t) / σ_t

Where

Variable	Meaning
X_t	log(VIX)
μ_t	rolling OU equilibrium estimate
σ_t	OU volatility estimate
Trading Rule

Positions are determined by the sign of the deviation from equilibrium.

If z_t > 0  → Short volatility
If z_t < 0  → Long volatility

The intuition is that large deviations from equilibrium tend to revert back toward the long-run mean.

Data

The strategy is tested on historical VIX data from the CBOE.

Dataset characteristics

Feature	Value
Start date	1990
End date	2026
Frequency	Daily
Variable used	log(VIX)

Using log(VIX) improves statistical stability and better aligns the data with the assumptions of the OU model.

Model Estimation

OU parameters are estimated using rolling window calibration.

For each window the model estimates

• mean reversion speed
• equilibrium level
• volatility parameter

This simulates real-time parameter learning and avoids look-ahead bias.

Backtest Framework

The repository implements a complete backtesting pipeline including

• signal generation
• daily returns
• cumulative PnL
• wealth process
• risk metrics

Key evaluation statistics include

• Sharpe ratio
• CAGR
• maximum drawdown

Results

Baseline results of the OU strategy

Metric	Value
Sharpe Ratio	~1.09
CAGR	~35.6%
Max Drawdown	−34.7%
Final Wealth	~11.6

The strategy captures periods where volatility moves away from equilibrium and subsequently mean-reverts.

These results should be interpreted cautiously since the backtest does not include

• transaction costs
• slippage
• volatility derivatives implementation constraints

Repository Structure
src/

data/
vix_loader.py
Load historical VIX data

models/
ou_estimation.py
Rolling OU parameter estimation

backtest/
backtest_ou.py
Mean reversion strategy implementation

plot_ou_results.py
Visualization of results

run_ou_vix.py
Main experiment pipeline
Running the Experiment

Execute the main script

python -m run_ou_vix

The pipeline will

load VIX data

estimate rolling OU parameters

generate trading signals

run the historical backtest

output performance metrics and plots

References

Relevant research on OU processes and mean-reversion trading includes

Holý, V. & Tomanová, P. (2018)
Estimation of Ornstein–Uhlenbeck process using ultra-high-frequency data with application to intraday pairs trading strategy.

Endres, S. & Stübinger, J. (2019)
Optimal trading strategies for Lévy-driven Ornstein–Uhlenbeck processes.

Wu, L. (2020)
Analytic value function for a pairs trading strategy with a Lévy-driven Ornstein–Uhlenbeck process.

Zhu, D. M., Yu, F. & Zhou, X. (2021)
Optimal pairs trading with dynamic mean-variance objective.

These works show that Ornstein–Uhlenbeck processes provide a tractable framework for modeling equilibrium-seeking financial variables and designing statistical arbitrage strategies.
Evaluate robustness across volatility regimes, shocks, and changing parameters

This combines stochastic process modeling, reinforcement learning, and financial market structure.
