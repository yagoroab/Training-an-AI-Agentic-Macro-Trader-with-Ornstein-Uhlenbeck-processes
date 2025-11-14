**Training an AI Agentic Macro Trader with Ornstein–Uhlenbeck Processes**

This repository contains the development of my IE University Final Degree Project.
The goal is to:

1. Simulate a universe of mean-reverting macro asset prices

using Ornstein–Uhlenbeck (OU) stochastic processes, calibrated and validated through statistical tests.

2. Build and train an AI agent capable of trading these OU-driven assets

by learning profitable mean-reversion strategies from synthetic environments.

This repo will evolve from early OU testing → to full agentic training.

**Current Progress**
✔️ OU Simulation Validation (Completed)

Implemented OU simulation tests using reference Python libraries

Verified correctness of OU paths

Checked calibration methods (MLE / Least Squares)

Confirmed recovery of true parameters from synthetic data

Notebook: OU Trials.ipynb

**Upcoming Work (Next Steps)**

Implement custom OU class (src/ou_process.py)

Build OU parameter estimator (src/ou_estimation.py)

Generate synthetic multi-asset datasets

Create trading environment (Gym-style)

Implement first-stage agent (baseline mean-reversion strategy)

Prepare reinforcement learning phase

**Repository Structure**
notebooks/
    OU Trials.ipynb           → OU simulation & parameter recovery tests

src/
    (future) ou_process.py    → Custom OU simulator
    (future) ou_estimation.py → Calibration methods
    (future) agent.py         → Trading agent logic

data/
    (future) synthetic datasets

models/
    (future) trained agents and saved policies

figures/
    (future) OU plots, learning curves, diagnostics


Each folder will fill up as the thesis progresses.

**How to View & Run the Notebooks**
View online (advisor-friendly):

You can open any notebook directly on GitHub — no setup required.

Run locally:
pip install -r requirements.txt
jupyter notebook


(requirements file will be added later)

**Thesis Objective (Short Summary)**

The objective of this thesis is to:

Model asset prices as stochastic mean-reverting OU processes

Estimate OU parameters from synthetic or real-like data

Build a simulated market environment reflecting realistic macro behaviour

Train an agent capable of trading based on statistical properties of the OU dynamics

Evaluate robustness across volatility regimes, shocks, and changing parameters

This combines stochastic process modeling, reinforcement learning, and financial market structure.
