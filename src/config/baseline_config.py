# src/config/baseline_config.py

"""
Frozen OU baseline configuration.
Everything that defines the baseline must live here (single source of truth).
"""

# Data / OU estimation
OU_WINDOW = 126
DT = 1.0  # daily steps

# Signal thresholds (hysteresis)
ENTRY_Z = 1.5
EXIT_Z = 0.8
Z_CAP = 3.0

# Costs (IMPORTANT: store in BPS to match backtest_ou signature)
COST_BPS = 5.0  # 5 bps per unit position change

# Train/test split (used only for reporting metrics)
SPLIT_DATE = "2006-01-01"

# Positioning / exposure
EXPOSURE = 0.10
MAX_LEVERAGE = 1.0

# Vol targeting
VOL_TARGET = 0.006
VOL_WINDOW = 20
MAX_VOL_MULT = 2.0
VOL_MULT_FLOOR = 0.50

# Regime filter
KAPPA_MIN = 0.01
HL_MAX = 80.0

# Turnover reduction
POS_EMA_ALPHA = 0.15
REBALANCE_THRESH = 0.15

# Carry proxy
CARRY_BPS_PER_DAY = 0.2  # keep as-is if this is what you used yesterday

