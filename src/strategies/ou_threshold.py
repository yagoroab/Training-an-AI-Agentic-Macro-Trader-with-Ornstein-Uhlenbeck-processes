import numpy as np


def ou_zscore(x_t, mu_t, kappa_t, sigma_t) -> float:
    """
    Robust z-score in log(VIX) space.
    """
    sigma_floor = 0.02

    if not np.isfinite(mu_t) or not np.isfinite(sigma_t):
        return np.nan

    sigma_eff = max(float(sigma_t), sigma_floor)
    z = (float(x_t) - float(mu_t)) / (sigma_eff + 1e-12)

    z_cap = 6.0
    return float(np.clip(z, -z_cap, z_cap))


def banded_position_from_z(
    z: float,
    prev_pos: float,
    z_entry: float = 1.25,
    z_exit: float = 0.35,
    z_cap: float = 3.0,
    max_leverage: float = 1.25,
    vxx_short_bias: float = 0.30,
) -> float:
    """
    Symmetric OU overlay around a structural short-VXX bias.

    Interpretation:
    - VXX has structural decay, so the neutral stance is a modest short.
    - OU signal tilts around that short bias.
    - Very negative z can still reduce the short or flip the strategy long.

    Behavior:
    - |z| <= z_exit  -> hold baseline short bias
    - z >= z_entry   -> increase short
    - z <= -z_entry  -> reduce short, possibly go long
    """
    if not np.isfinite(z):
        return float(prev_pos)

    z = float(z)
    z_entry = float(z_entry)
    z_exit = float(z_exit)
    z_cap = float(z_cap)
    max_leverage = float(max_leverage)
    vxx_short_bias = float(vxx_short_bias)

    # Cap bias so it cannot exceed max leverage
    vxx_short_bias = min(vxx_short_bias, max_leverage)

    # Inside neutral band: keep baseline short-VXX position
    if abs(z) <= z_exit:
        return float(-vxx_short_bias)

    # Strong positive z => larger short-VXX
    if z >= z_entry:
        if z >= z_cap:
            scaled = 1.0
        else:
            scaled = (z - z_entry) / max(1e-12, (z_cap - z_entry))
            scaled = float(np.clip(scaled, 0.0, 1.0))

        short_pos = vxx_short_bias + scaled * (max_leverage - vxx_short_bias)
        return float(-short_pos)

    # Strong negative z => reduce short bias and possibly go long
    if z <= -z_entry:
        az = abs(z)
        if az >= z_cap:
            scaled = 1.0
        else:
            scaled = (az - z_entry) / max(1e-12, (z_cap - z_entry))
            scaled = float(np.clip(scaled, 0.0, 1.0))

        # Range goes from -short_bias up to +max_leverage
        pos = -vxx_short_bias + scaled * (max_leverage + vxx_short_bias)
        return float(np.clip(pos, -max_leverage, max_leverage))

    # Intermediate region between exit and entry:
    # smoothly transition from baseline short bias toward directional tilt
    if z > z_exit:
        frac = (z - z_exit) / max(1e-12, (z_entry - z_exit))
        frac = float(np.clip(frac, 0.0, 1.0))
        short_pos = vxx_short_bias + frac * (0.65 * max_leverage - vxx_short_bias)
        return float(-short_pos)

    # z < -z_exit
    frac = (abs(z) - z_exit) / max(1e-12, (z_entry - z_exit))
    frac = float(np.clip(frac, 0.0, 1.0))
    pos = -vxx_short_bias + frac * (0.50 * max_leverage + vxx_short_bias)
    return float(np.clip(pos, -max_leverage, max_leverage))

