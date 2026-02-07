import numpy as np

def ou_zscore(x_t, mu_t, kappa_t, sigma_t) -> float:
    sigma_stat = sigma_t / np.sqrt(2.0 * kappa_t + 1e-12)
    return float((x_t - mu_t) / (sigma_stat + 1e-12))

def target_position_from_z(z: float, z_entry=1.5, z_exit=0.3, prev_pos=0) -> int:
    # mean reversion: high z => short, low z => long
    if z > z_entry:
        return -1
    if z < -z_entry:
        return +1
    if abs(z) < z_exit:
        return 0
    return int(prev_pos)
