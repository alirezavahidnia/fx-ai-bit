import numpy as np

def sharpe(returns, periods_per_year=252*24*4):  # ≈ 15m bars in trading days
    r = np.asarray(returns, dtype=float)
    if r.size == 0: return 0.0
    mu = r.mean()
    sigma = r.std(ddof=1) + 1e-12
    return (mu / sigma) * (periods_per_year ** 0.5)

def max_drawdown(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(ec)
    dd = (ec - peaks) / (peaks + 1e-12)
    return dd.min()
