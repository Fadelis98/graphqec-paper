"""Statistics behind Table 1, Fig. 2, Fig. S3 and Table S4 (the paper's Methods conventions).

* Block event: at least one of the k logical observables is wrong in a shot of
  r = rmax + 1 rounds; block rate p_b = B / N.
* Marginal event: one wrong observable; marginal rate p_m = L / (N k).
* Effective per-round rate p_c = 1 - (1 - p)^(1/r) (both definitions).
* Block interval: fixed-N 95% Wilson interval with the circuit shot as the
  sampling unit, mapped per round; zero failures report 0 with the one-sided
  95% upper bound 1 - 0.05^(1/N).
* Marginal interval: Wilson interval with the effective trial count
  N_eff = N k (1 - p_m) / D, where D = Var(X)/E[X] and X is the number of wrong
  observables per shot, measured from per-shot error vectors; for k = 1 it is
  the block interval; without per-shot vectors no interval is given, only the
  deterministic bound p_b/k <= p_m <= p_b.

The arithmetic matches the scripts that generated the paper's display items
operation for operation, so recomputation from the released counts and
vectors reproduces the Source Data values exactly.
"""

from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054
ALPHA = 0.05


def wilson(successes, trials, z: float = Z95):
    """Two-sided Wilson score interval; vectorised, accepts non-integer counts (n_eff)."""
    x = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = x / n
        den = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / den
        half = z * np.sqrt(np.clip(p * (1.0 - p), 0.0, None) / n + z * z / (4.0 * n * n)) / den
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    lo = np.where(x <= 0, 0.0, lo)
    hi = np.where(x >= n, 1.0, hi)
    return lo, hi


def zero_failure_upper(trials, alpha: float = ALPHA):
    """One-sided 95% upper bound for zero failures in N trials: 1 - alpha^(1/N)."""
    n = np.asarray(trials, dtype=float)
    return -np.expm1(np.log(alpha) / n)


def per_round_event(rate, rounds):
    """Effective event rate p_c = 1 - (1 - p_L)^(1/r)."""
    p = np.asarray(rate, dtype=float)
    r = np.asarray(rounds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.expm1(np.log1p(-np.clip(p, 0.0, 1.0)) / r)
    return np.where(p >= 1.0, 1.0, out)


def per_round_parity(rate, rounds):
    """Sensitivity only: 0.5 * (1 - (1 - 2 p_L)^(1/r)), undefined for p_L >= 0.5."""
    p = np.asarray(rate, dtype=float)
    r = np.asarray(rounds, dtype=float)
    safe = np.clip(p, 0.0, 0.5 - 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.expm1(np.log1p(-2.0 * safe) / r) / 2.0
    return np.where(p < 0.5, out, np.nan)


def deff_wilson(shots, k, logical_errors, var_x):
    """Marginal 95% interval with the shot as sampling unit; returns (lo, hi, deff, n_eff)."""
    n = np.asarray(shots, dtype=float)
    k = np.asarray(k, dtype=float)
    m = np.asarray(logical_errors, dtype=float)
    v = np.asarray(var_x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = m / (n * k)
        ex = m / n
        deff = v / ex
        n_eff = n * k * (1.0 - p) / deff
    lo, hi = wilson(p * n_eff, n_eff)
    return lo, hi, deff, n_eff


def var_x_from_errors(errors, shots: int) -> float:
    """Var(X) over all N shots from the error vectors of the failing shots (non-failing shots have X = 0)."""
    wrong = np.asarray(errors, dtype=bool).sum(axis=1).astype(np.int64)
    e_x = wrong.sum() / shots
    e_x2 = (wrong.astype(float) ** 2).sum() / shots
    return float(e_x2 - e_x ** 2)


def table1_metrics(shots, block_failures, logical_errors, k, rounds, var_x=None) -> dict:
    """All Table 1 / Fig. 2 / Fig. S3 quantities for one (code, decoder, p) cell.

    ``var_x`` is Var(X) measured from per-shot vectors (None when they were not preserved).
    """
    n, b, m, kk, r = (float(v) for v in (shots, block_failures, logical_errors, k, rounds))
    if b > m or m > kk * b or b > n:
        raise ValueError("count relation B <= L <= kB, B <= N violated")
    nan = float("nan")
    has_var = var_x is not None and np.isfinite(var_x)
    block = b / n
    marg = m / (n * kk)
    lo, hi = wilson(b, n)
    if kk == 1:
        method = "k1_identical_to_block"
    elif m == 0:
        method = "zero_failures_block_upper_bound"
    elif has_var:
        method = "deff_adjusted_wilson"
    else:
        method = "none_per_shot_vectors_not_preserved"
    m_lo, m_hi, deff, n_eff = deff_wilson(n, kk, m, var_x if has_var else nan)
    use_deff, k1 = method == "deff_adjusted_wilson", method == "k1_identical_to_block"
    deff_out = deff if (has_var and m > 0) else ((1.0 - block) if kk == 1 else nan)
    n_eff_out = n_eff if use_deff else (n if k1 else nan)
    shot_lo = m_lo if use_deff else (lo if k1 else nan)
    shot_hi = m_hi if use_deff else (hi if k1 else nan)
    zero_ub = per_round_event(zero_failure_upper(n), r) if b == 0 else nan

    def f(x):
        x = float(x)
        return None if np.isnan(x) else x

    return {"shots": int(shots), "block_failures": int(block_failures), "logical_errors": int(logical_errors),
            "k": int(k), "rounds": int(rounds), "block_per_shot": block, "marginal_per_shot": marg,
            "block_shot_wilson_lo": f(lo), "block_shot_wilson_hi": f(hi),
            "block_round_event": f(per_round_event(block, r)),
            "block_round_event_lo": f(per_round_event(lo, r)), "block_round_event_hi": f(per_round_event(hi, r)),
            "block_zero_ub_round_event": f(zero_ub),
            "marginal_round_event": f(per_round_event(marg, r)),
            "marginal_interval_method": method, "deff": f(deff_out), "n_eff": f(n_eff_out),
            "marginal_shot_lo": f(shot_lo), "marginal_shot_hi": f(shot_hi),
            "marginal_round_event_lo": f(per_round_event(shot_lo, r)),
            "marginal_round_event_hi": f(per_round_event(shot_hi, r)),
            "marginal_zero_ub_round_event": f(zero_ub) if m == 0 else None,
            "marginal_analytic_bound_lo": f(per_round_event(block / kk, r)),
            "marginal_analytic_bound_hi": f(per_round_event(block, r)),
            "marginal_dependence_safe_lo": f(per_round_event(lo / kk, r)),
            "marginal_dependence_safe_hi": f(per_round_event(hi, r)),
            "block_round_parity": f(per_round_parity(block, r)),
            "marginal_round_parity": f(per_round_parity(marg, r))}


def integer_deff_bounds(shots: int, block_failures: int, logical_errors: int, k: int) -> tuple[float, float]:
    """Exact range of D = Var(X)/E[X] given only (N, B, L) and X_i in {1..k} on failing shots."""
    n, b, m = shots, block_failures, logical_errors
    e_x = m / n
    q, rem = divmod(m, b)
    low_sum_sq = (b - rem) * q * q + rem * (q + 1) ** 2
    extra = m - b
    n_full, rest = divmod(extra, k - 1)
    high_sum_sq = n_full * k * k + (1 + rest) ** 2 * (1 if rest else 0) + (b - n_full - (1 if rest else 0))
    var_lo = low_sum_sq / n - e_x ** 2
    var_hi = high_sum_sq / n - e_x ** 2
    return var_lo / e_x, var_hi / e_x


def ols_slope(x, y) -> tuple[float, float, float]:
    """Least squares y = slope * x + intercept via ``lstsq`` on [x, 1] (as in the timing audit); (slope, intercept, R^2)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    a = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(a, y, rcond=None)
    yhat = a @ np.array([slope, intercept])
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(slope), float(intercept), float(r2)
