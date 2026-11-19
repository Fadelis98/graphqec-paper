"""Paired same-shot comparison used for the SI1000 study (Fig. 3, Table S6).

Block-failure ratio adapted/control with a 5,000-replicate paired shot bootstrap over the four
paired outcome cells (seed 10012026), exact McNemar test, and per-model Wilson summaries.
"""

import numpy as np


def unpack_errors(packed, k=12):
    """Bit-packed (little bit order) per-shot vectors -> (N, k) bool."""
    return np.unpackbits(np.asarray(packed), axis=1, count=k, bitorder="little").astype(bool)


def paired_statistics(control, adapted, bootstrap_seed=10012026):
    from scipy.stats import binomtest, norm

    control = np.asarray(control, dtype=bool)
    adapted = np.asarray(adapted, dtype=bool)
    if control.shape != adapted.shape or control.ndim != 2 or not len(control):
        raise ValueError("Paired vectors must have identical nonempty shot/output shape")
    shots, outputs = control.shape
    baseline = control.any(axis=1)
    candidate = adapted.any(axis=1)
    counts = np.bincount(2 * baseline.astype(int) + candidate.astype(int), minlength=4)
    improved, regressed = int(counts[2]), int(counts[1])
    base_fails, adapted_fails = int(baseline.sum()), int(candidate.sum())

    def metrics(errors):
        failures = int(errors.any(axis=1).sum())
        rate = failures / shots
        quantile = float(norm.ppf(.975))
        denominator = 1 + quantile**2 / shots
        center = (rate + quantile**2 / (2 * shots)) / denominator
        half = quantile * np.sqrt(rate * (1 - rate) / shots + quantile**2 / (4 * shots**2)) / denominator
        return {"shots": shots, "block_fails": failures, "logical_fails": int(errors.sum()),
                "block_ler": rate, "block_wilson_95": [float(center - half), float(center + half)],
                "block_zero_failure_upper_95": float(-np.expm1(np.log(.05) / shots)) if not failures else None,
                "marginal_ler": float(errors.mean()),
                "marginal_shot_cluster_se": float(errors.mean(axis=1).std(ddof=1) / np.sqrt(shots)) if shots > 1 else None}

    rng = np.random.default_rng(bootstrap_seed)
    resampled = rng.multinomial(shots, counts / shots, size=5000)
    denominator = resampled[:, 2] + resampled[:, 3]
    numerator = resampled[:, 1] + resampled[:, 3]
    estimable = base_fails > 0 and adapted_fails > 0 and bool((denominator > 0).all())
    ratios = numerator / denominator if estimable else None
    difference = (resampled[:, 1] - resampled[:, 2]) / shots
    upper_ratio = float(np.quantile(ratios, .95)) if estimable else None
    return {"control": metrics(control), "adapted": metrics(adapted),
            "paired_counts_00_01_10_11": counts.tolist(), "improved_shots": improved, "regressed_shots": regressed,
            "block_ratio": adapted_fails / base_fails if base_fails else None,
            "block_ratio_bootstrap_95": np.quantile(ratios, [.025, .975]).tolist() if estimable else None,
            "block_ratio_upper_95": upper_ratio,
            "block_difference": (adapted_fails - base_fails) / shots,
            "block_difference_bootstrap_95": np.quantile(difference, [.025, .975]).tolist(),
            "mcnemar_exact_two_sided_p": float(binomtest(improved, improved + regressed).pvalue) if improved + regressed else 1.0,
            "retention_ratio_upper_le_1_25": upper_ratio <= 1.25 if estimable else False,
            "bootstrap_replicates": 5000, "bootstrap_seed": bootstrap_seed,
            "interval_note": "Fixed checkpoints, paired shot bootstrap; not simultaneous across seeds/points; sparse/zero failures may be inconclusive"}


def paired_seed_mean_statistics(control, seed_errors, bootstrap_seed=10012026, replicates=5000):
    """Seed-averaged block failure rate and paired ratio vs control on shared test shots (Fig. 3).

    Resamples whole shots together with the joint (control, seed_1, ..., seed_m) outcome vector
    across 2^(m+1) categories so shared test shots are never pooled as m * N independent trials.
    """
    control = np.asarray(control, dtype=bool)
    seeds = [np.asarray(s, dtype=bool) for s in seed_errors]
    if control.ndim != 2 or not len(control) or not seeds or any(s.shape != control.shape for s in seeds):
        raise ValueError("Control and all seed error arrays must have identical nonempty (N, k) shape")
    shots = control.shape[0]
    m = len(seeds)
    num_cat = 1 << (m + 1)
    pattern = np.array([[(c >> k) & 1 for k in range(m, -1, -1)] for c in range(num_cat)])
    zero_fail = control.any(axis=1)
    seed_fails = [s.any(axis=1) for s in seeds]
    joint = zero_fail.astype(np.int64) * (1 << m)
    for idx, sf in enumerate(seed_fails):
        joint = joint + sf.astype(np.int64) * (1 << (m - 1 - idx))
    cat = np.bincount(joint, minlength=num_cat)
    boot = np.random.default_rng(bootstrap_seed).multinomial(shots, cat / shots, size=replicates)
    zero_b = boot @ pattern[:, 0]
    seeds_b = boot @ pattern[:, 1:].sum(axis=1)
    zero_count = int(zero_fail.sum())
    seed_counts = [int(sf.sum()) for sf in seed_fails]
    total_seed_fails = sum(seed_counts)
    rate = total_seed_fails / (float(m) * shots)
    rate_lo, rate_hi = np.quantile(seeds_b / (float(m) * shots), [0.025, 0.975])
    estimable = zero_count > 0 and total_seed_fails > 0 and bool((zero_b > 0).all())
    ratio = total_seed_fails / (float(m) * zero_count) if zero_count > 0 else None
    ratio_ci = np.quantile(seeds_b / (float(m) * zero_b), [0.025, 0.975]).tolist() if estimable else None
    return {"shots": shots, "num_seeds": m, "zero_block_fails": zero_count, "seed_block_fails": seed_counts,
            "mean_block_fails": total_seed_fails / float(m), "mean_block_ler": rate,
            "mean_block_bootstrap_95": [float(rate_lo), float(rate_hi)],
            "block_ratio": ratio, "block_ratio_bootstrap_95": ratio_ci,
            "joint_category_counts": cat.tolist(),
            "bootstrap_replicates": replicates, "bootstrap_seed": bootstrap_seed}
