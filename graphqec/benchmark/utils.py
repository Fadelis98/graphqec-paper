import time

import numpy as np
import torch
from scipy.stats import linregress

__all__ = [
    'timed',
    'format_table',
    'format_number',
    'get_lfr',
    'get_lfr_std',
    'log_lfr_fit_fn',
    'fit_log_lfr',
    'sub_threshold_fit_fn',
    'log_sub_threshold_fit_fn',
    'merge_subset_results',
]

def timed(func):
    """Helper function to time the execution of a function."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func()
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    else:
        start = time.time()
        result = func()
        end = time.time()
        elapsed_time = (end - start) * 1000
    return result, elapsed_time

def format_table(data, headers):
    """Helper function to format a table using built-in Python functionality."""
    col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *data)]
    header_row = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
    separator = "-+-".join("-" * width for width in col_widths)
    rows = [" | ".join(f"{str(item):<{width}}" for item, width in zip(row, col_widths)) for row in data]
    table = [header_row, separator] + rows
    return "\n".join(table)

def format_number(num: int | float) -> str:
    """Format large numbers with K, M, B, T suffixes"""
    for suffix in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1024.0:
            return f"{num:3.2f}{suffix}"
        num /= 1024.0
    return f"{num:.2f}T"


# metric functions

def get_lfr(error_rate, rmax):
    rmax = rmax + 1
    fid = 1 - 2*error_rate
    return (1-fid**(1/rmax))/2

def get_lfr_std(error_rate, rmax, num_samples):
    rmax = rmax + 1
    return (1/rmax)*(1 - 2*error_rate)**(1/rmax - 1)*np.sqrt(error_rate*(1 - error_rate)/num_samples)

def log_lfr_fit_fn(r,lfr,f0):
    return np.log(f0)+r*np.log(1-2*lfr)

def fit_log_lfr(
        logical_fidelities: np.ndarray,
        rounds: np.ndarray,
):
    
    if len(rounds) != len(logical_fidelities):
        raise ValueError("The number of rounds and logical fidelities must match.")

    log_fidelities = np.log(logical_fidelities)

    fit_results = linregress(rounds,log_fidelities)
    k = fit_results.slope
    b = fit_results.intercept

    epsilon = (1 - np.exp(k)) / 2
    f0 = np.exp(b)
    epsilon_std = k*(2*epsilon-1)/2*np.sqrt(fit_results.stderr)

    return f0,epsilon,epsilon_std

def sub_threshold_fit_fn(pn, a, p0, alpha, beta):
    """
    Fitting function for sub-threshold behavior.
    """
    p, n = pn
    return a * (p / p0) ** ((alpha * n**beta) / 2)

def log_sub_threshold_fit_fn(pn, a, p0, alpha, beta):
    """
    Logarithmic version of the fitting function for sub-threshold behavior.
    """
    A = np.log(a)
    B = alpha / 2
    C = - (alpha / 2) * np.log(p0)
    p, n = pn
    return A + B * n**beta * np.log(p) + C * n**beta

def merge_subset_results(means: np.ndarray[float], stds: np.ndarray[float]) -> dict:
    """
    Aggregate the mean and standard deviation values of multiple subsets to compute the global mean 
    and propagated uncertainties.

    Parameters:
        means: array of mean values from each subset.
        stds: array of standard deviation values from each subset.

    Returns:
        result: Dictionary containing the global mean and propagated standard deviation.
    """
    if len(means) != len(stds):
        raise ValueError("The lengths of 'means' and 'stds' must be the same.")

    # Global mean
    global_mean = np.mean(means, axis=0)

    # Propagated global standard deviation
    global_std = np.sqrt(np.sum(stds**2, axis=0) / len(stds))

    return global_mean, global_std
