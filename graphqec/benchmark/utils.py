import time
from typing import Tuple

import numpy as np
import torch
from scipy.stats import linregress

__all__ = [
    'timed',
    'format_table',
    'format_number',
    'fit_log_lfr',
    'extract_nkd_from_profile_name',
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


def fit_log_lfr(
        logical_fidelities: np.ndarray,
        rounds: np.ndarray,
):
    
    if len(rounds) != len(logical_fidelities):
        raise ValueError("The number of rounds and logical fidelities must match.")

    log_fidelities = np.log(logical_fidelities + 1e-10)

    fit_results = linregress(rounds,log_fidelities)
    k = fit_results.slope
    b = fit_results.intercept

    epsilon = (1 - np.exp(k)) / 2
    f0 = np.exp(b)
    epsilon_std = k*(2*epsilon-1)/2*np.sqrt(fit_results.stderr)

    return f0,epsilon,epsilon_std


def extract_nkd_from_profile_name(profile_name: str) -> Tuple[int, int, int]:
    """Helper to extract n, k, d from a profile string like "[[n,k,d]]" or "n,k,d".
    Assumes k is the number of logical qubits.
    """
    parts = profile_name.strip("[]").split(",")
    if len(parts) >= 3:
        try:
            n = int(parts[0].strip())
            k = int(parts[1].strip())
            d = int(parts[2].strip())
            return n, k, d
        except ValueError:
            pass
    print(
        f"Warning: Could not parse n, k, d from profile name: {profile_name}. Using (None, 1, None) as default (k=1)."
    )
    return -1, 1, -1  # Use -1 for n, d if parsing fails, but 1 for k (logical qubits)
