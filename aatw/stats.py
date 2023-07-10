"""Functions related to statistics"""

import numpy as np
from scipy import interpolate


def sample_from_pdf(pdf_func, start, end, num_samples, num_partition=1000):
    """Note: pdf_func must be vectorized"""
    x = np.linspace(start, end, num_partition)
    cdf = np.cumsum(pdf_func(x) * np.diff(x, prepend=x[0]))
    cdf /= cdf[-1]
    inverse_cdf = interpolate.interp1d(cdf, x)
    return inverse_cdf(np.random.uniform(0, 1, num_samples))


def bounded_sample(sampler, lower_bound=None, upper_bound=None):
    while True:
        sample = sampler()
        if lower_bound is not None and sample < lower_bound:
            continue
        if upper_bound is not None and sample > upper_bound:
            continue
        return sample


def poisson_process(rate, total_time):
    """Returns the array of occurance time given rate and total time.
    [rate] * [total_time] = 1
    """
    ts = []
    t = np.random.exponential(1/rate)
    while t < total_time:
        ts.append(t)
        t += np.random.exponential(1/rate)
    return ts