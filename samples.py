from numba import njit, prange, float64, int64
import numpy as np


@njit(float64(float64[::1], float64[::1]), cache=True)
def calculate_numerator_over_denominator(values_n, values_d):
    mean_d = values_d.mean()
    return float(0) if mean_d == 0 else values_n.mean() / mean_d


@njit(float64[::1](float64[::1], float64[::1], int64), cache=True, parallel=True)
def create_randomized(values_n, values_d, n_samples):
    array_size = values_n.size
    samples = np.zeros(shape=(n_samples,), dtype=np.float64)

    for i in prange(n_samples):
        # create random index, so we choose consistent values in the numerator and denominator
        random_index = np.random.choice(np.arange(array_size), replace=True, size=array_size)
        samples[i] = calculate_numerator_over_denominator(values_n[random_index], values_d[random_index])
    return samples
