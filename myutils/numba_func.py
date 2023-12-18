from numba import jit, f8, i8, b1, void
import numpy as np

# @jit(f8(f8[:, :], f8[:, :]), nopython=True)


@jit(nopython=True)
def calc_cos_sim_numba(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
