'''Useful tools to check values, compute others, etc.'''

import numpy as np


def is_vector_valid(x):
    if not isinstance(x, np.ndarray):
        return False
    if len(x.shape) == 1 and x.shape[0] < 1:
        return False
    if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1):
        return False
    if x.size == 0:
        return False
    return True


def is_theta_valid(theta):
    if not isinstance(theta, np.ndarray):
        return False
    if len(theta.shape) == 1 and theta.shape != (2,):
        return False
    if len(theta.shape) == 2 and theta.shape != (2, 1):
        return False
    return True


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray):
            return None

        new_col = np.empty((x.shape[0], 1))
        np.ndarray.fill(new_col, 1.)

        if len(x.shape) == 1:
            return np.concatenate(
                [new_col, np.array(list([item] for item in x))], axis=1)
        return np.concatenate([new_col, x], axis=1)
    except:
        return None
