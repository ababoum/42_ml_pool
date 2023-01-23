'''Understand and manipulate the notion of hypothesis in machine learning.'''

import numpy as np
from tools import add_intercept


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    try:
        if not isinstance(x, np.ndarray) or (len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1))\
                or (len(x.shape) == 1 and x.shape[0] < 1):
            return None
        if not isinstance(theta, np.ndarray) or (theta.shape != (2, 1) and theta.shape != (2, )):
            return None

        X = add_intercept(x)
        if not isinstance(X, np.ndarray):
            return None
        return np.matmul(X, theta)
    except:
        return None