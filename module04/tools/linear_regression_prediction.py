import numpy as np
from tools import is_matrix_valid, is_theta_valid, add_intercept


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    	x: has to be an numpy.array, a matrix of dimension m * n.
    	theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
    	y_hat as a numpy.array, a vector of dimension m * 1.
    	None if x or theta are empty numpy.array.
    	None if x or theta dimensions are not matching.
    	None if x or theta is not of expected type.
    Raises:
    	This function should not raise any Exception.
    """
    try:
        if not is_matrix_valid(x) or \
                not is_theta_valid(theta, x.shape[1] + 1):
            return None

        x_p = add_intercept(x)
        return x_p @ theta

    except:
        return None