'''Notion of gradient and gradient descent in machine learning'''


import numpy as np
from tools import is_vector_valid, is_theta_valid, add_intercept


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(x) or not is_vector_valid(y) or not is_theta_valid(theta):
        return None
    if x.size != y.size:
        return None
    x_p = add_intercept(x)
    res = np.zeros(theta.shape)
    res = (x_p.T @ ((x_p @ theta) - y))
    return res / y.size


if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382,
                 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287,
                 46.6793434, 59.5585554]).reshape((-1, 1))
    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(repr(simple_gradient(x, y, theta1)))
    # Output:
    # array([[-19.0342574], [-586.66875564]])
    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(repr(simple_gradient(x, y, theta2)))
    # Output:
    # array([[-57.86823748], [-2230.12297889]])