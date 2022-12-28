'''Understand and manipulate the notion of loss function in machine learning.'''


import numpy as np


def is_vector_valid(x):
    if not isinstance(x, np.ndarray):
        return False
    if len(x.shape) == 1 and x.shape[0] < 1:
        return False
    if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1):
        return False
    if not np.any(x):
        return False
    return True


def is_theta_valid(theta):
    if not isinstance(theta, np.ndarray):
        return False
    if len(theta.shape) == 1 and theta.shape != (2,):
        return False
    if len(theta.shape) == 2 and theta.shape != (2, 1):
        return False
    if not np.any(theta):
        return False
    return True


def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """
    if not is_vector_valid(y) or not is_vector_valid(y_hat):
        return None
    if y.size != y_hat.size:
        return None
    return np.vdot(y - y_hat, y - y_hat) / (2 * y.size)


if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    # Example 1:
    print(loss_(X, Y))
    # Output:
    # 2.142857142857143
    # Example 2:
    print(loss_(X, X))
    # Output:
    # 0.0

