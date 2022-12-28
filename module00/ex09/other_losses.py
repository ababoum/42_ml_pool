'''Deepen the notion of loss function in machine learning.
Implement MSE, RMSE, MAE, and R2score'''

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


def mse_elem(y, y_hat):
    if not is_vector_valid(y) or not is_vector_valid(y_hat):
        return None
    if y.size != y_hat.size:
        return None
    return (y - y_hat) ** 2


def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    elem = mse_elem(y, y_hat)
    if not isinstance(elem, np.ndarray):
        return None
    return np.sum(elem) / y.size


def rmse_elem(y, y_hat):
    if not is_vector_valid(y) or not is_vector_valid(y_hat):
        return None
    if y.size != y_hat.size:
        return None
    return (y - y_hat) ** 2


def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    elem = rmse_elem(y, y_hat)
    if not isinstance(elem, np.ndarray):
        return None
    return (np.sum(elem) / y.size) ** 0.5


def mae_elem(y, y_hat):
    if not is_vector_valid(y) or not is_vector_valid(y_hat):
        return None
    if y.size != y_hat.size:
        return None
    return abs(y - y_hat)


def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    elem = mae_elem(y, y_hat)
    if not isinstance(elem, np.ndarray):
        return None
    return (np.sum(elem) / y.size)


def r2score_elem(y):
    if not is_vector_valid(y):
        return None
    return (y - np.mean(y)) ** 2


def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    r_elem = r2score_elem(y)
    m_elem = mse_elem(y, y_hat)
    if not isinstance(r_elem, np.ndarray) or not isinstance(m_elem, np.ndarray):
        return None
    if np.sum(r_elem) == 0:
        return None
    return 1 - (np.sum(m_elem) / np.sum(r_elem))


if __name__ == "__main__":

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from math import sqrt
    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    # Mean squared error
    # your implementation
    print(mse_(x, y))
    # Output:
    # 4.285714285714286
    # sklearn implementation
    print(mean_squared_error(x, y))
    # Output:
    # 4.285714285714286
    # Root mean squared error
    # your implementation
    print(rmse_(x, y))
    # Output:
    # 2.0701966780270626
    # sklearn implementation not available: take the square root of MSE
    print(sqrt(mean_squared_error(x, y)))
    # Output:
    # 2.0701966780270626
    # Mean absolute error
    # your implementation
    print(mae_(x, y))
    # Output:
    # 1.7142857142857142
    # sklearn implementation
    print(mean_absolute_error(x, y))
    # Output:
    # 1.7142857142857142
    # R2-score
    # your implementation
    print(r2score_(x, y))
    # Output:
    # 0.9681721733858745
    # sklearn implementation
    print(r2_score(x, y))
    # Output:
    # 0.9681721733858745
