'''Notion of gradient and gradient descent in machine learning
> Fit function'''

import copy
import numpy as np
from tools import is_vector_valid, is_theta_valid, add_intercept
from vec_gradient import simple_gradient
from prediction import predict_


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(x) or not is_vector_valid(y) or not is_theta_valid(theta):
        return None
    if x.size != y.size:
        return None
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        return None
    if not isinstance(max_iter, int) or max_iter <= 0:
        return None

    new_theta = np.array(theta, dtype=np.float64)
    for _ in range(max_iter):
        gradient = simple_gradient(x, y, theta)
        for i in range(new_theta.size):
            new_theta[i] -= alpha * gradient[i]

    return new_theta


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))
    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(repr(theta1))
    # Output:
    # array([[1.40709365],
    # [1.1150909 ]])
    # Example 1:
    print(repr(predict_(x, theta1)))
    # Output:
    # array([[15.3408728 ],
    # [25.38243697],
    # [36.59126492],
    # [55.95130097],
    # [65.53471499]])
