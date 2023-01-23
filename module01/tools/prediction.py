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
    if not isinstance(x, np.ndarray) or (len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1))\
            or (len(x.shape) == 1 and x.shape[0] < 1):
        return None
    if not isinstance(theta, np.ndarray) or (theta.shape != (2, 1) and theta.shape != (2, )):
        return None

    X = add_intercept(x)
    if not isinstance(X, np.ndarray):
        return None
    return np.matmul(X, theta)


if __name__ == "__main__":
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([[5], [0]])
    print(repr(predict_(x, theta1)))
    # Ouput:
    # array([[5.], [5.], [5.], [5.], [5.]])
    # Do you remember why y_hat contains only 5’s here?
    # Example 2:
    theta2 = np.array([[0], [1]])
    print(repr(predict_(x, theta2)))
    # Output:
    # array([[1.], [2.], [3.], [4.], [5.]])
    # Do you remember why y_hat == x here?
    # Example 3:
    theta3 = np.array([[5], [3]])
    print(repr(predict_(x, theta3)))
    # Output:
    # array([[ 8.], [11.], [14.], [17.], [20.]])
    # Example 4:
    theta4 = np.array([[-3], [1]])
    print(repr(predict_(x, theta4)))
    # Output:
    # array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])

    print("ERRORS:")
    print(repr(predict_(np.array([]), np.array([[1], [2]]))))
    print(repr(predict_(np.array([[1], [2], [3]]), np.array([[1], [2], [3]]))))
    print(repr(predict_(np.array([[1, 5], [2, 3], [3, 2]]), np.array([[1], [2]]))))
