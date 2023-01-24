import numpy as np
from tools import is_matrix_valid


def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Return:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    None if y or y_hat is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not is_matrix_valid(y) or not is_matrix_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None

        return np.vdot(y_hat - y, y_hat - y) / (2 * y.size)
    except:
        return None


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Example 1:
    print(loss_(X, Y))
    # Output:
    # 2.142857142857143
    # Example 2:
    print(loss_(X, X))
    # Output:
    # 0.0