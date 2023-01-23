'''Implementation of the loss function'''


import numpy as np
from prediction import predict_


def is_vector_valid(x):
    if not isinstance(x, np.ndarray):
        return False
    if len(x.shape) == 1 and x.shape[0] < 1:
        return False
    if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1):
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


def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(y) or not is_vector_valid(y_hat):
        return None
    if y.size != y_hat.size:
        return None
    return (y - y_hat) ** 2 / (2 * y.size)


def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(y) or not is_vector_valid(y_hat):
        return None
    if y.size != y_hat.size:
        return None
    try:
        return np.sum(loss_elem_(y, y_hat))
    except:
        return None


if __name__ == "__main__":
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
    # Example 1:
    print(repr(loss_elem_(y1, y_hat1)))
    # Output:
    # array([[0.], [1], [4], [9], [16]])
    # Example 2:
    print(loss_(y1, y_hat1))
    # Output:
    # 3.0
    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
    # Example 3:
    print(loss_(y2, y_hat2))
    # Output:
    # 2.142857142857143
    # Example 4:
    print(loss_(y2, y2))
    # Output:
    # 0.0

    print('*' * 25)
    x = np.arange(1, 10)
    print(repr(loss_elem_(x, x)))
    print(repr(loss_(x, x)))

    print('*' * 25)
    y_hat = np.array([[1], [2], [3], [4]])
    y = np.array([[0], [0], [0], [0]])
    print(repr(loss_elem_(y, y_hat)))
    print(repr(loss_(y, y_hat)))
