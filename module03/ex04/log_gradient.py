import numpy as np
from tools import is_matrix_valid, is_theta_valid
from log_pred import logistic_predict_


def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not is_matrix_valid(x) or not is_matrix_valid(y) or not is_theta_valid(theta, x.shape[1] + 1):
            return None
        if x.shape[0] != y.shape[0]:
            return None

        grad = np.array([0] * theta.shape[0],
                        dtype=np.float64).reshape((-1, 1))
        y_pred = logistic_predict_(x, theta)

        grad[0, 0] = sum(y_pred_i - y_i for y_pred_i,
                         y_i in zip(y_pred, y)) / x.shape[0]
        for i in range(1, x.shape[1] + 1):
            grad[i, 0] = sum(
                (y_pred_i - y_i) * x_i for y_pred_i, y_i, x_i in zip(y_pred, y, x[:, i - 1])) / x.shape[0]
        return grad

    except:
        return None


if __name__ == "__main__":
    # Example 1:
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    print(log_gradient(x1, y1, theta1), end="\n\n")
    # Output:
    # array([[-0.01798621],
    # [-0.07194484]])
    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(log_gradient(x2, y2, theta2), end="\n\n")
    # Output:
    # array([[0.3715235 ],
    # [3.25647547]])
    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(log_gradient(x3, y3, theta3), end="\n\n")
    # Output:
    # array([[-0.55711039],
    # [-0.90334809],
    # [-2.01756886],
    # [-2.10071291],
    # [-3.27257351]])
