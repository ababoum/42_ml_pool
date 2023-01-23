'''A function which plots the data, the prediction line, and the loss'''


import numpy as np
from matplotlib import pyplot as plt
from vec_loss import loss_, is_vector_valid, is_theta_valid
from prediction import predict_


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(x) or not is_vector_valid(y) or \
            not is_theta_valid(theta) or x.size != y.size:
        print("Warning: plotting is impossible (wrong parameters)")
        return None

    y_hat = predict_(x, theta)
    plt.scatter(x, y)
    plt.plot(x, y_hat, 'r')
    for point, value in zip(x, y):
        plt.plot([point, point], [value, predict_(
            np.array([point]), theta)[0]], 'b--')
    plt.title(f'Cost: {2 * loss_(y, predict_(x, theta)):.6f}')
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482,
                 13.14755699, 18.60682298, 14.14329568])
    # Example 1:
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)

    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)

    plot_with_loss(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    plot_with_loss(np.array([0, 1]), np.array([0, 1]), np.array([1, 1]))
    plot_with_loss(np.array([0, 2]), np.array([0, 0]), np.array([-1, 1]))
