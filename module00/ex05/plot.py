'''A function to plot the data and the prediction line (or regression line)'''


import numpy as np
from matplotlib import pyplot as plt
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


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    if not is_vector_valid(x) or not is_vector_valid(y) or \
            not is_theta_valid(theta) or x.size != y.size:
        print("Warning: plotting is impossible (wrong parameters)")
        return None
    try:
        plt.scatter(x, y)
        plt.plot(x, predict_(x, theta), 'r')
        plt.show()
    except:
        print("Warning: plotting is impossible (wrong parameters)")
    return None


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    # Example 1:
    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)

    print('*' * 25)
    plot(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    plot(np.array([0, 1]), np.array([0, 1]), np.array([1, 1]))
    plot(np.array([0, 2]), np.array([0, 0]), np.array([-1, 1]))
