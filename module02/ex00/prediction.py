import numpy as np
from tools import is_matrix_valid, is_theta_valid, add_intercept


def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not matching.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not is_matrix_valid(x) or \
                not is_theta_valid(theta, x.shape[1] + 1):
            return None

        # x_p = add_intercept(x)
        # return x_p @ theta

        ret = np.empty((x.shape[0], 1))
        for i in range(x.shape[0]):
            ret[i] = theta[0]
            for j in range(1, x.shape[1] + 1):
                ret[i] += theta[j] * x[i][j - 1]
        return ret

    except:
        return None


if __name__ == "__main__":
    x = np.arange(1,13).reshape((4,-1))
    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta1))
    # Ouput:
    # array([[5.], [5.], [5.], [5.]])
    # Do you understand why y_hat contains only 5’s here?
    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta2))
    # Output:
    # array([[ 1.], [ 4.], [ 7.], [10.]])
    # Do you understand why y_hat == x[:,0] here?
    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print(simple_predict(x, theta3))
    # Output:
    # array([[ 9.64], [24.28], [38.92], [53.56]])
    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(simple_predict(x, theta4))
    # Output:
    # array([[12.5], [32. ], [51.5], [71. ]])