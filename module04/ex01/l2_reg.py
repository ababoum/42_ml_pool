import numpy as np
from tools import is_vector_valid


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(theta):
        return None

    try:
        ret = 0
        for i in range(1, theta.shape[0]):
            ret += theta[i] ** 2

        return float(ret)
    except:
        return None


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(theta):
        return None

    try:
        theta_calc = np.array(theta[1:])
        return float(np.vdot(theta_calc, theta_calc))
    except:
        return None


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Example 1:
    print(iterative_l2(x))
    # Output:
    911.0
    # Example 2:
    print(l2(x))
    # Output:
    911.0
    y = np.array([3,0.5,-6]).reshape((-1, 1))
    # Example 3:
    print(iterative_l2(y))
    # Output:
    36.25
    # Example 4:
    print(l2(y))
    # Output:
    36.25