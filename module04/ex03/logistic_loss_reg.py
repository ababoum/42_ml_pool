import numpy as np
from tools import is_vector_valid, is_matrix_valid


def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for loop. The two arrays must have the same shapes.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized loss as a float.
    None if y, y_hat, or theta is empty numpy.ndarray.
    None if y and y_hat do not share the same shapes.
    Raises:
    This function should not raise any Exception.
    """
    if not is_vector_valid(y) or \
        not is_vector_valid(y_hat) or \
            not is_vector_valid(theta):
        return None

    if y.shape != y_hat.shape:
        return None

    if not isinstance(lambda_, (int, float)):
        return None

    try:
        m = y.shape[0]
        y_calc = np.vdot(y, np.log(y_hat)) + np.vdot((1 - y), np.log(1 - y_hat))
        theta_calc = np.array(theta[1:])
        theta_sq = np.vdot(theta_calc, theta_calc)
        return float(-y_calc + lambda_ * theta_sq * 0.5) / m


    except:
        return None


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
    # Example :
    print(reg_log_loss_(y, y_hat, theta, .5))
    # Output:
    0.43377043716475955
    # Example :
    print(reg_log_loss_(y, y_hat, theta, .05))
    # Output:
    0.13452043716475953
    # Example :
    print(reg_log_loss_(y, y_hat, theta, .9))
    # Output:
    0.6997704371647596
