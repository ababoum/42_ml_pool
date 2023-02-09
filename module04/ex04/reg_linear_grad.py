import numpy as np
from tools import is_matrix_valid, is_vector_valid, is_theta_valid, add_intercept
from linear_regression_prediction import predict_

def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not is_matrix_valid(x) or \
            not is_vector_valid(y) or \
                not is_theta_valid(theta, x.shape[1] + 1):
            return None

        if not isinstance(lambda_, (int, float)):
            return None

        ret = []
        y_pred = predict_(x, theta)

        # 0th component
        sum0 = 0
        for i in range(x.shape[0]):
            sum0 += y_pred[i] - y[i]
        ret.append(sum0 / x.shape[0])

        # 1st to nth component
        for j in range(1, x.shape[1] + 1):
            sum = 0
            for i in range(x.shape[0]):
                sum += (y_pred[i] - y[i]) * x[i][j - 1]
            ret.append((sum / x.shape[0]) + (lambda_ * theta[j]) / x.shape[0])

        return np.array(ret).reshape(-1, 1)
    
    except Exception as e:
        print(e)
        return None


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
    lambda_: has to be a float.
    Return:
    A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    None if y, x or theta or lambda_ is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not is_matrix_valid(x) or \
            not is_vector_valid(y) or \
                not is_theta_valid(theta, x.shape[1] + 1):
            return None

        if not isinstance(lambda_, (int, float)):
            return None

        theta_calc = theta.copy()
        theta_calc[0] = 0
        y_pred = predict_(x, theta)
        x_p = add_intercept(x)

        return (x_p.T @ (y_pred - y) + (lambda_ * theta_calc)) / x.shape[0]
    
    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    # Example 1.1:
    print(reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99 ],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])
    # Example 1.2:
    print(vec_reg_linear_grad(y, x, theta, 1))
    # Output:
    # array([[ -60.99 ],
    # [-195.64714286],
    # [ 863.46571429],
    # [-644.52142857]])
    # Example 2.1:
    print(reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99 ],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])
    # Example 2.2:
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # Output:
    # array([[ -60.99 ],
    # [-195.86142857],
    # [ 862.71571429],
    # [-644.09285714]])
    # Example 3.1:
    print(reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99 ],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]])
    # Example 3.2:
    print(vec_reg_linear_grad(y, x, theta, 0.0))
    # Output:
    # array([[ -60.99 ],
    # [-196.07571429],
    # [ 861.96571429],
    # [-643.66428571]]