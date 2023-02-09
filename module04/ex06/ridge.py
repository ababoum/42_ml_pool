import numpy as np
from my_linear_regression import MyLinearRegression as my_lr
import matplotlib.pyplot as plt


class MyRidge(my_lr):
    """
    Description:
    My personnal Ridge regression class. It inherits from MyLinearRegression.
    Thetas must contain 2 parameters.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(np.copy(thetas.astype('float64')), alpha, max_iter)
        self.lambda_ = lambda_

        if not isinstance(lambda_, (int, float)) or lambda_ < 0:
            raise ("Lambda must be a positive float")

    def get_params_(self):
        return self.thetas

    def set_params(self, thetas):
        if not super().is_theta_valid(thetas):
            raise ("Wrong format for thetas")
        self.thetas = thetas

    def loss_elem_(self, y, y_hat):
        if not super().is_vector_valid(y) or \
                not super().is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        return (y - y_hat) ** 2 + self.lambda_ * np.sum(self.thetas ** 2)

    def loss_(self, y, y_hat):
        if not super().is_vector_valid(y) or \
                not super().is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        try:
            return np.sum(self.loss_elem_(y, y_hat)) / (2 * y.size)
        except:
            return None

    def fit_(self, x, y):
        try:
            if not super().is_vector_valid(x) or \
                    not super().is_vector_valid(y):
                return None
            if x.size != y.size:
                return None

            for _ in range(self.max_iter):
                gradient = self.gradient_(x, y)
                self.thetas[0] -= self.alpha * gradient[0]
                self.thetas[1] -= self.alpha * gradient[1]
        except Exception as e:
            print(e)
            return None

    def gradient_(self, x, y):
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
            if not super().is_vector_valid(x) or \
                not super().is_vector_valid(y) or \
                    not super().is_theta_valid(self.thetas):
                return None

            theta_calc = self.thetas.copy()
            theta_calc[0] = 0
            y_pred = super().predict_(x)
            x_p = super().add_intercept(x)

            return (x_p.T @ (y_pred - y) + (self.lambda_ * theta_calc)) / x.shape[0]

        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    x = np.arange(1, 6).astype('float64').reshape(-1, 1)
    y = np.array([11.52458968, 12.52458968, 13.52458968, 14.52458968, 15.52458968]).reshape(-1, 1)

    # Example 0:
    myridge0 = MyRidge(np.array([1, 1]).reshape(-1, 1), max_iter=10000)
    y_hat = myridge0.predict_(x)
    print(myridge0.thetas)

    print(f'Before: loss = {myridge0.loss_(y, y_hat)}')
    myridge0.fit_(x, y)
    y_hat = myridge0.predict_(x)
    print(myridge0.thetas)
    print(f'After: loss = {myridge0.loss_(y, y_hat)}')


    # Plot the result
    plt.scatter(x, y)
    plt.plot(x, y_hat, 'r')
    plt.show()


