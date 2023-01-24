'''A class that contains all methods necessary to perform linear regression'''

import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas)

        if not isinstance(self.alpha, (int, float)) or alpha <= 0 or alpha >= 1:
            raise ("Alpha must be strictly between 0 and 1")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ("Max_iter must be a positive integer")

    @staticmethod
    def is_matrix_valid(x):
        if not isinstance(x, np.ndarray):
            return False
        if len(x.shape) == 1 and x.shape[0] < 1:
            return False
        if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] < 1):
            return False
        if x.size == 0:
            return False
        return True

    @staticmethod
    def is_theta_valid(theta, n):
        if not isinstance(theta, np.ndarray):
            return False
        if len(theta.shape) == 1 and theta.shape != (n,):
            return False
        if len(theta.shape) == 2 and theta.shape != (n, 1):
            return False
        if theta.size == 0:
            return False
        return True

    @staticmethod
    def add_intercept(x):
        try:
            if not isinstance(x, np.ndarray):
                return None

            new_col = np.empty((x.shape[0], 1))
            np.ndarray.fill(new_col, 1.)

            if len(x.shape) == 1:
                return np.concatenate(
                    [new_col, np.array(list([item] for item in x))], axis=1)
            return np.concatenate([new_col, x], axis=1)
        except:
            return None

    @staticmethod
    def gradient(x, y, theta):
        try:
            if not MyLinearRegression.is_theta_valid(theta, x.shape[1] + 1):
                return None
            if not MyLinearRegression.is_matrix_valid(x) or \
                    not MyLinearRegression.is_matrix_valid(y):
                return None
            if x.shape[0] != y.shape[0]:
                return None
            x_p = MyLinearRegression.add_intercept(x)
            return x_p.T @ (x_p @ theta - y) / x.shape[0]
        except:
            return None

    def fit_(self, x, y):
        try:
            if not MyLinearRegression.is_matrix_valid(x) or \
                not MyLinearRegression.is_matrix_valid(y) or \
                    not MyLinearRegression.is_theta_valid(self.thetas, 1 + x.shape[1]):
                return None

            if x.shape[0] != y.shape[0]:
                return None

            new_theta = np.copy(self.thetas.astype('float64'))
            for _ in range(self.max_iter):
                grad = MyLinearRegression.gradient(x, y, new_theta)
                new_theta = new_theta - self.alpha * grad
            self.thetas = new_theta
        except:
            return None

    def predict_(self, x):
        try:
            if not MyLinearRegression.is_matrix_valid(x):
                return None

            X = MyLinearRegression.add_intercept(x)
            if not isinstance(X, np.ndarray):
                return None
            return X @ self.thetas
        except:
            return None

    def loss_elem_(self, y, y_hat):
        if not MyLinearRegression.is_matrix_valid(y) or \
                not MyLinearRegression.is_matrix_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        return (y - y_hat) ** 2

    def loss_(self, y, y_hat):
        if not MyLinearRegression.is_matrix_valid(y) or \
                not MyLinearRegression.is_matrix_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        try:
            return np.sum(self.loss_elem_(y, y_hat)) / (2 * y.size)
        except:
            return None

    @staticmethod
    def mse_(y, y_hat):
        if not MyLinearRegression.is_matrix_valid(y) or \
                not MyLinearRegression.is_matrix_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None

        return np.sum((y-y_hat) ** 2) / y.size

    def plot(self, x, y, plot_options=None):
        if not MyLinearRegression.is_matrix_valid(x) or \
                not MyLinearRegression.is_matrix_valid(y) or \
                x.shape[0] != y.shape[0]:
            print("Warning: plotting is impossible (wrong parameters)")
            return None

        try:
            if plot_options != None:
                plt.xlabel(plot_options['xlabel'])
                plt.ylabel(plot_options['ylabel'])
                plt.scatter(x, y, c='b', label=f'{plot_options["xdatalabel"]}')
                plt.plot(x, self.predict_(x), 'xg--',
                         label=f'{plot_options["ydatalabel"]}')
                plt.legend()
            else:
                plt.scatter(x, y, c='b')
                plt.plot(x, self.predict_(x), 'xg--')
            plt.show()
        except:
            print("Warning: plotting is impossible (wrong parameters)")
        return None

    def multiplot(self, feature, x, y, plot_options=None):
        if not MyLinearRegression.is_matrix_valid(x) or \
                not MyLinearRegression.is_matrix_valid(y) or \
                x.shape[0] != y.shape[0]:
            print("Warning: plotting is impossible (wrong parameters)")
            return None

        try:
            if plot_options != None:
                plt.xlabel(plot_options['xlabel'])
                plt.ylabel(plot_options['ylabel'])
                plt.scatter(feature, y, c='b', label=f'{plot_options["xdatalabel"]}')
                plt.plot(feature, self.predict_(x), 'og',
                         label=f'{plot_options["ydatalabel"]}')
                plt.legend()
            else:
                plt.scatter(feature, y, c='b')
                plt.plot(feature, self.predict_(x), 'og')
            plt.show()
        except:
            print("Warning: plotting is impossible (wrong parameters)")
        return None
