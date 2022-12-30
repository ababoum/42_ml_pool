'''a function which adds an extra column of 1's on the left side of a given vector
or matrix'''


import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.any(x):
        return None

    new_col = np.empty((x.shape[0], 1))
    np.ndarray.fill(new_col, 1.)

    if len(x.shape) == 1:
        return np.concatenate(
            [new_col, np.array(list([item] for item in x))], axis=1)
    return np.concatenate([new_col, x], axis=1)


if __name__ == "__main__":
    x = np.arange(1, 6)
    print(repr(add_intercept(x)))
    # Output:
    # array([[1., 1.],
    # [1., 2.],
    # [1., 3.],
    # [1., 4.],
    # [1., 5.]])
    # Example 2:
    y = np.arange(1, 10).reshape((3, 3))
    print(repr(add_intercept(y)))
    # Output:
    # array([[1., 1., 2., 3.],
    # [1., 4., 5., 6.],
    # [1., 7., 8., 9.]])
    z = np.arange(0)
    print(repr(add_intercept(z)))

