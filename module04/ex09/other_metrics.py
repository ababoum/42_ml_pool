import numpy as np
from tools import is_matrix_valid


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not is_matrix_valid(y) or not is_matrix_valid(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    try:
        true = sum(1 for i in range(len(y)) if y[i] == y_hat[i])
        false = sum(1 for i in range(len(y)) if y[i] != y_hat[i])
        return (true) / (false + true)
    except:
        return None


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """

    if not is_matrix_valid(y) or not is_matrix_valid(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    try:
        tp = sum(1 for i in range(len(y))
                 if y[i] == y_hat[i] and y[i] == pos_label)
        fp = sum(1 for i in range(len(y))
                 if y[i] != y_hat[i] and y[i] != pos_label)
        if tp == 0 and fp == 0:
            return 0
        return tp / (tp + fp)
    except:
        return None


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not is_matrix_valid(y) or not is_matrix_valid(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    try:
        tp = sum(1 for i in range(len(y))
                 if y[i] == y_hat[i] and y[i] == pos_label)
        fn = sum(1 for i in range(len(y))
                 if y[i] != y_hat[i] and y[i] == pos_label)
        if tp == 0 and fn == 0:
            return 0
        return tp / (tp + fn)
    except:
        return None


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not is_matrix_valid(y) or not is_matrix_valid(y_hat):
        return None
    if y.shape != y_hat.shape:
        return None
    try:
        return 2 * precision_score_(y, y_hat, pos_label) * recall_score_(y, y_hat, pos_label) / \
            (precision_score_(y, y_hat, pos_label) +
             recall_score_(y, y_hat, pos_label))
    except ZeroDivisionError:
        return 0
    except:
        return None


if __name__ == "__main__":

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Example 1:
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # Output:
    0.5
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # Output:
    0.5
    # Precision
    # your implementation
    print(precision_score_(y, y_hat))
    # Output:
    0.4
    # sklearn implementation
    print(precision_score(y, y_hat))
    # Output:
    0.4
    # Recall
    # your implementation
    print(recall_score_(y, y_hat))
    # Output:
    0.6666666666666666
    # sklearn implementation
    print(recall_score(y, y_hat))
    # Output:
    0.6666666666666666
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat))
    # Output:
    0.5
    # sklearn implementation
    print(f1_score(y, y_hat))
    # Output:
    0.5

    print('*' * 50)
    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                     'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                 'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # Output:
    0.625
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # Output:
    0.625
    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='dog'))
    # Output:
    0.6
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog'))
    # Output:
    0.6
    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='dog'))
    # Output:
    0.75
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='dog'))
    # Output:
    0.75
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='dog'))
    # Output:
    0.6666666666666665
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='dog'))
    # Output:
    0.6666666666666665

    print('*' * 50)
    # Example 3:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                     'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                 'dog', 'norminet', 'dog', 'norminet'])
    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='norminet'))
    # Output:
    0.6666666666666666
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='norminet'))
    # Output:
    0.6666666666666666
    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='norminet'))
    # Output:
    0.5
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='norminet'))
    # Output:
    0.5
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='norminet'))
    # Output:
    0.5714285714285715
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='norminet'))
    # Output:
    0.5714285714285715
