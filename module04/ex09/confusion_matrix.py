import numpy as np
import pandas as pd
from tools import is_matrix_valid


def confusion_matrix_(y, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will return a pandas DataFrame
    instead of a numpy array. (default=False)
    Return:
    The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
    None if any error.
    Raises:
    This function should not raise any Exception.
    """

    if y.shape != y_hat.shape or not is_matrix_valid(y) or not is_matrix_valid(y_hat):
        return None

    if labels == None:
        labels = np.unique(np.concatenate((y, y_hat), axis=0))
    else:
        if not isinstance(labels, list) or len(labels) == 0:
            return None
        labels = np.array(labels)

    ret = np.array([[0] * len(labels)] * len(labels))

    for i in range(len(y)):
        try:
            true_index = np.where(labels == y[i])[0][0]
            pred_index = np.where(labels == y_hat[i])[0][0]
            ret[true_index][pred_index] += 1
        except:
            pass

    if df_option:
        return pd.DataFrame(ret, index=labels, columns=labels)
    return ret


if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    y_hat = np.array([['norminet'], ['dog'], ['norminet'],
                     ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], [
                 'norminet'], ['dog'], ['norminet']])
    # Example 1:
    # your implementation
    print(confusion_matrix_(y, y_hat))
    # Output:
    # array([[0 0 0]
    # [0 2 1]
    # [1 0 2]])
    # sklearn implementation
    print(confusion_matrix(y, y_hat))
    # Output:
    # array([[0 0 0]
    # [0 2 1]
    # [1 0 2]])

    print('*' * 50)

    # Example 2:
    # your implementation
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    # Output:
    # array([[2 1]
    # [0 2]])
    # sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
    # Output:
    # array([[2 1]
    # [0 2]])

    print('*' * 50)

    # Example 3:
    print(confusion_matrix_(y, y_hat, df_option=True))
    # Output:
    # bird dog norminet
    # bird 0 0 0
    # dog 0 2 1
    # norminet 1 0 2
    # Example 2:
    print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
    # Output:
    # bird dog
    # bird 0 0
    # dog 0 2
