import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from data_splitter import data_splitter
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR


# Plotting the data only
if len(sys.argv) == 2 and sys.argv[1] == 'plot':

    # Load the data
    df = pd.read_csv('models.csv')
    mse_data = df['mse'].values
    model_x = (df['feature'] + '_' + df['degree'].astype(str)).values
    plt.rcParams.update({'figure.autolayout': True})
    plt.bar(range(1, 13), mse_data, align='center')
    plt.xticks(range(1, 13), model_x, rotation=60)
    plt.show()
    exit()


# Normalize the target data (function)
def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray) or x.size <= 1:
            return None

        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return None
        return (x - mean) / std

    except:
        return None


# Load the data
df = pd.read_csv(
    '../data/space_avocado.csv')

# Split the data into train and test
x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
y = np.array(df['target']).reshape((-1, 1))

# Normalize the target data
y = zscore(y)

# Split the data into train and test
x_train, x_test, y_train, y_test = data_splitter(x, y, 0.8)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# prepare the benchmark file output
file = open("models.csv", "w")
print('feature,degree,theta0,theta1,theta2,theta3,theta4,mse', file=file)

# Prepare plotting parameters
model_x = []
model_y = []

# Run linear regression models with polynomial hypothesis up to degree 4 on each feature
# WEIGHT VS TARGET
for p in range(1, 5):
    # Normalize the feature data
    x_train_normalized = zscore(x_train[:, 0])
    x_test_normalized = zscore(x_test[:, 0])
    lbl = df.columns[1]
    x_train_poly = add_polynomial_features(
        x_train_normalized.reshape(-1, 1), p)
    x_test_poly = add_polynomial_features(x_test_normalized.reshape(-1, 1), p)

    # Train the model
    model = MyLR(thetas=np.zeros(
        (x_train_poly.shape[1] + 1, 1)), alpha=1e-4, max_iter=100000)
    model.fit_(x_train_poly, y_train)

    # Evaluate the model
    y_pred = model.predict_(x_test_poly)
    test = model.mse_(y_test, y_pred)
    thetas = [[0]] * 5
    for i in range(p):
        thetas[i] = model.thetas[i]
    thetas_disp = ','.join([str(t[0]) for t in thetas])
    print(f'{lbl},{p},{thetas_disp},{test}', file=file)
    model_x.append(f'{lbl}_{p}')
    model_y.append(test)


# PROD_DISTANCE VS TARGET
for p in range(1, 5):
    # Normalize the feature data
    x_train_normalized = zscore(x_train[:, 1])
    x_test_normalized = zscore(x_test[:, 1])
    lbl = df.columns[2]
    x_train_poly = add_polynomial_features(
        x_train_normalized.reshape(-1, 1), p)
    x_test_poly = add_polynomial_features(x_test_normalized.reshape(-1, 1), p)

    # Train the model
    model = MyLR(thetas=np.zeros(
        (x_train_poly.shape[1] + 1, 1)), alpha=1e-4, max_iter=1000000)
    model.fit_(x_train_poly, y_train)

    # Evaluate the model
    y_pred = model.predict_(x_test_poly)
    test = model.mse_(y_test, y_pred)
    thetas = [[0]] * 5
    for i in range(p):
        thetas[i] = model.thetas[i]
    thetas_disp = ','.join([str(t[0]) for t in thetas])
    print(f'{lbl},{p},{thetas_disp},{test}', file=file)
    model_x.append(f'{lbl}_{p}')
    model_y.append(test)

# TIME_DELIVERY VS TARGET
for p in range(1, 5):
    # Normalize the feature data
    x_train_normalized = zscore(x_train[:, 2])
    x_test_normalized = zscore(x_test[:, 2])
    lbl = df.columns[3]
    x_train_poly = add_polynomial_features(
        x_train_normalized.reshape(-1, 1), p)
    x_test_poly = add_polynomial_features(x_test_normalized.reshape(-1, 1), p)

    # Train the model
    model = MyLR(thetas=np.zeros(
        (x_train_poly.shape[1] + 1, 1)), alpha=1e-4, max_iter=1000000)
    model.fit_(x_train_poly, y_train)

    # Evaluate the model
    y_pred = model.predict_(x_test_poly)
    test = model.mse_(y_test, y_pred)
    thetas = [[0]] * 5
    for i in range(p):
        thetas[i] = model.thetas[i]
    thetas_disp = ','.join([str(t[0]) for t in thetas])
    print(f'{lbl},{p},{thetas_disp},{test}', file=file)
    model_x.append(f'{lbl}_{p}')
    model_y.append(test)


plt.rcParams.update({'figure.autolayout': True})
plt.bar(range(1, 13), model_y, align='center')
plt.xticks(range(1, 13), model_x, rotation=60)
plt.show()
