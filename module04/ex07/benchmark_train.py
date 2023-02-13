import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from ridge import MyRidge as MyRdg


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


# Plotting the data only
def plot_comparison():
    # Load the data
    df = pd.read_csv('models.csv')
    mse_data = df['mse'].values
    model_x = (df['feature'] + '_' + df['degree'].astype(str) +
               ' / λ = ' + df['lambda'].astype(str)).values
    plt.rcParams.update({'figure.autolayout': True})

    # plot the comparison between the models (features / polynomial features)
    plt.bar(range(1, 13), mse_data, align='center')
    plt.xticks(range(1, 13), model_x, rotation=60)
    plt.show()

def plot_comparison_and_curve():
    # Load the data
    df = pd.read_csv('models.csv')
    mse_data = df['mse'].values
    model_x = (df['feature'] + '_' + df['degree'].astype(str) +
               ' / λ = ' + df['lambda'].astype(str)).values
    plt.rcParams.update({'figure.autolayout': True})

    # plot the comparison between the models (features / polynomial features)
    plt.bar(range(1, 13), mse_data, align='center')
    plt.xticks(range(1, 13), model_x, rotation=60)
    plt.show()

    ### plot the curve of the best model

    # retrieve raw data
    df_raw = pd.read_csv(
        '../data/space_avocado.csv')

    x_norm = zscore(np.array(df_raw['weight'])).reshape(-1, 1)
    y = np.array(df_raw['target']).reshape(-1, 1)

    # plot the best model: weight_1 vs target
    thetas = np.array(df[df['feature'] == 'weight'].iloc[0]
                    ['theta0':'theta1'].values).reshape(-1, 1)
    print(thetas)
    plt.scatter(df_raw['weight'], df_raw['target'], color='blue')
    for lmbd in range(0, 11, 2):
        plot_model = MyRdg(thetas=thetas, lambda_=lmbd/10)
        plot_model.fit_(x_norm, y)
        plt.plot(x_norm, plot_model.predict_(
            x_norm), label=f'lambda={lmbd/10}')
    plt.legend()
    plt.show()                                                                                                                                                                                                                                                        



if len(sys.argv) == 2 and sys.argv[1] == 'plot':
    plot_comparison_and_curve()
    exit()

# Load the data
df = pd.read_csv(
    '../data/space_avocado.csv')

# Split the data into train and test
x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
y = np.array(df['target']).reshape((-1, 1))

# Normalize the target data
# y = zscore(y)

# Split the data into train and test
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Split test data into validation and test
x_test, x_val, y_test, y_val = data_spliter(x_test, y_test, 0.5)

# prepare the benchmark file output
file = open("models.csv", "w")
print('feature,degree,theta0,theta1,theta2,theta3,theta4,mse,lambda', file=file)

# Prepare plotting parameters
model_x = []
model_y = []
default_alpha = 1e-3
default_max_iter = 10000

# Run linear regression models with polynomial hypothesis up to degree 4 on each feature


def handle_feature(x_train_set, y_train_set, x_test_set, y_test_set, x_val_set, y_val_set, lbl):
    for p in range(1, 5):
        # Normalize the feature data, and add polynomial features
        x_train_poly = add_polynomial_features(
            zscore(x_train_set).reshape(-1, 1), p)
        x_test_poly = add_polynomial_features(
            zscore(x_test_set).reshape(-1, 1), p)
        x_val_poly = add_polynomial_features(
            zscore(x_val_set).reshape(-1, 1), p)

        lambdas = {}
        for lmbda in range(0, 11, 2):
            # Train the model for each lambda
            model = MyRdg(
                thetas=np.zeros((x_train_poly.shape[1] + 1, 1)),
                alpha=default_alpha,
                max_iter=default_max_iter,
                lambda_=lmbda/10)
            model.fit_(x_train_poly, y_train_set)

            # Evaluate the model for each lambda (validation set)
            y_val_pred = model.predict_(x_val_poly)
            test = model.mse_(y_val_set, y_val_pred)
            lambdas[lmbda/10] = (test, np.array(model.thetas))

        # Select the best lambda
        scores = [lambdas[lmbda][0] for lmbda in lambdas]
        min_index = scores.index(min(scores))
        best_lambda = min_index * 0.2
        best_thetas = lambdas[best_lambda][1]

        thetas = [[0]] * 5
        for i in range(p):
            thetas[i] = best_thetas[i]
        thetas_display = ','.join([str(t[0]) for t in thetas])
        y_test_pred = model.predict_(x_test_poly)
        print(
            f'{lbl},{p},{thetas_display},{model.mse_(y_test_set, y_test_pred)},{best_lambda}', file=file)
        model_x.append(f'{lbl}_{p}')
        model_y.append(test)


# WEIGHT VS TARGET
handle_feature(x_train[:, 0], y_train, x_test[:, 0],
               y_test, x_val[:, 0], y_val, df.columns[1])
# PROD_DISTANCE VS TARGET
handle_feature(x_train[:, 1], y_train, x_test[:, 1],
               y_test, x_val[:, 1], y_val, df.columns[2])
# TIME_DELIVERY VS TARGET
handle_feature(x_train[:, 2], y_train, x_test[:, 2],
               y_test, x_val[:, 2], y_val, df.columns[3])

file.close()

