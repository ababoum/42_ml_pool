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
if len(sys.argv) == 2 and sys.argv[1] == 'plot':

    # Load the data
    df = pd.read_csv('models.csv')
    mse_data = df['mse'].values
    model_x = (df['feature'] + '_' + df['degree'].astype(str)).values
    plt.rcParams.update({'figure.autolayout': True})

    # plot the comparison between the models (features / polynomial features)
    plt.bar(range(1, 13), mse_data, align='center')
    plt.xticks(range(1, 13), model_x, rotation=60)
    plt.show()




# Load the data
df = pd.read_csv(
    '/mnt/nfs/homes/mababou/work/ml_pool/module04/data/space_avocado.csv')

# Split the data into train and test
x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
y = np.array(df['target']).reshape((-1, 1))

# Normalize the target data
# y = zscore(y)

# Split the data into train and test
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# prepare the benchmark file output
file = open("models.csv", "w")
print('feature,degree,theta0,theta1,theta2,theta3,theta4,mse,lambda', file=file)

# Prepare plotting parameters
model_x = []
model_y = []
default_alpha = 1e-4
default_max_iter = 100

# Run linear regression models with polynomial hypothesis up to degree 4 on each feature


def handle_feature(train_set, test_set, lbl):
    for p in range(1, 5):
        # Normalize the feature data
        x_train_normalized = zscore(train_set)
        x_test_normalized = zscore(test_set)
        x_train_poly = add_polynomial_features(
            x_train_normalized.reshape(-1, 1), p)
        x_test_poly = add_polynomial_features(
            x_test_normalized.reshape(-1, 1), p)

        lambdas = {}
        for lmbda in range(0, 11, 2):
            # Train the model for each lambda
            model = MyRdg(
                thetas=np.zeros((x_train_poly.shape[1] + 1, 1)),
                alpha=default_alpha,
                max_iter=default_max_iter,
                lambda_=lmbda/10)
            model.fit_(x_train_poly, y_train)

            # Evaluate the model for each lambda
            y_pred = model.predict_(x_test_poly)
            test = model.mse_(y_test, y_pred)
            lambdas[lmbda/10] = (test, np.array(model.thetas))

        # Select the best lambda
        scores = [lambdas[lmbda][0] for lmbda in lambdas]
        min_index = scores.index(min(scores))
        best_lambda = min_index * 0.2
        test = lambdas[best_lambda][0]
        best_thetas = lambdas[best_lambda][1]

        thetas = [[0]] * 5
        for i in range(p):
            thetas[i] = best_thetas[i]
        thetas_display = ','.join([str(t[0]) for t in thetas])
        print(f'{lbl},{p},{thetas_display},{test},{best_lambda}', file=file)
        model_x.append(f'{lbl}_{p}')
        model_y.append(test)


# WEIGHT VS TARGET
handle_feature(x_train[:, 0], x_test[:, 0], df.columns[1])
# PROD_DISTANCE VS TARGET
handle_feature(x_train[:, 1], x_test[:, 1], df.columns[2])
# TIME_DELIVERY VS TARGET
handle_feature(x_train[:, 2], x_test[:, 2], df.columns[3])

file.close()

# Plot the results
plt.rcParams.update({'figure.autolayout': True})
plt.bar(range(1, 13), model_y, align='center')
plt.xticks(range(1, 13), model_x, rotation=60)
plt.show()



# retrieve raw data
df_raw = pd.read_csv('/mnt/nfs/homes/mababou/work/ml_pool/module04/data/space_avocado.csv')
df = pd.read_csv('models.csv')

x_test_normalized = zscore(x_test[:, 0]).reshape(-1, 1)

# plot the best model: weight_1 vs target
thetas = np.array(df[df['feature'] == 'weight'].iloc[0]['theta0':'theta1'].values).reshape(-1, 1)
print(thetas)
plt.scatter(df_raw['weight'], df_raw['target'], color='blue')
for lmbd in range(0, 11, 2):
    plot_model = MyRdg(thetas=thetas, lambda_=lmbd/10)
    plot_model.fit_(x_test_normalized, y_test)
    plt.plot(x_test_normalized, plot_model.predict_(x_test_normalized), label=f'lambda={lmbd/10}')
plt.legend()
plt.show()
exit()
