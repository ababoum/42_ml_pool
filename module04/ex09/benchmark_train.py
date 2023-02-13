import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from data_spliter import data_spliter
from polynomial_model_extended import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as mylogr
from other_metrics import f1_score_


# Prepare plotting/training parameters
model_x = []
model_y = []
default_alpha = 0.5
default_max_iter = 10000

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


# Plot the f1score comparison
def plot_data():
    try:
        models_hist = pd.read_csv('models.csv')
        zipcodes = models_hist['zipcode']
        lambdas = models_hist['lambda']
        scores = models_hist['f1_score']
    except:
        print("models.csv file not found / corrupted")
        return

    for i in range(len(models_hist)):
        model_x.append(f'zipcode={zipcodes[i]};λ={lambdas[i]}')
        model_y.append(scores[i])
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'figure.autolayout': True})
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.bar(model_x, model_y)
    plt.xticks(rotation=45)
    plt.title('f1_score comparison')
    plt.xlabel('zipcode / λ')
    plt.ylabel('f1_score')
    plt.show()



# plot data only
if len(sys.argv) == 2 and sys.argv[1] == 'plot':
    plot_data()
    exit()


# Load the data
df_x = pd.read_csv(
    '../data/solar_system_census.csv')
df_y = pd.read_csv(
    '../data/solar_system_census_planets.csv')

# Split the data into train and test
x = np.array(df_x[['height', 'weight', 'bone_density']])
y = np.array(df_y['Origin']).reshape((-1, 1))

# Normalize the data
x = zscore(x)

# Split the data into train and test
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Split test data into validation and test
x_test, x_val, y_test, y_val = data_spliter(x_test, y_test, 0.5)

# prepare the benchmark file output
file = open("models.csv", "w")
print('zipcode,lambda,f1_score', file=file)



# Run logistic regression models with polynomial hypothesis equal to degree 3 with different values of lambda

p = 3
# Normalize the feature data, and add polynomial features
x_train_poly = add_polynomial_features(
    (x_train), p)
x_test_poly = add_polynomial_features(
    (x_test), p)
x_val_poly = add_polynomial_features(
    (x_val), p)


for zipcode in range(4):
    y_train_zipcode = np.where(y_train == zipcode, 1, 0)
    y_val_zipcode = np.where(y_val == zipcode, 1, 0)
    for lmbda in range(0, 11, 2):
        # Train the model for each lambda
        model = mylogr(
            theta=np.zeros((x_train_poly.shape[1] + 1, 1)),
            alpha=default_alpha,
            max_iter=default_max_iter,
            lambda_=lmbda/10)
        model.fit_(x_train_poly, y_train_zipcode)

        # Evaluate the model for each lambda (validation set)
        y_val_pred = model.predict_(x_val_poly)
        guesses = np.where(y_val_pred > 0.5, 1, 0)
        score = f1_score_(y_val_zipcode, guesses)

        # Save the model for each lambda
        print(f'{zipcode},{lmbda/10},{score}', file=file)
        model_x.append(f'zipcode={zipcode};λ={lmbda/10}')
        model_y.append(score)


file.close()
plot_data()