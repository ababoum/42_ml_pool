import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from data_spliter import data_spliter
from polynomial_model_extended import add_polynomial_features
from my_logistic_regression import MyLogisticRegression as mylogr
from other_metrics import f1_score_
from confusion_matrix import confusion_matrix_


# Prepare plotting/training parameters
model_x = []
model_y = []
default_alpha = 0.5
default_max_iter = 10000
best_lambda = 0

########################################
#### BEST MODEL FOUND -> LAMBA  = 0 ####
########################################


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
        models_hist = pd.read_csv(
            '/mnt/nfs/homes/mababou/work/ml_pool/module04/ex09/models.csv')
        # filter out the model to train
        models_hist = models_hist[models_hist['lambda'] != best_lambda]
        zipcodes = np.array(models_hist['zipcode'])
        lambdas = np.array(models_hist['lambda'])
        scores = np.array(models_hist['f1_score'])
    except Exception as e:
        print(e)
        print("models.csv file not found / corrupted")
        return

    for i in range(len(models_hist)):
        model_x.append(f'zipcode={zipcodes[i]};λ={lambdas[i]}')
        model_y.append(scores[i])
        print(
            f'The f1_score for zipcode={zipcodes[i]};λ={lambdas[i]} is {scores[i]}')
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'figure.autolayout': True})
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.bar(model_x, model_y)
    plt.xticks(rotation=45)
    plt.title('f1_score comparison')
    plt.xlabel('zipcode / λ')
    plt.ylabel('f1_score')
    plt.show()


# Load the data
df_x = pd.read_csv(
    '/mnt/nfs/homes/mababou/work/ml_pool/module04/data/solar_system_census.csv')
df_y = pd.read_csv(
    '/mnt/nfs/homes/mababou/work/ml_pool/module04/data/solar_system_census_planets.csv')

# Split the data into train and test
x = np.array(df_x[['height', 'weight', 'bone_density']])
y = np.array(df_y['Origin']).reshape((-1, 1))

# Normalize the data
x = zscore(x)

# Split the data into train and test
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))


# Run logistic regression models with polynomial hypothesis equal to degree 3 with different values of lambda

p = 3
# Normalize the feature data, and add polynomial features
x_train_poly = add_polynomial_features(
    (x_train), p)
x_test_poly = add_polynomial_features(
    (x_test), p)
x_all_poly = add_polynomial_features(
    (x), p)

models = []
y_pred_all_list = []
for zipcode in range(4):
    y_train_zipcode = np.where(y_train == zipcode, 1, 0)
    y_test_zipcode = np.where(y_test == zipcode, 1, 0)
    # Train the model for each lambda
    model = mylogr(
        theta=np.zeros((x_train_poly.shape[1] + 1, 1)),
        alpha=default_alpha,
        max_iter=default_max_iter,
        lambda_=best_lambda)
    models.append(model)
    model.fit_(x_train_poly, y_train_zipcode)

    # Evaluate the model for each lambda (validation set)
    y_val_pred = model.predict_(x_test_poly)
    guesses = np.where(y_val_pred > 0.5, 1, 0)
    score = f1_score_(y_test_zipcode, guesses)

    # Save the model for each lambda
    print(f'The f1_score for zipcode={zipcode};λ={best_lambda} is {score}')
    model_x.append(f'zipcode={zipcode};λ={best_lambda}')
    model_y.append(score)

    # Store data for the scatter plot
    y_pred_all_list.append(model.predict_(x_all_poly))

plot_data()

# Select the output with the highest probability (whole set)
y_pred_all_array = np.concatenate(y_pred_all_list, axis=1)
predictions = np.array(np.argmax(y_pred_all_array, axis=1)).reshape((-1, 1))

# plot the data for the best model


populations = []
for i in range(4):
    populations.append(
        [j for j in range(len(predictions)) if predictions[j] == i])

fig = plt.figure(figsize=(25, 12))

# weight vs height
fig.add_subplot(1, 3, 1)
for i in range(4):
    plt.scatter(x[populations[i], 0], x[populations[i], 1], marker='o',
                label=f'From planet {i}')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.legend()

# height vs bone density
fig.add_subplot(1, 3, 2)
for i in range(4):
    plt.scatter(x[populations[i], 1], x[populations[i], 2], marker='o',
                label=f'From planet {i}')
plt.xlabel('Height')
plt.ylabel('Bone density')
plt.legend()


# bone density vs weight
fig.add_subplot(1, 3, 3)
for i in range(4):
    plt.scatter(x[populations[i], 2], x[populations[i], 0], marker='o',
                label=f'From planet {i}')
plt.xlabel('Bone density')
plt.ylabel('Weight')
plt.legend()


plt.show()


# Plot the target data vs the predictions
matrix = confusion_matrix_(y, predictions, labels=[0, 1, 2, 3], df_option=True)
plt.figure(figsize = (10,7))
sn.heatmap(matrix, annot=True)
plt.show()