import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as MyLR


# Load data

df_x = pd.read_csv(
    "../data/solar_system_census.csv")
df_y = pd.read_csv(
    "../data/solar_system_census_planets.csv")
x = np.array(df_x[['weight', 'height', 'bone_density']])
y = np.array(df_y[['Origin']])


# Split data

x_train, x_test, y_train, y_test = MyLR.data_spliter(x, y, 0.8)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

y_train_list = []
y_test_list = []
models = []
for i in range(4):
    y_train_list.append(np.array([(0, 1)[int(item) == i]
             for item in y_train]).reshape((-1, 1)))
    y_test_list.append(np.array([(0, 1)[int(item) == i]
             for item in y_test]).reshape((-1, 1)))
    models.append(
        MyLR(theta=np.zeros((x.shape[1] + 1, 1)), alpha=1e-4, max_iter=100))

# Train the models
y_pred_all_list = []
y_pred_test_list = []

for i in range(4):
    models[i].fit_(x_train, y_train_list[i])
    y_pred_all_list.append(models[i].predict_(x)) # we do it on the whole dataset
    y_pred_test_list.append(models[i].predict_(x_test))

y_pred_all_array = np.concatenate(y_pred_all_list, axis=1)
predictions = np.array(np.argmax(y_pred_all_array, axis=1)).reshape((-1, 1))

predictions_test = np.array(np.argmax([y_pred_test_list[j][i] for j in range(4)], axis=1))

# Evaluate the model
correct_guesses = sum(
    [1 if predictions_test[i] == y_test[i] else 0 for i in range(len(predictions_test))])
print(
    f'The ratio of correct predictions is {100 * correct_guesses / len(predictions_test):.2f}%')

