import numpy as np
import pandas as pd
import sys
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR

# Load the data
df = pd.read_csv(
    '/mnt/nfs/homes/mababou/work/ml_pool/module02/data/space_avocado.csv')

# Split the data into train and test
x = np.array(df[['weight', 'prod_distance', 'time_delivery']])
y = np.array(df['target']).reshape((-1, 1))

# Split the data into train and test
x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# prepare the benchmark file output
file = open("models.csv", "w")
print('feature,degree,mse', file=file)

# Run linear regression models with polynomial hypothesis up to degree 4 on each feature

for i in range(3):
    for p in range(1, 5):
        x_train_poly = add_polynomial_features(x_train[:, i].reshape(-1, 1), p)
        x_test_poly = add_polynomial_features(x_test[:, i].reshape(-1, 1), p)

        # Train the model
        model = MyLR(thetas=np.zeros(
            (x_train_poly.shape[1] + 1, 1)), alpha=1e-7, max_iter=1000000)
        model.fit_(x_train_poly, y_train)
        y_pred = model.predict_(x_test_poly)
        test = model.mse_(y_test, y_pred)
        print(f'{df.columns[i+1]},{p},{test}', file=file)
