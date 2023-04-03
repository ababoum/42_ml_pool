import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_logistic_regression import MyLogisticRegression as MyLR

zipcode = 0

if len(sys.argv) != 2:
    print("Usage: python3 mono_log.py -zipcode=x (x being 0, 1, 2 or 3)")
    exit()

if sys.argv[1] in ["-zipcode=" + str(i) for i in range(3)]:
    zipcode = int(sys.argv[1][-1])
else:
    print("Usage: python3 mono_log.py -zipcode=x (x being 0, 1, 2 or 3)")
    exit()

# Load data

df_x = pd.read_csv(
    "../data/solar_system_census.csv")
df_y = pd.read_csv(
    "../data/solar_system_census_planets.csv")
x = np.array(df_x[['weight', 'height', 'bone_density']])
y = np.array([(0, 1)[int(item) == zipcode]
             for item in df_y['Origin']]).reshape((-1, 1))

mylr = MyLR(theta=np.zeros((x.shape[1] + 1, 1)), alpha=1e-4, max_iter=100000)

# Split data

x_train, x_test, y_train, y_test = MyLR.data_splitter(x, y, 0.8)
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# Train the model

mylr.fit_(x_train, y_train)

# Evaluate the model
y_pred = mylr.predict_(x_test)
y_pred = [1 if item > 0.5 else 0 for item in y_pred]
correct_guesses = sum(
    [1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_pred))])
print(f'Thetas are: {mylr.theta}')
print(
    f'The ratio of correct predictions is {100 * correct_guesses / len(y_pred):.2f}%')

# Plot the data
y_pred = mylr.predict_(x)
y_pred = [1 if item > 0.5 else 0 for item in y_pred]
trues = [i for i in range(len(y)) if y_pred[i] == 1]
falses = [i for i in range(len(y)) if y_pred[i] == 0]

fig = plt.figure(figsize=(25, 12))

# weight vs height
fig.add_subplot(1, 3, 1)
plt.scatter(x[trues, 0], x[trues, 1], color='green', marker='o',
            label=f'From planet {zipcode}')
plt.scatter(x[falses, 0], x[falses, 1], color='red',
            marker='o', label=f'Not from planet {zipcode}')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.legend()

# height vs bone density
fig.add_subplot(1, 3, 2)
plt.scatter(x[trues, 1], x[trues, 2], color='green', marker='o',
            label=f'From planet {zipcode}')
plt.scatter(x[falses, 1], x[falses, 2], color='red',
            marker='o', label=f'Not from planet {zipcode}')
plt.xlabel('Height')
plt.ylabel('Bone density')
plt.legend()


# bone density vs weight
fig.add_subplot(1, 3, 3)
plt.scatter(x[trues, 2], x[trues, 0], color='green', marker='o',
            label=f'From planet {zipcode}')
plt.scatter(x[falses, 2], x[falses, 0], color='red',
            marker='o', label=f'Not from planet {zipcode}')
plt.xlabel('Bone density')
plt.ylabel('Weight')
plt.legend()


plt.show()
