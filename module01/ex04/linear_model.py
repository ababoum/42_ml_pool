import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


data = pd.read_csv("../data/are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1, 1)
Yscore = np.array(data['Score']).reshape(-1, 1)
linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

print(MyLR.mse_(Yscore, Y_model1))
# 57.60304285714282
print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
print(MyLR.mse_(Yscore, Y_model2))
# 232.16344285714285
print(mean_squared_error(Yscore, Y_model2))
# 232.16344285714285


print('*' * 25)

# Linear regression
linear_model1.fit_(Xpill, Yscore)
linear_model2.fit_(Xpill, Yscore)

plot_options = {
    'xlabel': 'Quantity of blue pill (in micrograms)',
    'ylabel': 'Space driving score',
    'xdatalabel': 'S true (pills)',
    'ydatalabel': 'S predict (pills)'
}

linear_model1.plot(Xpill, Yscore, plot_options)
linear_model2.plot(Xpill, Yscore, plot_options)


print('*' * 25)

n = 6
theta0 = np.linspace(80, 96, n)
theta1 = np.linspace(-14, -4, 100)
colors = plt.get_cmap('magma', n)
fig, axe = plt.subplots(1, 1, figsize=(15, 10))
for t0, color in zip(theta0, colors(range(n))):
    l_loss = []
    for t1 in theta1:
        plot_model = MyLR(np.array([[t0], [t1]]))
        Y_hat = plot_model.predict_(Xpill)
        l_loss += [plot_model.loss_(Yscore, Y_hat)]
    axe.plot(theta1, l_loss,
             label=r"J($\theta_0$ = " + f"{t0}," + r"$\theta_1$)",
             lw=2.5,
             c=color)
plt.grid()
plt.legend()
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"cost function J($\theta_0 , \theta_1$)")
axe.set_ylim([10, 150])
plt.show()
