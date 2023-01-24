import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR


data = pd.read_csv(
    "../data/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
Yscore = np.array(data["Score"]).reshape(-1, 1)


thetas456 = [
    np.array([[-20], [160], [-80], [10], [-1]]),
    np.array([[1140], [-1850], [1110], [-305], [40], [-2]]),
    np.array([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]])
]

models = [add_polynomial_features(Xpill, i) for i in range(1, 7)]
MyLRs = [MyLR(
    thetas=(np.zeros((i + 1, 1)), thetas456[i-4])[i >= 4],
    alpha=(1e-7, 1e-10)[i >= 4],
    max_iter=100000)
    for i in range(1, 7)]
MSE_list = []

for i, model in enumerate(models):
    MyLRs[i].fit_(model, Yscore)
    y_pred = MyLRs[i].predict_(model)
    MSE_list.append(MyLRs[i].mse_(Yscore, y_pred))
    print(f'Degree {i+1} evaluation score: {MSE_list[i]}')
    print(f'Coefficients: {MyLRs[i].thetas}')

# Plots a bar plot showing the MSE score of the models in function of the polynomial
# degree of the hypothesis

plt.bar(range(1, 7), MSE_list)
plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.show()


# Plots the 6 models and the data points on the same figure.

plt.scatter(Xpill, Yscore, color='blue', label='Data points')
for i, model in enumerate(models):
    sample = np.linspace(np.min(Xpill), np.max(Xpill), 100).reshape(-1, 1)
    sample_extended = add_polynomial_features(sample, i + 1)
    plt.plot(sample, MyLRs[i].predict_(
        sample_extended), label=f'Degree {i + 1}')
    plt.legend()
plt.show()