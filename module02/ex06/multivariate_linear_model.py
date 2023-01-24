import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
import sys

args = sys.argv[1:]
if len(args) != 1:
    print("Usage: python3 multivariate_linear_model.py <1, 2, or 3>")
    exit(1)


##############################################################################
# 								PART ONE								 	 #
##############################################################################

if args[0] == '1':
    data = pd.read_csv("../data/spacecraft_data.csv")

    ## AGE VS SELL_PRICE ##
    X = np.array(data[['Age']])
    Y = np.array(data[['Sell_price']])
    myLR_age = MyLR(thetas=[[850.0], [-30.0]], alpha=2.5e-4, max_iter=100000)
    y_pred = myLR_age.predict_(X[:, 0].reshape(-1, 1))
    print(f'Before: {myLR_age.mse_(y_pred, Y)}')
    myLR_age.fit_(X[:, 0].reshape(-1, 1), Y)
    y_pred = myLR_age.predict_(X[:, 0].reshape(-1, 1))
    print(f'After: {myLR_age.mse_(y_pred, Y)}')
    print(myLR_age.thetas)
    myLR_age.plot(X, Y, plot_options={'xlabel': r'$x_1$' + ': age (in years))',
                                      'ylabel': 'y: sell price (in keuros)',
                                      'xdatalabel': 'Sell price',
                                      'ydatalabel': 'Predicted sell price'})

    ## THRUST POWER VS SELL_PRICE ##
    print('*' * 42)
    X = np.array(data[['Thrust_power']])
    Y = np.array(data[['Sell_price']])
    myLR_thrust = MyLR(thetas=[[20.0], [4.5]], alpha=2.5e-5, max_iter=100000)
    y_pred = myLR_thrust.predict_(X[:, 0].reshape(-1, 1))
    print(f'Before: {myLR_thrust.mse_(y_pred, Y)}')
    myLR_thrust.fit_(X[:, 0].reshape(-1, 1), Y)
    y_pred = myLR_thrust.predict_(X[:, 0].reshape(-1, 1))
    print(f'After: {myLR_thrust.mse_(y_pred, Y)}')
    print(myLR_thrust.thetas)
    myLR_thrust.plot(X, Y, plot_options={'xlabel': r'$x_2$' + ': thrust power (in km/s)',
                                         'ylabel': 'y: sell price (in keuros)',
                                         'xdatalabel': 'Sell price',
                                         'ydatalabel': 'Predicted sell price'})

    ## DISTANCE VS SELL_PRICE ##
    print('*' * 42)
    X = np.array(data[['Terameters']])
    Y = np.array(data[['Sell_price']])
    myLR_distance = MyLR(thetas=[[700.0], [-2.5]],
                         alpha=2.5e-5, max_iter=100000)
    y_pred = myLR_distance.predict_(X[:, 0].reshape(-1, 1))
    print(f'Before: {myLR_distance.mse_(y_pred, Y)}')
    myLR_distance.fit_(X[:, 0].reshape(-1, 1), Y)
    y_pred = myLR_distance.predict_(X[:, 0].reshape(-1, 1))
    print(f'After: {myLR_distance.mse_(y_pred, Y)}')
    print(myLR_distance.thetas)
    myLR_distance.plot(X, Y, plot_options={'xlabel': r'$x_3$' + ': distance totalizer value of spacecraft (in Tmeters)',
                                           'ylabel': 'y: sell price (in keuros)',
                                           'xdatalabel': 'Sell price',
                                           'ydatalabel': 'Predicted sell price'})

##############################################################################
# 								PART TWO								 	 #
##############################################################################

elif args[0] == '2':
    data = pd.read_csv("../data/spacecraft_data.csv")
    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    Y = np.array(data[['Sell_price']])
    myLR = MyLR(thetas=[[1.0], [1.0], [1.0], [1.0]],
                alpha=1e-5, max_iter=600000)

    y_pred = myLR.predict_(X)
    print(f'Before: {myLR.mse_(y_pred, Y)}')

    myLR.fit_(X, Y)
    print(myLR.thetas)

    y_pred = myLR.predict_(X)
    print(f'After: {myLR.mse_(y_pred, Y)}')

    myLR.multiplot(X[:, 0], X, Y, plot_options={'xlabel': r'$x_1$' + ': age (in years))',
                                        'ylabel': 'y: sell price (in keuros)',
                                        'xdatalabel': 'Sell price',
                                        'ydatalabel': 'Predicted sell price'})

    myLR.multiplot(X[:, 1], X, Y, plot_options={'xlabel': r'$x_2$' + ': thrust power (in km/s)',
                                         'ylabel': 'y: sell price (in keuros)',
                                         'xdatalabel': 'Sell price',
                                         'ydatalabel': 'Predicted sell price'})

    myLR.multiplot(X[:, 2], X, Y, plot_options={'xlabel': r'$x_3$' + ': distance totalizer value of spacecraft (in Tmeters)',
                                           'ylabel': 'y: sell price (in keuros)',
                                           'xdatalabel': 'Sell price',
                                           'ydatalabel': 'Predicted sell price'})
