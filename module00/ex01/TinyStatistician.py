'''Statistics functions'''

import numbers
import numpy as np
import math


class TinyStatistician:

    def __init__(self) -> None:
        pass

    def mean(self, x):
        '''Computes the mean value of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'mean' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        total = 0.0
        for item in x:
            total += item
        return float(total / len(x))

    def median(self, x):
        '''Computes the median value of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print(
                "Function 'median' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        if len(y) % 2 == 1:
            return float(y[len(y) // 2])
        return float((y[len(y) // 2] + y[-1 + len(y) // 2]) / 2)

    def quartile(self, x):
        '''Computes the first and third quartiles of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print(
                "Function 'quartile' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        lim1 = min(0, int(math.ceil(0.25 * len(y))) - 1)
        lim2 = min(0, int(math.ceil(0.75 * len(y))) - 1)
        return [y[lim1], y[lim2]]

    def percentile(self, x, p):
        '''Computes the expected percentile of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print(
                "Function 'percentile' takes only a list or an array of numbers as parameter")
            return None
        if p not in range(0, 101):
            print("The requested percentile must be between 0 and 100")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        res = min(0, int(math.ceil(p * (len(y)) / 100)) - 1)
        return float(y[int(res)])

    def var(self, x):
        '''Computes the variance of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'var' takes only a list or an array of numbers as parameter")
            return None
        if len(x) <= 1:
            return None
        mean = self.mean(x)
        return sum((item - mean) ** 2 for item in x) / (len(x) - 1)
        

    def std(self, x):
        '''Computes the standard deviation of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x):
            print("Function 'var' takes only a list or an array of numbers as parameter")
            return None
        var = self.var(x)
        if not var:
            return None
        return var ** 0.5


if __name__ == "__main__":
    tstat = TinyStatistician()
    a = [1, 42, 300, 10, 59]
    b = [1, 5, 7, 89]
    print(tstat.mean(a))
    # Expected result: 82.4
    print(tstat.mean(b))
    # Expected result: 25.5
    print(tstat.median(a))
    # Expected result: 42.0
    print(tstat.median(b))
    # # Expected result: 6.0
    print(tstat.quartile(a))
    # Expected result: [10.0, 59.0]
    print(tstat.quartile(b))
    # Expected result: [4.0, 27.5]

    print('*' * 25)
    print(tstat.percentile(a, 10))
    # 4.6
    print(tstat.percentile(a, 15))
    # 6.4
    print(tstat.percentile(a, 20))
    # 8.2
    print('*' * 25)

    print(tstat.var(a))
    # Expected result: 12279.439999999999
    print(tstat.var(b))
    # Expected result: 1348.75
    print(tstat.std(a))
    # Expected result: 110.81263465868862
    print(tstat.std(b))
    # Expected result: 36.72533185690771

    print('*' * 25)
    c = np.array(range(101))
    print(tstat.percentile(c, 50))
    print(tstat.percentile(c, 25))
    print(tstat.mean(c))
    print(tstat.median(c))

    print('*' * 25)
    data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
    epsilon = 1e-5
    err = "Error, grade 0 :("
    tstat = TinyStatistician()
    assert abs(tstat.mean(data) - 499.2) < epsilon, err
    assert abs(tstat.median(data) - 210.5) < epsilon, err
    quartile = tstat.quartile(data)
    assert abs(quartile[0] - 18) < epsilon, err
    assert abs(quartile[1] - 650) < epsilon, err
    assert abs(tstat.percentile(data, 10) - 3) < epsilon, err
    assert abs(tstat.percentile(data, 28) - 18) < epsilon, err
    assert abs(tstat.percentile(data, 83) - 754) < epsilon, err
    print(tstat.var(data))
    assert abs(tstat.var(data) - 654661) < epsilon * 1e5, err
    print(tstat.std(data))
    assert abs(tstat.std(data) - 809.11) < epsilon * 1e5, err
