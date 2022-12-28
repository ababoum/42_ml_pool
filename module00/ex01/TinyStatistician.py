'''Statistics functions'''

import numbers
import numpy as np

class TinyStatistician:

    def __init__(self) -> None:
        pass

    def mean(self, x):
        '''Computes the mean value of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x ):
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
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x ):
            print("Function 'median' takes only a list or an array of numbers as parameter")
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
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x ):
            print("Function 'quartile' takes only a list or an array of numbers as parameter")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        return [self.percentile(x, 25), self.percentile(x, 75)]

    def percentile(self, x, p):
        '''Computes the expected percentile of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x ):
            print("Function 'percentile' takes only a list or an array of numbers as parameter")
            return None
        if p not in range(0, 101):
            print("The wished percentile must be between 0 and 100")
            return None
        if len(x) < 1:
            return None
        y = list(x)
        y.sort()
        res = float(p * (len(y) - 1) / 100)
        lim1 = int(res)
        lim2 = int(res) + 1

        if res.is_integer():
            return float(y[lim1])
        else:
            return float(y[lim1] * (lim2 - res) + y[lim2] * (res - lim1))

    def var(self, x):
        '''Computes the variance of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x ):
            print("Function 'var' takes only a list or an array of numbers as parameter")
            return None
        if len(x) <= 1:
            return None
        return float(sum((item - self.mean(x)) ** 2 for item in x) / (len(x) - 1))

    def std(self, x):
        '''Computes the standard deviation of a list of numbers'''
        if not isinstance(x, (list, np.ndarray)) or any(not isinstance(item, (float, int, numbers.Number)) for item in x ):
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
    # Expected result: [5.0, 89.0]

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
