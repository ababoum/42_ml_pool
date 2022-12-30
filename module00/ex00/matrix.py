'''Matrix and Vector classes'''


import copy
import numbers


def is_matrix_derived(obj):
    if isinstance(obj, Matrix) and type(obj) != Matrix:
        return True
    return False


class Matrix:
    def __init__(self, *args) -> None:
        # shape initializer
        if len(args) == 1 and isinstance(args[0], tuple):
            shape = args[0]
            if len(shape) != 2 or any(not isinstance(elem, int) or elem <= 0 for elem in shape):
                raise Exception("Matrix constructor: wrong shape parameter")
            self.shape = shape
            self.data = [[0.] * shape[1]] * shape[0]

        # data initializer
        elif len(args) == 1 and isinstance(args[0], list):
            rows = args[0]
            if len(rows) < 1 or \
                any(not isinstance(row, list) or
                    len(row) < 1 or
                    any(not isinstance(elem, (int, float, numbers.Number)) for elem in row) for row in rows):
                raise Exception(
                    "Matrix constructor: list initializer must only contain lists of numbers")
            self.shape = (len(rows), len(rows[0]))
            if any(len(row) != self.shape[1] for row in rows):
                raise Exception(
                    "Matrix constructor: rows must be of equal length")
            self.data = rows

        # other parameters
        else:
            raise Exception("Matrix constructor: unrecognized parameter(s)")

    def T(self):
        cpy = copy.deepcopy(self.data)
        T_data = [[cpy[i][j]
                   for i in range(self.shape[0])] for j in range(self.shape[1])]
        if type(self) == Matrix:
            return Matrix(T_data)
        return type(self)(T_data)

    # add : only matrices of same dimensions.
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise Exception(
                "Addition is impossible between matrices and other objects")

        if self.shape != other.shape:
            raise Exception(
                "Addition is impossible between matrices of different shapes")

        cpy = copy.deepcopy(self.data)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cpy[i][j] += other.data[i][j]

        if is_matrix_derived(self) and is_matrix_derived(other) \
            and type(self) == type(other):
            return type(self)(cpy)
        return Matrix(cpy)

    def __radd__(self, other):
        if not isinstance(other, Matrix):
            raise Exception(
                "Addition is impossible between matrices and other objects")
        return self.__add__(other)

    # sub : only matrices of same dimensions.
    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise Exception(
                "Substraction is impossible between matrices and other objects")
        return self.__add__(other * -1)

    def __rsub__(self, other):
        if not isinstance(other, Matrix):
            raise Exception(
                "Substraction is impossible between matrices and other objects")
        return other.__sub__(self)

    # div : only scalars.
    def __truediv__(self, other):
        if not isinstance(other, (int, float, numbers.Number)):
            raise Exception("Matrices can only be divided by a scalar")
        if other == 0:
            raise Exception("Division by zero is not possible")
        return self * (1 / other)

    def __rtruediv__(self):
        raise Exception("Matrices cannot divide other objects")

    # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    def __mul__(self, other):
        if isinstance(other, (int, float, numbers.Number)):
            cpy = copy.deepcopy(self.data)
            data = [[cpy[i][j] * other
                     for j in range(self.shape[1])] for i in range(self.shape[0])]
            new_instance = Matrix(data)
            return new_instance

        elif is_matrix_derived(other) or is_matrix_derived(self):
            if self.shape[1] != other.shape[0]:
                raise Exception(
                    "Matrix / Vector multiplication impossible: incompatible shapes")
            cpy = copy.deepcopy(self.data)
            data = [[sum(cpy[row][i] * other.data[i][col]
                         for i in range(self.shape[1])) for col in range(other.shape[1])]
                    for row in range(self.shape[0])]
            if self.shape[0] != 1 and other.shape[1] != 1:
                return Matrix(data)
            elif type(self) != Matrix:
                return type(self)(data)
            return type(other)(data)

        elif type(other) == Matrix and type(self) == Matrix:
            if self.shape[1] != other.shape[0]:
                raise Exception(
                    "Matrix multiplication impossible: incompatible shapes")
            cpy = copy.deepcopy(self.data)
            data = [[sum(cpy[row][i] * other.data[i][col]
                         for i in range(self.shape[1])) for col in range(other.shape[1])]
                    for row in range(self.shape[0])]
            return Matrix(data)
        else:
            raise Exception(
                "Matrix multiplication is not allowed with types other \
than Matrix, Vector and scalars")

    def __rmul__(self, other):
        if isinstance(other, (int, float, numbers.Number)):
            return self.__mul__(other)
        elif isinstance(other, Matrix):
            return other.__mul__(self)
        else:
            raise Exception(
                "Matrix multiplication is not allowed with types other \
than Matrix, Vector and scalars")

    def __str__(self):
        return f'{type(self).__name__}({self.data})'

    def __repr__(self):
        return f'{type(self).__name__}({self.data})'


class Vector(Matrix):
    def __init__(self, data) -> None:
        if not isinstance(data, list) or \
            (len(data) == 1 and not isinstance(data[0], list)) or \
            (len(data) == 1 and any(not isinstance(item, (float, int)) for item in data[0])) or \
                (len(data) > 1 and any(not isinstance(i, list) or len(i) != 1 or not isinstance(i[0], (float, int)) for i in data)):
            raise Exception(
                "Vector constructor must contain 1 list of 1-number lists OR 1 list with 1 list of scalars")

        super().__init__(data)

    def dot(self, other):
        if not type(other) == Vector or self.shape != other.shape:
            raise Exception(
                "Dot product only possible between vectors of matching shapes")
        if self.shape[0] == 1:
            # row vector
            return sum(self.data[0][i] * other.data[0][i] for i in range(self.shape[1]))
        else:
            # column vector
            return sum(self.data[i][0] * other.data[i][0] for i in range(self.shape[0]))
