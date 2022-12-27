'''Matrix and Vector classes'''


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
                    any(not isinstance(elem, (int, float)) for elem in row) for row in rows):
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
        T_data = [[self.data[i][j]
                   for i in range(self.shape[0])] for j in range(self.shape[1])]
        return Matrix(T_data)

    # add : only matrices of same dimensions.
    def __add__(self, other):
        if self.shape != other.shape:
            raise Exception(
                "Addition is impossible between matrices of different shapes")

        add_data = self.data
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                add_data[i][j] += other.data[i][j]

        if type(self) == Vector and type(other) == Vector:
            return Vector(add_data)
        return Matrix(add_data)

    def __radd__(self, other):
        return self + other

    # sub : only matrices of same dimensions.
    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other - self

    # div : only scalars.
    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise Exception("Matrices can only be divided by a scalar")
        if other == 0:
            raise Exception("Division by zero is not possible")
        return self * (1 / other)

    def __rtruediv__(self):
        raise Exception("Matrices cannot divide other objects")

    # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[self.data[i][j] * other
                            for j in range(self.shape[1])] for i in range(self.shape[0])])

        elif type(other) == Matrix:
            if self.shape[1] != other.shape[0]:
                raise Exception(
                    "Matrix multiplication impossible: incompatible shapes")
            return Matrix([
                [sum(self.data[row][i] * other.data[i][col]
                     for i in range(self.shape[1])) for col in range(other.shape[1])]
                for row in range(self.shape[0])
            ])

        elif type(other) == Vector:
            if self.shape[1] != other.shape[0]:
                raise Exception(
                    "Matrix multiplication impossible: incompatible shapes")
            return Vector([
                [sum(self.data[row][i] * other.data[i][col]
                     for i in range(self.shape[1])) for col in range(other.shape[1])]
                for row in range(self.shape[0])
            ])
        else:
            raise Exception(
                "Matrix multiplication is not allowed with types other \
than Matrix, Vector and scalars")

    def __rmul__(self, other):
        return other * self

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


if __name__ == "__main__":
    m = Matrix((3, 3))
    print(m)

    m2 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    print(m2)

    m3 = m2.T()
    print(m3)

    m4 = m3 - m2
    print(m4)

    try:
        v3 = Vector([[1, 2], [3, 4]])
    except Exception as e:
        print(e)

    v = Vector([[1, 2, 3]]) 
    print(v.dot(v))
    v2 = Vector([[1]])
    print(v2.dot(v2))
    print(v2.shape)
