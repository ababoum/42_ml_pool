'''Matrix and Vector classes'''


class Matrix:
    def __init__(self) -> None:
        pass

    # add : only matrices of same dimensions.
    def __add__(self):
        pass

    def __radd__(self):
        pass

    # sub : only matrices of same dimensions.
    def __sub__(self):
        pass

    def __rsub__(self):
        pass

    # div : only scalars.
    def __truediv__(self):
        pass

    def __rtruediv__(self):
        pass

    # mul : scalars, vectors and matrices , can have errors with vectors and matrices,
    # returns a Vector if we perform Matrix * Vector mutliplication.
    def __mul__(self):
        pass

    def __rmul__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


class Vector(Matrix):
    def __init__(self) -> None:
        super().__init__()
