from matrix import Matrix, Vector

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)
# Output:
# (3, 2)
print(m1.T())
# Output:
# Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.T().shape)
# Output:
# (2, 3)
m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
print(m1.shape)
# Output:
# (2, 3)
print(m1.T())
# Output:
# Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.T().shape)
# Output:
# (3, 2)
m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
             [0.0, 2.0, 4.0, 6.0]])
m2 = Matrix([[0.0, 1.0],
             [2.0, 3.0],
             [4.0, 5.0],
             [6.0, 7.0]])
print(m1 * m2)
# Output:
# Matrix([[28., 34.], [56., 68.]])
m1 = Matrix([[0.0, 1.0, 2.0],
             [0.0, 2.0, 4.0]])
v1 = Vector([[1], [2], [3]])
print(m1 * v1)
# Output:
# Matrix([[8], [16]])
# Or: Vector([[8], [16]
v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
print(v1 + v2)
# Output:
# Vector([[3],[6],[11]])


print("=" * 42)


m = Matrix((3, 3))
print(m)

m2 = Matrix([[1.0, 2.0], [3.0, 4.0]])
print(m2)

m2b = Matrix([[1.0], [2.0], [3.0], [4.0]])
print(m2b)

print('1 ' + '*' * 25)


print(m2b + m2b)
print(m2b - m2b)
print(-1 * m2b)
print(m2b * -1)
print(m2b)

print('2 ' + '*' * 25)

m3 = m2.T()
print(m2)
print(m3)

print(m3 - m2)
print(m2 * m3)
print(m3 * m2)

print(m2)
print(m3)


print('3 ' + '*' * 25)

v2 = Vector([[1.0], [2.0], [3.0], [4.0]])
v3 = v2.T()
print(v2)
print(v3)

print(v2 * v3)
print(v3 * v2)

print(v2)
print(v3)

print('4 ' + '*' * 25)

v4 = Vector([[1.0], [2.0], [3.0], [4.0]])
m4 = Matrix([[1.0], [2.0], [3.0], [4.0]])
print(v4)
print(m4)

print(v4 + m4)
print(v4 - m4)
print(v4 * m4.T())


try:
    v3 = Vector([[1, 2], [3, 4]])
except Exception as e:
    print(e)

v = Vector([[1, 2, 3]])
print(v.dot(v))
v2 = Vector([[1]])
print(v2.dot(v2))
print(v2.shape)

print("*" * 25 + " ERROR MGMT " + "*" * 25)

try:
    m = Matrix((5))
    print(m)
except Exception as e:
    print(e)

try:
    m = Matrix([[1], [1, 2]])
    print(m)
except Exception as e:
    print(e)

try:
    m = Matrix([1], [1, 2])
    print(m)
except Exception as e:
    print(e)

try:
    m = Matrix([[1], ['lala'], [3]])
    print(m)
except Exception as e:
    print(e)

try:
    m = Matrix((5, 2)) * Matrix((3, 5))
    print(m)
except Exception as e:
    print(e)

try:
    m = Matrix([[1, 2], [3, 4]])
    m2 = Matrix((2, 3))
    print(m + m2)
except Exception as e:
    print(e)

try:
    m = Matrix((5, 2)) / 0
    print(m)
except Exception as e:
    print(e)


try:
    m = Matrix((5, 2))
    print(1 + m)
except Exception as e:
    print(e)