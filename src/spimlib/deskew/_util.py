import numpy

# yields the coefficients from highest to lowest degree
def extract_coefficients(p, degree):
    n = degree + 1
    sample_x = [ x for x in range(n) ]
    sample_y = [ p(x) for x in sample_x ]
    A = [ [ 0 for _ in range(n) ] for _ in range(n) ]
    for line in range(n):   
        for column in range(n):
            A[line][column] = sample_x[line] ** column
    c = numpy.linalg.solve(A, sample_y)
    return c[::-1]


def characteristic_polynomial(A):
    pass