import numpy


# yields the coefficients from highest to lowest degree
def extract_coefficients(p, degree):
    """Extract polynomial coefficients from a polynomial function.
    
    Given a polynomial function p(x) of specified degree, this function
    extracts the coefficients by evaluating the polynomial at n+1 points
    and solving a linear system to find the coefficients.
    
    Args:
        p (callable): A polynomial function that takes a single argument.
        degree (int): The degree of the polynomial.
        
    Returns:
        numpy.ndarray: Array of coefficients from highest to lowest degree.
    """
    n = degree + 1
    sample_x = [x for x in range(n)]
    sample_y = [p(x) for x in sample_x]
    A = [[0 for _ in range(n)] for _ in range(n)]
    for line in range(n):
        for column in range(n):
            A[line][column] = sample_x[line] ** column
    c = numpy.linalg.solve(A, sample_y)
    return c[::-1]


def characteristic_polynomial(A):
    """Compute the characteristic polynomial of a matrix.
    
    This function is currently a placeholder and not implemented.
    
    Args:
        A (numpy.ndarray): Input matrix.
        
    Returns:
        NotImplemented: This function is not yet implemented.
    """
    pass
