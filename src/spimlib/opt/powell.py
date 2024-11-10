"""Powell's method

"""
import math
from typing import Callable, Tuple

import numpy

from ..typing import NDArray


__R = (-1 + math.sqrt(5) / 2)
__c = 1 + __R


def goldsect_bracket(f : Callable[[float], float], x1 : float, incr : float, 
                     max_iter : int = 100) -> Tuple[float, float]:
    f1 = f(x1)
    x2 = x1 + incr
    f2 = f(x2)
    # determine downhill direction, change sign if needed
    if f2 > f1:  # need to change direction of h
        incr = -incr
        x2 = x1 + incr
        f2 = f(x2)
        if f2 > f1:
            return x2, x1 - incr
    for _ in range(max_iter):
        incr = __c * incr
        x3 = x2 + incr
        f3 = f(x3)
        if f3 > f2:
            return x1, x3
        x1, x2 = x2, x3
        f1, f2 = f2, f3
    raise Exception('golden section search bracketing did not find minimum')


def goldsect_search(f : Callable[[float],float], a : float, b : float,
                    tol : float=1e-9) -> Tuple[float,float]:
    n_iter = int(math.ceil(-2.078087 * math.log(tol / abs(b-a))))
    x1 = __R * a + (1 - __R) * b
    f1 = f(x1)
    x2 = (1 - __R) * a + __R * b
    f2 = f(x2)
    for _ in range(n_iter):
        if f1 > f2:
            a = x1
            x1, f1 = x2, f2
            x2 = (1 - __R) * a + __R * b
            f2 = f(x2)
        else:
            b = x2
            x2, f2 = x1, f1
            x1 = __R * a + (1 - __R) * b
            f1 = f(x1)
    return x1, f1 if f1 < f2 else x2, f2


def powell(F : Callable[[NDArray],float], x : NDArray, 
           n_iter : int = 30, h : float = 0.1, tol : float=1e-6):
    f = lambda s : F(x + s * v)
    n_par = len(x)
    dec_f = numpy.zeros(n_par)
    u = numpy.eye(n_par)
    for i in range(n_iter):
        x_prev, f_prev = x.copy()
        f_prev = f(x_prev)
        for j in range(n_par):
            v = u[j]
            a, b = goldsect_bracket(f, 0., h)
            s, f_min = goldsect_search(f, a, b)
            dec_f[j] = f_prev - f_min
            f_prev = f_min
            x = x + s * v
        v = x - x_prev  # v_{n+1} = x_0 - x_prev
        a, b = goldsect_bracket(f, 0., h)
        s, _ = goldsect_search(f, a, b)
        x = x + s * v
        if math.sqrt(numpy.dot(x-x_prev, x-x_prev) / n_iter) < tol:
            return x, i+1
        i_max = numpy.argmax(dec_f)
        for i in range(i_max, n_iter-1):
            u[j] = u[j+1]
        u[n_iter-1] = v
    raise Exception("powell's method did not converge")