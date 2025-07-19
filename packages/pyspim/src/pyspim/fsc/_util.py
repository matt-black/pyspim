import cupy

from ..typing import NDArray


def hamming_window(arr : NDArray):
    """apply hamming window to an N-dimensional array

    :param arr: input array
    :type arr: NDArray
    :returns: array with hamming window applied
    :rtype: NDArray
    """
    xp = cupy.get_array_module(arr)
    res = arr.copy().astype(xp.float32)
    for ax, sze in enumerate(arr.shape):
        fshape = [1,] * arr.ndim
        fshape[ax] = sze
        window = xp.hamming(sze).reshape(fshape)
        xp.power(window, (1.0/arr.ndim), out=window)
        res *= window
    return res
