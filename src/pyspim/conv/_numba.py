import numba


def make_separable_kernel(axis : int, kernel_radius : int):
    nhood = [(0,0),]*3
    nhood[axis] = (-kernel_radius, kernel_radius)

    @numba.stencil(neighborhood=tuple(nhood), standard_indexing=("b",))
    def kernel(a, b):
        val = 0
        for i in range(-kernel_radius, kernel_radius+1):
            val += a[i] * b[i+kernel_radius]
        return val
    
    return kernel