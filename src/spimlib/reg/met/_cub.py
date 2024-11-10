import os
import cupy

from ...typing import NDArray, CuLaunchParameters

## read in template module text
__linear_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'linear_cub.cu'
)
with open(__linear_module_path, 'r') as f:
    __linear_module_txt = f.read()


__cubspl_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'cubspl_cub.cu'
)
with open(__cubspl_module_path, 'r') as f:
    __cubspl_module_txt = f.read()


def make_kernel(function : str, interp_method : str, 
                block_size_x : int, 
                block_size_y : int, 
                block_size_z : int) -> cupy.RawKernel:
    expr1 = 'rval * mval'
    if function == 'nip':
        function_name = 'normalizedInnerProduct{:d}{:d}{:d}'.format(
            block_size_x, block_size_y, block_size_z
        )
        expr2, expr3 = 'rval * rval', 'mval * mval'
    elif function == 'cr':
        function_name = 'correlationRatio{:d}{:d}{:d}'.format(
            block_size_x, block_size_y, block_size_z
        )
        expr2, expr3 = '(rval-mu_ref) * (rval-mu_ref)', 'mval*mval'
    elif function == 'ncc':
        function_name = 'normalizedCrossCorrelation{:d}{:d}{:d}'.format(
            block_size_x, block_size_y, block_size_z
        )
        expr2 = '(rval-mu_ref) * (rval-mu_ref)'
        expr3 = '(mval-mu_mov) * (mval-mu_mov)'
    else:
        raise ValueError('invalid function type')
    if interp_method == 'linear':
        txt = __linear_module_txt
    elif interp_method == 'cubspl':
        txt = __cubspl_module_txt
    else:
        raise ValueError('invalid interpolation method')
    function_text = txt.format(
        block_size_x=block_size_x, block_size_y=block_size_y,
        block_size_z=block_size_z, function_name=function_name,
        expr1=expr1, expr2=expr2, expr3=expr3
    )
    return cupy.RawKernel(code=function_text, name=function_name,
                          backend='nvcc', options=('-std=c++14',))


def compute_kernel(kernel : cupy.RawKernel,
                   T : NDArray, reference : cupy.ndarray, moving : cupy.ndarray,
                   mu_reference : float, mu_moving : float,
                   sz_r : int, sy_r : int, sx_r : int,
                   sz_m : int, sy_m : int, sx_m : int,
                   launch_params : CuLaunchParameters) -> float:
    prods = cupy.zeros((3,), dtype=cupy.float32)
    T = cupy.asarray(T).astype(cupy.float32)
    kernel(
        launch_params[0], launch_params[1],
        (prods, T, reference, moving, mu_reference, mu_moving, 
         sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)
    )
    prods += cupy.finfo(cupy.float32).eps
    return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))