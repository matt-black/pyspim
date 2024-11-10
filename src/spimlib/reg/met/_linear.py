import os
import cupy

from ...typing import NDArray, CuLaunchParameters

## load the module as cupy.RawModule
__module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'linear.cu'
)
with open(__module_path, 'r') as f:
    __module_txt = f.read()

__kernel_names = (
    'normInnerProduct<unsigned short,float>',
    'normInnerProduct<unsigned short,double>',
    'correlationRatio<unsigned short,float>',
    'correlationRatio<unsigned short,double>',
    'normCrossCorrelation<unsigned short,float>',
    'normCrossCorrelation<unsigned short,double>',
    'normInnerProduct<float,float>',
    'normInnerProduct<float,double>',
    'correlationRatio<float,float>',
    'correlationRatio<float,double>',
    'normCrossCorrelation<float,float>',
    'normCrossCorrelation<float,double>',
)
__cuda_module = cupy.RawModule(code=__module_txt,
                               name_expressions=__kernel_names)
__cuda_module.compile()  # throw compiler errors here, if problems


def normalized_inner_product(T : NDArray,
                             reference : cupy.ndarray, moving : cupy.ndarray, 
                             sz_r : int, sy_r : int, sx_r : int, 
                             sz_m : int, sy_m : int, sx_m : int,
                             launch_params : CuLaunchParameters) -> float:
    T = cupy.asarray(T).astype(cupy.float32)
    prods = cupy.zeros((3,), dtype=cupy.float64)
    dtype = reference.dtype
    if dtype == cupy.uint16:
        kname = 'normInnerProduct<unsigned short,double>'
    else:
        kname = 'normInnerProduct<float,double>'
    kernel = __cuda_module.get_function(kname)
    kernel(
        launch_params[0], launch_params[1],
        (prods, T, reference, moving, sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)
    )
    prods += cupy.finfo(cupy.float64).eps
    nip = float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))
    print(nip)
    return nip


def correlation_ratio(T : NDArray,
                      reference : cupy.ndarray, moving : cupy.ndarray,
                      mu_reference : float, 
                      sz_r : int, sy_r : int, sx_r : int, 
                      sz_m : int, sy_m : int, sx_m : int,
                      launch_params : CuLaunchParameters) -> float:
    T = cupy.asarray(T).astype(cupy.float32)
    prods = cupy.zeros((3,), dtype=cupy.float64)
    dtype = reference.dtype
    if dtype == cupy.uint16:
        kname = 'correlationRatio<unsigned short,double>'
    else:
        kname = 'correlationRatio<float,double>'
    kernel = __cuda_module.get_function(kname)
    kernel(
        launch_params[0], launch_params[1],
        (prods, T, reference, moving, mu_reference,
         sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)
    )
    prods += cupy.finfo(cupy.float64).eps
    return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))


def normalized_cross_correlation(
        T : NDArray, reference : cupy.ndarray, moving : cupy.ndarray,
        mu_reference : float, mu_moving : float,
        sz_r : int, sy_r : int, sx_r : int, 
        sz_m : int, sy_m : int, sx_m : int,
        launch_params : CuLaunchParameters) -> float:
    T = cupy.asarray(T).astype(cupy.float32)
    prods = cupy.zeros((3,), dtype=cupy.float64)
    dtype = reference.dtype
    if dtype == cupy.uint16:
        kname = 'normCrossCorrelation<unsigned short,double>'
    else:
        kname = 'normCrossCorrelation<float,double>'
    kernel = __cuda_module.get_function(kname)
    kernel(
        launch_params[0], launch_params[1],
        (prods, T, reference, moving, mu_reference, mu_moving,
         sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)
    )
    prods += cupy.finfo(cupy.float64).eps
    return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))