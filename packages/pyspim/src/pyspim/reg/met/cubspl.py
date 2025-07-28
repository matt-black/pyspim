import os

import cupy

from ...typing import CuLaunchParameters, NDArray

## load the module as cupy.RawModule
__module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cubspl.cu")
with open(__module_path) as f:
    __module_txt = f.read()

__kernel_names = (
    "normInnerProduct<unsigned short,float>",
    "normInnerProduct<unsigned short,double>",
    "correlationRatio<unsigned short,float>",
    "correlationRatio<unsigned short,double>",
    "normCrossCorrelation<unsigned short,float>",
    "normCrossCorrelation<unsigned short,double>",
    "normInnerProduct<float,float>",
    "normInnerProduct<float,double>",
    "correlationRatio<float,float>",
    "correlationRatio<float,double>",
    "normCrossCorrelation<float,float>",
    "normCrossCorrelation<float,double>",
)
__cuda_module = cupy.RawModule(code=__module_txt, name_expressions=__kernel_names)
__cuda_module.compile()  # throw compiler errors here, if problems


def normalized_inner_product(
    T: NDArray,
    reference: cupy.ndarray,
    moving: cupy.ndarray,
    sz_r: int,
    sy_r: int,
    sx_r: int,
    sz_m: int,
    sy_m: int,
    sx_m: int,
    launch_params: CuLaunchParameters,
) -> float:
    T = cupy.asarray(T).astype(cupy.float32)
    prods = cupy.zeros((3,), dtype=cupy.float64)
    dtype = reference.dtype
    if dtype == cupy.uint16:
        kname = "normInnerProduct<unsigned short,double>"
    else:
        kname = "normInnerProduct<float,double>"
    kernel = __cuda_module.get_function(kname)
    kernel(
        launch_params[0],
        launch_params[1],
        (prods, T, reference, moving, sz_r, sy_r, sx_r, sz_m, sy_m, sx_m),
    )
    prods += cupy.finfo(cupy.float64).eps
    return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))

<<<<<<< HEAD
=======
called = False
num_gpus = cupy.cuda.runtime.getDeviceCount()
nones = [ None for _ in range(num_gpus) ]
kernels = nones[:]
streams = nones[:]
ref_dev = nones[:]
mov_dev = nones[:]
mu_ref_dev = nones[:]
prods_dev = nones[:]
>>>>>>> kcophenhagen-register-multigpu

def correlation_ratio(
    T: NDArray,
    reference: cupy.ndarray,
    moving: cupy.ndarray,
    mu_reference: float,
    sz_r: int,
    sy_r: int,
    sx_r: int,
    sz_m: int,
    sy_m: int,
    sx_m: int,
    launch_params: CuLaunchParameters,
) -> float:
    T = cupy.asarray(T).astype(cupy.float32)
    prods = cupy.zeros((3,), dtype=cupy.float64)
    dtype = reference.dtype
    if dtype == cupy.uint16:
        kname = "correlationRatio<unsigned short,double>"
    else:
        kname = "correlationRatio<float,double>"
    kernel = __cuda_module.get_function(kname)
    kernel(
        launch_params[0],
        launch_params[1],
        (prods, T, reference, moving, mu_reference, sz_r, sy_r, sx_r, sz_m, sy_m, sx_m),
    )
    prods += cupy.finfo(cupy.float64).eps
    return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))

    T = cupy.asarray(T).astype(cupy.float32)
    dtype = reference.dtype

    if dtype == cupy.uint16:
        kname = f'correlationRatio<unsigned short,{output_type}>'
    else:
        kname = f'correlationRatio<float,{output_type}>'

    arrays = []
    for i, gpu_id in enumerate(gpu_indices):
        with cupy.cuda.Device(gpu_id):
            T_dev = cupy.copy(T)
            if not called:
                kernels[i] = __cuda_module.get_function(kname)
                streams[i] = cupy.cuda.Stream()
    
                ref_dev[i] = cupy.asarray(reference)
                mov_dev[i] = cupy.asarray(moving)
                mu_ref_dev[i] = dtype_map[output_type](mu_reference)
                prods_dev[i] = cupy.zeros((3,), dtype=dtype_map[output_type])
    
            arrays.append((prods_dev[i], T_dev, ref_dev[i], mov_dev[i], mu_ref_dev[i], 
                       sz_r, sy_r, sx_r, sz_m, sy_m, sx_m, i, num_gpus))

    called = True

    for i, gpu_id in enumerate(gpu_indices):
        with cupy.cuda.Device(gpu_id), streams[i]:
            kernels[i](launch_params[0], launch_params[1], arrays[i], stream=streams[i])

    for stream in streams:
        stream.synchronize()
    
    prods = cupy.zeros((3,), dtype=dtype_map[output_type])

    for i, gpu_id in enumerate(gpu_indices):
        with cupy.cuda.Device(gpu_id):
            result = arrays[i][0]
            prods+=result
            print(result[0] / (cupy.sqrt(result[1]) * cupy.sqrt(result[2])))

    print(f'prods0 = {prods[0]} prods1 = {prods[1]} prods2 = {prods[2]}')
    return (prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))

def normalized_cross_correlation(
    T: NDArray,
    reference: cupy.ndarray,
    moving: cupy.ndarray,
    mu_reference: float,
    mu_moving: float,
    sz_r: int,
    sy_r: int,
    sx_r: int,
    sz_m: int,
    sy_m: int,
    sx_m: int,
    launch_params: CuLaunchParameters,
) -> float:
    T = cupy.asarray(T).astype(cupy.float32)
    prods = cupy.zeros((3,), dtype=cupy.float64)
    dtype = reference.dtype
    if dtype == cupy.uint16:
        kname = "normCrossCorrelation<unsigned short,double>"
    else:
        kname = "normCrossCorrelation<float,double>"
    kernel = __cuda_module.get_function(kname)
    kernel(
        launch_params[0],
        launch_params[1],
        (
            prods,
            T,
            reference,
            moving,
            mu_reference,
            mu_moving,
            sz_r,
            sy_r,
            sx_r,
            sz_m,
            sy_m,
            sx_m,
        ),
    )
    prods += cupy.finfo(cupy.float64).eps
    return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))
