import os
import cupy

from ...typing import NDArray, CuLaunchParameters

## load the module as cupy.RawModule
__module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'cubspl.cu'
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
                               name_expressions=__kernel_names,
                               options= ("-lineinfo",))
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
    return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))

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
   
# def correlation_ratio_multigpu(T : NDArray,
#                       reference : cupy.ndarray, moving : cupy.ndarray,
#                       mu_reference : float, 
#                       sz_r : int, sy_r : int, sx_r : int, 
#                       sz_m : int, sy_m : int, sx_m : int,
#                       launch_params : CuLaunchParameters) -> float:
    
#     #num_gpus = cupy.cuda.runtime.getDeviceCount()
#     num_gpus = 1
#     print(f"n_GPUs: {num_gpus:d}")
#     gpu_indices = [ i for i in range(num_gpus) ]
#     nones = [ None for i in range(num_gpus) ]

#     kernels = nones[:]
#     streams = nones[:]
#     arrays = []
                
#     T = cupy.asarray(T).astype(cupy.float32)
#     dtype = reference.dtype
#     prods = cupy.zeros((3,), dtype=cupy.float64)

#     if dtype == cupy.uint16:
#         kname = 'correlationRatio<unsigned short,double>'
#     else:
#         kname = 'correlationRatio<float,double>'

#     for i, gpu_id in enumerate(gpu_indices):
#         with cp.cuda.Device(gpu_id):
#             kernels[i] = __cuda_module.get_function(kname)
#             streams[i] = cupy.cuda.Stream()
#             ref_dev = cupy.asarray(reference)
#             mov_dev = cupy.asarray(moving)
#             T_dev = cupy.copy(T)
#             prods_dev = cupy.zeros((3,), dtype=cupy.float64)
#             z_start = np.int32(i * sz_m // num_gpus)
#             z_end = sz_m if i == num_gpus - 1 else (i + 1) * sz_m//num_gpus
#             mov_dev = mov_dev[z_start:z_end]
#             sz_m_dev, sy_m_dev, sx_m_dev = mov_dev.shape
#             if dtype == cupy.uint16:
#                 mu_ref_dev = cupy.array(mu_reference, dtype=cupy.float64)
#             else:
#                 mu_ref_dev = cupy.array(mu_reference, dtype=cupy.float32)
#             arrays.append((prods_dev, T_dev, ref_dev, mov_dev, mu_reference, 
#                            sz_r, sy_r, sx_r, sz_m_dev, sy_m_dev, sx_m_dev, z_start))
            
#     for i, gpu_id in enumerate(gpu_indices):
#         with cp.cuda.Device(gpu_id), streams[i]:
#             #kernels[i](launch_params[0], launch_params[1], arrays[i], stream=streams[i])
#             try:
#                 kernels[i](launch_params[0], launch_params[1], arrays[i], stream=streams[i])
#                 cupy.cuda.runtime.deviceSynchronize()
#             except cupy.cuda.driver.CUDADriverError as e:
#                 print(f"GPU {gpu_id} kernel failed: {e}")
#                 raise
#     for stream in streams:
#         stream.synchronize()

#     for i,gpu_id in enumerate(gpu_indices):
#         with cp.cuda.Device(gpu_id):
#             result = arrays[i][0]
#             prods+=result.get()
            
#     #kernel = __cuda_module.get_function(kname)
#     #kernel(
#     #    launch_params[0], launch_params[1],
#     #    (prods, T, reference, moving, mu_reference,
#     #     sz_r, sy_r, sx_r, sz_m, sy_m, sx_m)
#     #)
#     return float(prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2])))

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