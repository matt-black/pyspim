import os
from typing import List, Protocol, Tuple

import cupy

from ..typing import CuLaunchParameters, NDArray


KernelArgs = Tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray,
                   cupy.ndarray, cupy.ndarray,
                   cupy.ndarray, cupy.ndarray,
                   int, int, int, int, int, int, int, int]

class CallableKernel(Protocol):

    def __call__(self, 
                 lp0: Tuple[int,int,int], 
                 lp1: Tuple[int,int,int], 
                 args: KernelArgs, 
                 *, 
                 stream: cupy.cuda.Stream):
        ...


## read in template module text
__kernel_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernels.cu"
)
with open(__kernel_module_path) as f:
    __kernel_model_txt = f.read()


def make_kernel(
    function: str,
    interp_method: str,
    input_type: str,
    output_type: str,
) -> CallableKernel:
    if function == "nip":
        function_name = "normalizedInnerProduct"
        expr1 = "rval * mval"
        expr2, expr3 = "rval * rval", "mval * mval"
    elif function == "cr":
        function_name = "correlationRatio"
        expr1 = "(rval - mu_ref[0]) * mval"
        expr2, expr3 = "(rval - mu_ref[0]) * (rval - mu_ref[0])", "mval * mval"
    elif function == "ncc":
        function_name = "normalizedCrossCorrelation"
        expr1 = "(rval - mu_ref[0]) * (mval - mu_mov[0])"
        expr2, expr3 = "(rval - mu_ref[0]) * (rval - mu_ref[0])", "(mval - mu_mov[0]) * (mval - mu_mov[0])"
    else:
        raise ValueError("invalid function type")
    interp_type = interp_method.upper()
    module_txt = __kernel_model_txt.format(
        function_name=function_name,
        interp_type=interp_type,
        expr1=expr1,
        expr2=expr2,
        expr3=expr3
    )
    kernel_name = function_name+f"<{input_type},{output_type}>"
    print(kernel_name, flush=True)
    module = cupy.RawModule(
        code=module_txt, 
        name_expressions=(kernel_name,)
    )
    module.compile() # throw compiler errors here, if problems
    return module.get_function(kernel_name)


def initialize_computation(
    function: str,
    interp_method: str,
    reference: cupy.ndarray,
    moving: cupy.ndarray,
    mu_ref: cupy.float32,
    mu_mov: cupy.float32,
    sz_r: int, sy_r: int, sx_r: int,
    sz_m: int, sy_m: int, sx_m: int,
) -> Tuple[List[CallableKernel], List[cupy.cuda.Stream], List[KernelArgs]]:
    # move transform to GPU, initialize output
    T = cupy.concatenate(
        [cupy.eye(3), cupy.zeros((3,1))], axis=1
    ).astype(cupy.float32)
    # figure out input/output data types
    dtype = reference.dtype
    if dtype == cupy.uint16:
        input_type = "unsigned short"
    elif dtype == cupy.float32:
        input_type = "float"
    elif dtype == cupy.float64:
        input_type = "double"
    else:
        raise ValueError("invalid input type")
    output_type = "double" #TODO: allow other input types?

    n_gpu = cupy.cuda.runtime.getDeviceCount()
    kernels, streams, kernel_args = [], [], []
    for gpu_id in range(n_gpu):
            with cupy.cuda.Device(gpu_id):
                kernels.append(
                    make_kernel(
                        function, interp_method, input_type, output_type
                    )
                )
                streams.append(cupy.cuda.Stream())
                kernel_args.append(
                    (cupy.zeros((3,), dtype=cupy.float64), cupy.copy(T),
                     cupy.asarray(reference), cupy.asarray(moving),  # use asarray here b/c avoids an extra copy if we're on the device these are already on
                     cupy.copy(mu_ref), cupy.copy(mu_mov),
                     sz_r, sy_r, sx_r, sz_m, sy_m, sx_m, gpu_id, n_gpu)
                )

    return kernels, streams, kernel_args


def compute(
    T: NDArray,
    kernels: List[CallableKernel],
    streams: List[cupy.cuda.Stream],
    kernel_args: List[KernelArgs],
    launch_pars: CuLaunchParameters
) -> cupy.ndarray:
    # iterate over all of the available GPUs, computing the value for the part of the
    # array that each GPU is responsible for
    for gpu_id, (kernel, stream, args) in enumerate(zip(kernels, streams, kernel_args)):
        with cupy.cuda.Device(gpu_id):
            # swap the default transform with the one being tested
            T_dev = cupy.asarray(T, dtype=cupy.float32)
            new_args = (args[0], T_dev, *args[2:])
            kernel(launch_pars[0], launch_pars[1], new_args, stream=stream)
    # force stream synchronization, ensure all values computed
    for stream in streams:
        stream.synchronize()
    # accumulate results into a single output vector
    prods = cupy.zeros((3,), dtype=cupy.float64)
    n_gpu = len(kernels)
    for gpu_id in range(n_gpu):
        with cupy.cuda.Device(gpu_id):
            # first argument of kernel_args holds the result
            # TODO: does explicit copy to default device help here?
            result = kernel_args[gpu_id][0]
            prods += result
    # compute the actual value
    return prods[0] / (cupy.sqrt(prods[1]) * cupy.sqrt(prods[2]))
