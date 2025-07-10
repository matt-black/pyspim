These are scripts for profiling kernels/pipeline functions. 

For profiling cupy code, there are some utilities provided by the library, see https://docs.cupy.dev/en/stable/user_guide/performance.html for details. 

For details about how kernel invokations work in CuPy, see https://docs.cupy.dev/en/latest/reference/generated/cupy.RawKernel.html#cupy.RawKernel.__call__
Other details about RawModule/RawKernel can be found here: https://docs.cupy.dev/en/latest/user_guide/kernel.html


register.py 
===
example call:

INP_FLDR=/path/to/fruiting_body001/proc_ortho

python register.py \
    --view-a=$INP_FLDR/a.zarr \
    --view-b=$INP_FLDR/b.zarr \
    --metric="cr" --interp-method="cubspl" \
    --num-repeat=100 --output-type="double"
