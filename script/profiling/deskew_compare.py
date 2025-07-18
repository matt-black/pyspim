#!/usr/bin/env python3
"""Profile and compare deskewing methods."""

import os
import math
import time
from argparse import ArgumentParser
import itertools

import numpy
import tifffile
from tqdm import tqdm

from pyspim.data.dispim import uManagerAcquisitionOnePos
from pyspim.deskew import deskew_stage_scan


def main(
    input_path: os.PathLike,
    direction: int,
    preserve_dtype: bool,
    num_repeat: int,
    methods: list,
    use_tiff: bool = False,
) -> int:
    """Run deskewing methods and profile them."""
    # Load data based on input type
    if use_tiff:
        # Load directly from TIFF file
        vol = tifffile.imread(input_path)
        print(f"Loaded TIFF file: {input_path}")
    else:
        # Load from acquisition directory
        head = 'a' if direction == 1 else 'b'
        with uManagerAcquisitionOnePos(input_path, numpy) as acq:
            vol = acq.get(head, 0, 0)
        print(f"Loaded acquisition: {input_path}")
    
    print(f"Input volume shape: {vol.shape}")
    print(f"Testing methods: {methods}")
    
    # Test parameters
    pixel_size = 0.1625
    step_size = 0.5
    theta = math.pi / 4
    
    results = {}
    
    for method in methods:
        print(f"\nProfiling {method} method...")
        exec_times = numpy.zeros((num_repeat,))
        
        for idx in tqdm(range(num_repeat), desc=f"Profiling {method}"):
            start_time = time.perf_counter()
            result = deskew_stage_scan(
                vol, pixel_size, step_size, direction, theta,
                method=method, preserve_dtype=preserve_dtype
            )
            stop_time = time.perf_counter()
            exec_times[idx] = stop_time - start_time
            
            # Store the last result for comparison
            if idx == num_repeat - 1:
                results[method] = result
        
        mean_exec_time = numpy.mean(exec_times)
        sd_exec_time = numpy.std(exec_times)
        print(f"  {method}: {mean_exec_time:.3f} +/- {sd_exec_time:.3f} seconds")
        print(f"  Times: {exec_times.tolist()}")


    # If we have results from at least two methods, compare the first two
    if len(results) >= 2:
        return compare_first_two_results(results)


def compare_first_two_results(results: dict[str, numpy.ndarray]) -> int:
    """Compare the first two deskewed results from the profiling."""
    method1, method2 = list(results.keys())[:2]
    result1 = results[method1]
    result2 = results[method2]
    
    if result1.shape != result2.shape:
        print(f"✗ {method1} and {method2} results have different shapes: "
                f"{result1.shape} vs {result2.shape}")
        return 1

    # Handle both NumPy and CuPy arrays
    data1 = result1.get() if hasattr(result1, 'get') else result1
    data2 = result2.get() if hasattr(result2, 'get') else result2
    
    # Calculate differences
    diff = numpy.abs(data1 - data2)
    max_diff = numpy.max(diff)
    mean_diff = numpy.mean(diff)
    
    print(f"✓ Max difference: {max_diff:.2f}")
    print(f"✓ Mean difference: {mean_diff:.4f}")

    # Check if data1 and/or data2 contain NaN values
    if numpy.isnan(data1).any():
        print(f"✗ {method1} result contains NaN values")
    else:
        print(f"✓ {method1} result does not contain NaN values")

    if numpy.isnan(data2).any():
        print(f"✗ {method2} result contains NaN values")
    else:
        print(f"✓ {method2} result does not contain NaN values")

    # Calculate correlation over different blocks of the volume
    # to avoid memory issues with large volumes
    dim1_chunksize = 100
    dim2_chunksize = 100
    dim3_chunksize = 10

    corr_coefs = []
    dim1_iter = range(0, data1.shape[0], dim1_chunksize)
    dim2_iter = range(0, data1.shape[1], dim2_chunksize)
    dim3_iter = range(0, data1.shape[2], dim3_chunksize)

    num_blocks = 0
    for i,j,k in itertools.product(dim1_iter, dim2_iter, dim3_iter):
        num_blocks += 1
        block1 = data1[i:i + dim1_chunksize, j:j + dim2_chunksize, k:k + dim3_chunksize]
        block2 = data2[i:i + dim1_chunksize, j:j + dim2_chunksize, k:k + dim3_chunksize]

        # Some blocks result in NaN correlation coefficients, for example having all zeros
        with numpy.errstate(divide='ignore', invalid='ignore'):
            correlation = numpy.corrcoef(block1.flatten(), block2.flatten())[0, 1]

            if not numpy.isnan(correlation):
                corr_coefs.append(correlation)


    # Print correlation coefficient summary statistics
    corr_coefs = numpy.array(corr_coefs)
    print(f"✓ Correlation coefficients statistics splitting the data into {num_blocks} blocks of size {dim1_chunksize}x{dim2_chunksize}x{dim3_chunksize}: ")
    print(f"- min ={numpy.min(corr_coefs):.4f}, ")
    print(f"- q25 ={numpy.quantile(corr_coefs, 0.25):.4f}, ")
    print(f"- q50 ={numpy.median(corr_coefs):.4f}, ")
    print(f"- q75 ={numpy.quantile(corr_coefs, 0.75):.4f}, ")
    print(f"- max ={numpy.max(corr_coefs):.4f}, ")
    print(f"- mean={numpy.mean(corr_coefs):.4f}, ")
    print(f"- std ={numpy.std(corr_coefs):.4f}")

    
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--input-path", type=str, required=True,
                        help="path to input file")
    parser.add_argument("-d", "--direction", type=int, required=True,
                        choices=[-1, 1], help="stage scanning direction")
    parser.add_argument("-pdt", "--preserve-data-type", action="store_true",
                        help="cast output data to type of input data")
    parser.add_argument("-n", "--num-repeat", type=int, default=5,
                        help="number of times to repeat each method")
    parser.add_argument("-m", "--methods", nargs="+", 
                        choices=['raw', 'orthogonal'], 
                        default=['orthogonal'],
                        help="methods to test")
    parser.add_argument("-t", "--use-tiff", action="store_true",
                        help="load input as TIFF file instead of acquisition")
    args = parser.parse_args()
    
    ec = main(
        args.input_path,
        args.direction,
        args.preserve_data_type,
        args.num_repeat,
        args.methods,
        args.use_tiff,
    )
    exit(ec) 