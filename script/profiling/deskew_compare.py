#!/usr/bin/env python3
"""Profile and compare deskewing methods."""

import os
import math
import time
from argparse import ArgumentParser

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
    
    # Compare results if we have multiple methods
    if len(results) > 1:
        print("\nCOMPARISON:")
        print("-" * 30)
        
        # Check if outputs have same shape
        shapes = {k: v.shape for k, v in results.items()}
        if len(set(shapes.values())) == 1:
            print("✓ All methods produce same output shape")
        else:
            print("✗ Methods produce different output shapes:")
            for method, shape in shapes.items():
                print(f"  {method}: {shape}")
        
        # Compare first two methods
        method_names = list(results.keys())
        if len(method_names) >= 2:
            method1, method2 = method_names[0], method_names[1]
            result1 = results[method1]
            result2 = results[method2]
            
            if result1.shape == result2.shape:
                # Handle both NumPy and CuPy arrays
                if hasattr(result1, 'get'):
                    data1 = result1.get()
                else:
                    data1 = result1
                    
                if hasattr(result2, 'get'):
                    data2 = result2.get()
                else:
                    data2 = result2
                
                # Calculate correlation coefficient
                correlation = numpy.corrcoef(data1.flatten(), data2.flatten())[0, 1]
                
                # Calculate differences
                diff = numpy.abs(data1 - data2)
                max_diff = numpy.max(diff)
                mean_diff = numpy.mean(diff)
                
                print(f"✓ Correlation ({method1} vs {method2}): {correlation:.6f}")
                print(f"✓ Max difference: {max_diff:.2f}")
                print(f"✓ Mean difference: {mean_diff:.4f}")
                
                if correlation > 0.99:
                    print("✓ Results are highly correlated (>0.99)")
                elif correlation > 0.95:
                    print("✓ Results are well correlated (>0.95)")
                else:
                    print("⚠ Results have low correlation (<0.95)")
    
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