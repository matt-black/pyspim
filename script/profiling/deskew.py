import os
import math
import time
from argparse import ArgumentParser

import numpy
from tifffile import imread

from pyspim.deskew import ortho


def main(
    input_path: os.PathLike,
    direction: int,
    preserve_dtype: bool,
    num_repeat: int,
) -> int:
    vol = imread(input_path)
    exec_times = numpy.zeros((num_repeat,))
    for idx in range(num_repeat):
        start_time = time.perf_counter()
        _ = ortho.deskew_stage_scan(
            vol, 0.1625, 0.5, direction, math.pi/4, preserve_dtype
        )
        stop_time = time.perf_counter()
        exec_times[idx] = stop_time - start_time
    mean_exec_time = numpy.mean(exec_times)
    sd_exec_time = numpy.std(exec_times)
    print(f"Executed {num_repeat} times, {mean_exec_time:.2f} +/- {sd_exec_time:.2f}", flush=True)
    print(f"Exec Times: {exec_times.tolist()}", flush=True)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--input-path", type=str, required=True,
                        help="path to input file")
    parser.add_argument("-d", "--direction", type=int, required=True,
                        choices=[-1,1], help="stage scanning direction")
    parser.add_argument("-pdt", "--preserve-data-type", action="store_true",
                        help="cast output data to type of input data")
    parser.add_argument("-n", "--num-repeat", type=int, default=10)
    args = parser.parse_args()
    ec = main(
        args.input_path,
        args.direction,
        args.preserve_data_type,
        args.num_repeat,
    )
    exit(ec)
