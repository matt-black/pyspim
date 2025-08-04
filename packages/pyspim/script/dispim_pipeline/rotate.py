import os
from functools import partial
from typing import Generator, Tuple
from argparse import ArgumentParser

import zarr
import cupy
import numpy
import tifffile

from pyspim._matrix import rotation_about_point_matrix
from pyspim.interp import affine


def main(
    input_path : os.PathLike,
    output_path : os.PathLike,
    alpha : float, 
    beta : float,
    gamma : float,
    interp_method : str
) -> int:
    if not os.path.exists(input_path):
        print("input path does not exist!", flush=True)
        return -1
    store = tifffile.imread(input_path, aszarr=True)
    vol = zarr.open(store, 'r')
    cent = [s/2 for s in vol.shape[1:]]
    R = cupy.asarray(
        rotation_about_point_matrix(alpha, beta, gamma, *cent)
    ).astype(cupy.float32)
    out_shape = affine.output_shape_for_transform(R, vol.shape[1:])
    generator = partial(rotation_generator,
                        vol, R, interp_method, out_shape,
                        block_size_z=8, block_size_y=8, block_size_x=8)
    tifffile.imwrite(
        output_path,
        generator(),
        bigtiff=True,
        shape=out_shape,
        dtype="float32",
        photometric="minisblack",
        resolution=(1/0.1625, 1/0.1625),
        metadata={'axes' : "czyx", "spacing" : 0.1625, "units": "um"}
    )
    return 0


def rotation_generator(
    v : zarr.Array,
    R : cupy.ndarray,
    interp_method : str,
    output_shape : Tuple[int,int,int],
    block_size_z : int = 8,
    block_size_y : int = 8,
    block_size_x : int = 8
) -> Generator[numpy.ndarray,None,None]:
    n_chan = v.shape[0]
    for c in range(n_chan):
        _v = cupy.asarray(
            v.oindex[c,slice(None),slice(None),slice(None)]
        )
        yield affine.transform(_v, R, interp_method, 
                               preserve_dtype=False,
                               out_shp=output_shape,
                               block_size_z=block_size_z,
                               block_size_y=block_size_y,
                               block_size_x=block_size_x)


if __name__ == "__main__":
    parser = ArgumentParser("Rotation of Volumes")
    parser.add_argument('-i', '--input-path', type=str, required=True)
    parser.add_argument('-o', '--output-path', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=float, default=0.)
    parser.add_argument('-b', '--beta', type=float, default=0.)
    parser.add_argument('-g', '--gamma', type=float, default=0.)
    parser.add_argument('-m', '--interp-method', type=str,
                        choices=['nearest','linear','cubspl'])
    args = parser.parse_args()
    exit_code = main(args.input_path, args.output_path, 
                     args.alpha * numpy.pi/180, 
                     args.beta  * numpy.pi/180, 
                     args.gamma * numpy.pi/180, 
                     args.interp_method)