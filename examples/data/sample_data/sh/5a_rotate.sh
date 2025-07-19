#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=0:15:00
#SBATCH --job-name=rotate
#SBATCH --output=job-rot-%j.out

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate /scratch/gpfs/mb46/conda/spim

INP_FLDR="/scratch/gpfs/SHAEVITZ/out"

python rotate.py \
    --input-path=$INP_FLDR/decon_ana.ome.tif \
    --output-path=$INP_FLDR/decon_ana_rot.ome.tif \
    --alpha=0 --beta=90 --gamma=0 \
    --interp-method="cubspl"

# python deconvolve.py \
#     --a-zarr-path=$INP_FLDR/trans/a.zarr \
#     --b-zarr-path=$INP_FLDR/trans/b.zarr \
#     --output-path=$INP_FLDR/trans/decon_sep.ome.tif \
#     --function="additive" \
#     --sigma-az=3.66 --sigma-ay=1.175 --sigma-ax=1.175 \
#     --sigma-bz=3.66 --sigma-by=1.175 --sigma-bx=1.175 \
#     --kernel-radius-z=31 \
#     --kernel-radius-y=31 \
#     --kernel-radius-x=31 \
#     --num-iter=40 \
#     --verbose
