#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=100G
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00
#SBATCH --job-name=didec
#SBATCH --output=job-decon-%j.out

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate /scratch/gpfs/kc32/conda/pyspim3

# INP_FLDR="/scratch/gpfs/SHAEVITZ/exp/all_data/hela_250320/dgel1/cell007/proc_shear"

# python deconvolve.py \
#     --a-zarr-path=$INP_FLDR/trans_trs0/a.zarr \
#     --b-zarr-path=$INP_FLDR/trans_trs0/b.zarr \
#     --output-path=$INP_FLDR/trans_trs0/decon.ome.tif \
#     --function="additive" \
#     --sigma-az=3.66 --sigma-ay=1.175 --sigma-ax=1.175 \
#     --sigma-bz=3.66 --sigma-by=1.175 --sigma-bx=1.175 \
#     --kernel-radius-z=31 \
#     --kernel-radius-y=31 \
#     --kernel-radius-x=31 \
#     --num-iter=40 \
#     --distributed \
#     --verbose

INP_FLDR="/scratch/gpfs/kc32/ExampleDispimData/example_dispim_data/fruiting_body001/proc_ortho"


python deconvolve.py \
   --a-zarr-path=$INP_FLDR/trans_trs0/a.zarr \
   --b-zarr-path=$INP_FLDR/trans_trs0/b.zarr \
   --output-path=$INP_FLDR/trans_trs0/decon_eff20.ome.tif \
   --function="efficient" \
   --psf-a-path=/scratch/gpfs/kc32/ExampleDispimData/example_dispim_data/psfs/PSFA_500.npy \
   --psf-b-path=/scratch/gpfs/kc32/ExampleDispimData/example_dispim_data/psfs/PSFB_500.npy \
   --num-iter=20 \
   --distributed \
   --verbose
