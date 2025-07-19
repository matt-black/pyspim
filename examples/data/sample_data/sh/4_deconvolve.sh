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
conda activate /scratch/gpfs/mb46/conda/spim

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

INP_FLDR="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body005/proc_ortho"

python deconvolve.py \
   --a-zarr-path=$INP_FLDR/trans_trs0/a.zarr \
   --b-zarr-path=$INP_FLDR/trans_trs0/b.zarr \
   --output-path=$INP_FLDR/trans_trs0/decon_eff80.ome.tif \
   --function="efficient" \
   --psf-a-path=/scratch/gpfs/mb46/psfs/ortho/a.npy \
   --psf-b-path=/scratch/gpfs/mb46/psfs/ortho/b.npy \
   --num-iter=80 \
   --verbose

python deconvolve.py \
   --a-zarr-path=$INP_FLDR/trans_trs0/a.zarr \
   --b-zarr-path=$INP_FLDR/trans_trs0/b.zarr \
   --output-path=$INP_FLDR/trans_trs0/decon_eff40.ome.tif \
   --function="efficient" \
   --psf-a-path=/scratch/gpfs/mb46/psfs/ortho/a.npy \
   --psf-b-path=/scratch/gpfs/mb46/psfs/ortho/b.npy \
   --num-iter=40 \
   --verbose

python deconvolve.py \
   --a-zarr-path=$INP_FLDR/trans_trs0/a.zarr \
   --b-zarr-path=$INP_FLDR/trans_trs0/b.zarr \
   --output-path=$INP_FLDR/trans_trs0/decon_eff20.ome.tif \
   --function="efficient" \
   --psf-a-path=/scratch/gpfs/mb46/psfs/ortho/a.npy \
   --psf-b-path=/scratch/gpfs/mb46/psfs/ortho/b.npy \
   --num-iter=20 \
   --verbose