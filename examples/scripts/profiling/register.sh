#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=0:00:05
#SBATCH --job-name=dvreg
#SBATCH --output=job-prof_reg-%j.out

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate /scratch/gpfs/mb46/conda/spim

python register.py \
    --view-a=/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body001/proc_ortho/a.zarr \
    --view-b=/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body001/proc_ortho/b.zarr \
    --metric="cr" --interp-method="cubspl" \
    --output-type="float"
