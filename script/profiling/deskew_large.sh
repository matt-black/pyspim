#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=0:00:05
#SBATCH --job-name=deskew_prof
#SBATCH --output=job-prof_deskew-%j.out

module purge
module load cudatoolkit/12.6
# module load anaconda3/2024.6
# conda activate /scratch/gpfs/mb46/conda/spim

ACQ_PATH="/scratch/gpfs/SHAEVITZ/out/mito002_t/a_c00.tif"

ncu --set full --clock-control none \
    -k regex:deskew_stage_scan \
    -c 10 \
    -f -o ncu_deskew_output \
    --launch-skip-before-match 20 \
    python deskew_large.py \
        --input-path=$ACQ_PATH \
        --direction=1 \
        --preserve-data-type \
        --num-repeat=5

# nsys profile \
#     --force-overwrite=true \
#     --trace=cuda,nvtx,osrt \
#     --stats=true \