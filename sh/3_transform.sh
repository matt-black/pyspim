#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=0:30:00
#SBATCH --job-name=trans
#SBATCH --output=job-trans-%j.out

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate /scratch/gpfs/mb46/conda/spim

INP_FLDR="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body005/proc_ortho"

python transform.py \
    --input-folder=$INP_FLDR \
    --output-folder=$INP_FLDR/trans_trs0 --force \
    --reg-params=$INP_FLDR/reg_trs0/reg_params.json \
    --reg-transform=$INP_FLDR/reg_trs0/reg_transform.npy \
    --num-workers=$SLURM_CPUS_PER_TASK \
    --verbose