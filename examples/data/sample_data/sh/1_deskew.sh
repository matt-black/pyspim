#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --time=0:15:00
#SBATCH --job-name=deskew
#SBATCH --output=job-deskew-%j.out

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate /scratch/gpfs/mb46/conda/spim

ACQ_PATH="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body005"
METHOD="ortho"

python deskew_cpu.py \
    --input-folder=$ACQ_PATH \
    --data-type="umasi" \
    --output-folder=$ACQ_PATH/proc_$METHOD --force \
    --deskew-method=$METHOD \
    --step-size=0.5 --pixel-size=0.1625 \
    --channels="0" \
    --interp-method='cubspl' \
    --verbose
