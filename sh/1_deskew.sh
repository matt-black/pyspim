#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=50G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --job-name=deskew
#SBATCH --output=job-deskew-%j.out

module purge
#module load cudatoolkit/12.6

cd /home/mb46/dev/pyspim
source .venv/bin/activate
cd /home/mb46/dev/pyspim/examples/scripts/dispim_pipeline

ACQ_PATH="/scratch/gpfs/SHAEVITZ/mb46/fb_dispim/static/live/2day/LS_only/2023-10-20/48hr_live/acq001"
METHOD="shear"

python deskew_gpu.py \
    --input-folder=$ACQ_PATH \
    --data-type="umasi" \
    --output-folder=$ACQ_PATH/proc_$METHOD --force \
    --deskew-method=$METHOD \
    --step-size=0.5 --pixel-size=0.1625 \
    --channels="0" \
    --interp-method='cubspl' \
    --verbose
