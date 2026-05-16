#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=200G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00
#SBATCH --job-name=dvreg
#SBATCH --output=job-dvreg-%j.out

module purge
#module load cudatoolkit/12.6
module load anaconda3/2024.6

cd /home/mb46/dev/pyspim
source .venv/bin/activate
cd /home/mb46/dev/pyspim/examples/scripts/dispim_pipeline

INP_FLDR="/scratch/gpfs/SHAEVITZ/mb46/fb_dispim/static/live/2day/LS_only/2023-10-20/48hr_live/acq001/proc_shear"

python register.py \
	--input-folder=$INP_FLDR \
	--output-folder=$INP_FLDR/reg_trs0 \
	--crop-box-a="260,900,185,2025,500,1500" \
	--crop-box-b="260,900,185,2025,500,1500" \
    --channel=0 \
    --metric='cr' \
    --transform='t+r+s' \
    --bounds='10,10,10,5,5,5,0.05,0.05,0.05' \
    --interp-method='cubspl' \
	--downscale \
    --verbose --debug
