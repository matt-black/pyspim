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
module load cudatoolkit/12.6
module load nsight-systems/2025.3.1
module load anaconda3/2024.6
conda activate /scratch/gpfs/kc32/conda/pyspim

INP_FLDR="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body004/proc_ortho"

ncu --set full --clock-control none \
    -k regex:correlationRatio \
    -c 10 \
    -f -o ncu_output \
    --launch-skip-before-match 20 \
	python register.py \
    		--input-folder=$INP_FLDR \
    		--output-folder=$INP_FLDR/reg_trs0 \
    		--crop-box-a="115,275,167,437,500,850" \
		--crop-box-b="115,275,167,437,500,850" \
    		--channel=0 \
    		--metric='cr' \
    		--transform='t+r+s' \
    		--bounds='20,20,20,5,5,5,0.05,0.05,0.05' \
    		--interp-method='cubspl' \
    		--verbose

#
