#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=200G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00
#SBATCH --job-name=dvreg
#SBATCH --output=job-dvreg-4gpu-%j.out

module purge
module load anaconda3/2024.6
module load cudatoolkit/12.6
conda activate /scratch/gpfs/kc32/conda/pyspim3

#INP_FLDR="/scratch/gpfs/kc32/ExampleDispimData/example_dispim_data/fruiting_body001/proc_ortho"
INP_FLDRA="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body001/proc_ortho/a.zarr"
INP_FLDRB="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body001/proc_ortho/b.zarr"

python register.py \
        --view-a=$INP_FLDRA \
	--view-b=$INP_FLDRB \
        --metric='cr' \
        --interp-method='cubspl' \
	--output-type='float' \
	-n=1
