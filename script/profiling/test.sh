#!/bin/bash

module purge
module load cudatoolkit/12.6
module load anaconda3/2024.6
module load nsight-systems/2025.3.1
conda activate /scratch/gpfs/kc32/conda/pyspim3
module list

#INP_FLDR="/scratch/gpfs/kc32/ExampleDispimData/example_dispim_data/fruiting_body001/proc_ortho"
INP_FLDRA="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body001/proc_ortho/a.zarr"
INP_FLDRB="/projects/SHAEVITZ/mb46/fb_dispim/13hr/2025-06-06/fruiting_body001/proc_ortho/b.zarr"

#ncu --set full --clock-control none \
#    -k regex:correlationRatio \
#    -c 1 \
#    -f -o ncu_output \
#    --launch-skip-before-match 0 \
nsys profile --trace=cuda,nvtx,osrt \
	--output=nsys_register \
	    python register.py \
        	--view-a=$INP_FLDRA \
	        --view-b=$INP_FLDRB \
	        --metric='cr' \
	        --interp-method='cubspl' \
		--output-type='double' \
	        --num-repeat=1 \
	        --cupyx-benchmark
#python register.py \
#    --input-folder=$INP_FLDR \
#    --output-folder=$INP_FLDR/reg_trs0 \
#    --channel=0 \
#    --metric='cr' \
#    --transform='t+r+s' \
#    --bounds='20,20,20,5,5,5,0.05,0.05,0.05' \
#    --interp-method='cubspl' \
#    --verbose

#
