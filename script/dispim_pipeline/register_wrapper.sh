#!/bin/bash
module purge
module load cudatoolkit/12.6
module load anaconda3/2024.6
conda activate /scratch/gpfs/kc32/conda/pyspim3

INP_FLDR="/scratch/gpfs/kc32/ExampleDispimData/example_dispim_data/fruiting_body001/proc_ortho"

ncu --set full --clock-control none \
    -k regex:correlationRatio \
    -c 10 \
    -f -o ncu_output \
    --launch-skip-before-match 20 \
	python register.py \
   		--input-folder=$INP_FLDR \
    		--output-folder=$INP_FLDR/reg_trs0 \
    		--channel=0 \
    		--metric='cr' \
    		--transform='t+r+s' \
    		--bounds='20,20,20,5,5,5,0.05,0.05,0.05' \
    		--interp-method='cubspl' \
    		--verbose

#
