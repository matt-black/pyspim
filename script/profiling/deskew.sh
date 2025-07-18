#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=0:10:00
#SBATCH --job-name=deskew_prof
#SBATCH --output=job-prof_deskew-%j.out

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.6

# NOTE: Rob's pyspim conda env, change as needed
conda activate /scratch/gpfs/AKEY/rbierman/pyspim/conda

#ACQ_PATH="/scratch/gpfs/SHAEVITZ/out/mito002_t/a_c00.tif"
ACQ_PATH="/scratch/gpfs/rb3242/a_c00.tif"

# Determine whether to run each comparison and profiling step (0 = no, 1 = yes)
RUN_COMPARE=1
RUN_NSYS=0
RUN_NCU=0

# Compare deskew methods raw (cuda raw kernel) and orthogonal (numpy CPU)
# Timings July 17 2025 on the a_c00.tif dataset:
# - raw [3.402, 1.037] <-- note first run is much slower (due to data transfer?)
# - orthogonal [84.571, 81.526]
#
# Comparison of results for raw and orthogonal methods, are these close enough?:
# - ✓ Max difference: 65535.00
# - ✓ Mean difference: 74.4952
if [ $RUN_COMPARE -eq 1 ]; then
    python deskew_compare.py \
        --input-path=$ACQ_PATH \
        --direction=1 \
        --preserve-data-type \
        --num-repeat=2 \
        --methods raw orthogonal \
        --use-tiff
fi

# nsight systems profile generates a report#.nsys-rep and report#.sqlite
if [ $RUN_NSYS -eq 1 ]; then
    nsys profile \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --stats=true \
        python deskew_compare.py \
            --input-path=$ACQ_PATH \
            --direction=1 \
            --preserve-data-type \
            --num-repeat=1 \
            --methods raw \
            --use-tiff
fi

# nsight compute profile generates ncu_deskew_output.ncu-rep
if [ "$RUN_NCU" -eq 1 ]; then
    ncu --set full --clock-control none \
        -k deskew_kernel_u16 \
        -c 10 \
        -f -o ncu_deskew_output \
        python deskew_compare.py \
            --input-path="$ACQ_PATH" \
            --direction=1 \
            --preserve-data-type \
            --num-repeat=1 \
            --methods raw \
            --use-tiff
fi
