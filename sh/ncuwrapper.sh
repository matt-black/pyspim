#!/bin/bash

# NCU wrapper script with source code information
# Usage: ./ncu_wrapper.sh python foo.py
# Usage: ./ncu_wrapper.sh ./my_cuda_program arg1 arg2

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 python foo.py"
    echo "Example: $0 ./cuda_program arg1 arg2"
    exit 1
fi

OUTPUT_FILE="ncu_profile_$(date +%Y%m%d_%H%M%S)"

echo "Running ncu with source code information..."
echo "Command: $@"
echo "Output will be saved as: ${OUTPUT_FILE}.ncu-rep"

# Run ncu with source information flags
ncu \
    --output=${OUTPUT_FILE} \
    --force-overwrite \
    --source-folders=. \
    --source-folders=/path/to/your/source \
    --nvtx \
    --target-processes=all \
    --kernel-name-base=function \
    --launch-skip-before-match=10 \
    --launch-count=1 \
    "$@"

echo "Profiling complete!"
echo "To view with source code: ncu-ui ${OUTPUT_FILE}.ncu-rep"
