#!/bin/bash

# Check if output directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset_directory>"
    exit 1
fi

OUTPUT_DIR=$1

# Initialize conda for bash script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate crftm_py39
python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR --from_file

python dataset_toolkits/voxelize.py --output_dir $OUTPUT_DIR
python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR --from_file

conda activate trellis

python dataset_toolkits/extract_feature.py --output_dir $OUTPUT_DIR
python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR --from_file

python dataset_toolkits/encode_ss_latent.py --output_dir $OUTPUT_DIR
python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR --from_file

python dataset_toolkits/encode_latent.py --output_dir $OUTPUT_DIR
python dataset_toolkits/build_metadata.py --output_dir $OUTPUT_DIR --from_file