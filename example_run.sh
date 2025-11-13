#! /usr/bin/env sh
# Example shell script for running the Soo Locks orthoimagery pipeline.

# Define a base path to data for convenience
BASE_PATH="/Users/rdcrlrka/Research/Soo_locks"

# Run the pipeline
python generate_orthoimage.py \
-video_folder "${BASE_PATH}/20251001_imagery/video" \
-target_datetime "20251001171000" \
-inputs_folder "${BASE_PATH}/inputs" \
-output_folder "${BASE_PATH}/outputs" \
-refine_cameras 1