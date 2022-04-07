#!/bin/sh
#set -eo pipefail

CFARCONFIG=base_configs_downsample.15.yml

CLASSMODEL=models/VGG-FalsePositives-3.h5

LENGTHMODEL=models/length_CNN_logloss_fulldata.h5

# Execute xView3 inference pipeline using CLI arguments passed in
# 1) Path to directory with all data files for inference
# 2) Scene ID
# 3) Path to output CSV

if [ $# -lt 3 ]; then 
    echo "run_inference.sh: [#1 Path to directory with all data files for inference] [#2 Scene ID] [#3 Path to output CSV]"
else
    conda run --no-capture-output -n xview3 python3 predict_image.py --image_folder "$1" --scene_ids "$2" --output "$3" --cfar_configs_name=$CFARCONFIG --class_channel_indexes 0 1 --length_channel_indexes 0 1 3 --class_model_path $CLASSMODEL --length_model_path $LENGTHMODEL 2>&1 | tee -a /mnt/log.txt

fi

