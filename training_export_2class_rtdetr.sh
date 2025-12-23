#!/bin/bash

# Configuration file
CONFIG_FILE="configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml"
# Generate a timestamp for logging
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Create the logs directory if it doesn't exist
mkdir -p /opt/ml/output/model/logs

# Training the model
python -u tools/train.py -c ${CONFIG_FILE} -o use_gpu=true --use_vdl=true --vdl_log_dir=/opt/ml/output/model/logs --eval 2>&1 | tee /opt/ml/output/model/logs/training_${timestamp}.log

# Exporting the model and logging the output
python tools/export_model.py -c ${CONFIG_FILE} --output_dir /opt/ml/output/model 2>&1 | tee /opt/ml/output/model/logs/export_model_${timestamp}.log

# Setting permissions for the output directory
chmod -R 777 /opt/ml/output/model
