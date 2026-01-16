#!/bin/bash

# Configuration file
CONFIG_FILE="configs/ppyoloe/ppyoloe_plus_crn_s_50e_custom_2class.yml"
MODEL_NAME="ppyoloe_plus_crn_s_50e_custom_2class"  # The name of the model is also the folder name under output/model

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

#export model to onnx
pip install paddle2onnx

paddle2onnx --model_dir /opt/ml/output/model/${MODEL_NAME} \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file /opt/ml/output/model/${MODEL_NAME}/model.onnx 2>&1 | tee /opt/ml/output/model/logs/paddle2onnx_${timestamp}.log