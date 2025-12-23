#!/bin/bash

python deploy/python/infer.py --model_dir /opt/ml/models/ppyoloe_plus_crn_l_80e_custom --image_dir /opt/ml/input/dataset/custom --output_dir /opt/ml/output/labelled_images/ppyoloe_plus_crn_l_80e_custom --device GPU --batch_size 32 --save_results --threshold 0.9

python deploy/python/infer.py --model_dir /opt/ml/models/rtdetr_r101vd_6x_custom --image_dir /opt/ml/input/dataset/custom --output_dir /opt/ml/output/labelled_images/rtdetr_r101vd_6x_custom --device GPU --batch_size 32 --save_results --threshold 0.9

python label_and_compare.py

chmod -R 777 /opt/ml/output

