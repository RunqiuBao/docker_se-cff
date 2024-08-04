#!/bin/bash

set -x

cuda_idx='0, 2, 3'

config_path=/root/code/docker_se-cff/configs/config_unitreego_inference.yaml
data_root=/root/data/unitree-go-dataset/slam/
save_root=/root/code/docker_se-cff/experiments/unitree-go/ 
num_workers=3
NUM_PROC=1

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} --resume_cpt /root/code/docker_se-cff/experiments/unitree-go/weights/best.pth --only_resume_weight --only_test
