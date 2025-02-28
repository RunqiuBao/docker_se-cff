#!/bin/bash

set -x

cuda_idx='0,1,2,3'

config_path=/root/code/docker_pytorch_trainnn/configs/config_unitreego_trainyolox.yaml
data_root=/root/data/unitree-go-dataset/objdet/
save_root=/root/code/docker_pytorch_trainnn/experiments/test_trainyolox_unitree/
num_workers=4
NUM_PROC=4

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} --resume_cpt /root/code/docker_pytorch_trainnn/test/weights_objdet_unitree_trainyolox/best.pth --only_resume_weight # --only_test # --is_save_onnx  --only_resume_weight_from concentration_net
