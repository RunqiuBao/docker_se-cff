#!/bin/bash

set -x

cuda_idx='0,1,2,3'

config_path=/root/code/for_dgx/docker_pytorch_trainnn/configs/config_circleblob.yaml
data_root=/root/code/for_dgx/datasets/circle_blob/
save_root=/root/code/for_dgx/docker_pytorch_trainnn/experiments/circle_blob/
num_workers=4
NUM_PROC=4

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} #--resume_cpt /root/code/for_dgx/docker_pytorch_trainnn/experiments/circle_blob/weights/best.pth --only_test  #--only_resume_weight_from concentration_net

# # generate onnx model
# CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} --resume_cpt /root/code/for_dgx/docker_pytorch_trainnn/weights_mix_sim_real_imageonly/best.pth --only_test --is_save_onnx  #--only_resume_weight_from concentration_net
