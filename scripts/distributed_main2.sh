#!/bin/bash

set -x

cuda_idx='0'

config_path=/root/code/docker_pytorch_trainnn/configs/config_mujinextended.yaml
data_root=/root/data/mujin_datasets/jp_post_generated/
save_root=/root/code/docker_pytorch_trainnn/experiments/mujinextended2/
num_workers=0
NUM_PROC=1

CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} --save_term 500 --resume_cpt /root/code/docker_pytorch_trainnn/weights_jppost_rfdetr/best260306.pth --only_test --is_save_onnx #--only_resume_weight_from concentration_net

# # generate onnx model
# CUDA_VISIBLE_DEVICES=${cuda_idx} python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$RANDOM ../src/distributed_main.py --config_path ${config_path} --data_root ${data_root} --save_root ${save_root} --num_workers ${num_workers} --resume_cpt /root/code/for_dgx/docker_pytorch_trainnn/weights_mix_sim_real_imageonly/best.pth --only_test --is_save_onnx  #--only_resume_weight_from concentration_net
