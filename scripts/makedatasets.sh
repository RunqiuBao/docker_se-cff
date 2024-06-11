# python3 teod_makedataset.py \
#     --data_root /mnt/data/datasets/blender-vibration \
#     --dataset_type train \
#     --lmdb_dir /mnt/data/datasets/blender-vibration/train/lmdb/ \
#     --view4label_dir /mnt/data/datasets/blender-vibration/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_blendervib.yaml \
#     --seq_idx 0 

python3 teod_makedataset.py \
    --data_root /mnt/data/datasets/blender-vibration \
    --dataset_type valid \
    --lmdb_dir /mnt/data/datasets/blender-vibration/valid/lmdb/ \
    --view4label_dir /mnt/data/datasets/blender-vibration/valid/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_blendervib.yaml \
    --seq_idx 0 