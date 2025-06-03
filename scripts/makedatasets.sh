# python3 teod_makedataset.py \
#     --data_root /media/runqiu/t7shield/datasets/blender-vibration/raw-data/x-shape-vib-hdr/ \
#     --dataset_type train \
#     --lmdb_dir /media/runqiu/t7shield/datasets/blender-vibration/raw-data/x-shape-vib-hdr/train/lmdb/ \
#     --view4label_dir /media/runqiu/t7shield/datasets/blender-vibration/raw-data/x-shape-vib-hdr/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_blendervib.yaml \
#     --seq_idx 0 

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/datasets/unitree-go-dataset/slam/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/datasets/unitree-go-dataset/slam/test/lmdb/ \
#     --view4label_dir /home/runqiu/datasets/unitree-go-dataset/slam/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 0 \
#     --seq_idx 12 \
#     --calib_path /home/runqiu/datasets/unitree-go-dataset/calib.json 

# python3 teod_makedataset.py \
#     --data_root /root/data/unitree-go-dataset/objdet/ \
#     --dataset_type train \
#     --lmdb_dir /root/data/unitree-go-dataset/objdet/train/lmdb/ \
#     --view4label_dir /root/data/unitree-go-dataset/objdet/train/view4label/ \
#     --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 0 \
#     --seq_idx 0 \
#     --calib_path /root/data/unitree-go-dataset/calib.json

# python3 teod_makedataset.py \
#     --data_root /root/data/unitree-go-dataset/objdet/ \
#     --dataset_type train \
#     --lmdb_dir /root/data/unitree-go-dataset/objdet/train/lmdb/ \
#     --view4label_dir /root/data/unitree-go-dataset/objdet/train/view4label/ \
#     --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 1 \
#     --seq_idx 2 \
#     --calib_path /root/data/unitree-go-dataset/calib.json

# python3 teod_makedataset.py \
#     --data_root /root/data/unitree-go-dataset/objdet/ \
#     --dataset_type train \
#     --lmdb_dir /root/data/unitree-go-dataset/objdet/train/lmdb/ \
#     --view4label_dir /root/data/unitree-go-dataset/objdet/train/view4label/ \
#     --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 2 \
#     --seq_idx 3 \
#     --calib_path /root/data/unitree-go-dataset/calib.json

# python3 teod_makedataset.py \
#     --data_root /root/data/unitree-go-dataset/objdet/ \
#     --dataset_type train \
#     --lmdb_dir /root/data/unitree-go-dataset/objdet/train/lmdb/ \
#     --view4label_dir /root/data/unitree-go-dataset/objdet/train/view4label/ \
#     --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 3 \
#     --seq_idx 4 \
#     --calib_path /root/data/unitree-go-dataset/calib.json


# python3 teod_makedataset.py \
#     --data_root /root/data/unitree-go-dataset/objdet/ \
#     --dataset_type valid \
#     --lmdb_dir /root/data/unitree-go-dataset/objdet/valid/lmdb/ \
#     --view4label_dir /root/data/unitree-go-dataset/objdet/valid/view4label/ \
#     --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 0 \
#     --seq_idx 0 \
#     --calib_path /root/data/unitree-go-dataset/calib.json

# python3 teod_makedataset.py \
#     --data_root /root/data/unitree-go-dataset/objdet/ \
#     --dataset_type valid \
#     --lmdb_dir /root/data/unitree-go-dataset/objdet/valid/lmdb/ \
#     --view4label_dir /root/data/unitree-go-dataset/objdet/valid/view4label/ \
#     --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 1 \
#     --seq_idx 2 \
#     --calib_path /root/data/unitree-go-dataset/calib.json

# python3 teod_makedataset.py \
#     --data_root /root/data/unitree-go-dataset/objdet/ \
#     --dataset_type valid \
#     --lmdb_dir /root/data/unitree-go-dataset/objdet/valid/lmdb/ \
#     --view4label_dir /root/data/unitree-go-dataset/objdet/valid/view4label/ \
#     --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 2 \
#     --seq_idx 3 \
#     --calib_path /root/data/unitree-go-dataset/calib.json

python3 teod_makedataset.py \
    --data_root /root/data/unitree-go-dataset/slam/ \
    --dataset_type test \
    --lmdb_dir /root/data/unitree-go-dataset/slam/test/lmdb/ \
    --view4label_dir /root/data/unitree-go-dataset/slam/test/view4label/ \
    --config_path /root/code/docker_pytorch_trainnn/configs/config_datagen_unitreego.yaml \
    --seq_idx_toselect 2 \
    --seq_idx 22 \
    --calib_path /root/data/unitree-go-dataset/calib.json
