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

python3 teod_makedataset.py \
    --data_root /home/runqiu/datasets/unitree-go-dataset/slam/ \
    --dataset_type test \
    --lmdb_dir /home/runqiu/datasets/unitree-go-dataset/slam/test/lmdb/ \
    --view4label_dir /home/runqiu/datasets/unitree-go-dataset/slam/test/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx_toselect 1 \
    --seq_idx 14 \
    --calib_path /home/runqiu/datasets/unitree-go-dataset/calib.json

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/datasets/unitree-go-dataset/slam/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/datasets/unitree-go-dataset/slam/test/lmdb/ \
#     --view4label_dir /home/runqiu/datasets/unitree-go-dataset/slam/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx_toselect 2 \
#     --seq_idx 22 \
#     --calib_path /home/runqiu/datasets/unitree-go-dataset/calib.json

# python3 teod_makedataset.py \
#     --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego \
#     --dataset_type train \
#     --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/train/lmdb/ \
#     --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 0 \
#     --calib_path /home/runqiu/datasets/unitree-go-dataset/calib.json 
