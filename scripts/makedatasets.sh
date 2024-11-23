# python3 teod_makedataset.py \
#     --data_root /media/runqiu/t7shield/datasets/blender-vibration/raw-data/x-shape-vib-hdr/ \
#     --dataset_type train \
#     --lmdb_dir /media/runqiu/t7shield/datasets/blender-vibration/raw-data/x-shape-vib-hdr/train/lmdb/ \
#     --view4label_dir /media/runqiu/t7shield/datasets/blender-vibration/raw-data/x-shape-vib-hdr/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_blendervib.yaml \
#     --seq_idx 0 

# python3 teod_makedataset.py \
#     --data_root /media/runqiu/t7shield/datasets/blender-vibration \
#     --dataset_type test \
#     --lmdb_dir /media/runqiu/t7shield/datasets/blender-vibration/test/lmdb/ \
#     --view4label_dir /media/runqiu/t7shield/datasets/blender-vibration/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen.yaml \
#     --seq_idx 0 

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/Documents/datasets/test/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 0 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/Documents/datasets/test/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 1 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/Documents/datasets/test/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 2 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/Documents/datasets/test/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 3 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/Documents/datasets/test/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 4 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/ \
#     --dataset_type test \
#     --lmdb_dir /home/runqiu/Documents/datasets/test/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/test/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 5 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json

python3 teod_makedataset.py \
    --data_root /home/runqiu/Documents/datasets/eventstereoslam_general/object_detection_trainset/ \
    --dataset_type train \
    --lmdb_dir /home/runqiu/Documents/datasets/eventstereoslam_general/object_detection_trainset/train/lmdb/ \
    --view4label_dir /home/runqiu/Documents/datasets/eventstereoslam_general/object_detection_trainset/train/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 0 \
    --calib_path /home/runqiu/Documents/datasets/calib.json \
    --is_leftright_flipped

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/binpicking/ \
#     --dataset_type train \
#     --lmdb_dir /home/runqiu/Documents/datasets/binpicking/train/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/binpicking/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_binpicking.yaml \
#     --seq_idx 1 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json \
#     --is_leftright_flipped

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/binpicking/ \
#     --dataset_type train \
#     --lmdb_dir /home/runqiu/Documents/datasets/binpicking/train/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/binpicking/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_binpicking.yaml \
#     --seq_idx 2 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json \
#     --is_leftright_flipped

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/binpicking/ \
#     --dataset_type valid \
#     --lmdb_dir /home/runqiu/Documents/datasets/binpicking/valid/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/binpicking/valid/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_binpicking.yaml \
#     --seq_idx 0 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json \
#     --is_leftright_flipped

# python3 teod_makedataset.py \
#     --data_root /home/runqiu/Documents/datasets/binpicking/ \
#     --dataset_type valid \
#     --lmdb_dir /home/runqiu/Documents/datasets/binpicking/valid/lmdb/ \
#     --view4label_dir /home/runqiu/Documents/datasets/binpicking/valid/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_binpicking.yaml \
#     --seq_idx 1 \
#     --calib_path /home/runqiu/Documents/datasets/calib.json \
#     --is_leftright_flipped