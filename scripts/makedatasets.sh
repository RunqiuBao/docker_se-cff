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

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type train \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 0 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type train \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 1 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type train \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 2 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type train \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 3 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type train \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/train/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 4 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type valid \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/valid/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/valid/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 0 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type valid \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/valid/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/valid/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 1 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/ \
    --dataset_type valid \
    --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/valid/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/seqs/valid/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
    --seq_idx 2 \
    --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 