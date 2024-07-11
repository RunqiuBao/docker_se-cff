# python3 teod_makedataset.py \
#     --data_root /mnt/data/datasets/blender-vibration \
#     --dataset_type train \
#     --lmdb_dir /mnt/data/datasets/blender-vibration/train/lmdb/ \
#     --view4label_dir /mnt/data/datasets/blender-vibration/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_blendervib.yaml \
#     --seq_idx 0 

python3 teod_makedataset.py \
    --data_root /media/runqiu/t7shield/datasets/blender-vibration \
    --dataset_type test \
    --lmdb_dir /media/runqiu/t7shield/datasets/blender-vibration/test/lmdb/ \
    --view4label_dir /media/runqiu/t7shield/datasets/blender-vibration/test/view4label/ \
    --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen.yaml \
    --seq_idx 0 

# python3 teod_makedataset.py \
#     --data_root /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego \
#     --dataset_type train \
#     --lmdb_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/train/lmdb/ \
#     --view4label_dir /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/train/view4label/ \
#     --config_path /home/runqiu/code/docker_se-cff/configs/config_datagen_unitreego.yaml \
#     --seq_idx 0 \
#     --calib_path /media/runqiu/t7shield/stereo_eventcam_experiment/event_sequences/dataset/unitreego/calib.json 
