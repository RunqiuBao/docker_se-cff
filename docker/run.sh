# allow GUI application in docker container to reach host device.
# note GUI might still fail, try launching a few more times.
xhost +local:docker

docker run -it \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --gpus all \
    --ipc=host \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/usr/local:/usr/local" \
    --volume="$(pwd)/..:/root/code:rw" \
    --volume="/home/runqiu/datasets:/root/data:rw" \
    --volume="/dev/shm:/dev/shm:rw" \
    --name=dockerpytorch \
    dockerpytorch
echo "done"
