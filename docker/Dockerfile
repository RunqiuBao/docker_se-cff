# nvidiaが提供するcudaイメーjiをベースとしてダウンロード
FROM nvcr.io/nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Docker実行してシェルに入ったときの初期ディレクトリ（ワークディレクトリ）の設定
WORKDIR /root/

# nvidia-container-runtime（描画するための環境変数の設定）
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# install minimum apps
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get -y install git vim less cmake

# install opencv python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# install pytorch and useful python packages
RUN apt-get -y install python3-pip && pip3 install opencv-python numpy torch torchvision tensorboard scikit-image pytorch-msssim lmdb matplotlib hydra-core thop tqdm

# add useful commands to .bashrc
RUN echo 'alias kk="kill -9 %"' >> ~/.bashrc

# install mmdetection depends
RUN pip3 install -U openmim && mim install mmengine && mim install "mmcv>=2.0.0"
RUN cd /root/code/mmdetection && pip3 install -v -e .
