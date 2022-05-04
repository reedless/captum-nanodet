# run git clone https://github.com/RangiLyu/nanodet.git in dir above

# build Dockerfile from parent directory
# sudo docker build -t nanodet -f captum-nanodet/Dockerfile .

FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

# Updating the CUDA Linux GPG Repository Key
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    libgl1 libsm6 libxrender1 libglib2.0-0 \
	python3-pip \
    git \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

# install captum-nanodet requirements
COPY ./captum-nanodet/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install nanodet
COPY nanodet/ /nanodet/
WORKDIR /nanodet
RUN python3 -m pip install -r requirements.txt
RUN python3 setup.py develop

WORKDIR /app
