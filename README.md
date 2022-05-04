# captum-nanodet
Applying Captum library to ImageNat pretrained shufflenetv2 with [nanodet library](https://github.com/RangiLyu/nanodet)

# Instructions

1) Clone this repo

2) Clone https://github.com/RangiLyu/nanodet in the same directory

3) Build Dockerfile without changing directory `sudo docker build -t nanodet -f captum-nanodet/Dockerfile .`

4) `cd captum-nanodet`

5) `bash run_docker_gpu.bash`

6) `python3 main.py`