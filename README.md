## Installation issue

````
#GeForce RTX 3090 with CUDA capability sm_86 is not compatible with the current PyTorch installation.
#The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75.

#torch 1.7 stable -> cuda 11


MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .


#https://github.com/open-mmlab/mmdetection/issues/4052
````