
# SMHR: Single-stage Multiple 3D Hand mesh Reconstruction with weak supervison

## Introduction
* This repo is official **[PyTorch](https://pytorch.org)** implementation of **[End-to-end Weakly-supervised Multiple 3D Hand Mesh Reconstruction from Single Image](https://arxiv.org/abs/2204.08154)**. 

<p align="middle">
    <img src="assets/Demo.jpg", width="840" height="600">
</p>

## Demo on a random image
1. Download pre-trained MultiNet from [here](xxx=sharing)
2. Put the model at `demo` folder
3. Modify the config file at 'config' folder
4. run `python demo.py --gpu 0`
5. You can see `result_2D.jpg` and 3D viewer.

## Install
*   Environment
```
    conda create -n hands_wsp python=3.7
    conda activate hands_wsp

    # If you failed to install pytorch, you may try to modify your conda source: https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
    conda install -c pytorch pytorch==1.6.0 torchvision cudatoolkit=10.1
    
    # install pytorch3d from source if you are not using latest pytorch version
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    conda install pytorch3d -c pytorch3d

    pip install -r requirements.txt    
```

## Directory

### Root
The `${ROOT}` is described as below.
```
${ROOT}
|-- data
|-- lib
|-- outputs
|-- scripts
```
* `data` contains data loading codes and soft links to images and annotations directories.
* `lib` contains main codes for 3D multiple hand pose estimation.
* `outputs` contains log, trained models, imgs, and pretrained models.
* `scripts` contains running scripts.

