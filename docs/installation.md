
## Installation of Minkowski Engine 

We use [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) for sparse convolution of point cloud in our project.

`MinkowskiEngine==0.5.4` with `cudatoolkit=10.2` was used for the project.

First we creat a new environment
```
conda create -n box2mask python=3.7
conda activate box2mask
```

Setup the CUDA system environment variables like the example below:
```
cuda_version=10.2
# please set the right path to CUDA in your system, bellow is an example used for our system
export CUDA_HOME=/usr/lib/cuda-${cuda_version}/ 
export PATH=/usr/lib/cuda-${cuda_version}/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/lib/cuda-${cuda_version}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/usr/lib/cuda-${cuda_version}/
```

Next, we install pytorch with cudatoolkit and dependencies
```
conda install pytorch=1.8.1 torchvision cudatoolkit=${cuda_version} -c pytorch -c nvidia
```

Install dependencies for Minkowski Engine
```
pip install torch ninja
conda install openblas-devel -c anaconda 
```

We then install gcc version 7
`sudo apt install g++-7`  # For CUDA 10.2, must use GCC <= 8
> Make sure `g++-7 --version` is at least 7.4.0
> export CXX=g++-7

Install Minkowski Engine via pip:
```
pip install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps
```

For more detailed installation instruction, see [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
## Checking installations of Minkowski Engine

The following commands will clone the repository of Minkowski Engine and run an example segmentation model on an indoor point cloud:
```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
# code requires open3d
pip install open3d
python -m examples.indoor
```

## Install GIT repository and other dependencies
The following commands will clone Box2Mask repo on your machine and install the remaining dependencies. Note that you should still be using `box2mask` environemnt
```
git clone -b release https://github.com/jchibane/impseg.git box2mask
cd box2mask
conda env update --file env.yml
```

