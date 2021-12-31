# PRLMU

This repository is for the paper, "**Progressive Residual Learning with Memory Upgrade for Ultrasound Image Blind Super-resolution**", which has been submitted to JBHI.
## Requirements

+ python: 3.8.3
+ pytorch: 1.5.0
+ CUDA: 10.1
+ Ubuntu: 18.04
+ Python packages: pip3 install numpy opencv-python lmdb pyyaml

## Pre-trained Model & Dataset Preparation
Pretrained model of PRLMU and part of the test set are available at [BaiduYun](https://pan.baidu.com/s/1jBaxP-_KI7LRh0LLOey55g)(password:7x0w). After the dataset downloaded, run `codes/scripts/create_lmdb.py` to transform datasets to binary files(you need to modify the paths by yourself).

## Training   

Training code will be available after the paper is accepted.

## Testing
You need to modify the dataset path and the pre-trainied model path in `test_setting.yml` before testing.Then run the following command:
> cd codes/config/cascade \
> python test.py -opt=test_setting.yml

## Results

### Results under protocol 1

![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results_1.png)

### Results under protocol 2

![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results_2.png)

### Results on Real World

#### fetal head ultrasound images
![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results_3.png)

#### thyroid ultrasound images
![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results_4.png)
