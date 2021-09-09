# PRLMU

This repository is for the paper, "**Progressive Residual Learning with Memory Upgrade for Ultrasound Image Blind Super-resolution**", which has been submitted and is awaiting review.
## Requirements

+ python: 3.8.3
+ pytorch: 1.5.0
+ CUDA: 10.1
+ Ubuntu: 18.04
+ Python packages: pip3 install numpy opencv-python lmdb pyyaml

## Pre-trained Model & Dataset Preparation
Pretrained model of PRLMU and test sets are available at [BaiduYun](https://pan.baidu.com/s/1jBaxP-_KI7LRh0LLOey55g)(password:7x0w). After the dataset downloaded, run `codes/scripts/create_lmdb.py` to transform datasets to binary files(you need  to modify the paths by your self).

## Training   

Training code will be available after the paper received.

## Testing
You need to modify the dataset path and the pre-trainied model path in `test_setting.yml` before testing.Then run the following command:
> cd codes/config/cascade \
> python test.py -opt=test_setting.yml

## Results

### Results on CCA-US dataset

![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results1.png)

### Results on US-CASE dataset

![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results2.png)

### Results on Real World

#### fetal head ultrasound images
![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results3.png)

#### thyroid ultrasound images
![img](https://github.com/hengliusky/PRLMU/blob/main/pic/Results4.png)
