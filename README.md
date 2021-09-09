# PRLMU

This repository is for the paper, "**Progressive Residual Learning with Memory Upgrade for Ultrasound Image Blind Super-resolution**", which  is being submitted and awaiting review.

## Requirements

+ python: 3.8.3
+ pytorch: 1.5.0
+ CUDA: 10.1
+ Ubuntu: 18.04

## Pretrained Model & Dataset
Pretrained model of PRLMU is available at [BaiduYun](https://pan.baidu.com/s/1jBaxP-_KI7LRh0LLOey55g)(password:7x0w)
## Training   

Training code will be available after the paper received.

## Testing

+ run `codes/scripts/create_lmdb` to transform datasets to binary files(you need  to modify the paths by your self)
+ `cd codes/config/cascade`
+ `python test.py -opt=test_setting.yml`

## Results

Results on CCA-US dataset
