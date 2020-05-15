# FashionAI
[![Travis](https://img.shields.io/travis/Lmy0217/FashionAI.svg?label=Travis+CI)](https://travis-ci.com/github/Lmy0217/FashionAI) [![CircleCI](https://img.shields.io/circleci/project/github/Lmy0217/FashionAI.svg?label=CircleCI)](https://circleci.com/gh/Lmy0217/FashionAI) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Lmy0217/FashionAI/pulls)

This repo is code of [FashionAI Global Challenge—Attributes Recognition of Apparel](https://tianchi.aliyun.com/competition/introduction.htm?spm=a2c22.11190735.991137.11.23446d83RhZFij&raceId=231649&_lang=zh_CN) based on PyTorch. **This repo only for learning.**

## Environment
- Operating system: Ubuntu 17.10
- Data would take up to 25GB disk memory
- Memory cost would be around 20GB
- Dependencies: 
  - [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) with GPU
  - [PyTorch](https://github.com/pytorch/pytorch) with packages ([torchvision](https://github.com/pytorch/vision)) installed

## Prerequisites
- Download this repo
  ```bash
  git clone https://github.com/Lmy0217/FashionAI.git
  cd FashionAI
  ```

- Install requirements
  ```bash
  pip3 install -r requirements.txt
  ```

- (Unnecessary) Download the [Attributes Recognition of Apparel](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.7b463a26RhDo2u&raceId=231649) dataset and extract the tar file in the folder `datasets` (now, this folder should contain three folder named 'base', 'web' and 'rank' respectively)

## Usage
The training and testing scripts come with several options, which can be listed with the `--help` flag.
```bash
python3 main.py --help
```

To run the training and testing, simply run main.py. By default, the script runs resnet34 on attribute 'coat_length_labels' with 50 epochs.

To training and testing resnet34 on attribute 'collar_design_labels' with 100 epochs and some learning parameters:
```bash
python3 main.py --model 'resnet34' --attribute 'collar_design_labels' --epochs 100 --batch-size 128 --lr 0.01 --momentum 0.5
```

Every epoch trained model will be saved in the folder `save/[attribute]/[model]`.

## License
The code is licensed with the [MIT](LICENSE) license.
