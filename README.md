# [AAAI-23] TopicFM: Robust and Interpretable Topic-Assisted Feature Matching

The method first inferred the latent topics (high-level context information) for each image and then use them to explicitly learn robust feature representation for the matching task.

![Alt Text](demo/topicfm.gif)

**Overall Architecture:**

![Alt Text](demo/architecture_v4.png)

## TODO List

- [x] Release training and evaluation code on MegaDepth and ScanNet
- [ ] Evaluation on HPatches, Aachen Day&Night, and InLoc
- [ ] Evaluation for Image Matching Challenge

## Requirements

All experiments in this paper are implemented on the Ubuntu environment 
with a NVIDIA driver of at least 430.64 and CUDA 10.1.

First, create a virtual environment by anaconda as follows,

    conda create -n topicfm python=3.8 
    conda activate topicfm
    conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch
    pip install -r requirements.txt
    # using pip to install any missing packages

## Data Preparation

The proposed method is trained on the MegaDepth dataset and evaluated on the MegaDepth test, ScanNet, HPatches, Aachen Day and Night (v1.1), and InLoc dataset.
All these datasets are large, so we cannot include them in this code. 
The following descriptions help download these datasets. 

### MegaDepth

This dataset is used for both training and evaluation (Li and Snavely 2018). 
To use this dataset with our code, please follow the [instruction of LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md) (Sun et al. 2021)

### ScanNet 
We only use 1500 image pairs of ScanNet (Dai et al. 2017) for evaluation. 
Please download and prepare [test data](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf) of ScanNet
provided by [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md).

## Training

To train our model, we recommend to use GPUs card as much as possible, and each GPU should be at least 12GB.
In our settings, we train on 4 GPUs, each of which is 12GB. 
Please setup your hardware environment in `scripts/reproduce_train/outdoor.sh`.
And then run this command to start training.

    bash scripts/reproduce_train/outdoor.sh

 We then provide the trained model in `pretrained/model_best.ckpt`
## Evaluation

### MegaDepth

    bash scripts/reproduce_test/outdoor.sh

### ScanNet

    bash scripts/reproduce_test/indoor.sh

### HPatches, Aachen v1.1, InLoc

To evaluate on these datasets, we integrate our code to the image-matching-toolbox provided by Zhou et al. (2021).
The updated code is available [here](https://github.com/TruongKhang/image-matching-toolbox). 
The detailed evaluation will be provided soon.



### Runtime comparison

The runtime reported in the paper is measured by averaging runtime of 1500 image pairs of the ScanNet evaluation dataset.
The image size can be changed at `configs/data/scannet_test_1500.py`

    python visualization.py --method <method_name> --dataset_name "scannet" --measure_time --no_viz
    # note that method_name is in ["topicfm", "loftr"]

To measure time for LoFTR, please download the LoFTR's code as follows:

    git submodule update --init
    # download pretrained models
    mkdir third_party/loftr/pretrained 
    gdown --id 1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY -O third_party/loftr/pretrained/outdoor_ds.ckpt

## Citations
If you find this work useful, please cite this:

    @article{giang2022topicfm,
        title={TopicFM: Robust and Interpretable Topic-assisted Feature Matching},
        author={Giang, Khang Truong and Song, Soohwan and Jo, Sungho},
        journal={arXiv preprint arXiv:2207.00328},
        year={2022}
    }

## Acknowledgement
This code is built based on [LoFTR](https://github.com/zju3dv/LoFTR). We thank the authors for their useful source code.