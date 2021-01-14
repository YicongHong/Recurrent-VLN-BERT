# Recurrent VLN-BERT

Code of the Recurrent-VLN-BERT paper:
**A Recurrent Vision-and-Language BERT for Navigation**<br>
[**Yicong Hong**](http://www.yiconghong.me/), [Qi Wu](http://www.qi-wu.me/), [Yuankai Qi](https://sites.google.com/site/yuankiqi/home), [Cristian Rodriguez-Opazo](https://crodriguezo.github.io/), [Stephen Gould](http://users.cecs.anu.edu.au/~sgould/)<br>

[[Paper & Appendices](https://arxiv.org/abs/2011.13922) | [GitHub](https://github.com/YicongHong/Recurrent-VLN-BERT)]

## Prerequisites

### Installation

Install the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator).
Please find the versions of packages in our environment [here](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/main/recurrent-vln-bert.yml).

Install the [Pytorch-Transformers](https://github.com/huggingface/transformers).
In particular, we use [this version](https://github.com/huggingface/transformers/tree/067923d3267325f525f4e46f357360c191ba562e) (same as [OSCAR](https://github.com/microsoft/Oscar)) in our experiments.

### Data Preparation

Please follow the instructions below to prepare the data in directories:

- MP3D navigability graphs: `connectivity`
    - Download the [connectivity maps [23.8MB]](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
- R2R data: `data`
    - Download the [R2R data [5.8MB]](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R/data).
- Augmented data: `data/prevalent`
    - Download the [collected triplets in PREVALENT [1.5GB]](https://zenodo.org/record/4437864/files/prevalent_aug.json?download=1) (pre-processed for easy use).
- MP3D image features: `img_features`
    - Download the [Scene features [4.2GB]](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1) (ResNet-152-Places365).

### Initial OSCAR and PREVALENT weights

Please refer to [vlnbert_init.py](https://github.com/YicongHong/Recurrent-VLN-BERT/blob/main/r2r_src/vlnbert/vlnbert_init.py) to set up the directories.

- Pre-trained [OSCAR](https://github.com/microsoft/Oscar) weights
    - Download the `base-no-labels` following [this guide](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md).
- Pre-trained [PREVALENT](https://github.com/weituo12321/PREVALENT) weights
    - Download the `pytorch_model.bin` from [here](https://drive.google.com/drive/folders/1sW2xVaSaciZiQ7ViKzm_KbrLD_XvOq5y).

### Trained Network Weights

- Recurrent-VLN-BERT: `snap`
    - Download the [trained network weights [2.5GB]](https://zenodo.org/record/4437864/files/snap.zip?download=1) for our OSCAR-based and PREVALENT-based models.

## R2R Navigation

Please read Peter Anderson's VLN paper for the [R2R Navigation task](https://arxiv.org/abs/1711.07280).

Our code is based on the code structure of the [EnvDrop](https://github.com/airsplay/R2R-EnvDrop).

### Reproduce Testing Results

To replicate the performance reported in our paper, load the trained network weights and run validation:
```bash
bash run/test_agent.bash
```

You can simply switch between the OSCAR-based and the PREVALENT-based VLN models by changing the arguments `vlnmodel` (oscar or prevalent) and `load` (trained model paths).

### Training

#### Navigator

To train the network from scratch, simply run:
```bash
bash run/train_agent.bash
```
The trained Navigator will be saved under `snap/`.

## Citation
If you use or discuss our Recurrent VLN-BERT, please cite our paper:
```
@article{hong2020recurrent,
  title={A Recurrent Vision-and-Language BERT for Navigation},
  author={Hong, Yicong and Wu, Qi and Qi, Yuankai and Rodriguez-Opazo, Cristian and Gould, Stephen},
  journal={arXiv preprint arXiv:2011.13922},
  year={2020}
}
```
