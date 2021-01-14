# Entity-Graph-VLN

Code of the NeurIPS 2020 paper:
**Language and Visual Entity Relationship Graph for Agent Navigation**<br>
[**Yicong Hong**](http://www.yiconghong.me/), [Cristian Rodriguez-Opazo](https://crodriguezo.github.io/), [Yuankai Qi](https://sites.google.com/site/yuankiqi/home), [Qi Wu](http://www.qi-wu.me/), [Stephen Gould](http://users.cecs.anu.edu.au/~sgould/)<br>

[[Paper](https://papers.nips.cc/paper/2020/hash/56dc0997d871e9177069bb472574eb29-Abstract.html)] [[Supplemental](https://papers.nips.cc/paper/2020/file/56dc0997d871e9177069bb472574eb29-Supplemental.pdf)] [[GitHub](https://github.com/YicongHong/Entity-Graph-VLN)]

<p align="center">
<img src="teaser/f1.png" width="100%">
</p>

## Prerequisites

### Installation

Install the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator).

Please find the versions of packages in our environment [here](https://github.com/YicongHong/Entity-Graph-VLN/blob/master/entity_graph_vln.yml). In particular, we use:
- Python 3.6.9
- NumPy 1.18.1
- OpenCV 3.4.2
- PyTorch 1.3.0
- Torchvision 0.4.1

### Data Preparation

Please follow the instructions below to prepare the data in directories:

- `connectivity`
    - Download the [connectivity maps [23.8MB]](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
- `data`
    - Download the [R2R data [5.8MB]](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R/data).
    - Download the vocabulary and the [augmented data from EnvDrop [79.5MB]](https://github.com/airsplay/R2R-EnvDrop/tree/master/tasks/R2R/data).
- `img_features`
    - Download the [Scene features [4.2GB]](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1) (ResNet-152-Places365).
    - Download the pre-processed [Object features and vocabulary [1.3GB]](https://zenodo.org/record/4310441/files/objects.zip?download=1) ([Caffe Faster-RCNN](https://github.com/peteanderson80/bottom-up-attention)).

### Trained Network Weights

- `snap`
    - Download the trained [network weights [146.0MB]](https://zenodo.org/record/4310441/files/snap.zip?download=1)

## R2R Navigation

Please read Peter Anderson's VLN paper for the [R2R Navigation task](https://arxiv.org/abs/1711.07280).

Our code is based on the code structure of the [EnvDrop](https://github.com/airsplay/R2R-EnvDrop).

### Reproduce Testing Results

To replicate the performance reported in our paper, load the trained network weights and run validation:
```bash
bash run/agent.bash
```

### Training

#### Navigator

To train the network from scratch, first train a Navigator on the R2R training split:

Modify `run/agent.bash`, remove the argument for `--load` and set `--train listener`. Then,
```bash
bash run/agent.bash
```
The trained Navigator will be saved under `snap/`.

#### Speaker

You also need to train a [Speaker](https://github.com/airsplay/R2R-EnvDrop) for augmented training:
```bash
bash run/speak.bash
```
The trained Speaker will be saved under `snap/`.

#### Augmented Navigator

Finally, keep training the Navigator with the mixture of original data and [augmented data](http://www.cs.unc.edu/~airsplay/aug_paths.json):
```bash
bash run/bt_envdrop.bash
```
We apply a one-step learning rate decay to 1e-5 when training saturates.

## Citation
If you use or discuss our Entity Relationship Graph, please cite our paper:
```
@article{hong2020language,
  title={Language and Visual Entity Relationship Graph for Agent Navigation},
  author={Hong, Yicong and Rodriguez, Cristian and Qi, Yuankai and Wu, Qi and Gould, Stephen},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
