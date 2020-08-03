# Self-supervised learning through the eyes of a child

This repository contains code for reproducing the results reported in the following paper:

Orhan AE, Gupta VV, Lake BM (2020) [Self-supervised learning through the eyes of a child.](https://arxiv.org/abs/2007.16189) arXiv:2007.16289.

## Requirements

* pytorch == 1.5.1
* torchvision == 0.6.1

Slightly older versions will probably work fine as well.

## Datasets

This project uses the SAYCam dataset described in the following paper: 

Sullivan J, Mei M, Perfors A, Wojcik EH, Frank MC (2020) [SAYCam: A large, longitudinal audiovisual dataset recorded from the infant’s perspective.](https://psyarxiv.com/fy8zx/) PsyArXiv.

The dataset is hosted on the [Databrary](https://nyu.databrary.org/) repository for behavioral science. Unfortunately, we are unable to publicly share the SAYCam dataset here due to the terms of use. However, interested researchers can apply for access to the dataset with approval from their institution's IRB. 

In addition, this project also uses the Toybox dataset for evaluation purposes. The Toybox dataset is publicly available at [this address](https://aivaslab.github.io/toybox/).

## Code description

* [`moco`](https://github.com/eminorhan/baby-vision/tree/master/moco) directory contains helper files for training static and temporal MoCo models. The code here was modified from [Facebook's MoCo repository](https://github.com/facebookresearch/moco).
* [`temporal_classification.py`](https://github.com/eminorhan/baby-vision/blob/master/temporal_classification.py): trains temporal classification models as described in the paper.
* [`moco_img.py`](https://github.com/eminorhan/baby-vision/blob/master/moco_img.py): trains an image-based MoCo model as described in the paper. This code was modified from [Facebook's MoCo repository](https://github.com/facebookresearch/moco).
* [`moco_temp.py`](https://github.com/eminorhan/baby-vision/blob/master/moco_temp.py): trains a temporal MoCo model as described in the paper. This code was also modified from [Facebook's MoCo repository](https://github.com/facebookresearch/moco).
* [`moco_utils.py`](https://github.com/eminorhan/baby-vision/blob/master/moco_utils.py): some utility functions for MoCo training.
* [`linear_decoding.py`](https://github.com/eminorhan/baby-vision/blob/master/linear_decoding.py): evaluates self-supervised models on downstream linear classification tasks.
* [`linear_combination_maps.py`](https://github.com/eminorhan/baby-vision/blob/master/linear_combination_maps.py): plots spatial attention maps as in Figure 4b and Figure 6 in the paper.
* [`highly_activating_imgs.py`](https://github.com/eminorhan/baby-vision/blob/master/highly_activating_imgs.py): finds highly activating images for a given feature as in Figure 7b in the paper.
* [`selectivities.py`](https://github.com/eminorhan/baby-vision/blob/master/selectivities.py): measures the class selecitivity indices of all features in a given layer as in Figure 7a in the paper.
* [`hog_baseline.py`](https://github.com/eminorhan/baby-vision/blob/master/hog_baseline.py): runs the HOG baseline model as described in the paper.

For specific usage examples, please see the slurm scripts provided in the [`scripts`](https://github.com/eminorhan/baby-vision/tree/master/scripts) directory.

## Pre-trained models

We share below the pre-trained weights for our best self-supervised models trained on the SAYCam dataset. Four pre-trained models are provided below: temporal classification models trained on data from the individual children in the SAYCam dataset (TC-S, TC-A, TC-Y) and a temporal classification model trained on data from all three children (TC-SAY).

* [TC-S](https://drive.google.com/file/d/1DVJjpaGhoBPNmlO7jXpwEX3lSCk2ZUCa/view?usp=sharing) (69.4 MB)
* [TC-A](https://drive.google.com/file/d/1uQvJBbuy6P0uCW0HYs1wNgawRU8sGLhC/view?usp=sharing) (54.4 MB)
* [TC-Y](https://drive.google.com/file/d/1TTndiiiqSiCMdZjwYZPKQySZot4ipCrG/view?usp=sharing) (53.3 MB)
* [TC-SAY](https://drive.google.com/file/d/1zeidpBaXqqWCeeYj-fMI7V7x9EiAGH6Q/view?usp=sharing) (123.3 MB)

These models come with the classifier heads attached. To load these models, please do something along the lines of:

```python
import torch
import torchvision.models as models

model = models.mobilenet_v2(pretrained=False)
model.classifier = torch.nn.Linear(in_features=1280, out_features=n_out, bias=True)
model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load('TC-SAY.tar')
model.load_state_dict(checkpoint['model_state_dict'])

```

where `n_out` should be 6269 for TC-SAY, 2765 for TC-S, 1786 for TC-A, and 1718 for TC-Y. The differences here are due to the different lengths of the datasets. To use these models for a different task, you can detach the classifier head and attach a new classifier head of the appropriate size, e.g.:

```python
model.module.classifier = torch.nn.Linear(in_features=1280, out_features=new_n_out, bias=True).cuda()
```

where `new_n_out` is the new output dimensionality. We also intend to release models fine-tuned on ImageNet in the near future for wider applicability.

## Acknowledgments

We are very grateful to the volunteers who contributed recordings to the SAYCam dataset. We thank Jessica Sullivan for her generous assistance with the dataset. We also thank the team behind the Toybox dataset, as well as the developers of Pytorch and torchvision for making this work possible. This project was partly funded by the NSF Award 1922658 NRT-HDR: FUTURE Foundations, Translation, and Responsibility for Data Science.
