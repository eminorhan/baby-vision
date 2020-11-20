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

* [`temporal_classification.py`](https://github.com/eminorhan/baby-vision/blob/master/temporal_classification.py): trains temporal classification models as described in the paper. This file uses code recycled from the PyTorch ImageNet training [example](https://github.com/pytorch/examples/tree/master/imagenet).
* [`read_saycam.py`](https://github.com/eminorhan/baby-vision/blob/master/read_saycam.py): SAYCam video-to-image reader.
* [`moco`](https://github.com/eminorhan/baby-vision/tree/master/moco) directory contains helper files for training static and temporal MoCo models. The code here was modified from [Facebook's MoCo repository](https://github.com/facebookresearch/moco).
* [`moco_img.py`](https://github.com/eminorhan/baby-vision/blob/master/moco_img.py): trains an image-based MoCo model as described in the paper. This code was modified from [Facebook's MoCo repository](https://github.com/facebookresearch/moco).
* [`moco_temp.py`](https://github.com/eminorhan/baby-vision/blob/master/moco_temp.py): trains a temporal MoCo model as described in the paper. This code was also modified from [Facebook's MoCo repository](https://github.com/facebookresearch/moco).
* [`moco_utils.py`](https://github.com/eminorhan/baby-vision/blob/master/moco_utils.py): some utility functions for MoCo training.
* [`linear_decoding.py`](https://github.com/eminorhan/baby-vision/blob/master/linear_decoding.py): evaluates self-supervised models on downstream linear classification tasks.
* [`linear_combination_maps.py`](https://github.com/eminorhan/baby-vision/blob/master/linear_combination_maps.py): plots spatial attention maps as in Figure 4b and Figure 6 in the paper.
* [`highly_activating_imgs.py`](https://github.com/eminorhan/baby-vision/blob/master/highly_activating_imgs.py): finds highly activating images for a given feature as in Figure 7b in the paper.
* [`selectivities.py`](https://github.com/eminorhan/baby-vision/blob/master/selectivities.py): measures the class selecitivity indices of all features in a given layer as in Figure 7a in the paper.
* [`hog_baseline.py`](https://github.com/eminorhan/baby-vision/blob/master/hog_baseline.py): runs the HOG baseline model as described in the paper.
* [`imagenet_finetuning.py`](https://github.com/eminorhan/baby-vision/blob/master/imagenet_finetuning.py): ImageNet evaluations.
* [`feature_animation.py`](https://github.com/eminorhan/baby-vision/blob/master/feature_animation.py) and [`feature_animation_class.py`](https://github.com/eminorhan/baby-vision/blob/master/feature_animation_class.py): Some tools for visualizing the learned features.

For specific usage examples, please see the slurm scripts provided in the [`scripts`](https://github.com/eminorhan/baby-vision/tree/master/scripts) directory.

## Pre-trained models

### ResNeXt 
Since the publication of the paper, we have found that training larger capacity models for longer with the temporal classification objective significantly improves the evaluation results. Hence, we provide below pre-trained `resnext50_32x4d` type models that are currently our best models. We encourage people to use these new models instead of the `mobilenet_v2` type models reported in the paper (the pre-trained `mobilenet_v2` models reported in the paper are also provided below for the record). Four pre-trained `resnext50_32x4d` models are provided here: temporal classification models trained on data from the individual children in the SAYCam dataset (`TC-S-resnext`, `TC-A-resnext`, `TC-Y-resnext`) and a temporal classification model trained on data from all three children (`TC-SAY-resnext`). These models were all trained on the SAYCam dataset for 11 epochs (with batch size 256) with the following data augmentation pipeline:

```python
import torchvision.transforms as tr

tr.Compose([
        tr.RandomApply([tr.ColorJitter(0.8, 0.8, 0.8, 0.4)], p=0.9),
        tr.RandomGrayscale(p=0.2),
        tr.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        tr.RandomHorizontalFlip(),
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Here are some evaluation results for these `resnext50_32x4d` models (to download the models, click on the links over the model names):

| Model | Toybox (*iid*) | Toybox (*exemplar*) | ImageNet (*linear*) | ImageNet (*1% ft + linear*) | 
| ----- |:--------------:|:-------------------:|:-------------------:|:---------------------------:|
| [`TC-SAY-resnext`](https://drive.google.com/file/d/107pX69UW2iigRHHNu1iYnuwC4-dRvIM0/view?usp=sharing)  | **88.1** | **53.7** | **32.6** | **42.1** |
| [`TC-S-resnext`](https://drive.google.com/file/d/1OXVgeskTtKqSiVCFwyIfJWWZ_B1X5-1a/view?usp=sharing)    | 85.2 | 49.2 | 29.2 | -- |
| [`TC-A-resnext`](https://drive.google.com/file/d/1Jn-u_MYxCnfKskZvTNc_SDoFEa7E8xe4/view?usp=sharing)    | 85.0 | 48.1 | 26.5 | -- |
| [`TC-Y-resnext`](https://drive.google.com/file/d/1jE55bbKpzUyuyDgr2aiFiSE1SMl6YYAg/view?usp=sharing)    | 82.9 | 49.6 | 24.8 | -- |

Here, **ImageNet (*linear*)** refers to the top-1 validation accuracy on ImageNet with only a linear classifier trained on top of the frozen features, and **ImageNet (*1% ft + linear*)** is similar but with the entire model first fine-tuned on 1% of the ImageNet training data (~12800 images). Note that these are results from a single run, so you might observe slightly different numbers.

These models come with the classifier heads attached. To load these models, please do something along the lines of:

```python
import torch
import torchvision.models as models

model = models.resnext50_32x4d(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=n_out, bias=True)
model = torch.nn.DataParallel(model).cuda()

checkpoint = torch.load('TC-SAY-resnext.tar')
model.load_state_dict(checkpoint['model_state_dict'])
```

where `n_out` should be 6269 for `TC-SAY-resnext`, 2765 for `TC-S-resnext`, 1786 for `TC-A-resnext`, and 1718 for `TC-Y-resnext`. In addition, please find below the best performing ImageNet models reported above: a model with a linear classifier trained on top of the frozen features of `TC-SAY-resnext` (`TC-SAY-resnext-IN-linear`) and a model that was first fine-tuned with 1% of the ImageNet training data (`TC-SAY-resnext-IN-1pt-linear`):

* [`TC-SAY-resnext-IN-linear`](https://drive.google.com/file/d/1h6tV24CaBzYVgk0EmzRInYXB9PMVYZRo/view?usp=sharing)
* [`TC-SAY-resnext-IN-1pt-linear`](https://drive.google.com/file/d/1Ue0LY8b6-wIUGa_PVAtyaRQdHCIh5HGt/view?usp=sharing)

You can load these models in the same way as described above. Since these are ImageNet models, `n_out` should be set to 1000.

### MobileNet 
The following are the pre-trained `mobilenet_v2` type models reported in the paper:

* [TC-S-mobilenet](https://drive.google.com/file/d/1DVJjpaGhoBPNmlO7jXpwEX3lSCk2ZUCa/view?usp=sharing) (69.4 MB)
* [TC-A-mobilenet](https://drive.google.com/file/d/1uQvJBbuy6P0uCW0HYs1wNgawRU8sGLhC/view?usp=sharing) (54.4 MB)
* [TC-Y-mobilenet](https://drive.google.com/file/d/1TTndiiiqSiCMdZjwYZPKQySZot4ipCrG/view?usp=sharing) (53.3 MB)
* [TC-SAY-mobilenet](https://drive.google.com/file/d/1zeidpBaXqqWCeeYj-fMI7V7x9EiAGH6Q/view?usp=sharing) (123.3 MB)

## Acknowledgments

We are very grateful to the volunteers who contributed recordings to the SAYCam dataset. We thank Jessica Sullivan for her generous assistance with the dataset. We also thank the team behind the Toybox dataset, as well as the developers of PyTorch and torchvision for making this work possible. This project was partly funded by the NSF Award 1922658 NRT-HDR: FUTURE Foundations, Translation, and Responsibility for Data Science.
