# Self-supervised learning through the eyes of a child

This repository contains code for reproducing the results reported in the following paper:

Orhan AE, Gupta VV, Lake BM (2020) [Self-supervised learning through the eyes of a child.](https://arxiv.org/abs/2007.xxxxx) arXiv:2007.xxxxx.

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

## Pre-trained models

We share below the pre-trained weights for our best self-supervised models trained on the SAYCam dataset. Four pre-trained models are provided below: temporal classification models trained on data from the individual children in the SAYCam dataset (TC-S, TC-A, TC-Y) and a temporal classification model trained on data from all three children (TC-SAY).

* [TC-S](https://drive.google.com/file/d/1DVJjpaGhoBPNmlO7jXpwEX3lSCk2ZUCa/view?usp=sharing) (69.4 MB)
* [TC-A](https://drive.google.com/file/d/1uQvJBbuy6P0uCW0HYs1wNgawRU8sGLhC/view?usp=sharing) (54.4 MB)
* [TC-Y](https://drive.google.com/file/d/1TTndiiiqSiCMdZjwYZPKQySZot4ipCrG/view?usp=sharing) (53.3 MB)
* [TC-SAY](https://drive.google.com/file/d/1zeidpBaXqqWCeeYj-fMI7V7x9EiAGH6Q/view?usp=sharing) (123.3 MB)

## Acknowledgments

We are very grateful to the volunteers who contributed recordings to the SAYCam dataset. We thank Jessica Sullivan for her generous assistance with the dataset. We also thank the team behind the Toybox dataset, as well as the developers of Pytorch and torchvision for making this work possible. This project was partly funded by the NSF Award 1922658 NRT-HDR: FUTURE Foundations, Translation, and Responsibility for Data Science.
