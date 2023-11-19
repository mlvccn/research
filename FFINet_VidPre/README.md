# FFINet

This repository contains the implementation code for paper:

### FFINet: Fast Fourier Inception Networks for Occluded Video Prediction  ([paper](https://arxiv.org/pdf/2306.10346.pdf)) TMM'2023
## Introduction

<p align="center">
    <img src="./readme_figures/overall_framework.png" width="600"> <br>
</p>

Video prediction is a pixel-level task that generates future frames by employing the historical frames. There often exist continuous complex motions, such as object overlapping and scene occlusion in video, which poses great challenges to this task. Previous works either fail to well capture the long-term temporal dynamics or do not handle the occlusion masks. To address these issues, we develop the fully convolutional Fast Fourier Inception Networks for video prediction, termed FFINet, which includes two primary components, i.e., the occlusion inpainter and the spatiotemporal translator. The former adopts the fast Fourier convolutions to enlarge the receptive field, such that the missing areas (occlusion) with complex geometric structures are filled by the inpainter. The latter employs the stacked Fourier transform inception module to learn the temporal evolution by group convolutions and the spatial movement by channel-wise Fourier convolutions, which captures both the local and the global spatiotemporal features. This encourages generating more realistic and high-quality future frames. To optimize the model, the recovery loss is imposed to the objective, i.e., minimizing the mean square error between the ground-truth frame and the recovery frame.

## Dependencies

* torch=1.9.0
* scikit-image=0.19.3
* numpy=1.21.5
* argparse
* tqdm=4.64.1
* addict=2.4.0
* fvcore=0.1.5
* hickle=5.0.2
* opencv-python=4.6.0
* pandas=1.3.5
* pillow=9.2.0

## Overview

* `API/` contains dataloaders and metrics.
* `main.py` is the executable python file with possible arguments.
* `model.py` contains the FFINet model.
* `exp.py` is the core file for training, validating, and testing pipelines.
* `modules.py` contains the component  such as FFT Inception and FourierUnit.

## Prepare Dateset

```
  python mask_generator.py      #produce the mask for test
  cd ./data/moving_mnist        
  bash download_mmnist.sh       #download the mmnist dataset
```

## Start Training

```
  cd ./script
  sh mnist_train_nomask.sh      #train the model without mask
  sh mnist_tarin_mask.sh        #tarin the model with mask
```

## Quantitative results on Moving MNIST

|                 | MSE  | MAE  | SSIM  |
|:---------------:|:----:|:----:|:-----:|
| FFINet wo/ mask | 19.2 | 60.4 | 0.958 |
| FFINet w/ mask  | 21.7 | 65.8 | 0.952 |

## Qualitative results on Moving MNIST

without maks

<p align="center">
    <img src="./readme_figures/qualitative_nomask.png" width="600"> <br>
</p>

with mask

<p align="center">
    <img src="./readme_figures/qualitative_mask.png" width="600"> <br>
</p>


## Citation

If you find this repo useful, please cite the following papers.
```
@article{li-tmm2023-ffinet,
  author    = {Ping Li and Chenhan Zhang and Xianghua Xu},
  title     = {Fast Fourier Inception Networks for Occluded Video Predcition},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2023},
  doi       = {10.1109/TMM.2023.3310330},
}
```
## Contact
If you have any questions, feel free to contact us through email (201050044@hdu.edu.cn)

## Acknowledgements
We would like to thank to the authors of [SimVP](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9879439) for making their [source code](https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction) public, which significantly accelerated the development of FFINet.
