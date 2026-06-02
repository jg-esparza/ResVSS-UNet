# ResVSS-UNet

This is the official code repository for "ResVSS-UNet: A ResNet-Vision State Space Decoder for Efficient Medical Image Segmentation". 

## Abstract

Automated medical image segmentation remains challenging due to the need to capture both fine-grained local details and long-range spatial context. 
Conventional CNN-based U-Net variants are effective at modeling local structures but often struggle to represent global dependencies, while Transformer-based approaches can improve global modeling at the cost of quadratic attention complexity and increased computation. 
Vision State Space Models (VSSMs) provide an alternative that enables long-range dependency modeling with linear complexity. 
In this work, we propose ResVSS-UNet, a hybrid segmentation framework that couples a ResNet encoder with a VSSM-based decoder to strengthen contextual representation during reconstruction. 
We evaluate the proposed model on three public benchmarks spanning different modalities and anatomical targets: dermoscopy (ISIC17), gastrointestinal endoscopy (Kvasir-SEG), and breast ultrasound (BUSI). 
Experimental results demonstrate that ResVSS-UNet achieves competitive performance across datasets and attains the best results on Kvasir-SEG, while maintaining favorable computational efficiency compared with attention-heavy segmentation backbones. These findings suggest that VSSM-based decoding is a promising direction for building accurate and efficient medical image segmentation models.

## Installation

### Requirements

- Ubuntu-24.04
- NVIDIA GPU
- CUDA 12.0

### Install environment

```
conda env create -f environment.yml
conda activate resvss_unet
```

## Model Summary

Display model summary from `torchinfo`.
```
python main.py
```

## Prepare the pre_trained weights
The weights used by ResVSS-UNet could be downloaded from [VM-UNet](https://github.com/JCruan519/VM-UNet). 
- The pre-trained weights should be stored in `./pretrained_weights/`.
- Set `use_pretrained_ckpt: True` in `config.yaml` file. 

## Acknowledgments

Special thanks to the authors for providing their research and public resources.

- [VMamba](https://github.com/MzeroMiko/VMamba).
- [VM-UNet](https://github.com/JCruan519/VM-UNet).