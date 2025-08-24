# GBP-Net
This is the code of TMI 2025 Paper “Glancing Beyond Patch: Spatial Contextual Cues for 3D Neuron Segmentation”.


## Overview

Accurate segmentation of neurons in 3D fluorescence microscopy images is essential for advancing neuroscience. Prevalent methods split a volume into patches and process each patch separately due to computational resource limitations. However, they fail to capture global neuronal morphology across multiple patches, which results in discontinuous segmentation and poses a challenge for subsequent neuronal reconstruction. 
In this paper, we propose a dual U-Net architecture termed ``Glancing Beyond Patch'' Network (GBP-Net) to incorporate contextual information into segmentation.

## Screenshots

![GBPNet](pic.png)
*The architecture of ``Glancing Beyond Patch'' Network*

## Installation

Our code framework is based on PyTorch connectomics. https://github.com/zudi-lin/pytorch_connectomics



## Datasett

We have also uploaded a sample image from our ZBFWB dataset.
For more information about the zbfwb dataset, please refer to X. Du, Z. Yue, J. Wei, W. Li, M. Chen, T. Chen, H. Hu, H. Ren, Z. Jia, X. Ning et al., “Central nervous system atlas of larval zebrafish constructed using the morphology of single excitatory and inhibitory neurons,” bioRxiv, pp. 2025–06, 2025.
