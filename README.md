# GBP-Net

This is the code of TMI 2025 Paper “Glancing Beyond Patch: Spatial Contextual Cues for 3D Neuron Segmentation”.


In addition, we have also uploaded a sample image from our ZBFWB dataset.
For more information about the zbfwb dataset, please refer to X. Du, Z. Yue, J. Wei, W. Li, M. Chen, T. Chen, H. Hu, H. Ren, Z. Jia, X. Ning et al., “Central nervous system atlas of larval zebrafish constructed using the morphology of single excitatory and inhibitory neurons,” bioRxiv, pp. 2025–06, 2025.

# GBPNet


## Overview

Accurate segmentation of neurons in 3D fluorescence microscopy images is essential for advancing neuroscience. Prevalent methods split a volume into patches and process each patch separately due to computational resource limitations. However, they fail to capture global neuronal morphology across multiple patches, which results in discontinuous segmentation and poses a challenge for subsequent neuronal reconstruction. 
In this paper, we propose a dual U-Net architecture termed ``Glancing Beyond Patch'' Network (GBP-Net) to incorporate contextual information into segmentation.

## Screenshots

![Screenshot 1](aa.png)
*Figure 1: Description of aa.png*

![Screenshot 2](bb.png)
*Figure 2: Description of bb.png*

## Installation

Our code framework is based on PyTorch connectomics. https://github.com/zudi-lin/pytorch_connectomics


### Prerequisites
- Conda (Miniconda or Anaconda)

### Setup Environment

