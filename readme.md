# MaskRCNN-Pytorch

Welcome to MaskRCNN-Pytorch repository! This repository provides a comprehensive C++ development workflow designed to help you train models, convert ONNX format, and C++ deploy. The workflow is primarily demonstrated using MaskRCNN as an example. The model training and ONNX model conversion are implemented using the PyTorch deep learning framework, while the C++ model deployment is accomplished using the ONNX Runtime.

## Features

- Provides sample code and training data for MaskRCNN model training.
- Uses the PyTorch framework for model training and weight saving.
- Supports converting trained PyTorch models to ONNX format.
- Implements C++ model deployment using the ONNX Runtime.
- Enables model deployment on different platforms (x86, arm64) and operating systems (Linux, Windows).
- Offers CPU and GPU acceleration options to cater to different hardware requirements.

## Usage Guide

This section describes how to use the project, step by step.

1. Clone the repository to your local machine
2. Install the dependencies. We recommend using conda to create a new environment with the dependencies installed. For example, create a new conda environment with Python 3.8 installed:

```
conda create -n maskrcnn python=3.8
conda activate maskrcnn
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
Note: Please install Pytorch according to the configuration of your computer. Some other import packages can be installed according to requirements. Alternative, you can also install them directly via pip:

```
pip install -r requirements.txt
```

3. Train the MaskRCNN Pytorch models by running the command. The path to the dataset, and training model parameters configuration, etc. can be modified in the python script `train.py`.

```
python train.py
```

4. Converting the trained models to ONNX format (run script `onnx_model_export.py`) and running them in onnxruntime. Details on deploying ONNX models in C++ can be found in `readme.md` of `deploy/C++deploy`

Please note that the repository provides a basic development workflow example. You can extend and customize it to suit your requirements.

## Project Structure
The project structure is:

```bash
root/
│
├── backbone/
│   ├── feature_pyramid_network.py
│   └── resnet50_fpn_model.py
│
├── coco_dataset/
│   ├── annotations/
│   └── images/
│ 
├── deploy/
│   └── onnx_cpp/
│
├── images/
│
├── network_file/
│   ├── faster_rcnn_framework.py
│   ├── maskrcnn.py
│   └── <other files>
│
├── record/
│   ├── det_results.json
│   └── <other files>
│
├── train.py
│
├── predict.py
│
├── validation.py
│
└── onnx_model_export.py
```
The `coco_dataset` directory contains the sample dataset in coco format used for training the MaskRCNN models. The `network_files` directory contains the Pytorch implementation of the MaskRCNN model. The `images` directory is used for testing trained model. The `deploy/onnx_cpp` directory is a submodule for C++ deploying the ONNX models using onnxruntime.

## Contribution

We highly appreciate contributions from the community to improve MaskRCNN-Pytorch repository. You can contribute by submitting issues, feedback, bug fixes, or feature enhancements. Please refer to the repository for detailed contribution guidelines.

We hope you find this repository helpful and encourage you to explore and contribute to its development.