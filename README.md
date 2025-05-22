# SSN\_Reach: Scalable and Data-Efficient Verification for Semantic Segmentation Neural Networks

## Overview

**SSN\_Reach** is a verification framework designed to scale semantic segmentation network verification efficiently. It balances computational performance and memory usage to enable evaluation across a wide range of hardware configurations.

## Instructions for Running Benchmarks

Before running the benchmarks, please ensure the following steps are completed:

1. **Install NNV**
   Download and install the [NNV (Neural Network Verification)](https://github.com/verivital/nnv) toolbox from its official repository.

2. **Add NNV to MATLAB Path**
   Make sure to add NNV to your MATLAB path before running any experiments.

3. **Download ONNX Models**
   The ONNX models used in our experiments are not included in this repository. However, download links are provided where applicable.

4. **Hardware and Memory Considerations**
   Our technique is designed to be memory-efficient. The experiments in the paper were conducted on a Vortex workstation with 48 GB of GPU memory. If you are using a GPU with less memory, you can reduce memory usage by adjusting the relevant hyperparameters in the code—specifically, by increasing the number of data injection sequences and reducing the amount of data injected per sequence. This allows the data to be processed sequentially rather than all at once. Although this may increase runtime, it enables successful execution on devices with more limited GPU resources.