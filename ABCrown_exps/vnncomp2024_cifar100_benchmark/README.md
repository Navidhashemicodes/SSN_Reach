## CIFAR-100 Neural Network Verification Benchmark for VNN-COMP 2024

This is the re-proposed CIFAR-100 benchmark for VNN-COMP 2024. The original benchmark [cifar100_tinyimagenet](https://github.com/Lucas110550/CIFAR100_TinyImageNet_ResNet) was from VNN-COMP 2022.

This benchmark has reduced complexity to improve its accessibility to new tools and new VNN-COMP participants.
The original benchmark had 4 models for CIFAR-100, and this benchmark only contains only 2 of them, which hopefully will reduce the effort to support this benchmark.
The largest model (`CIFAR100_resnet_super.onnx`) was removed. The TinyImageNet model was moved to a separate benchmark.

CIFAR100 images are 32x32x3 (same as CIFAR10), and the models contain FC, conv, and ReLU layers only.

For more information about the models, you can check out the original repo in 2022 (here, we only kept the `CIFAR100_resnet_medium` and `CIFAR100_resnet_large` models; other models were removed to reduce complexity).

### Setup 

PyTorch is required, which can be installed by `pip install torch`.

To generate specficiations:
```bash
python generate_properties.py SEED
```
