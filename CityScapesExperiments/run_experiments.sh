#!/bin/bash

echo "Starting experiments!"

# frankfurt_000000_000294
python verify.py --config configs/segformer/frankfurt_000000_000294_leftImg8bit_Np=34_config.yaml > frankfurt_000000_000294_segformer_Np=34.txt
echo "Finished with segformer/frankfurt_000000_000294_leftImg8bit_Np=34"
python verify.py --config configs/mask2former/frankfurt_000000_000294_leftImg8bit_Np=34_config.yaml > frankfurt_000000_000294_mask2former_Np=34.txt
echo "Finished with mask2former/frankfurt_000000_000294_leftImg8bit_Np=34"

python verify.py --config configs/segformer/frankfurt_000000_000294_leftImg8bit_Np=68_config.yaml > frankfurt_000000_000294_segformer_Np=68.txt
echo "Finished with segformer/frankfurt_000000_000294_leftImg8bit_Np=68"
python verify.py --config configs/mask2former/frankfurt_000000_000294_leftImg8bit_Np=68_config.yaml > frankfurt_000000_000294_mask2former_Np=68.txt
echo "Finished with mask2former/frankfurt_000000_000294_leftImg8bit_Np=68"

python verify.py --config configs/segformer/frankfurt_000000_000294_leftImg8bit_Np=102_config.yaml > frankfurt_000000_000294_segformer_Np=102.txt
echo "Finished with segformer/frankfurt_000000_000294_leftImg8bit_Np=102"
python verify.py --config configs/mask2former/frankfurt_000000_000294_leftImg8bit_Np=102_config.yaml > frankfurt_000000_000294_mask2former_Np=102.txt
echo "Finished with mask2former/frankfurt_000000_000294_leftImg8bit_Np=102"


# frankfurt_000000_001751
python verify.py --config configs/segformer/frankfurt_000000_001751_leftImg8bit_Np=34_config.yaml > frankfurt_000000_001751_segformer_Np=34.txt
echo "Finished with segformer/frankfurt_000000_001751_leftImg8bit_Np=34"
python verify.py --config configs/mask2former/frankfurt_000000_001751_leftImg8bit_Np=34_config.yaml > frankfurt_000000_001751_mask2former_Np=34.txt
echo "Finished with mask2former/frankfurt_000000_001751_leftImg8bit_Np=34"

python verify.py --config configs/segformer/frankfurt_000000_001751_leftImg8bit_Np=68_config.yaml > frankfurt_000000_001751_segformer_Np=68.txt
echo "Finished with segformer/frankfurt_000000_001751_leftImg8bit_Np=68"
python verify.py --config configs/mask2former/frankfurt_000000_001751_leftImg8bit_Np=68_config.yaml > frankfurt_000000_001751_mask2former_Np=68.txt
echo "Finished with mask2former/frankfurt_000000_001751_leftImg8bit_Np=68"

python verify.py --config configs/segformer/frankfurt_000000_001751_leftImg8bit_Np=102_config.yaml > frankfurt_000000_001751_segformer_Np=102.txt
echo "Finished with segformer/frankfurt_000000_001751_leftImg8bit_Np=102"
python verify.py --config configs/mask2former/frankfurt_000000_001751_leftImg8bit_Np=102_config.yaml > frankfurt_000000_001751_mask2former_Np=102.txt
echo "Finished with mask2former/frankfurt_000000_001751_leftImg8bit_Np=102"


# frankfurt_000000_006589
python verify.py --config configs/segformer/frankfurt_000000_006589_leftImg8bit_Np=34_config.yaml > frankfurt_000000_006589_segformer_Np=34.txt
echo "Finished with segformer/frankfurt_000000_006589_leftImg8bit_Np=34"
python verify.py --config configs/mask2former/frankfurt_000000_006589_leftImg8bit_Np=34_config.yaml > frankfurt_000000_006589_mask2former_Np=34.txt
echo "Finished with mask2former/frankfurt_000000_006589_leftImg8bit_Np=34"

python verify.py --config configs/segformer/frankfurt_000000_006589_leftImg8bit_Np=68_config.yaml > frankfurt_000000_006589_segformer_Np=68.txt
echo "Finished with segformer/frankfurt_000000_006589_leftImg8bit_Np=68"
python verify.py --config configs/mask2former/frankfurt_000000_006589_leftImg8bit_Np=68_config.yaml > frankfurt_000000_006589_mask2former_Np=68.txt
echo "Finished with mask2former/frankfurt_000000_006589_leftImg8bit_Np=68"

python verify.py --config configs/segformer/frankfurt_000000_006589_leftImg8bit_Np=102_config.yaml > frankfurt_000000_006589_segformer_Np=102.txt
echo "Finished with segformer/frankfurt_000000_006589_leftImg8bit_Np=102"
python verify.py --config configs/mask2former/frankfurt_000000_006589_leftImg8bit_Np=102_config.yaml > frankfurt_000000_006589_mask2former_Np=102.txt
echo "Finished with mask2former/frankfurt_000000_006589_leftImg8bit_Np=102"
