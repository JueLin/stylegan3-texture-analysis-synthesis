# Texture Representation via Analysis and Synthesis with Generative Adversarial Networks

The code is largely adapted from https://github.com/NVlabs/stylegan3

## Requirements

Please refer to the Requirements section of https://github.com/NVlabs/stylegan3 for creating the conda environment

## Dataset 

To be announced

## Pretrained Model

Pretrained models, including both generator and encoder (as describled in the paper) can be downloaded from [here](https://drive.google.com/file/d/1zKNcfjXvf5Shr5QkFXXs6ZDTxnv5qDcm/view?usp=share_link)

## Train a Synthesis Model

> python train.py --outdir ~/training-runs --cfg stylegan3-r --data PATH_TO_DATASET --gpus 1 --batch 32 --gamma 2 --batch-gpu 16 --snap 10  --mirror 1 --metrics none

## Generate Texture Crops

> python synthesize_homogeneous_crops.py --g_path PATH_TO_GENERATOR --e_path PATH_TO_ENCODER --n_textures 10 --samples_per_texture 5 --sigma 0.1 --outdir ~/synthesizeCrops 

## Train a Encoder

> python train_encoder.py --g_path PATH_TO_GENERATOR --outdir ~/trainEncoder

## Test a Encoder

> python test_encoder.py --data PATH_TO_DATASET --g_path PATH_TO_GENERATOR --e_path PATH_TO_ENCODER --outdir ~/testEncoder

## Iterative Refinement via Optimization-based GAN Inversion

> python train_latent.py --data PATH_TO_DATASET --g_path PATH_TO_GENERATOR --e_path PATH_TO_ENCODER --outdir ~/trainLatent

## Reference

Please cite the following paper if you use the provided data and/or code:

~~~bibtex
@article{textureRep,
author = {Jue Lin and Zhiwei Xu and Gaurav Sharma and Thrasyvoulos N. Pappas},
title = {Texture Representation via Analysis and Synthesis with Generative Adversarial Networks},
journal = {e-Prime - Advances in Electrical Engineering, Electronics and Energy},
note = {accepted for publication, to appear}
}
~~~

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan3

## Disclaimer: 

The code is provided "as is" with ABSOLUTELY NO WARRANTY expressed or implied. Use at your own risk.
