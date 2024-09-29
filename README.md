# OOTDiffusion-Training
This repository is an unofficial implementation of OOTDiffusion training part.

> **OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on** [[arXiv paper](https://arxiv.org/abs/2403.01779)]<br>
> [Yuhao Xu](http://levihsu.github.io/), [Tao Gu](https://github.com/T-Gu), [Weifeng Chen](https://github.com/ShineChen1024), [Chengcai Chen](https://www.researchgate.net/profile/Chengcai-Chen)<br>
> Xiao-i Research


The code supports the training on [VITON-HD](https://github.com/shadow2496/VITON-HD) (half-body) and [Dress Code](https://github.com/aimagelab/dress-code) (full-body).

<img width="786" alt="image" src="https://github.com/user-attachments/assets/578cdc18-57c1-4d05-abe1-82e20bbe7566">

## Installation
1. Clone the repository

```sh
git clone https://github.com/coolistener/OOTDiffusion-Training
```

2. Create a conda environment and install the required packages

```sh
conda create -n ootd python==3.10
conda activate ootd
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
```

## Training

1. Prepare the dataset as the following structure：

   1.1 VITON-HD

   ```
   OOTDiffusion/VITON-HD
   ├── train/test
   │   ├── agnostic-mask
   │   ├── agnostic-v3.2
   │   ├── cloth
   │   ├── cloth-mask
   │   ├── cloth-warp
   │   ├── cloth-warp-mask
   │   ├── cloth_caption
   │   ├── gt_cloth_warped_mask
   │   ├── image
   │   ├── image-densepose
   │   ├── image-parse-agnostic-v3.2
   │   ├── image-parse-v3
   │   ├── openpose_img
   │   └── openpose_json
   ├── test_pairs.txt
   └── train_pairs.txt
   ```

   1.2 DressCode

   ```
   OOTDiffusion/DressCode
   ├── dresses
   ├── lower_body
   ├── multigarment_test_triplets.txt
   ├── readme.txt
   ├── test_pairs_paired.txt
   ├── test_pairs_unpaired.txt
   ├── train_pairs_paired.txt
   └── upper_body
   ```

2. Prepare models

   - [clip-vit-large-patch14]([openai/clip-vit-large-patch14 · Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14))
   - [stable-diffusion-v1-5]([runwayml/stable-diffusion-v1-5 · Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5))

   It's suggested that put the models as the following structure：

   ```
   OOTDiffusion/models
   ├── clip-vit-large-patch14
   ├── humanparsing
   ├── openpose
   └── stable-diffusion-v1-5
   ```

   You can also put the models in way you like, **but don't forget change the models paths in args**.

3. Run the training

```sh
cd OOTDiffusion/run
bash train.sh
```

## Inference

Replace the model path with your model path, especially the unet path.

Take the Half-body model as the example:

```python
#!ootd/inference_ootd_hd.py
VIT_PATH = "../models/clip-vit-large-patch14"
VAE_PATH = "../models/stable-diffusion-v1-5"
UNET_PATH = "./train/checkpoints_hd/epoch_50"
MODEL_PATH = "../models/stable-diffusion-v1-5"
```

1. Half-body model

```sh
python run_ootd.py --model_path <model-image-path> --cloth_path <cloth-image-path> --scale 2.0 --sample 4
```

2. Full-body model 

> Garment category must be paired: 0 = upperbody; 1 = lowerbody; 2 = dress

```sh
python run_ootd.py --model_path <model-image-path> --cloth_path <cloth-image-path> --model_type dc --category 2 --scale 2.0 --sample 4
```

## Citation
```
@article{xu2024ootdiffusion,
  title={OOTDiffusion: Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on},
  author={Xu, Yuhao and Gu, Tao and Chen, Weifeng and Chen, Chengcai},
  journal={arXiv preprint arXiv:2403.01779},
  year={2024}
}
```
