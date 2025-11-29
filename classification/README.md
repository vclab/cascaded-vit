# Cascaded-ViT Classification

This document outlines the results and steps to run classification experiments undertaken with ImageNet-1K for Cascaded-ViT. 

<!-- | Model Name         | Top-1 (%) | Params (M) | FLOPs (M) | Throughput<br>(GPU/M4 Pro/RyzenAI) | Energy on M4 Pro  (mJ/Img) | Weights|
|--------------------|-----------|------------|-----------|------------------|---------------------------|---------------------------------|
| CascadedViT-S      | 62.0      | 1.9        | 67        |  25740/5775/1453 |   471                     | [CascadedViT_S](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_S.pth)       |
| CascadedViT-M      | 69.9      | 3.5        | 173       |  20464/           |   568                     | [CascadedViT_M](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_M.pth)       |
| CascadedViT-L      | 73.0      | 7.0        | 249       |  17335/           |   588                     | [CascadedViT_L](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_L.pth)       |
| CascadedViT-XL     | 75.5      | 9.8        | 366       |  11934/           |   653                     | [CascadedViT_XL](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_XL.pth)       | -->

| Model Name     | Top-1 (%) | Params (M) | FLOPs (M) | Throughput (img/s)<br>GPU / M4 Pro / RyzenAI | Energy (mJ/Img)<br>on M4 Pro | Weights |
|----------------|-----------|------------|-----------|----------------------------------------------|------------------------------|---------|
| CascadedViT-S  | 62.0      | 1.9        | 67        | 25740 / 5775 / 1453                          | 471                          | [pth](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_s.pth)/[CoreML](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_s.mlpackage.zip) |
| CascadedViT-M  | 69.9      | 3.5        | 173       | 20464 / 3717 / 867                           | 568                          | [pth](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_m.pth)/[CoreML](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_m.mlpackage.zip) |
| CascadedViT-L  | 73.0      | 7.0        | 249       | 17335 / 2978 / 667                           | 588                          | [pth](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_l.pth)/[CoreML](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_l.mlpackage.zip) |
| CascadedViT-XL | 75.5      | 9.8        | 366       | 11934 / 1910 / 423                           | 653                          | [pth](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_xl.pth)/[CoreML](https://github.com/vclab/cascaded-vit/releases/download/v1.0/cascadedvit_xl.mlpackage.zip) |

## Requirements 

###  Training and Evaluation

1. ImageNet-1K

    Download the ImageNet dataset for training and evaluation from the [official page](https://www.image-net.org) or [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge).

    Then run the following script from within the val folder to create the directory structure for val
    ```
    cd path/to/imagenet/val
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash -s .
    ```

    This will result in the following directory structure:

    ```
    ImageNet/
    |            
    +--CLS-LOC/
       |    
       +--train/
       |  |    
       |  +--n01440764/
       |     |    
       |     +--n01440764_10026.JPEG
       |     +--n01440764_10026.JPEG
       |     .
       |     .
       |  .
       |  .
       |
       +--val/
       |  |    
       |  +--n01440764/
       |     |
       |     +--ILSVRC2012_val_00000293.JPEG
       |     +--ILSVRC2012_val_00002138.JPEG
       |     .
       |     .
       |  .
       |  .
    ```

2. Install dependencies

    ```
    pip install requirements.txt
    ```

### Inference and Latency Test on Different Hardware


1.  **RyzenAI:** follow these [setup instructions](https://ryzenai.docs.amd.com/en/latest/inst.html)

2. **Apple Silicon (M4 Pro):** ```pip install torch==2.4.1```

## Training

Use the below commands to train CascadedViT on ImageNet-1K classification using a single GPU:

<details open>
<summary>CascadedViT_S</summary>

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model CascadedViT_S --data-path path/to/ImageNet --output_dir /path/to/save/checkpoints --epochs 300 --lr 9e-4 --weight-decay 0.0125 --clip-grad 0.015  --clip-mode 'norm' --min-lr 1e-4 --mixup 0.6 --cutmix 0.8 
```
</details>

<details close>
<summary>CascadedViT_M</summary>

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model CascadedViT_M --data-path path/to/ImageNet --output_dir /path/to/save/checkpoints --epochs 300 --lr 9e-4 --weight-decay 0.0125 --clip-grad 0.015  --clip-mode 'norm' --min-lr 1e-4 --mixup 0.6 --cutmix 0.8 
```
</details>

<details close>
<summary>CascadedViT_L</summary>

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model CascadedViT_L --data-path path/to/ImageNet --output_dir /path/to/save/checkpoints --epochs 300 --lr 9e-4 --weight-decay 0.0125 --clip-grad 0.015  --clip-mode 'norm' --min-lr 1e-4 --mixup 0.6 --cutmix 0.8 
```
</details>

<details close>
<summary>CascadedViT_XL</summary>

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model CascadedViT_XL --data-path path/to/ImageNet --output_dir path/to/save/checkpoints --epochs 300 --lr 9e-4 --weight-decay 0.0125 --clip-grad 0.015  --clip-mode 'norm' --min-lr 1e-4 --mixup 0.6 --cutmix 0.8 
```
</details>

## Knowledge Distillation

For the following student models:
1. CascadedViT_S
2. CascadedViT_M

Download the weights for CascadedViT_L teacher from the model zoo.
To train CascadedViT_L student, download the weights for CascadedViT_XL teacher from the model zoo.

<details close>
<summary>CascadedViT_S</summary>

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model CascadedViT_S --batch-size 3072 --num_workers 48 --data-path path/to/ImageNet --output_dir path/to/save/checkpoints --epochs 300  --lr 9e-4 --weight-decay 0.0125 --clip-grad 0.025  --clip-mode 'norm' --min-lr 9e-5 --distillation-type soft --teacher-model 'CascadedViT_L' --teacher-path path/to/cascadedvit_l.pth --mixup 0.1 --cutmix 0.1 --distillation-alpha 0.5 --distillation-tau 2.0
```
</details>

<details close>
<summary>CascadedViT_M</summary>

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model CascadedViT_M --batch-size 3072 --num_workers 48 --data-path path/to/ImageNet --output_dir path/to/save/checkpoints --epochs 300  --lr 9e-4 --weight-decay 0.0125 --clip-grad 0.025  --clip-mode 'norm' --min-lr 9e-5 --distillation-type soft --teacher-model 'CascadedViT_L' --teacher-path path/to/cascadedvit_l.pth --mixup 0.1 --cutmix 0.1 --distillation-alpha 0.5 --distillation-tau 2.0
```
</details> 

<details close>
<summary>CascadedViT_L</summary>

```
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 --use_env main.py --model CascadedViT_L --batch-size 3072 --num_workers 48 --data-path path/to/ImageNet --output_dir path/to/save/checkpoints --epochs 300  --lr 9e-4 --weight-decay 0.0125 --clip-grad 0.025  --clip-mode 'norm' --min-lr 9e-5 --distillation-type soft --teacher-model 'CascadedViT_XL' --teacher-path path/to/cascadedvit_xl.pth --mixup 0.1 --cutmix 0.1 --distillation-alpha 0.5 --distillation-tau 2.0
```
</details>

## Evaluation
Use the below command to evaluate a pre-trained CascadedViT_L on ImageNet-1K with a single GPU:

```
python main.py --eval --model CascadedViT_L --resume path/to/cascadedvit_l.pth --data-path path/to/ImageNet
```

## Credits
We express our gratitude to [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT) for the backbone. We further thank [Swin Transformer](https://github.com/microsoft/swin-transformer), [LeViT](https://github.com/facebookresearch/LeViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [PyTorch](https://github.com/pytorch/pytorch) for their superb codebases.
