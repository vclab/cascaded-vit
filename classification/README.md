# Cascaded-ViT Classification

This document outlines the classification experiments undertaken with ImageNet-1K for Cascaded-ViT. 

<!-- | Model Name         | Top-1 (%) | Params (M) | FLOPs (M) | Throughput<br>(GPU/M4 Pro/RyzenAI) | Energy on M4 Pro  (J/Img) | Weights|
|--------------------|-----------|------------|-----------|------------------|---------------------------|---------------------------------|
| CascadedViT-S      | 62.0      | 1.9        | 67        |  25740/5775/1453 |   471                     | [CascadedViT_S](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_S.pth)       |
| CascadedViT-M      | 69.9      | 3.5        | 173       |  20464/           |   568                     | [CascadedViT_M](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_M.pth)       |
| CascadedViT-L      | 73.0      | 7.0        | 249       |  17335/           |   588                     | [CascadedViT_L](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_L.pth)       |
| CascadedViT-XL     | 75.5      | 9.8        | 366       |  11934/           |   653                     | [CascadedViT_XL](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_XL.pth)       | -->

| Model Name     | Top-1 (%) | Params (M) | FLOPs (M) | Throughput (img/s)<br>GPU / M4 Pro / RyzenAI | Energy (J/Img)<br>on M4 Pro | Weights |
|----------------|-----------|------------|-----------|----------------------------------------------|------------------------------|---------|
| CascadedViT-S  | 62.0      | 1.9        | 67        | 25740 / 5775 / 1453                          | 471                          | [CascadedViT_S](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_S.pth) |
| CascadedViT-M  | 69.9      | 3.5        | 173       | 20464 / 3717 / 867                           | 568                          | [CascadedViT_M](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_M.pth) |
| CascadedViT-L  | 73.0      | 7.0        | 249       | 17335 / 2978 / 667                           | 588                          | [CascadedViT_L](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_L.pth) |
| CascadedViT-XL | 75.5      | 9.8        | 366       | 11934 / 1910 / 423                           | 653                          | [CascadedViT_XL](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_XL.pth) |

### Requirements for Training and Evaluation

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

### Requirements for Inference and Latency Test on Different Hardware


1.  **RyzenAI:** follow these [setup instructions](https://ryzenai.docs.amd.com/en/latest/inst.html)

2. **Apple Silicon (M4 Pro):** ```pip install torch==2.4.1```

