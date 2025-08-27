# Cascaded-ViT Downstream Tasks

This document outlines the results and steps to run downstream detection and segmentation experiments undertaken with MS-COCO for Cascaded-ViT.

## Object Detection
|Model | Pretrain | Lr Schd | Box AP | AP@50 | AP@75 | Model/Log | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|CascadedViT-L | ImageNet-1k | 2x | 31.4 | 50.7 | 32.6 | [cascadedvit_l_det_2x.pth](https://github.com/vclab/cascaded-vit/releases/download/v1.0-downstream/cascadedvit_l_det_2x.pth)/[log](https://github.com/vclab/cascaded-vit/releases/download/v1.0-downstream/20250711_022415.log.json) |

## Instance Segmentation

|Model | Pretrain | Lr Schd | Mask AP | AP@50 | AP@75 | Model/Log | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|CascadedViT-L | ImageNet-1k | 1x | 29.1 | 50.2 | 29.7 | [cascadedvit_l_segm.pth](https://github.com/vclab/cascaded-vit/releases/download/v1.0-downstream/cascadedvit_l_segm.pth)/[log](https://github.com/vclab/cascaded-vit/releases/download/v1.0-downstream/segmentation.log.json) |

## Requirements

1. MS-COCO

    Download the MS-COCO 2017 dataset either [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets) or use the below commands:

    1. Create data/coco directory to move there
        ```
        cd downstream/data/coco
        ```
    2. Dowload the dataset zip files
        ```
        wget http://images.cocodataset.org/zips/train2017.zip
        wget http://images.cocodataset.org/zips/val2017.zip
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        ```
    3. Unzip the files
        ```
        unzip train2017.zip
        unzip val2017.zip
        unzip annotations_trainval2017.zip
        ```

2. Dependencies

    We suggest `python3.10` with `torch==1.11.0+cu113` for the downstream experiments

    Use the below commands to install openmim, mmcv and mmdet:

    ```
    pip install -U openmim
    pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
    mim install mmdet==2.28.2
    ```

## Evaluation
Download the relevant model weights and run the below commands to check

<details open>
<summary>Object Detection</summary>

```
bash ./dist_test.sh configs/retinanet_cascadedvit_l_fpn_1x_coco.py cascadedvit_l_det_2x.pth 1 --eval bbox
```
Where 1 refers to the number of GPUs
</details>

<details close>
<summary>Instance Segmentation</summary>

```
bash ./dist_test.sh configs/mask_rcnn_cascadedvit_l_fpn_1x_coco.py cascadedvit_l_segm.pth 1 --eval bbox segm
```
Where 1 refers to the number of GPUs
</details>

## Training
Download the ImageNet-1K pretrained CascadedViT-L [weights](https://github.com/vclab/cascaded-vit/releases/download/v1.0/CascadedViT_L.pth) and follow the below commands to train the model.

<details open>
<summary>Object Detection</summary>

```
python train.py configs/retinanet_cascadedvit_l_fpn_1x_coco.py --gpu-id 0 --cfg-options model.backbone.pretrained=/path/to/pretrained/model.pth
```
</details>

<details close>
<summary>Instance Segmentation</summary>

```
python train.py configs/mask_rcnn_cascadedvit_l_fpn_1x_coco.py --gpu-id 0 --cfg-options model.backbone.pretrained=/path/to/pretrained/model.pth
```

</details>
<br>

For a full list of arguments refer to [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#training-on-multiple-gpus)

## Credits
We express our gratitude to [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT), [MMDetection](https://github.com/open-mmlab/mmdetection), [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and [PoolFormer](https://github.com/sail-sg/poolformer/tree/main/detection) for their superb codebases.