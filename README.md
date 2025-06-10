# Interpretable Image Classification via Non-parametric Part Prototype Learning
[[`Paper`](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Interpretable_Image_Classification_via_Non-parametric_Part_Prototype_Learning_CVPR_2025_paper.html)] [[`BibTeX`](https://github.com/zijizhu/proto-non-param?tab=readme-ov-file#cite-our-work)]

This repository presents the official PyTorch implementation for the paper "Interpretable Image Classification via Non-parametric Part Prototype Learning".

## Environment Setup

Setting up the environment involves the following steps:
- Install required Python packages listed in `requirements.txt`,
- Install `dinov2` without its dependencies for better compatibility (mainly to exclude `xformers`). We use commit `e1277af` of `dinov2` repository. Additionally, please ensure the PyTorch version meets `torch<2.4.0` to avoid performance regression.

To install Python packages used in this repository:
```sh
pip install -r requirements.txt
```

To install `dinov2` without installing `xformers`:
```sh
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
git checkout e1277af2ba9496fbadf7aec6eba56e8d882d1e35
pip install --no-deps -e .
```

## Data Preparation

Please prepare the dataset as follows:
- Download CUB-200-2011 dataset from the [official website](https://www.vision.caltech.edu/datasets/cub_200_2011/).
- Perform offline data augmentation on the dataset following previous works, such as by modifying [this script](https://github.com/JackeyWang96/TesNet/blob/master/preprocess_data/img_aug.py).
- For evaluation, please also download the object [segmentation masks](https://data.caltech.edu/records/w9d68-gec53).
- Make sure all the data folders are placed under one dataset root directory, which looks like the following after extracting archives:

```
dataset-root
├── CUB_200_2011/
├── cub200_cropped/
└── segmentations/
```

## Training and Evaluation

To train the model on CUB-200-2011, run the following command after replacing the arguments with dataset root directory and logging directory:
```sh
python train.py --data-dir <dataset-root> --log-dir <log-directory>
```

To evaluate the trained model:
- First change line 19 of [`eval/utils.py`](https://github.com/zijizhu/proto-non-param/blob/main/eval/utils.py#L19) so it points to `<dataset-root>`.
- Run the following command after changing the argument to the previous logging directory that contains training artifacts (keep `ckpt.pth` at the end):
```sh
python evaluate.py --ckpt-path <log-directory>/ckpt.pth
```

## Cite Our Work
```
@InProceedings{Zhu_2025_CVPR,
    author    = {Zhu, Zhijie and Fan, Lei and Pagnucco, Maurice and Song, Yang},
    title     = {Interpretable Image Classification via Non-parametric Part Prototype Learning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {9762-9771}
}
```
