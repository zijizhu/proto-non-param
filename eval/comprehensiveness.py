from collections import defaultdict
from logging import getLogger
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import InterpolationMode, resized_crop
from tqdm import tqdm

from .utils import mean, std

logger = getLogger(__name__)


@torch.no_grad()
def get_attn_maps(outputs: dict[str, torch.Tensor], labels: torch.Tensor):
    patch_prototype_logits = outputs["patch_prototype_logits"]

    batch_size, n_patches, C, K = patch_prototype_logits.shape
    H = W = int(sqrt(n_patches))

    patch_prototype_logits = rearrange(patch_prototype_logits, "B (H W) C K -> B C K H W", H=H, W=W)
    patch_prototype_logits = patch_prototype_logits[torch.arange(labels.numel()), labels, ...]  # B K H W

    pooled_logits = F.avg_pool2d(patch_prototype_logits, kernel_size=(2, 2,), stride=2)
    return patch_prototype_logits, pooled_logits


class CUBEvalDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 input_size: int = 224,
                 transform=None):
        super().__init__(
            root=(Path(root) / "cub200_cropped" / "test_cropped").as_posix(),
            transform=transform,
            target_transform=None
        )
        self.input_size = input_size
        root = Path(root)

        path_df = pd.read_csv(root / "CUB_200_2011" / "images.txt", header=None, names=["image_id", "image_path"],
                              sep=" ")
        bbox_df = pd.read_csv(root / "CUB_200_2011" / "bounding_boxes.txt", header=None,
                              names=["image_id", "x", "y", "w", "h"], sep=" ")
        self.bbox_df = path_df.merge(bbox_df, on="image_id")

        self.segmentation_map_root = root / "segmentations"

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = Image.open(im_path).convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        class_dir, sample_filename = Path(im_path).parts[-2:]
        seg_map = read_image(
            (self.segmentation_map_root / class_dir / sample_filename.replace("jpg", "png")).as_posix())

        row = self.bbox_df[self.bbox_df["image_path"] == "/".join(Path(im_path).parts[-2:])].iloc[0]
        x, y, w, h = tuple(row[["x", "y", "w", "h"]].values.flatten().astype(int))
        seg_map_cropped = resized_crop(
            seg_map,
            top=y, left=x, height=h, width=w,
            size=(224, 224,),
            interpolation=InterpolationMode.NEAREST_EXACT
        )
        if seg_map_cropped.size(0) > 1:
            seg_map_cropped = seg_map_cropped[0, :, :].unsqueeze(0)

        seg_map_cropped = (seg_map_cropped > 200).to(torch.long)
        return im, label, seg_map_cropped, index


def norm_and_thresh(activations: torch.Tensor, threshold: float = 0.7):
    """Converts each activation map to a binary mask by normalizing and thresholding"""
    B, H, H, W = activations.shape
    max_values = F.adaptive_max_pool2d(activations, output_size=(1, 1,))
    min_valies = -F.adaptive_max_pool2d(-activations, output_size=(1, 1,))
    normalized_activations = (activations - min_valies) / (max_values - min_valies)

    binary_activations = (normalized_activations >= threshold).to(dtype=torch.float32)

    return binary_activations


def calculate_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-6):
    intersection = torch.sum(torch.logical_and(mask1, mask2))
    union = torch.sum(torch.logical_or(mask1, mask2))
    iou = intersection / (union + eps)
    return iou


def batch_IoU_binary(batch_activations: torch.Tensor, batch_gt_foreground_maps: torch.Tensor, threshold: float = 0.7):
    B, K, H, W = batch_activations.shape

    binary_activations = norm_and_thresh(batch_activations, threshold=threshold)
    binary_activations_union = (binary_activations.sum(dim=1) > 0).to(dtype=torch.long)

    batch_IoUs = []
    for pred_fg_map, gt_fg_map in zip(binary_activations_union, batch_gt_foreground_maps):
        IoU = calculate_iou(pred_fg_map, gt_fg_map)
        batch_IoUs.append(IoU.item())

    return batch_IoUs


@torch.no_grad()
def evaluate_comprehensiveness(net: nn.Module,
                               data_root: str,
                               threshold: float = 0.6,
                               topk: int = 5,
                               num_classes: int = 200,
                               device: torch.device = torch.device("cpu"),
                               input_size: tuple[int, int] = (224, 224,)):
    normalize = T.Normalize(mean=mean, std=std)
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        normalize
    ])

    eval_dataset = CUBEvalDataset(root=data_root, transform=transform)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=64, num_workers=2, shuffle=True)

    net.to(device)
    net.eval()

    IoUs = []
    for b, batch in enumerate(tqdm(eval_dataloader)):
        images, targets, seg_map_cropped, indices = tuple(item.to(device=device) for item in batch)
        B, _, INPUT_H, INPUT_W = images.shape
        if hasattr(net, 'get_attn_maps'):
            _, batch_activations = net.get_attn_maps(images, targets)
        elif hasattr(net, 'push_forward'):
            _, all_batch_activations = net.push_forward(images)
            B, CK, H, W = all_batch_activations.shape
            K = CK // num_classes
            proto_indices = (targets * K).unsqueeze(dim=-1).repeat(1, K)
            proto_indices += torch.arange(K).to(
                device=device)  # The indexes of prototypes belonging to the ground-truth class of each image
            proto_indices = proto_indices[:, :, None, None].repeat(1, 1, H, W)
            gt_batch_activations = torch.gather(all_batch_activations, 1,
                                                proto_indices)  # (B, proto_per_class, fea_size, fea_size)

            max_vals = F.adaptive_max_pool2d(gt_batch_activations, output_size=(1, 1,))
            topk_batch_activations = torch.gather(gt_batch_activations, 1,
                                                  max_vals.topk(dim=1, k=min(topk, K)).indices.repeat(1, 1, H, W))

            batch_activations = F.avg_pool2d(topk_batch_activations, kernel_size=(2, 2,), stride=2)
        else:
            outputs = net(images, targets)
            batch_activations = get_attn_maps(outputs, labels=targets)
        batch_activations_resized = F.interpolate(batch_activations, size=(INPUT_H, INPUT_W,), mode="bilinear")

        IoUs += batch_IoU_binary(batch_activations_resized, seg_map_cropped, threshold=threshold)

    score = sum(IoUs) / len(IoUs)
    logger.info(f"Comprehensiveness Score with Threshold {threshold:.1f}: {score:.4f}")
