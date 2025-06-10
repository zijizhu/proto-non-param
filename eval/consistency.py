"""
Migrated from https://github.com/hqhQAQ/EvalProtoPNet/tree/main
"""
import argparse
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

from .utils import Cub2011Eval, id_to_bbox, id_to_part_loc, in_bbox, mean, part_num, std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perturb_img(norm_img, std=0.2, eps=0.25):
    noise = torch.zeros(norm_img.shape).normal_(mean=0, std=std).to(device=device)
    noise = torch.clip(noise, min=-eps,
                       max=eps)  # Constrain the maximum absolute value, ensuring that the noise is imperceptible by humans
    perturb_img = norm_img + noise
    return perturb_img


@torch.no_grad()
def get_corresponding_object_parts(ppnet, args, half_size, use_noise=False):
    ppnet.eval()
    ppnet_without_ddp = ppnet.module if hasattr(ppnet, 'module') else ppnet
    img_size = ppnet_without_ddp.img_size
    proto_per_class = ppnet_without_ddp.num_prototypes_per_class

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = Cub2011Eval(args.data_path, train=False, transform=transform)  # CUB test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=10,
                                              pin_memory=True, drop_last=False, shuffle=False)
    num_classes = args.nb_classes

    # Infer on the whole test dataset
    all_proto_acts, all_targets, all_img_ids = [], [], []
    for _, (data, targets, img_ids) in enumerate(tqdm(test_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)

        if use_noise:  # This is used when calculating stability score
            data = perturb_img(data)

        _, proto_acts = ppnet_without_ddp.push_forward(data)
        # Select the prototypes belonging to the ground-truth class of each image
        fea_size = proto_acts.shape[-1]
        proto_indices = (targets * proto_per_class).unsqueeze(dim=-1).repeat(1, proto_per_class)
        proto_indices += torch.arange(proto_per_class).to(
            device=device)  # The indexes of prototypes belonging to the ground-truth class of each image
        proto_indices = proto_indices[:, :, None, None].repeat(1, 1, fea_size, fea_size)
        proto_acts = torch.gather(proto_acts, 1, proto_indices)  # (B, proto_per_class, fea_size, fea_size)

        all_proto_acts.append(proto_acts.cpu().detach())
        all_targets.append(targets.cpu())
        all_img_ids.append(img_ids)
    all_proto_acts = torch.cat(all_proto_acts, dim=0).numpy()  # The activation maps of all test images
    all_targets = torch.cat(all_targets, dim=0).numpy()  # The categories of all test images
    all_img_ids = torch.cat(all_img_ids, dim=0).numpy()  # The image ids of all test images

    # Enumerate all the classes, thus enumerate all the prototypes
    all_proto_to_part, all_proto_part_mask = [], []
    '''
    The length of `all_proto_to_part` is 2000, each element indicates the corresponding object parts of a prototype on the images.
    The length of `all_proto_part_mask` is 2000, each element indicates the existing (non-masked) object parts on the images of a prototype.
    '''
    for test_image_label in tqdm(range(num_classes)):
        arr_ids = np.nonzero(all_targets == test_image_label)[0]
        class_proto_acts = all_proto_acts[arr_ids]  # Get the activation maps of all the images of current class
        img_ids = all_img_ids[arr_ids]  # Get the image ids of all the images of current class

        # Get part annotations on all the images of current class
        class_part_labels, class_part_masks = [], []
        '''
        `class_part_labels` save the part labels of images in this class.
        `class_part_masks` save the part masks of images in this class.
        '''
        for _, img_id in enumerate(img_ids):
            # Get part annotations
            part_labels, part_mask = [], np.zeros(part_num, )
            bbox = id_to_bbox[img_id]
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            part_locs = id_to_part_loc[img_id]
            for part_loc in part_locs:
                part_id = part_loc[0] - 1  # The id of current object part (begin from 0)
                part_mask[part_id] = 1  # The current object part exists in current image
                loc_x, loc_y = part_loc[1] - bbox_x1, part_loc[2] - bbox_y1
                ratio_x, ratio_y = loc_x / (bbox_x2 - bbox_x1), loc_y / (
                            bbox_y2 - bbox_y1)  # Fit the bounding boxes' coordinates to the cropped images
                re_loc_x, re_loc_y = int(img_size * ratio_x), int(img_size * ratio_y)
                part_labels.append([part_id, re_loc_x, re_loc_y])
            class_part_labels.append(part_labels)
            class_part_masks.append(part_mask)

        # Enumerate the prototypes of current class
        for proto_idx in range(proto_per_class):
            img_num = len(img_ids)
            proto_to_part = np.zeros((img_num,
                                      part_num))  # Element = 1 -> the prototype corresponds to this object part on this image, element = 0 otherwise
            for img_idx in range(img_num):
                part_labels = class_part_labels[img_idx]  # Get the part labels of current image
                activation_map = class_proto_acts[
                    img_idx, proto_idx]  # Get the activation map of current prototype on current image
                upsampled_activation_map = cv2.resize(activation_map, dsize=(img_size, img_size),
                                                      interpolation=cv2.INTER_CUBIC)

                max_indice = np.where(upsampled_activation_map == upsampled_activation_map.max())
                max_indice = (max_indice[0][0], max_indice[1][0])
                region_pred = (max(0, max_indice[0] - half_size), min(img_size, max_indice[0] + half_size),
                               max(0, max_indice[1] - half_size), min(img_size, max_indice[
                    1] + half_size))  # Get the corresponding region of current prototype, (y1, y2, x1, x2)

                # Get the corresponding object parts of current prototype
                for part_label in part_labels:
                    part_id, loc_x_gt, loc_y_gt = part_label[0], part_label[1], part_label[2]
                    if in_bbox((loc_y_gt, loc_x_gt), region_pred):
                        proto_to_part[img_idx, part_id] = 1

            proto_to_part = np.stack(proto_to_part, axis=0)
            class_part_masks = np.stack(class_part_masks, axis=0)
            all_proto_to_part.append(proto_to_part)
            all_proto_part_mask.append(class_part_masks)

    return all_proto_to_part, all_proto_part_mask


def evaluate_consistency(ppnet: nn.Module, args: argparse.Namespace, save_dir: str | Path, half_size=36,
                         part_thresh=0.8):
    all_proto_to_part, all_proto_part_mask = get_corresponding_object_parts(ppnet, args, half_size)
    with open(Path(save_dir) / "part_prototype.pkl", "wb") as fp:
        pkl.dump(dict(all_proto_to_part=all_proto_to_part, all_proto_part_mask=all_proto_part_mask), file=fp)

    all_proto_consis = []
    '''
    The length of `all_proto_consis` is 2000, each element indicates the consistency of a prototype, 1 -> consistent, 0 -> non-consistent.
    '''
    # Enumerate all the prototypes to calculate consistency score
    for proto_idx in range(len(all_proto_to_part)):
        proto_to_part = all_proto_to_part[proto_idx]
        proto_part_mask = all_proto_part_mask[proto_idx]
        assert ((
                            1. - proto_part_mask) * proto_to_part).sum() == 0  # Assert that the prototype does not correspond to an object part that cannot be visualized (not in the part annotations)
        proto_to_part_sum = proto_to_part.sum(axis=0)
        proto_part_mask_sum = proto_part_mask.sum(axis=0)
        proto_part_mask_sum = np.where(proto_part_mask_sum == 0, proto_part_mask_sum + 1,
                                       proto_part_mask_sum)  # Eliminate the 0 elements in all_part_mask_sum~(prevent 0 from being denominator), it doesn't affect the evaluation result
        mean_part_float = proto_to_part_sum / proto_part_mask_sum
        mean_part = (mean_part_float >= part_thresh).astype(
            np.int32)  # The prototope is determined to be non-consistent if  no element in the averaged corresponding object parts exceeds `part_thresh`

        if mean_part.sum() == 0:
            all_proto_consis.append(0)
        else:
            all_proto_consis.append(1)

    all_proto_consis = np.array(all_proto_consis)
    consistency_score = all_proto_consis.mean() * 100

    return consistency_score
