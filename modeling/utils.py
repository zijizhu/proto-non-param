import re
from copy import deepcopy
from logging import Logger

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def block_expansion_dino(state_dict: dict[str, torch.Tensor], n_splits: int = 3, freeze_layer_norm: bool = True):
    """Perform Block Expansion on a ViT described in https://arxiv.org/abs/2404.17245"""
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)

    block_indices = np.arange(0, n_blocks).reshape((n_splits, -1,))
    block_indices = np.concatenate([block_indices, block_indices[:, -1:]], axis=-1)

    n_splits, n_block_per_split = block_indices.shape
    new_block_indices = list((i + 1) * n_block_per_split - 1 for i in range(n_splits))

    expanded_state_dict = dict()
    learnable_param_names, zero_param_names = [], []

    for dst_idx, src_idx in enumerate(block_indices.flatten()):
        src_keys = [k for k in state_dict if f"blocks.{src_idx}" in k]
        dst_keys = [k.replace(f"blocks.{src_idx}", f"blocks.{dst_idx}") for k in src_keys]

        block_state_dict = dict()

        for src_k, dst_k in zip(src_keys, dst_keys):
            if ("mlp.fc2" in dst_k or "attn.proj" in dst_k) and (dst_idx in new_block_indices):
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
                zero_param_names.append(dst_k)
            else:
                block_state_dict[dst_k] = state_dict[src_k]

        expanded_state_dict.update(block_state_dict)

        if dst_idx in new_block_indices:
            learnable_param_names += dst_keys

    expanded_state_dict.update({k: v for k, v in state_dict.items() if "block" not in k})

    if not freeze_layer_norm:
        learnable_param_names += ["norm.weight", "norm.bias"]

    return expanded_state_dict, len(block_indices.flatten()), learnable_param_names, zero_param_names


def append_blocks(state_dict: dict[str, torch.Tensor], n_splits: int = 1, freeze_layer_norm: bool = True):
    """Append new ViT blocks with zero-ed MLPs and Attention Projection. Other weights initialized using last layer"""
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)

    src_block_idx = n_blocks - 1
    src_keys = [k for k in state_dict if f"blocks.{src_block_idx}" in k]  # keys of parameters to copy from

    expanded_state_dict = deepcopy(state_dict)
    learnable_param_names, zero_param_names = [], []
    for i in range(n_splits):
        dst_block_idx = n_blocks + i
        dst_keys = [k.replace(f"blocks.{src_block_idx}", f"blocks.{dst_block_idx}") for k in src_keys]

        block_state_dict = dict()
        for src_k, dst_k in zip(src_keys, dst_keys):
            if "mlp.fc2" in dst_k or "attn.proj" in dst_k:
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
                zero_param_names.append(dst_k)
            else:
                block_state_dict[dst_k] = state_dict[src_k]
        expanded_state_dict.update(block_state_dict)
        learnable_param_names += dst_keys

    if not freeze_layer_norm:
        learnable_param_names += ["norm.weight", "norm.bias"]

    return expanded_state_dict, n_blocks + n_splits, learnable_param_names, zero_param_names


def print_parameters(net: nn.Module, logger: Logger):
    logger.info("Learnable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            msg = name + ("(zero-ed)" if param.detach().sum() == 0 else "")
            logger.info(msg)

"""
The following functions are adapted from https://github.com/tfzhou/ProtoSeg
"""

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def sinkhorn_knopp(out, n_iterations=3, epsilon=0.05, use_gumbel=False):
    L = torch.exp(out / epsilon).t()  # shape: [K, B,]
    K, B = L.shape

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(n_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indices = torch.argmax(L, dim=1)
    if use_gumbel:
        L = F.gumbel_softmax(L, tau=0.5, hard=True)
    else:
        L = F.one_hot(indices, num_classes=K).to(dtype=torch.float32)

    return L, indices
