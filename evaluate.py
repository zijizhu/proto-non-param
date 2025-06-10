#!/usr/bin/env python3
import sys
import logging
from logging import Logger
from pathlib import Path
import argparse

import lightning as L
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CUBDataset
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import PCA, PNP
from eval.comprehensiveness import evaluate_comprehensiveness
from eval.distinctiveness import evaluate_distinctiveness
from eval.stability import evaluate_stability
from eval.consistency import evaluate_consistency

@torch.inference_mode()
def eval_accuracy(model: nn.Module, dataloader: DataLoader, logger: Logger, device: torch.device):
    model.eval()
    correct = 0
    total = 0

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]

        outputs = model(images)

        predicted = torch.argmax(outputs["class_logits"], dim=-1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    logger.info(f"Accuracy: {acc:.4f}")

    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    L.seed_everything(args.seed)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    assert "hparams" in ckpt and "state_dict" in ckpt
    hparams, state_dict = argparse.Namespace(**ckpt["hparams"]), ckpt["state_dict"]

    log_dir = Path(hparams.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logger = logging.getLogger(__name__)

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])

    if hparams.dataset == "CUB":
        logger.info("Test on CUB-200-2011")
        n_classes = 200
        dataset_dir = Path(hparams.data_root) / "cub200_cropped"

        dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(), transforms=transforms)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)
    else:
        raise NotImplementedError(f"Dataset {hparams.dataset} is not implemented")

    if "dinov2" in hparams.backbone:
        if hparams.num_splits and hparams.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=hparams.backbone,
                n_splits=hparams.num_splits,
                mode="block_expansion",
                freeze_norm_layer=True
            )
        else:
            backbone = DINOv2Backbone(name=hparams.backbone)
        dim = backbone.dim
    elif "dino" in hparams.backbone:
        backbone = DINOBackboneExpanded(
            name=hparams.backbone,
            n_splits=hparams.num_splits,
            mode="block_expansion",
            freeze_norm_layer=True
        )
        dim = backbone.dim
    else:
        raise NotImplementedError(f"Backbone {hparams.backbone} not implemented.")

    # Can be substituted with other off-the-shelf methods
    fg_extractor = PCA(bg_class=n_classes, compare_fn="le", threshold=0.5)

    net = PNP(
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=hparams.num_prototypes,
        n_classes=n_classes,
        gamma=hparams.gamma,
        temperature=hparams.temperature,
        sa_init=hparams.sa_initial_value,
        use_sinkhorn=True,
        norm_prototypes=False
    )
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)

    logger.info("Evaluating accuracy...")
    eval_accuracy(model=net, dataloader=dataloader_test, logger=logger, device=device)

    logger.info("Evaluating consistency...")
    consistency_args = {
        "nb_classes": n_classes,
        "data_path": hparams.data_root,
        "test_batch_size": 128
    }

    net.img_size = 224
    net.num_prototypes_per_class = net.n_prototypes
    consistency_score = evaluate_consistency(net, argparse.Namespace(**consistency_args), save_dir=log_dir)
    logger.info(f"Network consistency score: {consistency_score.item()}")

    logger.info("Evaluating stability...")
    stability_score  = evaluate_stability(net, argparse.Namespace(**consistency_args))
    logger.info(f"Network stability score: {stability_score.item()}")

    logger.info("Evaluating distinctiveness...")
    evaluate_distinctiveness(net, hparams.data_root, device=device)

    logger.info("Evaluating comprehensiveness...")
    evaluate_comprehensiveness(net, hparams.data_root, device=device)


if __name__ == '__main__':
    main()
