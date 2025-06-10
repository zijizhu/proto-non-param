#!/usr/bin/env python3
import sys
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path
import argparse

import lightning as L
import torch
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from data import CUBDataset
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import PCA, PNP, PNPCriterion
from modeling.utils import print_parameters


def train(model: nn.Module, criterion: nn.Module | None, dataloader: DataLoader, epoch: int,
          optimizer: optim.Optimizer | None, logger: Logger, device: torch.device):
    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]

        outputs = model(images, labels=labels)

        if criterion is not None and optimizer is not None:
            loss_dict = criterion(outputs, batch)  # type: dict[str, torch.Tensor]
            loss = sum(val for key, val in loss_dict.items() if not key.startswith("_"))

            if not isinstance(loss, torch.Tensor):
                raise ValueError

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for k, v in loss_dict.items():
                running_losses[k] += v.item() * dataloader.batch_size

        mca_train(outputs["class_logits"], labels)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        logger.info(f"EPOCH {epoch} train {k}: {loss_avg:.4f}")

    epoch_acc_train = mca_train.compute().item()
    logger.info(f"EPOCH {epoch} train acc: {epoch_acc_train:.4f}")


@torch.inference_mode()
def test(model: nn.Module, dataloader: DataLoader, epoch: int,
         logger: Logger, device: torch.device):
    model.eval()
    mca_test = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels = batch[:2]

        outputs = model(images)

        mca_test(outputs["class_logits"], labels)

    epoch_acc_test = mca_test.compute().item()
    logger.info(f"EPOCH {epoch} test acc: {epoch_acc_test:.4f}")

    return epoch_acc_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--dataset", type=str, default="CUB", choices=["CUB"])

    parser.add_argument("--backbone", type=str, default="dinov2_vitb14", choices=["dinov2_vitb14", "dinov2_vits14"])
    parser.add_argument("--num-splits", type=int, default=1)

    # Model related hyperparameters
    parser.add_argument("--num-prototypes", type=int, default=5, help="Number of prototypes per class")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sa-initial-value", type=float, default=0.5)

    # Optimization hyperparameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--classifier-lr", type=float, default=1.0e-6)
    parser.add_argument("--fine-tuning-start-epoch", type=int, default=1)

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
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

    L.seed_everything(args.seed)

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])

    if args.dataset == "CUB":
        logger.info("Train on CUB-200-2011")
        n_classes = 200
        dataset_dir = Path(args.data_root) / "cub200_cropped"
        dataset_train = CUBDataset((dataset_dir / "train_cropped_augmented").as_posix(),
                                   transforms=transforms)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, num_workers=8, shuffle=True)

        dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                                   transforms=transforms)
        dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    if "dinov2" in args.backbone:
        if args.num_splits and args.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=args.backbone,
                n_splits=args.num_splits,
                mode="block_expansion",
                freeze_norm_layer=True
            )
        else:
            backbone = DINOv2Backbone(name=args.backbone)
        dim = backbone.dim
    elif "dino" in args.backbone:
        backbone = DINOBackboneExpanded(
            name=args.backbone,
            n_splits=args.num_splits,
            mode="block_expansion",
            freeze_norm_layer=True
        )
        dim = backbone.dim
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")

    # Can be substituted with other off-the-shelf methods
    fg_extractor = PCA(bg_class=n_classes, compare_fn="le", threshold=0.5)

    net = PNP(
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=args.num_prototypes,
        n_classes=n_classes,
        gamma=args.gamma,
        temperature=args.temperature,
        sa_init=args.sa_initial_value,
        use_sinkhorn=True,
        norm_prototypes=False
    )
    criterion = PNPCriterion(l_ppd_coef=0.8, n_prototypes=args.num_prototypes, num_classes=n_classes)

    net.to(device)

    best_epoch, best_test_epoch = 0, 0.0

    for epoch in range(args.epochs):
        is_fine_tuning = epoch >= args.fine_tuning_start_epoch

        # Stage 2 training
        if is_fine_tuning:
            logger.info("Start fine-tuning backbone...")
            for name, param in net.named_parameters():
                param.requires_grad = ("backbone" not in name) and ("fg_extractor" not in name)

            net.backbone.set_requires_grad()

            param_groups = [{'params': net.backbone.learnable_parameters(),
                             'lr': args.backbone_lr}]
            param_groups += [{'params': net.classifier.parameters(), 'lr': args.classifier_lr}]

            optimizer = optim.Adam(param_groups)

            net.optimizing_prototypes = False
        # Stage 1 training
        else:
            for params in net.parameters():
                params.requires_grad = False
            optimizer = None
            net.optimizing_prototypes = True

        if epoch > 0:
            net.initializing = False

        print_parameters(net=net, logger=logger)
        logger.info(f"net.initializing: {net.initializing}")
        logger.info(f"net.optimizing_prototypes: {net.optimizing_prototypes}")

        train(
            model=net,
            criterion=criterion if is_fine_tuning else None,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer if is_fine_tuning else None,
            logger=logger,
            device=device
        )

        epoch_acc_test = test(model=net, dataloader=dataloader_test, epoch=epoch, logger=logger, device=device)
        
        torch.save(
            dict(
                state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                hparams=vars(args),
            ),
            log_dir / "ckpt.pth"
        )
        logger.info("Model saved as ckpt.pth")

        if epoch_acc_test > best_test_epoch:
            best_val_acc = epoch_acc_test
            best_epoch = epoch

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
