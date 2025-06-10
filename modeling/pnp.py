from math import sqrt

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import nn

from .utils import momentum_update, sinkhorn_knopp


class PCA(nn.Module):
    def __init__(self, compare_fn: str = "le", threshold: float = 0.5, n_components: int = 1,
                 bg_class: int = 200) -> None:
        super().__init__()
        self.compare_fn = torch.ge if compare_fn == "ge" else torch.le
        self.threshold = threshold
        self.n_components = n_components
        self.bg_class = bg_class

    def forward(self, x: torch.Tensor, y: torch.Tensor, cls_tokens: torch.Tensor | None = None, attn_maps=None):
        B, n_patches, dim = x.shape
        H = W = int(sqrt(n_patches))
        U, _, _ = torch.pca_lowrank(
            x.reshape(-1, dim),
            q=self.n_components, center=True, niter=10
        )
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()  # shape: [B*H*W, 1]
        U_scaled = U_scaled.reshape(B, H, W)

        pseudo_patch_labels = torch.where(
            self.compare_fn(U_scaled, other=self.threshold),
            repeat(y, "B -> B H W", H=H, W=W),
            self.bg_class
        )

        return pseudo_patch_labels.to(dtype=torch.long)  # B H W


class ScoreAggregation(nn.Module):
    def __init__(self, init_val: float = 0.2, n_classes: int = 200, n_prototypes: int = 5) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.full((n_classes, n_prototypes,), init_val, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        n_classes, n_prototypes = self.weights.shape
        sa_weights = F.softmax(self.weights, dim=-1) * n_prototypes
        x = x * sa_weights  # B C K
        x = x.sum(-1)  # B C
        return x


class PNP(nn.Module):
    def __init__(self, backbone: nn.Module, fg_extractor: nn.Module,
                 *,
                 always_norm_patches: bool = True, gamma: float = 0.999, n_prototypes: int = 5,
                 n_classes: int = 200, norm_prototypes=False, temperature: float = 0.2,
                 sa_init: float = 0.5, dim: int = 768, use_sinkhorn: bool = True):
        super().__init__()
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.C = n_classes + 1
        self.backbone = backbone
        self.use_sinkhorn = use_sinkhorn
        self.fg_extractor = fg_extractor

        self.dim = dim
        self.register_buffer("prototypes", torch.randn(self.C, self.n_prototypes, self.dim))
        self.temperature = temperature

        nn.init.trunc_normal_(self.prototypes, std=0.02)

        self.classifier = ScoreAggregation(init_val=sa_init, n_classes=n_classes, n_prototypes=n_prototypes)

        self.optimizing_prototypes = True
        self.initializing = True
        self.always_norm_patches = always_norm_patches
        self.norm_prototypes = norm_prototypes

    @staticmethod
    def online_clustering(prototypes: torch.Tensor,
                          patch_tokens: torch.Tensor,
                          patch_prototype_logits: torch.Tensor,
                          patch_labels: torch.Tensor,
                          *,
                          gamma: float = 0.999,
                          use_sinkhorn: bool = True,
                          use_gumbel: bool = False):
        """Updates the prototypes based on the given inputs.
        This function updates the prototypes based on the patch tokens,
        patch-prototype logits, labels, and patch labels.

        Args:
            prototypes (torch.Tensor): A tensor of shape [C, K, dim,], representing K prototypes for each of C classes.
            patch_tokens: A tensor of shape [B, n_patches, dim,], which is the feature from backbone.
            patch_prototype_logits: The logits between patches and prototypes of shape [B, n_patches, C, K].
            patch_labels: A tensor of shape [B, H, W,] of type torch.long representing the (generated) patch-level class labels.
            labels: A tensor of shape [B,] of type torch.long representing the image-level class labels.
            gamma: A float indicating the coefficient for momentum update.
            use_gumbel: A boolean indicating whether to use gumbel softmax for patch assignments.
        """
        B, H, W = patch_labels.shape
        C, K, dim = prototypes.shape

        patch_labels_flat = patch_labels.flatten()  # shape: [B*H*W,]
        patches_flat = rearrange(patch_tokens, "B n_patches dim -> (B n_patches) dim")
        L = rearrange(patch_prototype_logits, "B n_patches C K -> (B n_patches) C K")

        P_old = prototypes.clone()
        P_new = prototypes.clone()

        part_assignment_maps = torch.empty_like(patch_labels_flat)

        for c in patch_labels.unique().tolist():
            class_fg_mask = patch_labels_flat == c  # shape: [B*H*W,]
            I_c = patches_flat[class_fg_mask]  # shape: [N, dim]
            L_c = L[class_fg_mask, c, :]  # shape: [N, K,]
            if use_sinkhorn:
                L_c_assignment, L_c_assignment_indices = sinkhorn_knopp(L_c, use_gumbel=use_gumbel)  # shape: [N, K,], [N,]
            else:
                L_c_assignment, L_c_assignment_indices = L_c, L_c.argmax(dim=-1)
            P_c_new = torch.mm(L_c_assignment.t(), I_c)  # shape: [K, dim]

            P_c_old = P_old[c, :, :]

            P_new[c, ...] = momentum_update(P_c_old, P_c_new, momentum=gamma)

            part_assignment_maps[class_fg_mask] = L_c_assignment_indices + c * K

        part_assignment_maps = rearrange(part_assignment_maps, "(B H W) -> B (H W)", B=B, H=H, W=W)

        return part_assignment_maps, P_new

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, *, use_gumbel: bool = False):
        assert (not self.training) or (labels is not None)

        patch_tokens, raw_patch_tokens, cls_tokens = self.backbone(x)  # shape: [B, n_patches, dim,]

        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)

        patch_prototype_logits = einsum(patch_tokens, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")

        image_prototype_logits = patch_prototype_logits.max(1).values  # shape: [B, C, K,], C=n_classes+1

        class_logits = self.classifier(image_prototype_logits[:, :-1, :])
        class_logits = class_logits / self.temperature

        outputs = dict(
            patch_prototype_logits=patch_prototype_logits,  # shape: [B, n_patches, C, K,]
            image_prototype_logits=image_prototype_logits,  # shape: [B, C, K,]
            class_logits=class_logits  # shape: [B, n_classes,]
        )

        if labels is not None:
            raw_patch_tokens = F.normalize(raw_patch_tokens, p=2, dim=-1)
            pseudo_patch_labels = self.fg_extractor(raw_patch_tokens.detach(), labels, cls_tokens=cls_tokens)
            pseudo_patch_labels = pseudo_patch_labels.detach()

            part_assignment_maps, new_prototypes = self.online_clustering(
                prototypes=self.prototypes,
                patch_tokens=raw_patch_tokens.detach(),
                patch_prototype_logits=patch_prototype_logits.detach(),
                patch_labels=pseudo_patch_labels,
                gamma=self.gamma,
                use_gumbel=use_gumbel,
                use_sinkhorn=self.use_sinkhorn
            )

            if self.training and self.optimizing_prototypes:
                self.prototypes = F.normalize(new_prototypes, p=2, dim=-1) if self.norm_prototypes else new_prototypes

            outputs.update({
                "patches": raw_patch_tokens,
                "part_assignment_maps": part_assignment_maps,  # B n_patches
                "pseudo_patch_labels": pseudo_patch_labels  # B H W
            })

        return outputs

    def get_attn_maps(self, images: torch.Tensor, labels: torch.Tensor):
        outputs = self(images, labels)
        patch_prototype_logits = outputs["patch_prototype_logits"]

        batch_size, n_patches, C, K = patch_prototype_logits.shape
        H = W = int(sqrt(n_patches))

        patch_prototype_logits = rearrange(patch_prototype_logits, "B (H W) C K -> B C K H W", H=H, W=W)
        patch_prototype_logits = patch_prototype_logits[torch.arange(labels.numel()), labels, ...]  # B K H W

        pooled_logits = F.avg_pool2d(patch_prototype_logits, kernel_size=(2, 2,), stride=2)
        return patch_prototype_logits, pooled_logits

    def push_forward(self, x: torch.Tensor):
        patch_tokens, _, cls_tokens = self.backbone(x)  # shape: [B, n_patches, dim,]
        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)
        if not self.initializing:
            patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm,
                                            "B n_patches dim, C K dim -> B n_patches C K")
        else:
            patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm,
                                            "B n_patches dim, C K dim -> B n_patches C K")
        batch_size, n_patches, C, K = patch_prototype_logits.shape
        H = W = int(sqrt(n_patches))
        prototype_logits = rearrange(patch_prototype_logits[:, :, :-1, :], "B (H W) C K -> B (C K) H W", H=H, W=W)
        return None, F.avg_pool2d(prototype_logits, kernel_size=(2, 2,), stride=2)


class PNPCriterion(nn.Module):
    def __init__(
            self,
            l_ppd_coef: float = 0,
            l_ppd_temp: float = 0.1,

            num_classes: int = 200,
            n_prototypes: int = 5,
            bg_class_weight: float = 0.1
    ) -> None:
        super().__init__()
        self.l_ppd_coef = l_ppd_coef
        self.l_ppd_temp = l_ppd_temp

        self.xe = nn.CrossEntropyLoss()

        self.C = num_classes
        self.K = n_prototypes
        self.class_weights = torch.tensor([1] * self.C * self.K + [bg_class_weight] * self.K)

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]):
        logits = outputs["class_logits"]
        patch_prototype_logits = outputs["patch_prototype_logits"]
        part_assignment_maps = outputs["part_assignment_maps"]

        labels = batch[1]

        loss_dict = dict()
        loss_dict["l_y"] = self.xe(logits, labels)

        if self.l_ppd_coef != 0:
            l_ppd = self.ppd_criterion(
                patch_prototype_logits,
                part_assignment_maps,
                class_weight=self.class_weights.to(dtype=torch.float32, device=logits.device),
                temperature=self.l_ppd_temp
            )
            loss_dict["l_ppd"] = self.l_ppd_coef * l_ppd
            loss_dict["_l_ppd_unadjusted"] = l_ppd

        return loss_dict
    
    @staticmethod
    def ppd_criterion(patch_prototype_logits: torch.Tensor,
                      patch_prototype_assignments: torch.Tensor,
                      class_weight: torch.Tensor,
                      temperature: float = 0.1):
        patch_prototype_logits = rearrange(patch_prototype_logits, "B N C K -> B (C K) N") / temperature
        loss = F.cross_entropy(patch_prototype_logits, target=patch_prototype_assignments, weight=class_weight)
        return loss
