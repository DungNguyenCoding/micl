"""Dataset transforms. Plotting intentionally lives elsewhere."""

from __future__ import annotations

import torch
from torchvision import transforms

from bayesfl.config import DataConfig


class Cutout:
    """Tensor Cutout matching the paper setting (one 16x16 hole by default)."""

    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        self.n_holes = int(n_holes)
        self.length = int(length)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.n_holes <= 0:
            return img
        h, w = int(img.shape[-2]), int(img.shape[-1])
        out = img.clone()
        for _ in range(self.n_holes):
            y = int(torch.randint(0, h, (1,)).item())
            x = int(torch.randint(0, w, (1,)).item())
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            out[..., y1:y2, x1:x2] = 0
        return out


def build_transform(cfg: DataConfig, *, train: bool):
    ops = []
    if cfg.dataset == "cifar10" and train and cfg.augment:
        ops.append(
            transforms.RandomCrop(
                32,
                padding=cfg.crop_padding,
                fill=cfg.crop_fill,
            )
        )
        if cfg.random_flip:
            ops.append(transforms.RandomHorizontalFlip())
        if cfg.autoaugment_policy == "cifar10":
            # The paper/released code uses CIFAR10 AutoAugment.  Torchvision's
            # canonical CIFAR10 policy provides the same augmentation family
            # without vendoring the authors' third-party helper file.
            ops.append(
                transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.CIFAR10
                )
            )

    ops.append(transforms.ToTensor())
    if cfg.dataset == "cifar10" and train and cfg.augment and cfg.cutout_holes > 0:
        ops.append(Cutout(cfg.cutout_holes, cfg.cutout_length))
    ops.append(transforms.Normalize(tuple(cfg.normalize_mean), tuple(cfg.normalize_std)))
    return transforms.Compose(ops)
