"""Dataset transforms. Plotting intentionally lives elsewhere."""

from __future__ import annotations

from torchvision import transforms

from bayesfl.config import DataConfig


def build_transform(cfg: DataConfig, *, train: bool):
    ops = []
    if cfg.dataset == "cifar10" and train and cfg.augment:
        ops.append(transforms.RandomCrop(32, padding=cfg.crop_padding))
        if cfg.random_flip:
            ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(tuple(cfg.normalize_mean), tuple(cfg.normalize_std)))
    return transforms.Compose(ops)
