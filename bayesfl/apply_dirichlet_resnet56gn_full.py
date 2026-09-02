from pathlib import Path
import re

ROOT = Path('.')

def replace_once(text, pattern, repl, label, flags=0):
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        raise RuntimeError(f'{label}: expected one match, found {n}')
    return new

# ---------- config.py ----------
p = ROOT/'config.py'
text = p.read_text()
m = re.search(r'@dataclass\nclass DataConfig:\n(?P<body>.*?)(?=\n\n@dataclass\nclass ModelConfig:)', text, re.S)
if not m:
    raise RuntimeError('DataConfig block not found')
body = m.group('body')
if 'partition_mode:' not in body:
    addition = (
        '    # legacy: original Poisson + fixed label-count partition.\n'
        '    # dirichlet: Poisson-scarce client sizes with Dirichlet class mixtures.\n'
        '    partition_mode: str = "legacy"\n'
        '    dirichlet_alpha: float = 0.1\n'
    )
    body = body.rstrip() + '\n' + addition
    text = text[:m.start('body')] + body + text[m.end('body'):]

if 'data.partition_mode must be legacy or dirichlet' not in text:
    needle = '        if self.data.labels_per_client not in range(1, 11):\n            raise ValueError("data.labels_per_client must be between 1 and 10")\n'
    if needle not in text:
        raise RuntimeError('config validation insertion point not found')
    addition = needle + (
        '        partition_mode = str(self.data.partition_mode).strip().lower()\n'
        '        if partition_mode not in {"legacy", "dirichlet"}:\n'
        '            raise ValueError("data.partition_mode must be legacy or dirichlet")\n'
        '        self.data.partition_mode = partition_mode\n'
        '        if float(self.data.dirichlet_alpha) <= 0.0:\n'
        '            raise ValueError("data.dirichlet_alpha must be positive")\n'
    )
    text = text.replace(needle, addition, 1)
p.write_text(text)

# ---------- dataset.py block replacement ----------
p = ROOT/'dataset.py'
text = p.read_text()
new_block = r'''def _float_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def partition_filename(
    partition_dir: str | Path,
    seed: int,
    num_clients: int,
    labels_per_client: int,
    mean_samples: float,
    label_pairing_mode: str = "uniform",
    partition_mode: str = "legacy",
    dirichlet_alpha: float = 0.1,
) -> Path:
    """Return a stable filename without changing legacy cache names."""
    mode = str(partition_mode).strip().lower()
    safe_mean = _float_token(mean_samples)

    if mode == "dirichlet":
        dataset_name = (
            os.environ.get("AIRCOMP_DATASET", "mnist").strip().lower()
            or "mnist"
        )
        safe_alpha = _float_token(dirichlet_alpha)
        return Path(partition_dir) / (
            f"{dataset_name}_seed{seed}_k{num_clients}_"
            f"dirichlet_a{safe_alpha}_m{safe_mean}.json"
        )

    pairing = str(label_pairing_mode).strip().lower()
    suffix = "" if pairing == "uniform" else f"_{pairing}"
    return Path(partition_dir) / (
        f"mnist_seed{seed}_k{num_clients}_l{labels_per_client}_"
        f"m{safe_mean}{suffix}.json"
    )


def _sample_client_labels(
    rng: np.random.Generator,
    labels_per_client: int,
    label_pairing_mode: str,
) -> List[int]:
    """Sample labels for the original fixed-label partition mode."""
    labels_per_client = int(labels_per_client)
    mode = str(label_pairing_mode).strip().lower()

    if mode == "random_nonadjacent":
        if labels_per_client != 2:
            raise ValueError(
                "random_nonadjacent pairing requires labels_per_client=2"
            )
        allowed_pairs = [
            (a, b)
            for a in range(10)
            for b in range(a + 1, 10)
            if abs(a - b) > 1
        ]
        pair = allowed_pairs[int(rng.integers(0, len(allowed_pairs)))]
        return [int(pair[0]), int(pair[1])]

    if mode != "uniform":
        raise ValueError(f"Unknown label pairing mode: {mode!r}")

    if labels_per_client == 10:
        return list(range(10))

    return sorted(
        int(v)
        for v in rng.choice(
            10,
            size=labels_per_client,
            replace=False,
        ).tolist()
    )


def _build_label_pools(
    targets: np.ndarray,
    rng: np.random.Generator,
) -> tuple[Dict[int, np.ndarray], Dict[int, int]]:
    pools: Dict[int, np.ndarray] = {}
    offsets: Dict[int, int] = {}
    for label in range(10):
        indices = np.flatnonzero(targets == label).astype(np.int64)
        rng.shuffle(indices)
        pools[label] = indices
        offsets[label] = 0
    return pools, offsets


def _take_from_pool(
    *,
    label: int,
    count: int,
    pools: Dict[int, np.ndarray],
    offsets: Dict[int, int],
) -> np.ndarray:
    """Take unique examples from one label pool without replacement."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    pool = pools[int(label)]
    start = int(offsets[int(label)])
    end = start + int(count)
    if end > len(pool):
        raise RuntimeError(
            f"Partition requested {end} examples from class {label}, "
            f"but only {len(pool)} are available without replacement. "
            "Reduce mean_samples_per_client or use a larger dataset."
        )
    offsets[int(label)] = end
    return pool[start:end]


def prepare_partitions(
    data_cfg: DataConfig,
    seed: int,
    partition_dir: str | Path,
    force: bool = False,
) -> Path:
    """Create one deterministic partition file shared by all methods.

    ``legacy`` preserves the original scarce/non-IID partition rule.

    ``dirichlet`` preserves the existing data-scarcity budget: each client
    first receives a Poisson-distributed total sample count with mean
    ``mean_samples_per_client``.  Its ten-class mixture is then sampled as

        pi_k ~ Dirichlet(alpha * 1_10)
        n_k,* ~ Multinomial(n_k, pi_k)

    where ``alpha=data_cfg.dirichlet_alpha``.  Examples are allocated without
    replacement, so the same image never belongs to two clients.
    """
    partition_dir = Path(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    pairing_mode = str(getattr(data_cfg, "label_pairing_mode", "uniform"))
    partition_mode = str(getattr(data_cfg, "partition_mode", "legacy")).strip().lower()
    dirichlet_alpha = float(getattr(data_cfg, "dirichlet_alpha", 0.1))

    path = partition_filename(
        partition_dir,
        seed,
        data_cfg.num_clients,
        data_cfg.labels_per_client,
        data_cfg.mean_samples_per_client,
        pairing_mode,
        partition_mode,
        dirichlet_alpha,
    )
    if path.exists() and not force:
        return path

    ensure_mnist(data_cfg.root)
    targets = _training_targets(data_cfg.root)
    rng = np.random.default_rng(seed)
    pools, offsets = _build_label_pools(targets, rng)

    clients: Dict[str, Dict[str, object]] = {}

    for client_id in range(data_cfg.num_clients):
        n_samples = int(rng.poisson(data_cfg.mean_samples_per_client))
        n_samples = max(data_cfg.min_samples_per_client, n_samples)

        if partition_mode == "dirichlet":
            proportions = rng.dirichlet(
                np.full(10, dirichlet_alpha, dtype=np.float64)
            )
            counts = rng.multinomial(n_samples, proportions).astype(np.int64)
            client_labels = [int(i) for i in np.flatnonzero(counts > 0)]
        elif partition_mode == "legacy":
            client_labels = _sample_client_labels(
                rng,
                data_cfg.labels_per_client,
                pairing_mode,
            )
            base = n_samples // len(client_labels)
            remainder = n_samples % len(client_labels)
            counts = np.zeros(10, dtype=np.int64)
            for i, label in enumerate(client_labels):
                counts[int(label)] = base + (1 if i < remainder else 0)
        else:
            raise ValueError(f"Unknown partition_mode: {partition_mode!r}")

        client_indices: List[int] = []
        for label in range(10):
            selected = _take_from_pool(
                label=label,
                count=int(counts[label]),
                pools=pools,
                offsets=offsets,
            )
            client_indices.extend(int(v) for v in selected.tolist())

        rng.shuffle(client_indices)
        distance = float(data_cfg.bs_radius_m * math.sqrt(rng.uniform(0.0, 1.0)))
        distance = max(1.0, distance)

        clients[str(client_id)] = {
            "indices": client_indices,
            "labels": client_labels,
            "class_counts": {
                str(label): int(counts[label])
                for label in range(10)
                if int(counts[label]) > 0
            },
            "distance_m": distance,
            "num_examples": len(client_indices),
        }

    all_indices = [
        int(index)
        for client in clients.values()
        for index in client["indices"]
    ]
    if len(all_indices) != len(set(all_indices)):
        raise RuntimeError("Partition contains duplicate training indices across clients")

    payload = {
        "seed": seed,
        "num_clients": data_cfg.num_clients,
        "partition_mode": partition_mode,
        "labels_per_client": data_cfg.labels_per_client,
        "label_pairing_mode": pairing_mode,
        "mean_samples_per_client": data_cfg.mean_samples_per_client,
        "dirichlet_alpha": dirichlet_alpha if partition_mode == "dirichlet" else None,
        "dirichlet_allocation": (
            "client_label_proportions" if partition_mode == "dirichlet" else None
        ),
        "bs_radius_m": data_cfg.bs_radius_m,
        "dataset_train_size": int(len(targets)),
        "total_selected_examples": int(len(all_indices)),
        "unique_selected_examples": int(len(set(all_indices))),
        "clients": clients,
    }

    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return path


'''
pattern = r'def partition_filename\(.*?(?=def load_partition_metadata\()'
m = re.search(pattern, text, re.S)
if not m:
    raise RuntimeError('dataset partition block not found')
text = text[:m.start()] + new_block + text[m.end():]
p.write_text(text)

# ---------- resnet56_gn.py ----------
(ROOT/'resnet56_gn.py').write_text(r'''"""CIFAR ResNet-56 with GroupNorm for federated/Bayesian experiments."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

CIFAR10_RESNET56_GN_PARAMETER_COUNT = 855_770


def _group_count(channels: int) -> int:
    if channels == 16:
        return 4
    if channels in {32, 64}:
        return 8
    raise ValueError(f"Unsupported ResNet-56 channel count: {channels}")


class CIFARResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.gn1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        self.gn2 = nn.GroupNorm(groups, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.GroupNorm(groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.relu(x + identity)


class CIFAR10ResNet56GN(nn.Module):
    """CIFAR ResNet-56 (6n+2, n=9) with GroupNorm instead of BatchNorm.

    Stem: 3x3 Conv 3->16 + GN
    Stage 1: 9 basic blocks, 16 channels
    Stage 2: 9 basic blocks, 32 channels; first block stride 2
    Stage 3: 9 basic blocks, 64 channels; first block stride 2
    Head: global average pooling + Linear(64, 10)

    Projection shortcuts use 1x1 Conv + GroupNorm at stage transitions.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(4, 16)
        self.stage1 = self._make_stage(16, 16, blocks=9, first_stride=1)
        self.stage2 = self._make_stage(16, 32, blocks=9, first_stride=2)
        self.stage3 = self._make_stage(32, 64, blocks=9, first_stride=2)
        self.fc = nn.Linear(64, num_classes)

        actual = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if num_classes == 10 and actual != CIFAR10_RESNET56_GN_PARAMETER_COUNT:
            raise RuntimeError(
                f"ResNet-56-GN parameter count changed: {actual:,} != "
                f"{CIFAR10_RESNET56_GN_PARAMETER_COUNT:,}"
            )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        first_stride: int,
    ) -> nn.Sequential:
        layers = [
            CIFARResNetBasicBlock(in_channels, out_channels, first_stride)
        ]
        for _ in range(1, blocks):
            layers.append(CIFARResNetBasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)
''')

# ---------- cifar10_support.py ----------
p = ROOT/'cifar10_support.py'
text = p.read_text()
if 'CIFAR10_RESNET56_GN_PARAMETER_COUNT' not in text:
    needle = 'CIFAR10_PARAMETER_COUNT = 78_042\n'
    if needle not in text:
        raise RuntimeError('CIFAR10_PARAMETER_COUNT line not found')
    text = text.replace(needle, needle + '''\nfrom resnet56_gn import (\n    CIFAR10ResNet56GN,\n    CIFAR10_RESNET56_GN_PARAMETER_COUNT,\n)\n\nCIFAR10_PARAMETER_COUNTS = {\n    "paper_cnn": CIFAR10_PARAMETER_COUNT,\n    "cifar_residual_cnn": CIFAR10_PARAMETER_COUNT,\n    "resnet56_gn": CIFAR10_RESNET56_GN_PARAMETER_COUNT,\n}\n''', 1)

new_factory = r'''def cifar10_parameter_count(model_name: str = "paper_cnn") -> int:
    model = _cifar_build_model(model_name, 10)
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _cifar_build_model(
    *args: Any,
    **kwargs: Any,
) -> nn.Module:
    name = str(kwargs.get("name", args[0] if len(args) >= 1 else "paper_cnn"))
    name = name.strip().lower()
    num_classes = int(kwargs.get("num_classes", args[1] if len(args) >= 2 else 10))

    if num_classes != 10:
        raise ValueError("CIFAR-10 extension requires num_classes=10")

    if name in {"paper_cnn", "cifar_residual_cnn"}:
        model = CIFAR10ResidualCNN(num_classes=num_classes)
        expected = CIFAR10_PARAMETER_COUNT
    elif name == "resnet56_gn":
        model = CIFAR10ResNet56GN(num_classes=num_classes)
        expected = CIFAR10_RESNET56_GN_PARAMETER_COUNT
    else:
        raise ValueError(
            f"Unsupported CIFAR-10 model: {name!r}; expected "
            "paper_cnn, cifar_residual_cnn, or resnet56_gn"
        )

    count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if count != expected:
        raise RuntimeError(
            f"CIFAR-10 model parameter count changed: {count:,} != {expected:,}"
        )
    return model


'''
pattern = r'def cifar10_parameter_count\(\).*?(?=def install_cifar10_overrides\()'
if not re.search(pattern, text, re.S):
    # maybe already partially modified, use broader signature
    pattern = r'def cifar10_parameter_count\(.*?(?=def install_cifar10_overrides\()'
text = replace_once(text, pattern, new_factory, 'cifar factory block', re.S)

# dynamic assertion in install function
text = text.replace(
    '        if count != CIFAR10_PARAMETER_COUNT:\n            raise AssertionError(\n                f"Expected CIFAR-10 model dimension {CIFAR10_PARAMETER_COUNT:,}, got {count:,}"\n            )\n',
    '        supported = set(CIFAR10_PARAMETER_COUNTS.values())\n        if count not in supported:\n            raise AssertionError(\n                f"Unsupported CIFAR-10 model dimension {count:,}; "\n                f"expected one of {sorted(supported)}"\n            )\n',
    1,
)
p.write_text(text)

# ---------- models.py dynamic Ray assertion ----------
p = ROOT/'models.py'
text = p.read_text()
if 'CIFAR10_PARAMETER_COUNTS as _AIRCOMP_CIFAR_D_BY_MODEL' not in text:
    text = text.replace(
        '        CIFAR10_PARAMETER_COUNT as _AIRCOMP_CIFAR_D,\n',
        '        CIFAR10_PARAMETER_COUNT as _AIRCOMP_CIFAR_D,\n        CIFAR10_PARAMETER_COUNTS as _AIRCOMP_CIFAR_D_BY_MODEL,\n',
        1,
    )
    marker = '    # Replace model factory.\n'
    text = text.replace(
        marker,
        '    _AIRCOMP_CIFAR_D_VALUES = set(_AIRCOMP_CIFAR_D_BY_MODEL.values())\n\n' + marker,
        1,
    )
text = text.replace('        if count != _AIRCOMP_CIFAR_D:\n', '        if count not in _AIRCOMP_CIFAR_D_VALUES:\n', 1)
text = text.replace(
    '                f"Expected CIFAR-10 model dimension "\n                f"{_AIRCOMP_CIFAR_D:,}, got {count:,}"\n',
    '                f"Unsupported CIFAR-10 model dimension {count:,}; "\n                f"expected one of {sorted(_AIRCOMP_CIFAR_D_VALUES)}"\n',
    1,
)
p.write_text(text)

# ---------- main_cifar10.py ----------
(ROOT/'main_cifar10.py').write_text(r'''from __future__ import annotations

"""CIFAR-10 entry point with Ray-safe dataset/model overrides."""

import os

os.environ["AIRCOMP_DATASET"] = "cifar10"

from cifar10_support import install_cifar10_overrides

install_cifar10_overrides()

import main as core_main  # noqa: E402


if __name__ == "__main__":
    print(
        "CIFAR-10 extension active: RGB 32x32. "
        "Model architecture is selected by model.name in the YAML."
    )
    core_main.main()
''')

print('Applied Dirichlet-scarce + ResNet-56-GN source modification.')

# ---------- tests ----------
tests_dir = ROOT / 'tests'
tests_dir.mkdir(parents=True, exist_ok=True)

(tests_dir / 'test_dirichlet_scarce_partition.py').write_text(r'''import json
import numpy as np

import dataset
from config import DataConfig


def test_dirichlet_partition_is_unique_deterministic_and_scarce(tmp_path, monkeypatch):
    targets = np.repeat(np.arange(10, dtype=np.int64), 5000)
    monkeypatch.setattr(dataset, "ensure_mnist", lambda root: None)
    monkeypatch.setattr(dataset, "_training_targets", lambda root: targets)

    cfg = DataConfig(
        root="unused",
        num_clients=40,
        mean_samples_per_client=50,
        min_samples_per_client=1,
        bs_radius_m=200.0,
        partition_mode="dirichlet",
        dirichlet_alpha=0.1,
    )

    p1 = dataset.prepare_partitions(cfg, 12025, tmp_path / "a")
    p2 = dataset.prepare_partitions(cfg, 12025, tmp_path / "b")
    a = json.loads(p1.read_text())
    b = json.loads(p2.read_text())

    assert a == b
    assert a["partition_mode"] == "dirichlet"
    assert a["dirichlet_alpha"] == 0.1

    all_indices = [
        int(index)
        for client in a["clients"].values()
        for index in client["indices"]
    ]
    assert len(all_indices) == len(set(all_indices))
    assert a["total_selected_examples"] == len(all_indices)
    assert a["unique_selected_examples"] == len(all_indices)
    assert 1500 <= len(all_indices) <= 2500

    active_classes = [len(c["labels"]) for c in a["clients"].values()]
    assert min(active_classes) >= 1
    assert float(np.mean(active_classes)) < 6.0

    for client in a["clients"].values():
        assert sum(int(v) for v in client["class_counts"].values()) == int(
            client["num_examples"]
        )
''')

(tests_dir / 'test_resnet56_gn.py').write_text(r'''import pytest
import torch
from torch import nn

from resnet56_gn import (
    CIFAR10ResNet56GN,
    CIFARResNetBasicBlock,
    CIFAR10_RESNET56_GN_PARAMETER_COUNT,
)
from serialization import ParameterLayout


def test_resnet56_gn_structure_and_size():
    model = CIFAR10ResNet56GN()
    assert sum(p.numel() for p in model.parameters()) == 855_770
    assert CIFAR10_RESNET56_GN_PARAMETER_COUNT == 855_770
    assert sum(isinstance(m, CIFARResNetBasicBlock) for m in model.modules()) == 27
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert sum(isinstance(m, nn.GroupNorm) for m in model.modules()) > 0
    y = model(torch.randn(2, 3, 32, 32))
    assert y.shape == (2, 10)


def test_resnet56_gn_bayesian_torch_adapter_dimension():
    adapter_module = pytest.importorskip("bayesian_torch_adapter")
    model = CIFAR10ResNet56GN()
    layout = ParameterLayout(model)
    adapter = adapter_module.BayesianTorchParameterAdapter(model, layout)
    assert layout.total_numel == 855_770
    assert adapter.coordinate_count() == 855_770
    assert adapter.coordinate_count("groupnorm") > 0
''')

# ---------- configs ----------
import copy
import yaml

config_dir = ROOT / 'configs' / 'cifar_priority'
pyro_src = config_dir / 'cifar_selected_e1_lr006_cosine.yaml'
bt_src = config_dir / 'cifar_selected_e1_lr006_cosine_bayesian_torch.yaml'

if not pyro_src.exists():
    raise RuntimeError(f'Missing base config: {pyro_src}')
if not bt_src.exists():
    raise RuntimeError(f'Missing base config: {bt_src}')

for src, dst_name, backend in [
    (pyro_src, 'cifar_dirichlet_a01_resnet56gn_pyro.yaml', 'pyro'),
    (bt_src, 'cifar_dirichlet_a01_resnet56gn_bayesian_torch.yaml', 'bayesian_torch'),
]:
    cfg = yaml.safe_load(src.read_text()) or {}
    cfg = copy.deepcopy(cfg)

    data = cfg.setdefault('data', {})
    data['partition_mode'] = 'dirichlet'
    data['dirichlet_alpha'] = 0.1
    # Preserve the scarce-data budget from the current dense baseline.
    data['mean_samples_per_client'] = 50
    data['min_samples_per_client'] = 1
    # Fixed-label controls are ignored in Dirichlet mode, but keep them valid.
    data['labels_per_client'] = int(data.get('labels_per_client', 1))

    model = cfg.setdefault('model', {})
    model['name'] = 'resnet56_gn'

    training = cfg.setdefault('training', {})
    training['bayesian_backend'] = backend

    sparse = cfg.setdefault('sparse', {})
    sparse['enabled'] = False
    sparse['keep_ratio'] = 1.0

    dst = config_dir / dst_name
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('Created', dst)

print('Created validation tests and Dirichlet/ResNet-56 configs.')
