import torch

from config import SimulationConfig
from runtime_utils import resolve_backend, should_pin_memory


def test_cpu_loader_never_pins_memory():
    assert should_pin_memory(True, torch.device("cpu")) is False


def test_explicit_local_backend_is_respected():
    cfg = SimulationConfig()
    cfg.runtime.backend = "local"
    assert resolve_backend(cfg) == "local"


def test_explicit_ray_backend_is_respected():
    cfg = SimulationConfig()
    cfg.runtime.backend = "ray"
    assert resolve_backend(cfg) == "ray"


def test_auto_uses_local_for_native_windows_cuda(monkeypatch):
    cfg = SimulationConfig()
    cfg.runtime.backend = "auto"
    cfg.runtime.client_device = "cuda"
    monkeypatch.setattr("runtime_utils.platform.system", lambda: "Windows")
    assert resolve_backend(cfg) == "local"
