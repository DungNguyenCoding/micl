import torch

from runtime_utils import should_pin_memory


def test_cpu_server_never_pins_cuda_memory():
    assert should_pin_memory(True, torch.device("cpu")) is False
    assert should_pin_memory(False, torch.device("cpu")) is False
