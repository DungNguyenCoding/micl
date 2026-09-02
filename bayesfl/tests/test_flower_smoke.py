"""Import-level Flower integration test.

Real five-round simulations are in scripts/run_smoke_*.sh because they download data
and can consume GPU minutes; pytest should stay fast and offline.
"""

import pytest


def test_flower_and_ray_imports():
    pytest.importorskip("flwr")
    pytest.importorskip("ray")
