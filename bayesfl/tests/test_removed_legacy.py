from pathlib import Path


def test_wireless_pyro_two_phase_sources_are_absent():
    root = Path(__file__).resolve().parents[1]
    for name in [
        "aircomp.py",
        "wireless.py",
        "bayesian_protocol.py",
        "bayes_vi.py",
        "sparse_posterior.py",
        "cifar10_support.py",
    ]:
        assert not (root / name).exists(), name
