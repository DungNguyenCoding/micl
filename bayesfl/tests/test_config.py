from pathlib import Path

from bayesfl.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_all_configs_load():
    for path in (ROOT / "scripts/configs").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.training.rounds >= 1
        assert cfg.federation.clients_per_round <= cfg.federation.num_clients
