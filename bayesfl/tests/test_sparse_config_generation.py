import importlib.util
from pathlib import Path
import pytest
from bayesfl.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_sparse_configs', ROOT/'scripts/generate_sparse_configs.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
MNIST_FOLA = ROOT/'scripts/configs/run_mnist_fixed1c_s10_official_fola_n100_seed0_e10_b32_lr001_lam0p01_r150.yaml'
MNIST_AVG = ROOT/'scripts/configs/run_mnist_fixed1c_s10_official_fedavg_n100_seed0_e10_b32_lr001_r150.yaml'
CIFAR_FOLA = ROOT/'scripts/configs/run_cifar_basiccnn_a0p1_official_fola_cos400_n20_f0p5_seed0_e10_b32_lr002_lam0p1_r150.yaml'
CIFAR_AVG = ROOT/'scripts/configs/run_cifar_basiccnn_a0p1_official_fedavg_n20_f0p5_seed0_e10_b32_lr002_r150.yaml'

@pytest.mark.parametrize('fola,avg,policy,d', [(MNIST_FOLA,MNIST_AVG,'strict',545810),
                                               (CIFAR_FOLA,CIFAR_AVG,'floor_for_score',878538)])
def test_generator_preserves_locked_training_and_uses_one_budget(tmp_path,fola,avg,policy,d):
    source = load_config(fola)
    before = [p.read_bytes() for p in (fola,avg)]
    manifest = generator.generate(fola,avg,tmp_path,tag='test',rounds=1000,dense_budget_rounds=150,
                                  keep_ratios=[.5,.1],precision_policy=policy)
    assert len(manifest['runs']) == 8 and manifest['d'] == d
    assert [p.read_bytes() for p in (fola,avg)] == before
    budget = 16*source.federation.clients_per_round*d*150
    sparse_costs = {}
    for run in manifest['runs']:
        cfg=load_config(run['config'])
        assert cfg.communication.max_communication_bytes == budget
        assert cfg.communication.deterministic_client_schedule
        assert cfg.data == source.data and cfg.model == source.model
        expected = load_config(fola if cfg.method=='fola' else avg)
        expected.training.rounds=1000
        assert cfg.training == expected.training
        assert cfg.fola == expected.fola
        if cfg.sparse_enabled:
            sparse_costs.setdefault(cfg.compression.keep_ratio,set()).add(run['round_array_bytes'])
    assert all(len(costs)==1 for costs in sparse_costs.values())
    assert len((tmp_path/'queue.txt').read_text().splitlines())==8
    with pytest.raises(FileExistsError):
        generator.generate(fola,avg,tmp_path,tag='test',precision_policy=policy)


def test_cifar_strict_is_not_silently_floored_or_partially_generated(tmp_path):
    with pytest.raises(ValueError,match='Zero initial omega'):
        generator.generate(CIFAR_FOLA,CIFAR_AVG,tmp_path,tag='test')
    assert not list(tmp_path.glob('*.yaml'))


def test_no_silent_seed_mismatch_or_filename_collision(tmp_path):
    with pytest.raises(ValueError,match='Duplicate'):
        generator.generate(MNIST_FOLA,MNIST_AVG,tmp_path,tag='test',keep_ratios=[.5,.5])
    with pytest.raises(ValueError,match='tag'):
        generator.generate(MNIST_FOLA,MNIST_AVG,tmp_path,tag='../escape')
