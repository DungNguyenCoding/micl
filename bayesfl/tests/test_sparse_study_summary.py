import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('summarize_sparse_study',ROOT/'scripts/summarize_sparse_study.py')
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_no_phantom_runs_from_a_study_manifest(tmp_path):
    with pytest.raises(FileNotFoundError,match='No actual run'):
        summary.collect(ROOT/'scripts/configs/bayesian_sparse/mnist_fixed_s10',outputs_dir=tmp_path)
