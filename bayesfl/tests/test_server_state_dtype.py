import numpy as np

from serialization import normalize_server_state_dtypes


def test_proposed_server_state_preserves_sub_float32_precision_offsets():
    mean = np.zeros(4, dtype=np.float64)
    precision = np.asarray([400.0 + 1.0e-7, 400.0 - 2.0e-7], dtype=np.float64)
    normalized = normalize_server_state_dtypes("proposed", [mean, precision])
    assert normalized[0].dtype == np.float32
    assert normalized[1].dtype == np.float64
    np.testing.assert_array_equal(normalized[1], precision)
    assert np.any(normalized[1] != 400.0)


def test_deterministic_server_state_remains_float32():
    model = np.asarray([1.0, 2.0], dtype=np.float64)
    normalized = normalize_server_state_dtypes("fedavg", [model])
    assert normalized[0].dtype == np.float32
