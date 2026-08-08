from logger import METRIC_FIELDS


def test_v142_deterministic_update_diagnostics_are_logged():
    for name in (
        "global_model_update_l2",
        "ideal_model_update_l2",
        "received_model_update_l2",
    ):
        assert name in METRIC_FIELDS
