from logger import METRIC_FIELDS


def test_aircomp_diagnostics_are_logged():
    required = {
        "path_loss_reference_m",
        "aircomp_retained_magnitude_ratio",
        "aircomp_distorted_to_ideal_norm_ratio",
        "aircomp_ideal_l2",
        "aircomp_received_l2",
        "aircomp_delta_bar",
    }
    assert required.issubset(set(METRIC_FIELDS))
