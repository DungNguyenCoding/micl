from logger import CLIENT_FIELDS, METRIC_FIELDS


def test_phase_and_aircomp_diagnostics_are_logged():
    required_metrics = {
        "logical_round",
        "physical_round",
        "phase",
        "phase1_train_loss",
        "phase2_train_loss",
        "path_loss_reference_m",
        "deterministic_reference_power_mode",
        "aircomp_retained_magnitude_ratio",
        "aircomp_distorted_to_ideal_norm_ratio",
        "precision_aircomp_nmse",
        "mean_aircomp_nmse",
        "posterior_precision_mean",
    }
    assert required_metrics.issubset(set(METRIC_FIELDS))

    required_clients = {
        "logical_round",
        "physical_round",
        "phase",
        "local_precision_mean",
        "local_precision_delta_l2",
        "local_precision_gradient_l2_mean",
        "local_nu_l2",
        "local_implied_mean_l2",
    }
    assert required_clients.issubset(set(CLIENT_FIELDS))
