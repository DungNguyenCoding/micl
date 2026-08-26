from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_precision_is_optimized_in_rho_coordinate():
    source = (ROOT / "bayes_vi.py").read_text(encoding="utf-8")
    assert "__log_precision" not in source
    assert "rho = torch.nn.Parameter(initial_rho.clone())" in source
    assert "dtype=torch.float64" in source
    assert "elbo.differentiable_loss" in source


def test_phase2_receives_round_start_and_next_precision():
    client_source = (ROOT / "client.py").read_text(encoding="utf-8")
    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "prior_global_precision=prior_global_precision" in client_source
    assert "round_start_precision" in server_source
    assert "[mu_t, rho_{t+1}, rho_t]" in server_source
