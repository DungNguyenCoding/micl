from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_precision_path_uses_float64_master_and_transport():
    bayes = (ROOT / "bayes_vi.py").read_text(encoding="utf-8")
    aggregation = (ROOT / "aggregation.py").read_text(encoding="utf-8")
    client = (ROOT / "client.py").read_text(encoding="utf-8")
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    assert "global_precision, dtype=torch.float64" in bayes
    assert "precision=precision.cpu().numpy().astype(np.float64)" in bayes
    assert "output_dtype=np.float64" in aggregation
    assert "dtype=np.float64" in client
    assert "aggregation.parameters[0].astype(np.float64)" in server
