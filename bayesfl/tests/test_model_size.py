from models import build_model, count_parameters


def test_paper_cnn_has_disclosed_dimension():
    assert count_parameters(build_model()) == 62_346
