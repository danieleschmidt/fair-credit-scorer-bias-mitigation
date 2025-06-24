import importlib


def test_modules_documented():
    assert importlib.util.find_spec("src.data_loader_preprocessor") is not None
    assert importlib.util.find_spec("src.baseline_model") is not None
