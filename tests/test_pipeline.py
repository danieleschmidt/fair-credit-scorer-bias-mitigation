import pytest

from src.evaluate_fairness import run_pipeline

@pytest.mark.parametrize('method', ['baseline', 'reweight', 'postprocess'])
def test_run_pipeline_returns_metrics(method):
    results = run_pipeline(method=method, test_size=0.5)
    assert 'accuracy' in results
    assert 'overall' in results
    assert 'by_group' in results
    assert 'false_positive_rate_difference' in results['overall'].index
    assert 'accuracy_difference' in results['overall'].index


def test_run_pipeline_output_json(tmp_path):
    out_file = tmp_path / "metrics.json"
    results = run_pipeline(method="baseline", test_size=0.5, output_path=str(out_file))
    assert out_file.exists()
    import json
    with out_file.open() as f:
        loaded = json.load(f)
    assert results["accuracy"] == loaded["accuracy"]


def test_run_pipeline_random_state_deterministic():
    r1 = run_pipeline(method="baseline", test_size=0.5, random_state=123)
    r2 = run_pipeline(method="baseline", test_size=0.5, random_state=123)
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["overall"].equals(r2["overall"])
