import pytest

from src.evaluate_fairness import run_pipeline, run_cross_validation

@pytest.mark.parametrize('method', ['baseline', 'reweight', 'postprocess', 'expgrad'])
def test_run_pipeline_returns_metrics(method):
    results = run_pipeline(method=method, test_size=0.5)
    assert 'accuracy' in results
    assert 'overall' in results
    assert 'by_group' in results
    assert 'false_positive_rate_difference' in results['overall'].index
    assert 'accuracy_difference' in results['overall'].index
    assert 'balanced_accuracy_difference' in results['overall'].index
    assert 'precision_difference' in results['overall'].index
    assert 'recall_difference' in results['overall'].index
    assert 'f1_difference' in results['overall'].index
    assert 'log_loss_difference' in results['overall'].index
    assert 'roc_auc_difference' in results['overall'].index
    assert 'false_positive_rate' in results['overall'].index
    assert 'false_negative_rate' in results['overall'].index
    assert 'false_discovery_rate_difference' in results['overall'].index
    assert 'true_positive_rate_difference' in results['overall'].index
    assert 'true_negative_rate_difference' in results['overall'].index
    assert 'log_loss_difference' in results['overall'].index
    assert 'demographic_parity_ratio' in results['overall'].index
    assert 'equalized_odds_ratio' in results['overall'].index
    assert 'false_positive_rate_ratio' in results['overall'].index
    assert 'false_negative_rate_ratio' in results['overall'].index
    assert 'true_positive_rate_ratio' in results['overall'].index
    assert 'true_negative_rate_ratio' in results['overall'].index
    assert 'accuracy_ratio' in results['overall'].index
    assert 'false_positive_rate_ratio' in results['overall'].index
    assert 'false_negative_rate_ratio' in results['overall'].index
    assert 'true_positive_rate_ratio' in results['overall'].index
    assert 'true_negative_rate_ratio' in results['overall'].index
    assert 'accuracy_ratio' in results['overall'].index


def test_run_pipeline_output_json(tmp_path):
    out_file = tmp_path / "metrics.json"
    results = run_pipeline(method="baseline", test_size=0.5, output_path=str(out_file))
    assert out_file.exists()
    import json
    with out_file.open() as f:
        loaded = json.load(f)
    assert results["accuracy"] == loaded["accuracy"]
    assert "overall" in loaded
    assert "by_group" in loaded
    assert "true_positive_rate_difference" in loaded["overall"]
    assert "true_negative_rate_difference" in loaded["overall"]
    assert "log_loss_difference" in loaded["overall"]
    assert "demographic_parity_ratio" in loaded["overall"]
    assert "equalized_odds_ratio" in loaded["overall"]
    assert "false_positive_rate_ratio" in loaded["overall"]
    assert "false_negative_rate_ratio" in loaded["overall"]
    assert "true_positive_rate_ratio" in loaded["overall"]
    assert "true_negative_rate_ratio" in loaded["overall"]
    assert "accuracy_ratio" in loaded["overall"]


def test_run_pipeline_random_state_deterministic():
    r1 = run_pipeline(method="baseline", test_size=0.5, random_state=123)
    r2 = run_pipeline(method="baseline", test_size=0.5, random_state=123)
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["overall"].equals(r2["overall"])


def test_run_pipeline_threshold():
    r_default = run_pipeline(method="baseline", test_size=0.5, random_state=0)
    r_thresh = run_pipeline(
        method="baseline", test_size=0.5, random_state=0, threshold=0.3
    )
    # Expect accuracy to differ when using a custom threshold
    assert r_default["accuracy"] != r_thresh["accuracy"]


@pytest.mark.parametrize('method', ['baseline', 'reweight', 'postprocess', 'expgrad'])
def test_run_cross_validation_returns_metrics(method):
    results = run_cross_validation(method=method, cv=3)
    assert 'accuracy' in results
    assert 'overall' in results
    assert 'by_group' in results
    assert 'overall_std' in results
    assert 'by_group_std' in results
    assert 'folds' in results
    assert len(results['folds']) == 3
    for fold in results['folds']:
        assert 'accuracy' in fold
        assert 'overall' in fold
        assert 'by_group' in fold
    assert 'false_positive_rate_difference' in results['overall'].index
    assert 'false_positive_rate' in results['overall'].index
    assert 'false_negative_rate' in results['overall'].index
    assert 'false_discovery_rate_difference' in results['overall'].index
    assert 'accuracy_difference' in results['overall'].index
    assert 'true_positive_rate_difference' in results['overall'].index
    assert 'true_negative_rate_difference' in results['overall'].index
    assert 'demographic_parity_ratio' in results['overall'].index
    assert 'equalized_odds_ratio' in results['overall'].index


def test_run_cross_validation_output_json(tmp_path):
    out_file = tmp_path / "cv_metrics.json"
    results = run_cross_validation(method="baseline", cv=2, output_path=str(out_file))
    assert out_file.exists()
    import json
    with out_file.open() as f:
        loaded = json.load(f)
    assert results["accuracy"] == loaded["accuracy"]
    assert "overall" in loaded
    assert "by_group" in loaded
    assert "overall_std" in loaded
    assert "by_group_std" in loaded
    assert "folds" in loaded
    assert len(loaded["folds"]) == 2
    for fold in loaded["folds"]:
        assert "accuracy" in fold
        assert "overall" in fold
        assert "by_group" in fold
    assert "true_positive_rate_difference" in loaded["overall"]
    assert "true_negative_rate_difference" in loaded["overall"]
    assert "false_positive_rate" in loaded["overall"]
    assert "false_negative_rate" in loaded["overall"]
    assert "false_discovery_rate_difference" in loaded["overall"]
    assert "log_loss_difference" in loaded["overall"]
    assert "demographic_parity_ratio" in loaded["overall"]
    assert "equalized_odds_ratio" in loaded["overall"]
    assert "false_positive_rate_ratio" in loaded["overall"]
    assert "false_negative_rate_ratio" in loaded["overall"]
    assert "true_positive_rate_ratio" in loaded["overall"]
    assert "true_negative_rate_ratio" in loaded["overall"]
    assert "accuracy_ratio" in loaded["overall"]


def test_run_pipeline_custom_data_path(tmp_path):
    data_file = tmp_path / "alt.csv"
    run_pipeline(method="baseline", test_size=0.5, data_path=str(data_file))
    assert data_file.exists()


def test_run_cross_validation_custom_data_path(tmp_path):
    data_file = tmp_path / "alt_cv.csv"
    run_cross_validation(method="baseline", cv=2, data_path=str(data_file))
    assert data_file.exists()


def test_run_cross_validation_threshold():
    res_default = run_cross_validation(method="baseline", cv=2, random_state=0)
    res_thresh = run_cross_validation(
        method="baseline", cv=2, random_state=0, threshold=0.3
    )
    assert res_default["accuracy"] != res_thresh["accuracy"]
