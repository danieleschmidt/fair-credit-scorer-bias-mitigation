from fair_credit_scorer_bias_mitigation import run_pipeline


def test_includes_fairness_metrics():
    results = run_pipeline(method="baseline", test_size=0.5)
    assert "equalized_odds_difference" in results["overall"].index
