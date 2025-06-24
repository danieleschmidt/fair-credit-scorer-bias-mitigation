# Architecture Review

## Dependencies
- scikit-learn
- pandas
- numpy
- matplotlib
- fairlearn
- networkx
- pytest
- pytest-cov

## Module Graph
- **__init__** depends on: architecture_review
- **architecture_review** has no internal dependencies
- **baseline_model** has no internal dependencies
- **bias_mitigator** has no internal dependencies
- **data_loader_preprocessor** has no internal dependencies
- **evaluate_fairness** depends on: baseline_model, bias_mitigator, data_loader_preprocessor, fairness_metrics
- **fairness_metrics** has no internal dependencies
- **run_tests** has no internal dependencies
