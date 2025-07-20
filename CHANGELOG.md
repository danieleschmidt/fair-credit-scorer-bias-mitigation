# Changelog

All notable changes to this project will be documented in this file.

## [0.1.5] - 2025-07-20

### Added
- Comprehensive module docstrings for all core modules with usage examples and detailed feature descriptions
- Test-driven development approach for docstring quality assurance (tests/test_module_docstrings.py)
- Configuration reset functionality (reset_config()) for improved test isolation
- Pytest configuration fixture for automatic configuration cleanup between tests

### Fixed
- Test isolation issues with configuration singleton pattern
- Test coverage improved to 87% with all 95 tests passing

### Changed
- Enhanced documentation quality across fairness_metrics.py, bias_mitigator.py, baseline_model.py, data_loader_preprocessor.py, evaluate_fairness.py, architecture_review.py, and run_tests.py
- Added tests/conftest.py for better test configuration management

## [0.1.4] - 2025-07-20

### Added
- Centralized configuration management system with YAML support
- Environment variable override support (FAIRNESS_* prefix)
- Configurable model parameters (max_iter, solver)
- Configurable data processing parameters (column names, synthetic data generation)
- Configuration validation with specific error messages
- Hot-reload functionality and custom configuration file support

### Changed
- baseline_model.py now uses configuration defaults for LogisticRegression parameters
- data_loader_preprocessor.py uses configurable paths, column names, and data generation settings
- Added PyYAML>=6.0 dependency for configuration management

### Fixed
- Backward-compatible imports for package and module usage
- Maintained function signature compatibility while adding configuration support

## [0.1.3] - 2025-07-20

### Added
- Comprehensive error handling improvements with specific exception types
- New train_test_split_validated function with input validation
- Enhanced test suite for error handling scenarios (10 new test cases)

### Fixed
- Replaced generic RuntimeError with specific exceptions (ValueError, TypeError, etc.)
- Improved error messages for better debugging and user experience
- Enhanced pandas parsing error handling (EmptyDataError, ParserError)

### Changed
- Error handling in data_loader_preprocessor.py and evaluate_fairness.py now provides specific error types
- Updated BACKLOG.md with WSJF-prioritized tasks for autonomous development

## [0.1.2] - 2025-07-19

### Added
- Impact-ranked backlog with WSJF-based prioritization (BACKLOG.md)
- Autonomous development framework for continuous iteration

### Fixed
- Eliminated all 93 scikit-learn deprecation warnings by specifying liblinear solver
- Ensures compatibility with future scipy versions (1.18.0+)

### Changed
- Updated bias mitigation to use explicit solver configuration
- Improved code quality and future-proofing

## [0.1.1] - 2025-06-27
### Changed
- Compute log-loss using probability scores for accuracy
- Added pre-commit configuration and development requirements
- CLI now logs output with --verbose option
- Version bumped for package release

## [0.1.0] - 2025-06-26


### Added
- Initial implementation of credit scoring pipeline with bias mitigation
- Command line interface and public API
- Comprehensive API usage guide and architecture documentation
- Trade-off analysis discussing fairness vs accuracy

### Changed
- Pinned dependency versions for reproducible installs
- Added automated linting and security checks
