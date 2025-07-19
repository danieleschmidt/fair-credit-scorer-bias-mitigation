# Changelog

All notable changes to this project will be documented in this file.

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
