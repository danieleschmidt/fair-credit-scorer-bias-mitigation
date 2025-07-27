# Fair Credit Scorer - Bias Mitigation

Welcome to the Fair Credit Scorer documentation! This project provides a comprehensive framework for building fair credit scoring models with bias mitigation capabilities, along with enterprise-grade DevSecOps automation.

## 🎯 What is Fair Credit Scorer?

Fair Credit Scorer is a Python package that helps you:

- **Build Fair Models**: Create credit scoring models that are both accurate and fair
- **Mitigate Bias**: Apply various bias mitigation techniques to reduce demographic disparities
- **Measure Fairness**: Compute comprehensive fairness metrics to evaluate model performance
- **Automate DevSecOps**: Benefit from automated repository hygiene and security practices

## 🚀 Key Features

### Fair Credit Scoring
- **Multiple Mitigation Methods**: Baseline, re-weighting, post-processing, and adversarial approaches
- **Comprehensive Metrics**: 20+ fairness and performance metrics
- **Cross-Validation Support**: Robust evaluation with statistical significance testing
- **CLI Interface**: Easy-to-use command-line interface for experiments

### DevSecOps Automation
- **Repository Hygiene Bot**: Automated security and compliance management
- **CI/CD Pipeline**: Comprehensive testing, security scanning, and deployment
- **Monitoring**: Built-in health checks, metrics, and alerting
- **Container Support**: Docker and container orchestration ready

## 📊 Quick Example

```python
from fair_credit_scorer import run_pipeline

# Run fairness evaluation with bias mitigation
results = run_pipeline(
    method="reweight",
    data_path="credit_data.csv",
    test_size=0.3,
    cv=5
)

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Bias: {results['demographic_parity_difference']:.3f}")
```

## 🛠 Installation

```bash
# Install from PyPI (coming soon)
pip install fair-credit-scorer-bias-mitigation

# Or install from source
git clone https://github.com/username/fair-credit-scorer-bias-mitigation.git
cd fair-credit-scorer-bias-mitigation
pip install -e .
```

## 📚 Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and basic setup
- **[User Guide](user-guide/basic-usage.md)**: Comprehensive usage documentation
- **[Development](development/contributing.md)**: Information for contributors
- **[DevSecOps](devsecops/repo-hygiene-bot.md)**: Automation and operations
- **[Reference](reference/api.md)**: API documentation and configuration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/username/fair-credit-scorer-bias-mitigation/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- [Fairlearn](https://fairlearn.org/) for fairness metrics and algorithms
- [scikit-learn](https://scikit-learn.org/) for machine learning foundations
- The fair ML research community for bias mitigation techniques

---

!!! tip "Quick Start"
    New to Fair Credit Scorer? Start with our [Quick Start Guide](getting-started/quickstart.md) to get up and running in minutes!