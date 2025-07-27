# ADR-0002: Python as Primary Language

## Status
Accepted

## Context
The Fair Credit Scorer project requires a programming language that excels in:
- Machine learning and data science capabilities
- Bias mitigation and fairness libraries
- Statistical analysis and mathematical computations
- Rapid prototyping and development
- Strong ecosystem for scientific computing

Alternatives considered:
- R: Strong statistical capabilities but limited general-purpose programming
- Java/Scala: Robust ecosystem but steeper learning curve for ML
- JavaScript/TypeScript: Web-native but limited ML/statistical libraries

## Decision
We will use Python as the primary programming language for the Fair Credit Scorer project.

Rationale:
- Exceptional ML/AI ecosystem (scikit-learn, pandas, numpy)
- Dedicated fairness libraries (Fairlearn, AIF360)
- Large community and extensive documentation
- Rapid development and prototyping capabilities
- Strong testing frameworks (pytest)
- Excellent package management (pip, conda)

## Consequences

### Positive
- Access to comprehensive ML and fairness libraries
- Fast development cycles and easy prototyping
- Large talent pool familiar with Python
- Strong community support and documentation
- Excellent tooling for data science workflows
- Easy integration with Jupyter notebooks for exploration

### Negative
- Performance overhead compared to compiled languages
- Global Interpreter Lock (GIL) limits true parallelism
- Dynamic typing can lead to runtime errors
- Package dependency management can be complex

### Neutral
- Code will be interpreted rather than compiled
- Development environment requires Python runtime
- Package versions need careful management

## Related Decisions
- ADR-0003: Choice of Fairlearn library builds on Python ecosystem
- ADR-0005: Pytest testing framework aligns with Python best practices

## Notes
Python 3.8+ required for modern features and security updates. Virtual environments strongly recommended for dependency isolation.