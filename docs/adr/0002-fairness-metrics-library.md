# ADR-0002: Fairness Metrics Library Choice

## Status

Accepted

Date: 2025-01-27

## Context

The project requires a comprehensive fairness metrics library to evaluate bias in credit scoring models. The choice of library significantly impacts:

- Available fairness metrics and definitions
- Bias mitigation techniques
- Integration with existing ML pipeline (scikit-learn)
- Long-term maintenance and community support
- Regulatory compliance capabilities

Key requirements:
- Support for classification fairness metrics (demographic parity, equalized odds, etc.)
- Integration with scikit-learn pipelines
- Active development and maintenance
- Good documentation and examples
- Support for post-processing mitigation techniques

## Decision

We will use **Fairlearn** as the primary fairness metrics and bias mitigation library.

Fairlearn provides:
- Comprehensive set of fairness metrics aligned with academic research
- Native scikit-learn integration
- Multiple mitigation approaches (preprocessing, in-processing, post-processing)
- Active Microsoft-backed development
- Strong documentation and community support

## Alternatives Considered

### Alternative 1: IBM AIF360 (AI Fairness 360)
- **Description**: Comprehensive fairness toolkit from IBM Research
- **Pros**: 
  - Extensive fairness metrics catalog (70+ metrics)
  - Multiple mitigation algorithms
  - Research-backed implementations
  - Support for multiple ML frameworks
- **Cons**: 
  - Heavier dependency footprint
  - More complex API surface
  - Less seamless scikit-learn integration
  - Potential corporate lock-in concerns
- **Reason for rejection**: Complexity overhead outweighs benefits for our use case

### Alternative 2: Custom Implementation
- **Description**: Implement fairness metrics from scratch
- **Pros**: 
  - Full control over implementations
  - Minimal dependencies
  - Custom optimizations possible
- **Cons**: 
  - High development and maintenance cost
  - Risk of implementation errors
  - Limited peer review
  - Reinventing well-established solutions
- **Reason for rejection**: Not cost-effective given existing quality solutions

### Alternative 3: TensorFlow Fairness Indicators
- **Description**: Google's fairness evaluation library
- **Pros**: 
  - Well-tested at scale
  - Integration with TensorFlow ecosystem
  - Strong visualization capabilities
- **Cons**: 
  - TensorFlow dependency overhead
  - Limited mitigation techniques
  - Less suitable for scikit-learn workflows
- **Reason for rejection**: Architectural mismatch with our scikit-learn-based approach

## Consequences

### Positive
- Seamless integration with existing scikit-learn pipeline
- Access to well-tested fairness metrics implementations
- Strong community support and documentation
- Regular updates and maintenance from Microsoft
- Simplified dependency management

### Negative
- Dependency on Microsoft-maintained library
- Fewer total metrics compared to AIF360
- Limited to fairness approaches supported by Fairlearn

### Neutral
- Standard MIT license allows commercial use
- Active GitHub community for issue resolution
- Compatible with our Python 3.8+ requirement

## Implementation

1. Add `fairlearn==0.12.0` to project dependencies
2. Update existing fairness metrics implementations to use Fairlearn
3. Integrate Fairlearn's mitigation techniques (ExponentiatedGradient, etc.)
4. Update documentation to reference Fairlearn concepts and terminology
5. Add Fairlearn citation to project acknowledgments

## References

- [Fairlearn Documentation](https://fairlearn.org/)
- [Fairlearn GitHub Repository](https://github.com/fairlearn/fairlearn)
- [AIF360 Comparison Study](https://arxiv.org/abs/1810.01943)
- [Fairness Definitions Explained (Verma & Rubin)](https://fairware.cs.umass.edu/papers/Verma.pdf)

## Notes

This decision can be revisited if Fairlearn's development stagnates or if project requirements expand beyond Fairlearn's capabilities. The modular architecture allows for future library migration if needed.