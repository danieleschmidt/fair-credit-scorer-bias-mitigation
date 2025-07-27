# ADR-0003: Bias Mitigation Approach

## Status

Accepted

Date: 2025-01-27

## Context

Credit scoring models can exhibit bias against protected groups, leading to unfair lending decisions. This is both an ethical concern and a regulatory compliance issue. The project needs to implement bias mitigation techniques that:

- Reduce disparate impact while maintaining model performance
- Comply with fair lending regulations (ECOA, Fair Housing Act)
- Provide transparent and explainable results
- Support multiple mitigation strategies for comparison
- Allow for trade-off analysis between fairness and accuracy

The choice of mitigation approach affects model architecture, training pipeline, and regulatory defensibility.

## Decision

We will implement a **multi-strategy approach** supporting three main bias mitigation categories:

1. **Preprocessing**: Sample reweighting to balance representation
2. **In-processing**: Exponentiated Gradient (reduction approach)
3. **Post-processing**: Threshold optimization for equalized odds

This allows comparison of different approaches and selection based on specific use case requirements.

## Alternatives Considered

### Alternative 1: Single Post-processing Approach
- **Description**: Only implement threshold optimization after training
- **Pros**: 
  - Simple to implement
  - No changes to training process
  - Easy to explain to stakeholders
- **Cons**: 
  - Limited effectiveness for some bias types
  - May not address root causes
  - Less flexibility for different scenarios
- **Reason for rejection**: Insufficient for comprehensive bias mitigation

### Alternative 2: Adversarial Debiasing Only
- **Description**: Use adversarial training to remove bias signals
- **Pros**: 
  - Theoretically elegant approach
  - Can address complex bias patterns
  - End-to-end trainable
- **Cons**: 
  - Complex implementation and tuning
  - Less interpretable results
  - Training instability risks
  - Not supported by Fairlearn
- **Reason for rejection**: Too complex for initial implementation

### Alternative 3: Preprocessing Only (Data Augmentation)
- **Description**: Only use data-based approaches like SMOTE variants
- **Pros**: 
  - Preserves standard ML pipeline
  - Easy to understand and validate
  - Regulatory-friendly approach
- **Cons**: 
  - May not address all bias sources
  - Limited effectiveness for representation bias
  - Can introduce artificial patterns
- **Reason for rejection**: Insufficient coverage of bias mitigation spectrum

## Consequences

### Positive
- Comprehensive bias mitigation coverage across the ML pipeline
- Ability to compare effectiveness of different approaches
- Flexibility to choose optimal strategy per use case
- Regulatory compliance through multiple documented approaches
- Educational value for understanding bias mitigation trade-offs

### Negative
- Increased implementation complexity
- More testing and validation required
- Longer development timeline
- More complex user interface and documentation

### Neutral
- Code modularity allows selective use of techniques
- Performance comparisons provide insights for future improvements
- Establishes foundation for advanced techniques

## Implementation

1. **Preprocessing Module** (`bias_mitigator.py`):
   - Sample reweighting based on group membership
   - Integration with data loading pipeline

2. **In-processing Module**:
   - Exponentiated Gradient implementation via Fairlearn
   - Constraint-based optimization for fairness metrics

3. **Post-processing Module**:
   - Threshold optimization for equalized odds
   - Calibration-aware threshold selection

4. **Evaluation Framework**:
   - Unified metrics collection across all approaches
   - Trade-off visualization and reporting

5. **CLI Integration**:
   - Method selection via command line parameters
   - Batch comparison mode for all approaches

## References

- [A Reductions Approach to Fair Classification (Agarwal et al.)](https://arxiv.org/abs/1803.02453)
- [Equality of Opportunity in Supervised Learning (Hardt et al.)](https://arxiv.org/abs/1610.02413)
- [Fairness Through Awareness (Dwork et al.)](https://arxiv.org/abs/1104.3913)
- [Fairlearn User Guide](https://fairlearn.org/v0.8.0/user_guide/index.html)

## Notes

This multi-strategy approach provides a strong foundation for bias mitigation research and practical deployment. Individual strategies can be enhanced or replaced as new techniques become available in the fairness ML community.