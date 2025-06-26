# Fairness and Accuracy Trade-offs

Bias mitigation methods often reduce model performance in exchange for fairer
predictions. The experiments in this repository illustrate this effect:

- **Baseline** logistic regression achieves around 0.83 accuracy but shows
a notable gap in false positive rates between groups.
- **Sample reweighting** lowers the false positive rate difference from
  roughly 0.28 to 0.21 while accuracy drops slightly to about 0.79.
- **Post-processing** via equalized odds produces a similar reduction in
disparity but may further decrease overall accuracy depending on the
selected threshold.

Choosing a technique requires balancing fairness goals against the cost of
reduced predictive power. Cross-validation results can quantify this
tension by comparing mean accuracy and fairness metrics across folds.
