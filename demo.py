#!/usr/bin/env python3
"""Demo: Bias detection and mitigation on a synthetic credit dataset.

Generates 500 samples with 2 protected groups, trains a logistic regression
classifier, measures bias before and after two mitigation strategies, and
prints a full FairnessReport.

Run:
    ~/anaconda3/bin/python3 demo.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fair_credit import BiasDetector, BiasMitigator, FairnessReport


# -----------------------------------------------------------------------
# 1. Synthetic credit dataset (500 samples, 2 groups)
# -----------------------------------------------------------------------

def make_credit_dataset(n: int = 500, seed: int = 42):
    """Generate a synthetic credit scoring dataset with introduced bias.

    Features: income, debt_ratio, payment_history, credit_utilisation
    Protected attribute: group (0 = majority, 1 = minority)
    Label: creditworthy (1) / not creditworthy (0)

    Bias is introduced by shifting the minority group's predicted
    probability downward (disparate impact).
    """
    rng = np.random.RandomState(seed)

    # Protected attribute — 60 / 40 split
    protected = (rng.rand(n) > 0.6).astype(int)

    # Features
    income = rng.normal(50_000, 15_000, n)
    income[protected == 1] -= 10_000          # minority earns less on average
    debt_ratio = rng.uniform(0.1, 0.6, n)
    payment_history = rng.uniform(0, 1, n)
    credit_util = rng.uniform(0.1, 0.8, n)

    X = np.column_stack([income, debt_ratio, payment_history, credit_util])

    # True creditworthiness — driven by financial features
    log_odds = (
        0.00003 * income
        - 2.0 * debt_ratio
        + 3.0 * payment_history
        - 1.5 * credit_util
        - 0.5 * protected   # inject group-based bias
        - 1.0
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > 0.5).astype(int)

    return X, y, protected


X, y, protected = make_credit_dataset(500)
print(f"Dataset: {X.shape[0]} samples | "
      f"positive rate: {y.mean():.2%} | "
      f"group 0: {(protected==0).sum()}  group 1: {(protected==1).sum()}")

# -----------------------------------------------------------------------
# 2. Train/test split & baseline model
# -----------------------------------------------------------------------

(X_tr, X_te, y_tr, y_te,
 prot_tr, prot_te) = train_test_split(
    X, y, protected, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

baseline_clf = LogisticRegression(max_iter=300, random_state=42)
baseline_clf.fit(X_tr_sc, y_tr)
y_pred_baseline = baseline_clf.predict(X_te_sc)
y_prob_baseline = baseline_clf.predict_proba(X_te_sc)[:, 1]

# -----------------------------------------------------------------------
# 3. Measure bias BEFORE mitigation
# -----------------------------------------------------------------------

detector = BiasDetector("group")
results_baseline = detector.detect(
    y_te, y_pred_baseline, prot_te,
    X=X_te_sc, individual_fairness_k=5,
)
print("\n" + detector.summary())

# -----------------------------------------------------------------------
# 4a. Mitigation — Strategy A: Sample reweighting
# -----------------------------------------------------------------------

mitigator = BiasMitigator(strategy="reweight")
weights = mitigator.compute_sample_weights(y_tr, prot_tr)

reweight_clf = LogisticRegression(max_iter=300, random_state=42)
reweight_clf.fit(X_tr_sc, y_tr, sample_weight=weights)
y_pred_rw = reweight_clf.predict(X_te_sc)

detector_rw = BiasDetector("group")
results_rw = detector_rw.detect(
    y_te, y_pred_rw, prot_te,
    X=X_te_sc, individual_fairness_k=5,
)
print("\n" + detector_rw.summary())

# -----------------------------------------------------------------------
# 4b. Mitigation — Strategy B: Threshold adjustment
# -----------------------------------------------------------------------

mitigator_thr = BiasMitigator(
    strategy="threshold", fairness_criterion="demographic_parity"
)
mitigator_thr.fit_thresholds(y_tr, baseline_clf.predict_proba(X_tr_sc)[:, 1], prot_tr)
y_pred_thr = mitigator_thr.predict(y_prob_baseline, prot_te)
print("\nThresholds:", mitigator_thr.get_thresholds())

detector_thr = BiasDetector("group")
results_thr = detector_thr.detect(
    y_te, y_pred_thr, prot_te,
    X=X_te_sc, individual_fairness_k=5,
)
print("\n" + detector_thr.summary())

# -----------------------------------------------------------------------
# 5. Fairness Report
# -----------------------------------------------------------------------

report = FairnessReport(
    model_name="LogisticRegression",
    protected_attribute_name="group",
)
report.add_baseline(results_baseline)
report.add_mitigated("reweight", results_rw)
report.add_mitigated("threshold_adjustment", results_thr)

print("\n" + report.render())

# Save text report
report.save("fairness_audit_report.txt")
print("\nReport saved to fairness_audit_report.txt")

# Save JSON
with open("fairness_audit_report.json", "w") as f:
    f.write(report.to_json())
print("JSON report saved to fairness_audit_report.json")
