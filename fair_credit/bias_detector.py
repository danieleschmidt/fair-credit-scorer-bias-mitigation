"""BiasDetector: measure fairness metrics across protected attributes.

Supported metrics:
  - Demographic Parity Difference / Ratio
  - Equalized Odds Difference (TPR gap + FPR gap)
  - Equal Opportunity Difference (TPR gap only)
  - Individual Fairness (consistency score)
  - Per-group accuracy, precision, recall, F1

All computations use only numpy and scikit-learn — no exotic deps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors


class BiasDetector:
    """Compute fairness metrics for a binary classifier.

    Parameters
    ----------
    protected_attribute_name : str
        Human-readable label for the protected attribute (used in reports).

    Examples
    --------
    >>> detector = BiasDetector("gender")
    >>> results = detector.detect(y_true, y_pred, protected)
    >>> print(results["demographic_parity_difference"])
    """

    def __init__(self, protected_attribute_name: str = "protected_attribute") -> None:
        self.protected_attribute_name = protected_attribute_name
        self._results: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        y_true: Sequence,
        y_pred: Sequence,
        protected: Sequence,
        *,
        X: Optional[np.ndarray] = None,
        individual_fairness_k: int = 5,
    ) -> Dict[str, Any]:
        """Compute all fairness metrics.

        Parameters
        ----------
        y_true : array-like of 0/1
            Ground-truth labels.
        y_pred : array-like of 0/1
            Model predictions.
        protected : array-like
            Protected attribute values (binary: 0 / 1).
        X : ndarray, optional
            Feature matrix for individual fairness computation.
            If None, individual fairness is skipped.
        individual_fairness_k : int
            Number of nearest neighbours for individual fairness.

        Returns
        -------
        dict
            Nested dict with keys:
            ``demographic_parity_difference``, ``demographic_parity_ratio``,
            ``equalized_odds_difference``, ``equal_opportunity_difference``,
            ``individual_fairness_score`` (if X provided),
            ``per_group`` (dict of per-group metrics).
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        protected = np.asarray(protected)

        groups = np.unique(protected)
        if len(groups) < 2:
            raise ValueError("Need at least 2 distinct groups in `protected`.")

        per_group = self._per_group_metrics(y_true, y_pred, protected, groups)

        results: Dict[str, Any] = {}
        results["groups"] = groups.tolist()
        results["per_group"] = per_group

        # --- Demographic Parity ---
        sel_rates = np.array([per_group[g]["selection_rate"] for g in groups])
        results["demographic_parity_difference"] = float(sel_rates.max() - sel_rates.min())
        min_rate = sel_rates.min()
        results["demographic_parity_ratio"] = (
            float(min_rate / sel_rates.max()) if sel_rates.max() > 0 else 1.0
        )

        # --- Equalized Odds (max gap across TPR and FPR) ---
        tpr_vals = np.array([per_group[g]["tpr"] for g in groups])
        fpr_vals = np.array([per_group[g]["fpr"] for g in groups])
        tpr_diff = float(tpr_vals.max() - tpr_vals.min())
        fpr_diff = float(fpr_vals.max() - fpr_vals.min())
        results["equalized_odds_difference"] = max(tpr_diff, fpr_diff)
        results["tpr_difference"] = tpr_diff
        results["fpr_difference"] = fpr_diff

        # --- Equal Opportunity (TPR gap only) ---
        results["equal_opportunity_difference"] = tpr_diff

        # --- Individual Fairness ---
        if X is not None:
            results["individual_fairness_score"] = self._individual_fairness(
                X, y_pred, k=individual_fairness_k
            )

        # --- Overall accuracy ---
        results["overall_accuracy"] = float(accuracy_score(y_true, y_pred))

        self._results = results
        return results

    def summary(self) -> str:
        """Return a human-readable summary of the last ``detect()`` call."""
        if self._results is None:
            return "No results yet — call detect() first."
        r = self._results
        lines = [
            f"=== Fairness Metrics [{self.protected_attribute_name}] ===",
            f"Overall accuracy              : {r['overall_accuracy']:.4f}",
            f"Demographic parity difference : {r['demographic_parity_difference']:.4f}",
            f"Demographic parity ratio      : {r['demographic_parity_ratio']:.4f}",
            f"Equalized odds difference     : {r['equalized_odds_difference']:.4f}",
            f"  TPR difference              : {r['tpr_difference']:.4f}",
            f"  FPR difference              : {r['fpr_difference']:.4f}",
            f"Equal opportunity difference  : {r['equal_opportunity_difference']:.4f}",
        ]
        if "individual_fairness_score" in r:
            lines.append(f"Individual fairness score     : {r['individual_fairness_score']:.4f}")
        lines.append("")
        lines.append("Per-group metrics:")
        for g, m in r["per_group"].items():
            lines.append(
                f"  Group {g}: acc={m['accuracy']:.3f}  prec={m['precision']:.3f}"
                f"  rec={m['recall']:.3f}  f1={m['f1']:.3f}"
                f"  sel_rate={m['selection_rate']:.3f}"
                f"  TPR={m['tpr']:.3f}  FPR={m['fpr']:.3f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _per_group_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected: np.ndarray,
        groups: np.ndarray,
    ) -> Dict[Any, Dict[str, float]]:
        metrics: Dict[Any, Dict[str, float]] = {}
        for g in groups:
            mask = protected == g
            yt, yp = y_true[mask], y_pred[mask]
            if len(yt) == 0:
                continue

            # Confusion-matrix-derived rates
            tn, fp, fn, tp = _safe_confusion(yt, yp)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            metrics[g] = {
                "n": int(mask.sum()),
                "selection_rate": float(yp.mean()),
                "accuracy": float(accuracy_score(yt, yp)),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "tpr": tpr,
                "fpr": fpr,
            }
        return metrics

    @staticmethod
    def _individual_fairness(
        X: np.ndarray, y_pred: np.ndarray, k: int = 5
    ) -> float:
        """Consistency score: fraction of k-NN pairs that agree on prediction.

        Score of 1.0 = perfectly individually fair (similar individuals get
        the same prediction); lower values indicate more inconsistency.
        """
        X = np.asarray(X, dtype=float)
        y_pred = np.asarray(y_pred)
        k = min(k, len(X) - 1)
        if k <= 0:
            return 1.0

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        agreements = []
        for i, neighbours in enumerate(indices):
            neighbours = neighbours[neighbours != i][:k]  # exclude self
            agree = np.mean(y_pred[neighbours] == y_pred[i])
            agreements.append(agree)

        return float(np.mean(agreements))


def _safe_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (tn, fp, fn, tp) handling degenerate cases."""
    classes = np.array([0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)
