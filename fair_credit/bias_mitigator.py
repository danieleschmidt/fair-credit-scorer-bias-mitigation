"""BiasMitigator: two strategies to reduce bias in binary classifiers.

Strategy A — Reweighting (pre-processing)
    Compute sample weights so that each (group × label) cell has equal
    representation.  Pass the weights to ``model.fit(..., sample_weight=w)``.

Strategy B — Threshold Adjustment (post-processing)
    Find per-group classification thresholds that equalise the positive-
    prediction rate (demographic parity) or the TPR (equal opportunity).

Both strategies require only numpy/scikit-learn.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BiasMitigator:
    """Bias mitigation strategies for binary classifiers.

    Parameters
    ----------
    strategy : {'reweight', 'threshold', 'both'}
        Which strategy to use.  'both' computes weights AND thresholds.
    fairness_criterion : {'demographic_parity', 'equal_opportunity'}
        Criterion used for threshold adjustment.
    """

    VALID_STRATEGIES = {"reweight", "threshold", "both"}
    VALID_CRITERIA = {"demographic_parity", "equal_opportunity"}

    def __init__(
        self,
        strategy: str = "both",
        fairness_criterion: str = "demographic_parity",
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self.VALID_STRATEGIES}")
        if fairness_criterion not in self.VALID_CRITERIA:
            raise ValueError(f"fairness_criterion must be one of {self.VALID_CRITERIA}")
        self.strategy = strategy
        self.fairness_criterion = fairness_criterion

        # Fitted state
        self._sample_weights: Optional[np.ndarray] = None
        self._thresholds: Optional[Dict[Any, float]] = None
        self._groups: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Strategy A: Reweighting
    # ------------------------------------------------------------------

    def compute_sample_weights(
        self,
        y: Sequence,
        protected: Sequence,
    ) -> np.ndarray:
        """Compute sample weights that balance label distribution across groups.

        Each (group, label) cell gets weight ``P(group) * P(label) / P(group, label)``,
        so that weighted marginal distributions are independent.

        Parameters
        ----------
        y : array-like of 0/1
            Training labels.
        protected : array-like
            Protected attribute values.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Sample weights (sum to n_samples).
        """
        y = np.asarray(y)
        protected = np.asarray(protected)
        n = len(y)

        groups = np.unique(protected)
        labels = np.unique(y)

        weights = np.ones(n, dtype=float)

        p_group = {g: np.mean(protected == g) for g in groups}
        p_label = {l: np.mean(y == l) for l in labels}

        for g in groups:
            for l in labels:
                mask = (protected == g) & (y == l)
                p_joint = mask.mean()
                if p_joint > 0:
                    w = (p_group[g] * p_label[l]) / p_joint
                    weights[mask] = w

        # Normalise so weights sum to n_samples
        weights = weights / weights.mean()

        self._sample_weights = weights
        return weights

    # ------------------------------------------------------------------
    # Strategy B: Threshold Adjustment
    # ------------------------------------------------------------------

    def fit_thresholds(
        self,
        y_true: Sequence,
        y_prob: Sequence,
        protected: Sequence,
        *,
        n_thresholds: int = 100,
    ) -> Dict[Any, float]:
        """Learn per-group decision thresholds that satisfy the fairness criterion.

        Parameters
        ----------
        y_true : array-like of 0/1
            Ground-truth labels (used for equal opportunity).
        y_prob : array-like of float in [0, 1]
            Predicted probabilities for the positive class.
        protected : array-like
            Protected attribute values.
        n_thresholds : int
            Resolution of the threshold search grid.

        Returns
        -------
        dict
            Mapping ``{group_value: threshold}``.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob, dtype=float)
        protected = np.asarray(protected)

        groups = np.unique(protected)
        self._groups = groups
        thresholds_grid = np.linspace(0.0, 1.0, n_thresholds)

        if self.fairness_criterion == "demographic_parity":
            target = self._target_selection_rate(y_prob, protected, groups)
            self._thresholds = self._fit_demographic_parity(
                y_prob, protected, groups, thresholds_grid, target
            )
        else:  # equal_opportunity
            target = self._target_tpr(y_true, y_prob, protected, groups)
            self._thresholds = self._fit_equal_opportunity(
                y_true, y_prob, protected, groups, thresholds_grid, target
            )

        return dict(self._thresholds)

    def predict(
        self,
        y_prob: Sequence,
        protected: Sequence,
    ) -> np.ndarray:
        """Apply fitted per-group thresholds to produce binary predictions.

        Parameters
        ----------
        y_prob : array-like of float
            Predicted probabilities.
        protected : array-like
            Protected attribute values.

        Returns
        -------
        np.ndarray of 0/1
        """
        if self._thresholds is None:
            raise RuntimeError("Call fit_thresholds() before predict().")

        y_prob = np.asarray(y_prob, dtype=float)
        protected = np.asarray(protected)
        y_pred = np.zeros(len(y_prob), dtype=int)

        for g, thr in self._thresholds.items():
            mask = protected == g
            y_pred[mask] = (y_prob[mask] >= thr).astype(int)

        return y_pred

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _target_selection_rate(
        y_prob: np.ndarray,
        protected: np.ndarray,
        groups: np.ndarray,
    ) -> float:
        """Weighted average selection rate across groups (used as target)."""
        rates = []
        weights = []
        for g in groups:
            mask = protected == g
            rates.append((y_prob[mask] >= 0.5).mean())
            weights.append(mask.sum())
        return float(np.average(rates, weights=weights))

    @staticmethod
    def _target_tpr(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        protected: np.ndarray,
        groups: np.ndarray,
    ) -> float:
        """Weighted average TPR across groups (used as target)."""
        rates = []
        weights = []
        for g in groups:
            mask = protected == g
            pos_mask = mask & (y_true == 1)
            if pos_mask.sum() > 0:
                tpr = ((y_prob[pos_mask] >= 0.5)).mean()
                rates.append(tpr)
                weights.append(pos_mask.sum())
        if not rates:
            return 0.5
        return float(np.average(rates, weights=weights))

    @staticmethod
    def _fit_demographic_parity(
        y_prob: np.ndarray,
        protected: np.ndarray,
        groups: np.ndarray,
        grid: np.ndarray,
        target_rate: float,
    ) -> Dict[Any, float]:
        thresholds: Dict[Any, float] = {}
        for g in groups:
            mask = protected == g
            probs_g = y_prob[mask]
            best_thr = 0.5
            best_diff = float("inf")
            for thr in grid:
                rate = (probs_g >= thr).mean()
                diff = abs(rate - target_rate)
                if diff < best_diff:
                    best_diff = diff
                    best_thr = thr
            thresholds[g] = float(best_thr)
        return thresholds

    @staticmethod
    def _fit_equal_opportunity(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        protected: np.ndarray,
        groups: np.ndarray,
        grid: np.ndarray,
        target_tpr: float,
    ) -> Dict[Any, float]:
        thresholds: Dict[Any, float] = {}
        for g in groups:
            mask = protected == g
            pos_mask = mask & (y_true == 1)
            if pos_mask.sum() == 0:
                thresholds[g] = 0.5
                continue
            probs_pos = y_prob[pos_mask]
            best_thr = 0.5
            best_diff = float("inf")
            for thr in grid:
                tpr = (probs_pos >= thr).mean()
                diff = abs(tpr - target_tpr)
                if diff < best_diff:
                    best_diff = diff
                    best_thr = thr
            thresholds[g] = float(best_thr)
        return thresholds

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_sample_weights(self) -> np.ndarray:
        """Return computed sample weights (requires prior call to compute_sample_weights)."""
        if self._sample_weights is None:
            raise RuntimeError("Call compute_sample_weights() first.")
        return self._sample_weights

    def get_thresholds(self) -> Dict[Any, float]:
        """Return fitted thresholds (requires prior call to fit_thresholds)."""
        if self._thresholds is None:
            raise RuntimeError("Call fit_thresholds() first.")
        return dict(self._thresholds)
