"""
Experimental Framework for Fairness Research.

Provides a comprehensive framework for conducting rigorous fairness experiments
with statistical validation, reproducibility controls, and publication-ready results.

Research contributions:
- Standardized experimental protocols for fairness research
- Statistical significance testing with multiple testing correction
- Automated hypothesis testing and p-value computation
- Publication-ready result formatting and visualization
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from statsmodels.stats.multitest import multipletests

from ..fairness_metrics import compute_fairness_metrics
from ..logging_config import get_logger

logger = get_logger(__name__)


class HypothesisType(Enum):
    """Types of research hypotheses."""
    SUPERIORITY = "superiority"  # H1: Method A > Method B
    NON_INFERIORITY = "non_inferiority"  # H1: Method A >= Method B - δ
    EQUIVALENCE = "equivalence"  # H1: |Method A - Method B| < δ
    DIFFERENCE = "difference"  # H1: Method A ≠ Method B


class StatisticalTest(Enum):
    """Statistical tests for hypothesis testing."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    FRIEDMAN = "friedman"
    PERMUTATION = "permutation"
    BOOTSTRAP = "bootstrap"


class EffectSizeMetric(Enum):
    """Effect size metrics."""
    COHENS_D = "cohens_d"
    GLASS_DELTA = "glass_delta"
    CLIFF_DELTA = "cliff_delta"
    ETA_SQUARED = "eta_squared"


@dataclass
class ResearchHypothesis:
    """Research hypothesis definition."""
    name: str
    description: str
    hypothesis_type: HypothesisType
    null_hypothesis: str
    alternative_hypothesis: str
    primary_metric: str
    significance_level: float = 0.05
    effect_size_threshold: float = 0.2
    power: float = 0.8

    def __post_init__(self):
        """Validate hypothesis parameters."""
        if not 0 < self.significance_level < 1:
            raise ValueError("Significance level must be between 0 and 1")
        if not 0 < self.power < 1:
            raise ValueError("Statistical power must be between 0 and 1")


@dataclass
class ExperimentalCondition:
    """Experimental condition definition."""
    name: str
    description: str
    algorithm: BaseEstimator
    parameters: Dict[str, Any]
    preprocessing_steps: List[Callable] = field(default_factory=list)

    def get_configured_algorithm(self) -> BaseEstimator:
        """Get algorithm with configured parameters."""
        alg = clone(self.algorithm)
        alg.set_params(**self.parameters)
        return alg


@dataclass
class StatisticalResult:
    """Statistical test result."""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    is_significant: bool
    corrected_p_value: Optional[float] = None


@dataclass
class ExperimentResult:
    """Complete experiment result."""
    experiment_id: str
    timestamp: datetime
    hypothesis: ResearchHypothesis
    conditions: List[ExperimentalCondition]
    performance_results: Dict[str, Dict[str, float]]
    fairness_results: Dict[str, Dict[str, Any]]
    statistical_results: List[StatisticalResult]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'timestamp': self.timestamp.isoformat(),
            'hypothesis': {
                'name': self.hypothesis.name,
                'description': self.hypothesis.description,
                'hypothesis_type': self.hypothesis.hypothesis_type.value,
                'null_hypothesis': self.hypothesis.null_hypothesis,
                'alternative_hypothesis': self.hypothesis.alternative_hypothesis,
                'primary_metric': self.hypothesis.primary_metric,
                'significance_level': self.hypothesis.significance_level
            },
            'conditions': [
                {
                    'name': cond.name,
                    'description': cond.description,
                    'algorithm': str(type(cond.algorithm).__name__),
                    'parameters': cond.parameters
                }
                for cond in self.conditions
            ],
            'performance_results': self.performance_results,
            'fairness_results': self.fairness_results,
            'statistical_results': [
                {
                    'test_name': result.test_name,
                    'test_statistic': result.test_statistic,
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'confidence_interval': result.confidence_interval,
                    'interpretation': result.interpretation,
                    'is_significant': result.is_significant,
                    'corrected_p_value': result.corrected_p_value
                }
                for result in self.statistical_results
            ],
            'metadata': self.metadata
        }


class StatisticalValidation:
    """
    Statistical validation framework for fairness research.

    Provides comprehensive statistical testing capabilities with proper
    multiple testing correction and effect size computation.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = 'bonferroni',
        effect_size_metric: EffectSizeMetric = EffectSizeMetric.COHENS_D,
        bootstrap_samples: int = 1000
    ):
        """
        Initialize statistical validation framework.

        Args:
            alpha: Significance level
            correction_method: Multiple testing correction method
            effect_size_metric: Effect size metric to compute
            bootstrap_samples: Number of bootstrap samples
        """
        self.alpha = alpha
        self.correction_method = correction_method
        self.effect_size_metric = effect_size_metric
        self.bootstrap_samples = bootstrap_samples

        logger.info(f"StatisticalValidation initialized with α={alpha}, "
                   f"correction={correction_method}")

    def compare_algorithms(
        self,
        results_a: np.ndarray,
        results_b: np.ndarray,
        test_type: StatisticalTest = StatisticalTest.T_TEST,
        paired: bool = False
    ) -> StatisticalResult:
        """
        Compare two algorithms using statistical tests.

        Args:
            results_a: Performance results for algorithm A
            results_b: Performance results for algorithm B
            test_type: Statistical test to use
            paired: Whether to use paired test

        Returns:
            Statistical test result
        """
        # Validate inputs
        if len(results_a) != len(results_b) and paired:
            raise ValueError("Paired test requires equal sample sizes")

        # Perform statistical test
        if test_type == StatisticalTest.T_TEST:
            if paired:
                statistic, p_value = stats.ttest_rel(results_a, results_b)
            else:
                statistic, p_value = stats.ttest_ind(results_a, results_b)
            test_name = "Paired t-test" if paired else "Independent t-test"

        elif test_type == StatisticalTest.MANN_WHITNEY:
            statistic, p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
            test_name = "Mann-Whitney U test"

        elif test_type == StatisticalTest.WILCOXON:
            statistic, p_value = stats.wilcoxon(results_a, results_b)
            test_name = "Wilcoxon signed-rank test"

        elif test_type == StatisticalTest.BOOTSTRAP:
            return self._bootstrap_comparison(results_a, results_b)

        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        # Compute effect size
        effect_size = self._compute_effect_size(results_a, results_b)

        # Compute confidence interval
        ci = self._compute_confidence_interval(results_a, results_b)

        # Interpret results
        is_significant = p_value < self.alpha
        interpretation = self._interpret_result(p_value, effect_size, is_significant)

        return StatisticalResult(
            test_name=test_name,
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            is_significant=is_significant
        )

    def multiple_comparisons(
        self,
        results_list: List[np.ndarray],
        condition_names: List[str],
        test_type: StatisticalTest = StatisticalTest.T_TEST
    ) -> List[StatisticalResult]:
        """
        Perform multiple pairwise comparisons with correction.

        Args:
            results_list: List of result arrays for each condition
            condition_names: Names of experimental conditions
            test_type: Statistical test to use

        Returns:
            List of statistical results with corrected p-values
        """
        comparisons = []
        p_values = []

        # Perform all pairwise comparisons
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                result = self.compare_algorithms(
                    results_list[i], results_list[j], test_type
                )

                # Add comparison names
                result.test_name = f"{condition_names[i]} vs {condition_names[j]}: {result.test_name}"

                comparisons.append(result)
                p_values.append(result.p_value)

        # Apply multiple testing correction
        if len(p_values) > 1:
            rejected, corrected_p_values, _, _ = multipletests(
                p_values, alpha=self.alpha, method=self.correction_method
            )

            # Update results with corrected p-values
            for i, comparison in enumerate(comparisons):
                comparison.corrected_p_value = corrected_p_values[i]
                comparison.is_significant = rejected[i]
                comparison.interpretation = self._interpret_result(
                    corrected_p_values[i], comparison.effect_size, rejected[i]
                )

        return comparisons

    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = None
    ) -> Dict[str, float]:
        """
        Perform statistical power analysis.

        Args:
            effect_size: Expected effect size
            sample_size: Sample size per group
            alpha: Significance level (uses instance default if None)

        Returns:
            Power analysis results
        """
        if alpha is None:
            alpha = self.alpha

        # Simplified power calculation for t-test
        # In practice, would use specialized power analysis libraries

        from scipy.stats import norm

        # Critical value
        z_alpha = norm.ppf(1 - alpha / 2)

        # Power calculation (simplified)
        z_beta = z_alpha - effect_size * np.sqrt(sample_size / 2)
        power = 1 - norm.cdf(z_beta)

        # Required sample size for target power
        target_power = 0.8
        z_power = norm.ppf(target_power)
        required_n = 2 * ((z_alpha + z_power) / effect_size) ** 2

        return {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'alpha': alpha,
            'power': power,
            'required_sample_size_80_power': int(np.ceil(required_n))
        }

    def _compute_effect_size(self, results_a: np.ndarray, results_b: np.ndarray) -> float:
        """Compute effect size between two result sets."""
        if self.effect_size_metric == EffectSizeMetric.COHENS_D:
            # Cohen's d
            pooled_std = np.sqrt(
                ((len(results_a) - 1) * np.var(results_a, ddof=1) +
                 (len(results_b) - 1) * np.var(results_b, ddof=1)) /
                (len(results_a) + len(results_b) - 2)
            )

            if pooled_std == 0:
                return 0.0

            return (np.mean(results_a) - np.mean(results_b)) / pooled_std

        elif self.effect_size_metric == EffectSizeMetric.CLIFF_DELTA:
            # Cliff's delta (non-parametric effect size)
            greater = 0
            less = 0

            for a in results_a:
                for b in results_b:
                    if a > b:
                        greater += 1
                    elif a < b:
                        less += 1

            return (greater - less) / (len(results_a) * len(results_b))

        else:
            # Default to Cohen's d
            return self._compute_cohens_d(results_a, results_b)

    def _compute_cohens_d(self, results_a: np.ndarray, results_b: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        pooled_std = np.sqrt(
            ((len(results_a) - 1) * np.var(results_a, ddof=1) +
             (len(results_b) - 1) * np.var(results_b, ddof=1)) /
            (len(results_a) + len(results_b) - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (np.mean(results_a) - np.mean(results_b)) / pooled_std

    def _compute_confidence_interval(
        self,
        results_a: np.ndarray,
        results_b: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for the difference in means."""
        diff_mean = np.mean(results_a) - np.mean(results_b)

        # Pooled standard error
        pooled_var = (
            ((len(results_a) - 1) * np.var(results_a, ddof=1) +
             (len(results_b) - 1) * np.var(results_b, ddof=1)) /
            (len(results_a) + len(results_b) - 2)
        )

        standard_error = np.sqrt(pooled_var * (1/len(results_a) + 1/len(results_b)))

        # Critical value
        df = len(results_a) + len(results_b) - 2
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df)

        # Confidence interval
        margin_error = t_critical * standard_error

        return (diff_mean - margin_error, diff_mean + margin_error)

    def _bootstrap_comparison(self, results_a: np.ndarray, results_b: np.ndarray) -> StatisticalResult:
        """Perform bootstrap comparison of two algorithms."""
        # Observed difference
        observed_diff = np.mean(results_a) - np.mean(results_b)

        # Bootstrap resampling
        bootstrap_diffs = []

        for _ in range(self.bootstrap_samples):
            # Resample with replacement
            bootstrap_a = np.random.choice(results_a, len(results_a), replace=True)
            bootstrap_b = np.random.choice(results_b, len(results_b), replace=True)

            bootstrap_diff = np.mean(bootstrap_a) - np.mean(bootstrap_b)
            bootstrap_diffs.append(bootstrap_diff)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # P-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        # Effect size
        effect_size = self._compute_effect_size(results_a, results_b)

        # Confidence interval from bootstrap
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)

        is_significant = p_value < self.alpha
        interpretation = self._interpret_result(p_value, effect_size, is_significant)

        return StatisticalResult(
            test_name="Bootstrap test",
            test_statistic=observed_diff,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            is_significant=is_significant
        )

    def _interpret_result(self, p_value: float, effect_size: float, is_significant: bool) -> str:
        """Interpret statistical result in plain language."""
        # Effect size interpretation (Cohen's conventions)
        if abs(effect_size) < 0.2:
            effect_desc = "negligible"
        elif abs(effect_size) < 0.5:
            effect_desc = "small"
        elif abs(effect_size) < 0.8:
            effect_desc = "medium"
        else:
            effect_desc = "large"

        direction = "positive" if effect_size > 0 else "negative"

        significance_desc = "statistically significant" if is_significant else "not statistically significant"

        return (f"The difference is {significance_desc} (p = {p_value:.4f}) "
                f"with a {effect_desc} {direction} effect size (d = {effect_size:.3f}).")


class ResearchProtocol:
    """
    Research protocol for conducting systematic fairness experiments.

    Implements standardized protocols for reproducible fairness research
    following academic best practices.
    """

    def __init__(
        self,
        protocol_name: str,
        description: str,
        cv_folds: int = 5,
        cv_repeats: int = 3,
        random_state: int = 42,
        test_size: float = 0.2
    ):
        """
        Initialize research protocol.

        Args:
            protocol_name: Name of the research protocol
            description: Description of the protocol
            cv_folds: Number of cross-validation folds
            cv_repeats: Number of CV repeats
            random_state: Random seed for reproducibility
            test_size: Test set size
        """
        self.protocol_name = protocol_name
        self.description = description
        self.cv_folds = cv_folds
        self.cv_repeats = cv_repeats
        self.random_state = random_state
        self.test_size = test_size

        # Cross-validation strategy
        self.cv_strategy = RepeatedStratifiedKFold(
            n_splits=cv_folds,
            n_repeats=cv_repeats,
            random_state=random_state
        )

        logger.info(f"ResearchProtocol '{protocol_name}' initialized")

    def evaluate_algorithm(
        self,
        algorithm: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate algorithm using the research protocol.

        Args:
            algorithm: Algorithm to evaluate
            X: Feature matrix
            y: Target vector
            sensitive_attrs: Sensitive attributes
            metrics: Performance metrics to compute

        Returns:
            Evaluation results
        """
        if metrics is None:
            metrics = ['accuracy', 'roc_auc', 'precision', 'recall']

        logger.info(f"Evaluating {type(algorithm).__name__} using {self.protocol_name}")

        results = {
            'algorithm': type(algorithm).__name__,
            'parameters': algorithm.get_params(),
            'performance_metrics': {},
            'fairness_metrics': {},
            'cv_scores': {},
            'metadata': {
                'cv_folds': self.cv_folds,
                'cv_repeats': self.cv_repeats,
                'total_iterations': self.cv_folds * self.cv_repeats
            }
        }

        # Cross-validation evaluation
        for metric in metrics:
            if metric == 'accuracy':
                scores = cross_val_score(
                    algorithm, X, y, cv=self.cv_strategy, scoring='accuracy'
                )
            elif metric == 'roc_auc':
                scores = cross_val_score(
                    algorithm, X, y, cv=self.cv_strategy, scoring='roc_auc'
                )
            elif metric == 'precision':
                scores = cross_val_score(
                    algorithm, X, y, cv=self.cv_strategy, scoring='precision'
                )
            elif metric == 'recall':
                scores = cross_val_score(
                    algorithm, X, y, cv=self.cv_strategy, scoring='recall'
                )

            results['cv_scores'][metric] = scores
            results['performance_metrics'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'scores': scores.tolist()
            }

        # Fairness evaluation on full dataset
        algorithm_fitted = clone(algorithm)
        algorithm_fitted.fit(X, y)
        predictions = algorithm_fitted.predict(X)

        for attr_name in sensitive_attrs.columns:
            overall, by_group = compute_fairness_metrics(
                y, predictions, sensitive_attrs[attr_name]
            )

            results['fairness_metrics'][attr_name] = {
                'overall': overall,
                'by_group': by_group.to_dict() if hasattr(by_group, 'to_dict') else by_group
            }

        logger.info("Algorithm evaluation completed")
        return results

    def compare_algorithms(
        self,
        algorithms: List[BaseEstimator],
        algorithm_names: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame,
        primary_metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms using the research protocol.

        Args:
            algorithms: List of algorithms to compare
            algorithm_names: Names for the algorithms
            X: Feature matrix
            y: Target vector
            sensitive_attrs: Sensitive attributes
            primary_metric: Primary metric for statistical comparison

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(algorithms)} algorithms using {self.protocol_name}")

        if len(algorithms) != len(algorithm_names):
            raise ValueError("Number of algorithms must match number of names")

        # Evaluate each algorithm
        algorithm_results = []
        cv_scores_by_algorithm = []

        for alg, name in zip(algorithms, algorithm_names):
            result = self.evaluate_algorithm(alg, X, y, sensitive_attrs)
            result['name'] = name
            algorithm_results.append(result)
            cv_scores_by_algorithm.append(result['cv_scores'][primary_metric])

        # Statistical comparison
        statistical_validator = StatisticalValidation()
        statistical_results = statistical_validator.multiple_comparisons(
            cv_scores_by_algorithm, algorithm_names
        )

        # Overall comparison results
        comparison_results = {
            'protocol': self.protocol_name,
            'primary_metric': primary_metric,
            'algorithms': algorithm_results,
            'statistical_comparisons': [result.__dict__ for result in statistical_results],
            'summary': self._generate_comparison_summary(algorithm_results, statistical_results)
        }

        logger.info("Algorithm comparison completed")
        return comparison_results

    def _generate_comparison_summary(
        self,
        algorithm_results: List[Dict[str, Any]],
        statistical_results: List[StatisticalResult]
    ) -> Dict[str, Any]:
        """Generate summary of algorithm comparison."""
        # Find best performing algorithm
        best_algorithm = max(
            algorithm_results,
            key=lambda x: x['performance_metrics']['accuracy']['mean']
        )

        # Count significant differences
        significant_comparisons = [r for r in statistical_results if r.is_significant]

        # Effect size distribution
        effect_sizes = [r.effect_size for r in statistical_results]

        return {
            'best_algorithm': best_algorithm['name'],
            'best_performance': best_algorithm['performance_metrics']['accuracy']['mean'],
            'total_comparisons': len(statistical_results),
            'significant_comparisons': len(significant_comparisons),
            'effect_size_stats': {
                'mean': np.mean(effect_sizes),
                'std': np.std(effect_sizes),
                'max': np.max(effect_sizes),
                'min': np.min(effect_sizes)
            }
        }


class ExperimentalFramework:
    """
    Comprehensive experimental framework for fairness research.

    Orchestrates complete research experiments with proper controls,
    statistical validation, and publication-ready results.
    """

    def __init__(
        self,
        output_dir: str = "research_output",
        random_state: int = 42
    ):
        """
        Initialize experimental framework.

        Args:
            output_dir: Directory for saving results
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Initialize components
        self.statistical_validator = StatisticalValidation()
        self.protocol = ResearchProtocol("DefaultProtocol", "Standard research protocol")

        # Results storage
        self.experiments: List[ExperimentResult] = []

        logger.info(f"ExperimentalFramework initialized with output_dir={output_dir}")

    def conduct_experiment(
        self,
        experiment_name: str,
        hypothesis: ResearchHypothesis,
        conditions: List[ExperimentalCondition],
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> ExperimentResult:
        """
        Conduct a complete research experiment.

        Args:
            experiment_name: Name of the experiment
            hypothesis: Research hypothesis to test
            conditions: Experimental conditions to compare
            X: Feature matrix
            y: Target vector
            sensitive_attrs: Sensitive attributes

        Returns:
            Complete experiment results
        """
        logger.info(f"Conducting experiment: {experiment_name}")

        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Evaluate each condition
        condition_results = {}
        performance_results = {}
        fairness_results = {}
        cv_scores_by_condition = {}

        for condition in conditions:
            logger.info(f"Evaluating condition: {condition.name}")

            # Get configured algorithm
            algorithm = condition.get_configured_algorithm()

            # Apply preprocessing if specified
            X_processed = X.copy()
            for preprocessing_step in condition.preprocessing_steps:
                X_processed = preprocessing_step(X_processed)

            # Evaluate using research protocol
            result = self.protocol.evaluate_algorithm(
                algorithm, X_processed, y, sensitive_attrs
            )

            condition_results[condition.name] = result
            performance_results[condition.name] = result['performance_metrics']
            fairness_results[condition.name] = result['fairness_metrics']
            cv_scores_by_condition[condition.name] = result['cv_scores'][hypothesis.primary_metric]

        # Statistical hypothesis testing
        statistical_results = []

        if len(conditions) >= 2:
            # Pairwise comparisons
            condition_names = [cond.name for cond in conditions]
            cv_scores_list = [cv_scores_by_condition[name] for name in condition_names]

            statistical_results = self.statistical_validator.multiple_comparisons(
                cv_scores_list, condition_names
            )

        # Create experiment result
        experiment_result = ExperimentResult(
            experiment_id=experiment_id,
            timestamp=datetime.now(),
            hypothesis=hypothesis,
            conditions=conditions,
            performance_results=performance_results,
            fairness_results=fairness_results,
            statistical_results=statistical_results,
            metadata={
                'experiment_name': experiment_name,
                'dataset_info': {
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'n_sensitive_attrs': sensitive_attrs.shape[1]
                },
                'random_state': self.random_state
            }
        )

        # Save experiment result
        self._save_experiment_result(experiment_result)

        # Add to experiments list
        self.experiments.append(experiment_result)

        logger.info(f"Experiment {experiment_name} completed")
        return experiment_result

    def generate_research_report(
        self,
        experiment_results: List[ExperimentResult] = None,
        report_name: str = "research_report"
    ) -> str:
        """
        Generate comprehensive research report.

        Args:
            experiment_results: Experiment results to include (defaults to all)
            report_name: Name for the report file

        Returns:
            Path to generated report
        """
        if experiment_results is None:
            experiment_results = self.experiments

        logger.info(f"Generating research report with {len(experiment_results)} experiments")

        report_path = self.output_dir / f"{report_name}.html"

        # Generate HTML report
        html_content = self._generate_html_report(experiment_results)

        with open(report_path, 'w') as f:
            f.write(html_content)

        # Generate visualizations
        self._generate_report_visualizations(experiment_results, report_name)

        logger.info(f"Research report generated: {report_path}")
        return str(report_path)

    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to disk."""
        result_path = self.output_dir / f"{result.experiment_id}.json"

        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.debug(f"Experiment result saved: {result_path}")

    def _generate_html_report(self, experiment_results: List[ExperimentResult]) -> str:
        """Generate HTML research report."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Fairness Research Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .experiment { border: 1px solid #ccc; padding: 20px; margin: 20px 0; }
        .hypothesis { background-color: #f0f8ff; padding: 10px; margin: 10px 0; }
        .results { background-color: #f8f8f8; padding: 10px; margin: 10px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .significant { color: #d32f2f; font-weight: bold; }
        .not-significant { color: #388e3c; }
    </style>
</head>
<body>
        """

        html += "<h1>Fairness Research Report</h1>\n"
        html += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n"
        html += f"<p>Total experiments: {len(experiment_results)}</p>\n"

        # Executive Summary
        html += "<h2>Executive Summary</h2>\n"
        html += self._generate_executive_summary_html(experiment_results)

        # Individual experiment results
        html += "<h2>Experiment Results</h2>\n"

        for result in experiment_results:
            html += '<div class="experiment">\n'
            html += f"<h3>Experiment: {result.metadata.get('experiment_name', result.experiment_id)}</h3>\n"

            # Hypothesis
            html += '<div class="hypothesis">\n'
            html += "<h4>Research Hypothesis</h4>\n"
            html += f"<p><strong>Name:</strong> {result.hypothesis.name}</p>\n"
            html += f"<p><strong>Description:</strong> {result.hypothesis.description}</p>\n"
            html += f"<p><strong>Null Hypothesis:</strong> {result.hypothesis.null_hypothesis}</p>\n"
            html += f"<p><strong>Alternative Hypothesis:</strong> {result.hypothesis.alternative_hypothesis}</p>\n"
            html += "</div>\n"

            # Performance Results
            html += '<div class="results">\n'
            html += "<h4>Performance Results</h4>\n"
            html += self._generate_performance_table_html(result.performance_results)
            html += "</div>\n"

            # Statistical Results
            if result.statistical_results:
                html += '<div class="results">\n'
                html += "<h4>Statistical Analysis</h4>\n"
                html += self._generate_statistical_table_html(result.statistical_results)
                html += "</div>\n"

            html += "</div>\n"

        html += """
</body>
</html>
        """

        return html

    def _generate_executive_summary_html(self, experiment_results: List[ExperimentResult]) -> str:
        """Generate executive summary section."""
        summary = "<ul>\n"

        for result in experiment_results:
            exp_name = result.metadata.get('experiment_name', result.experiment_id)
            significant_results = [r for r in result.statistical_results if r.is_significant]

            summary += f"<li><strong>{exp_name}:</strong> "
            summary += f"{len(significant_results)} out of {len(result.statistical_results)} "
            summary += "comparisons showed statistically significant differences</li>\n"

        summary += "</ul>\n"
        return summary

    def _generate_performance_table_html(self, performance_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate performance results table."""
        html = "<table>\n"
        html += "<tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>ROC AUC</th></tr>\n"

        for alg_name, metrics in performance_results.items():
            html += "<tr>\n"
            html += f"<td>{alg_name}</td>\n"

            for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
                if metric in metrics:
                    mean_val = metrics[metric]['mean']
                    std_val = metrics[metric]['std']
                    html += f"<td>{mean_val:.4f} ± {std_val:.4f}</td>\n"
                else:
                    html += "<td>N/A</td>\n"

            html += "</tr>\n"

        html += "</table>\n"
        return html

    def _generate_statistical_table_html(self, statistical_results: List[StatisticalResult]) -> str:
        """Generate statistical results table."""
        html = "<table>\n"
        html += "<tr><th>Comparison</th><th>Test</th><th>p-value</th><th>Effect Size</th><th>Significant</th></tr>\n"

        for result in statistical_results:
            html += "<tr>\n"
            html += f"<td>{result.test_name}</td>\n"
            html += f"<td>{result.test_name.split(':')[-1].strip()}</td>\n"

            p_val = result.corrected_p_value if result.corrected_p_value else result.p_value
            html += f"<td>{p_val:.4f}</td>\n"
            html += f"<td>{result.effect_size:.3f}</td>\n"

            if result.is_significant:
                html += '<td class="significant">Yes</td>\n'
            else:
                html += '<td class="not-significant">No</td>\n'

            html += "</tr>\n"

        html += "</table>\n"
        return html

    def _generate_report_visualizations(self, experiment_results: List[ExperimentResult], report_name: str):
        """Generate visualizations for the research report."""
        # Performance comparison plots
        for i, result in enumerate(experiment_results):
            if len(result.performance_results) >= 2:
                plt.figure(figsize=(10, 6))

                # Extract accuracy means and stds
                algorithms = list(result.performance_results.keys())
                accuracies = [result.performance_results[alg]['accuracy']['mean'] for alg in algorithms]
                errors = [result.performance_results[alg]['accuracy']['std'] for alg in algorithms]

                plt.bar(algorithms, accuracies, yerr=errors, capsize=5, alpha=0.7)
                plt.title(f"Performance Comparison - {result.metadata.get('experiment_name', f'Experiment {i+1}')}")
                plt.ylabel('Accuracy')
                plt.xticks(rotation=45)
                plt.tight_layout()

                plot_path = self.output_dir / f"{report_name}_performance_{i+1}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()


# Example usage and CLI interface
def main():
    """CLI interface for experimental framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Fairness Research Experimental Framework")
    parser.add_argument("--demo", action="store_true", help="Run demonstration experiment")
    parser.add_argument("--output-dir", default="research_output", help="Output directory")

    args = parser.parse_args()

    if args.demo:
        # Create demonstration experiment
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=2, n_clusters_per_class=2, flip_y=0.05,
            random_state=42
        )

        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        # Create synthetic sensitive attributes
        sensitive_attrs_df = pd.DataFrame({
            'group_a': np.random.binomial(1, 0.3, len(X)),
            'group_b': np.random.choice([0, 1, 2], len(X))
        })

        # Initialize experimental framework
        framework = ExperimentalFramework(output_dir=args.output_dir)

        # Define hypothesis
        hypothesis = ResearchHypothesis(
            name="Algorithm Comparison",
            description="Compare logistic regression vs random forest",
            hypothesis_type=HypothesisType.SUPERIORITY,
            null_hypothesis="Random Forest accuracy <= Logistic Regression accuracy",
            alternative_hypothesis="Random Forest accuracy > Logistic Regression accuracy",
            primary_metric="accuracy"
        )

        # Define experimental conditions
        conditions = [
            ExperimentalCondition(
                name="LogisticRegression",
                description="Baseline logistic regression",
                algorithm=LogisticRegression(),
                parameters={'max_iter': 1000}
            ),
            ExperimentalCondition(
                name="RandomForest",
                description="Random forest classifier",
                algorithm=RandomForestClassifier(),
                parameters={'n_estimators': 100, 'random_state': 42}
            )
        ]

        # Conduct experiment
        result = framework.conduct_experiment(
            experiment_name="Demo_Algorithm_Comparison",
            hypothesis=hypothesis,
            conditions=conditions,
            X=X_df,
            y=y_series,
            sensitive_attrs=sensitive_attrs_df
        )

        # Generate report
        report_path = framework.generate_research_report()

        print("Demo experiment completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Report generated: {report_path}")

        # Print summary
        print("\nExperiment Summary:")
        print(f"- Experiment ID: {result.experiment_id}")
        print(f"- Conditions tested: {len(result.conditions)}")
        print(f"- Statistical tests performed: {len(result.statistical_results)}")

        for stat_result in result.statistical_results:
            print(f"- {stat_result.test_name}: p={stat_result.p_value:.4f}, "
                  f"significant={stat_result.is_significant}")


if __name__ == "__main__":
    main()
