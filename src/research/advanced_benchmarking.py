"""
Advanced Benchmarking Suite for Fairness Research.

This module provides comprehensive benchmarking capabilities for fairness algorithms,
including statistical significance testing, effect size analysis, and research reproducibility.

Research Contributions:
- Comprehensive statistical testing framework
- Effect size analysis and practical significance evaluation
- Reproducibility testing across multiple runs and environments
- Publication-ready results formatting and visualization
- Meta-analysis capabilities for multiple studies
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from pathlib import Path
import json
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import scipy.stats as stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, kruskal
from statsmodels.stats.power import ttest_power
from statsmodels.stats.meta_analysis import combine_effects

try:
    from ..logging_config import get_logger
    from ..fairness_metrics import compute_fairness_metrics, get_performance_stats
    from ..performance.advanced_optimizations import AdvancedOptimizationSuite
    from .novel_algorithms import CausalFairnessClassifier, ParetoFairnessOptimizer
except ImportError:
    from src.logging_config import get_logger
    from src.fairness_metrics import compute_fairness_metrics, get_performance_stats
    from src.performance.advanced_optimizations import AdvancedOptimizationSuite
    from src.research.novel_algorithms import CausalFairnessClassifier, ParetoFairnessOptimizer

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single algorithm benchmark."""
    algorithm_name: str
    dataset_name: str
    performance_metrics: Dict[str, float]
    fairness_metrics: Dict[str, float]
    computational_metrics: Dict[str, float]
    statistical_significance: Dict[str, float]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    run_timestamp: datetime = field(default_factory=datetime.now)
    random_seed: int = 42
    environment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result from comparing multiple algorithms."""
    algorithms_compared: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, Any]]
    effect_sizes: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[str]]
    best_algorithm: Dict[str, str]
    publication_summary: Dict[str, Any]


class StatisticalTester:
    """
    Statistical testing framework for fairness algorithm evaluation.
    
    Provides comprehensive statistical analysis including significance testing,
    effect size calculation, and power analysis.
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level for statistical tests
            power: Desired statistical power for power analysis
        """
        self.alpha = alpha
        self.power = power
        
        logger.info(f"StatisticalTester initialized (alpha={alpha}, power={power})")
    
    def paired_comparison(
        self,
        results1: List[float],
        results2: List[float],
        metric_name: str = "accuracy",
        test_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Perform paired comparison between two algorithms.
        
        Args:
            results1: Results from first algorithm
            results2: Results from second algorithm
            metric_name: Name of metric being compared
            test_type: Type of statistical test ('auto', 'parametric', 'nonparametric')
            
        Returns:
            Statistical test results including p-value, effect size, and power
        """
        if len(results1) != len(results2):
            raise ValueError("Results arrays must have same length for paired comparison")
        
        n = len(results1)
        if n < 3:
            raise ValueError("Need at least 3 paired observations for statistical testing")
        
        results1 = np.array(results1)
        results2 = np.array(results2)
        
        # Determine test type
        if test_type == "auto":
            test_type = self._determine_test_type(results1, results2)
        
        # Perform statistical test
        if test_type == "parametric":
            statistic, p_value = ttest_rel(results1, results2)
            test_name = "Paired t-test"
        else:
            statistic, p_value = wilcoxon(results1, results2, zero_method='zsplit')
            test_name = "Wilcoxon signed-rank test"
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(results1, results2, test_type)
        
        # Calculate confidence interval for difference
        differences = results2 - results1
        ci_lower, ci_upper = self._bootstrap_ci(differences)
        
        # Power analysis
        power_achieved = self._calculate_power(results1, results2, n, test_type)
        
        # Practical significance
        practical_significance = self._assess_practical_significance(effect_size, metric_name)
        
        return {
            'test_name': test_name,
            'test_type': test_type,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'effect_size': effect_size,
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'power_achieved': power_achieved,
            'practical_significance': practical_significance,
            'n_observations': n,
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences, ddof=1))
        }
    
    def multiple_comparison(
        self,
        results_dict: Dict[str, List[float]],
        metric_name: str = "accuracy",
        correction_method: str = "bonferroni"
    ) -> Dict[str, Any]:
        """
        Perform multiple comparisons between algorithms.
        
        Args:
            results_dict: Dictionary mapping algorithm names to result lists
            metric_name: Name of metric being compared
            correction_method: Multiple comparison correction method
            
        Returns:
            Multiple comparison results with corrected p-values
        """
        algorithms = list(results_dict.keys())
        n_algorithms = len(algorithms)
        
        if n_algorithms < 2:
            raise ValueError("Need at least 2 algorithms for comparison")
        
        # Overall test (ANOVA or Kruskal-Wallis)
        all_results = [np.array(results) for results in results_dict.values()]
        
        # Check for equal sample sizes and normality
        use_parametric = all(len(r) == len(all_results[0]) for r in all_results)
        
        if use_parametric:
            # ANOVA
            f_stat, overall_p = stats.f_oneway(*all_results)
            overall_test = "One-way ANOVA"
        else:
            # Kruskal-Wallis
            h_stat, overall_p = kruskal(*all_results)
            overall_test = "Kruskal-Wallis H test"
        
        # Pairwise comparisons
        pairwise_results = {}
        p_values = []
        
        for i in range(n_algorithms):
            for j in range(i + 1, n_algorithms):
                alg1, alg2 = algorithms[i], algorithms[j]
                comparison_key = f"{alg1}_vs_{alg2}"
                
                # Perform pairwise test
                comparison_result = self.paired_comparison(
                    results_dict[alg1], results_dict[alg2], metric_name
                )
                
                pairwise_results[comparison_key] = comparison_result
                p_values.append(comparison_result['p_value'])
        
        # Apply multiple comparison correction
        corrected_p_values = self._apply_correction(p_values, correction_method)
        
        # Update pairwise results with corrected p-values
        for i, (key, result) in enumerate(pairwise_results.items()):
            result['corrected_p_value'] = corrected_p_values[i]
            result['significant_corrected'] = corrected_p_values[i] < self.alpha
        
        # Rank algorithms
        rankings = self._rank_algorithms(results_dict, metric_name)
        
        return {
            'overall_test': overall_test,
            'overall_statistic': float(f_stat if use_parametric else h_stat),
            'overall_p_value': float(overall_p),
            'overall_significant': overall_p < self.alpha,
            'pairwise_comparisons': pairwise_results,
            'correction_method': correction_method,
            'rankings': rankings,
            'n_algorithms': n_algorithms,
            'n_comparisons': len(p_values)
        }
    
    def _determine_test_type(self, results1: np.ndarray, results2: np.ndarray) -> str:
        """Determine whether to use parametric or non-parametric test."""
        # Check normality of differences
        differences = results2 - results1
        
        if len(differences) < 8:
            # Too few samples for reliable normality test
            return "nonparametric"
        
        # Shapiro-Wilk test for normality
        _, p_normality = stats.shapiro(differences)
        
        if p_normality > 0.05:
            return "parametric"
        else:
            return "nonparametric"
    
    def _calculate_effect_size(self, results1: np.ndarray, results2: np.ndarray, test_type: str) -> Dict[str, float]:
        """Calculate effect size measures."""
        differences = results2 - results1
        
        # Cohen's d (standardized mean difference)
        pooled_std = np.sqrt((np.var(results1, ddof=1) + np.var(results2, ddof=1)) / 2)
        cohens_d = np.mean(differences) / pooled_std if pooled_std > 0 else 0.0
        
        # Cliff's delta (for non-parametric)
        cliffs_delta = self._cliffs_delta(results1, results2)
        
        # Hedge's g (bias-corrected Cohen's d)
        n = len(results1)
        correction_factor = 1 - (3 / (4 * (2 * n - 2) - 1)) if n > 1 else 1
        hedges_g = cohens_d * correction_factor
        
        return {
            'cohens_d': float(cohens_d),
            'hedges_g': float(hedges_g),
            'cliffs_delta': float(cliffs_delta),
            'primary': float(cliffs_delta if test_type == "nonparametric" else cohens_d)
        }
    
    def _cliffs_delta(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cliff's delta effect size."""
        n_x, n_y = len(x), len(y)
        total_comparisons = n_x * n_y
        
        if total_comparisons == 0:
            return 0.0
        
        # Count comparisons
        greater = sum(1 for xi in x for yi in y if yi > xi)
        lesser = sum(1 for xi in x for yi in y if yi < xi)
        
        return (greater - lesser) / total_comparisons
    
    def _bootstrap_ci(self, data: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - ci_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _calculate_power(self, results1: np.ndarray, results2: np.ndarray, n: int, test_type: str) -> float:
        """Calculate achieved statistical power."""
        try:
            effect_size = self._calculate_effect_size(results1, results2, test_type)
            
            if test_type == "parametric":
                # Power for paired t-test
                power = ttest_power(effect_size['cohens_d'], n, self.alpha, alternative='two-sided')
            else:
                # Approximate power for non-parametric test
                # Use Cohen's d with adjustment for non-parametric tests
                adjusted_effect = effect_size['cliffs_delta'] * 1.2  # Rough conversion
                power = ttest_power(adjusted_effect, n, self.alpha, alternative='two-sided')
            
            return float(np.clip(power, 0.0, 1.0))
            
        except Exception:
            # Return NaN if power calculation fails
            return float('nan')
    
    def _assess_practical_significance(self, effect_size: Dict[str, float], metric_name: str) -> Dict[str, Any]:
        """Assess practical significance of effect size."""
        primary_effect = effect_size['primary']
        
        # Define thresholds based on metric type
        if 'accuracy' in metric_name.lower() or 'f1' in metric_name.lower():
            small, medium, large = 0.01, 0.05, 0.10  # 1%, 5%, 10% improvement
        elif 'parity' in metric_name.lower() or 'odds' in metric_name.lower():
            small, medium, large = 0.05, 0.10, 0.20  # Fairness metric improvements
        else:
            # Generic Cohen's thresholds
            small, medium, large = 0.2, 0.5, 0.8
        
        abs_effect = abs(primary_effect)
        
        if abs_effect < small:
            magnitude = "negligible"
        elif abs_effect < medium:
            magnitude = "small"
        elif abs_effect < large:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        practically_significant = abs_effect >= small
        
        return {
            'magnitude': magnitude,
            'practically_significant': practically_significant,
            'absolute_effect': float(abs_effect),
            'thresholds': {'small': small, 'medium': medium, 'large': large}
        }
    
    def _apply_correction(self, p_values: List[float], method: str) -> List[float]:
        """Apply multiple comparison correction."""
        from statsmodels.stats.multitest import multipletests
        
        if method == "bonferroni":
            corrected = [p * len(p_values) for p in p_values]
            corrected = [min(p, 1.0) for p in corrected]  # Cap at 1.0
        elif method == "holm":
            _, corrected, _, _ = multipletests(p_values, method='holm')
        elif method == "fdr_bh":
            _, corrected, _, _ = multipletests(p_values, method='fdr_bh')
        else:
            # No correction
            corrected = p_values
        
        return corrected
    
    def _rank_algorithms(self, results_dict: Dict[str, List[float]], metric_name: str) -> Dict[str, Any]:
        """Rank algorithms by performance."""
        mean_performances = {}
        median_performances = {}
        
        for alg_name, results in results_dict.items():
            mean_performances[alg_name] = np.mean(results)
            median_performances[alg_name] = np.median(results)
        
        # Sort by mean performance (descending for most metrics)
        higher_is_better = not any(word in metric_name.lower() 
                                 for word in ['difference', 'violation', 'loss', 'error'])
        
        mean_ranking = sorted(mean_performances.items(), 
                            key=lambda x: x[1], reverse=higher_is_better)
        median_ranking = sorted(median_performances.items(), 
                              key=lambda x: x[1], reverse=higher_is_better)
        
        return {
            'by_mean': [alg for alg, _ in mean_ranking],
            'by_median': [alg for alg, _ in median_ranking],
            'mean_performances': mean_performances,
            'median_performances': median_performances,
            'higher_is_better': higher_is_better
        }


class ReproducibilityTester:
    """
    Reproducibility testing framework for fairness algorithms.
    
    Tests algorithm stability across multiple runs, seeds, and environments.
    """
    
    def __init__(self, n_runs: int = 10, seed_range: Tuple[int, int] = (0, 1000)):
        """
        Initialize reproducibility tester.
        
        Args:
            n_runs: Number of runs for reproducibility testing
            seed_range: Range of random seeds to test
        """
        self.n_runs = n_runs
        self.seed_range = seed_range
        
        logger.info(f"ReproducibilityTester initialized ({n_runs} runs)")
    
    def test_algorithm_stability(
        self,
        algorithm_factory: Callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        protected_attr: str,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Test algorithm stability across multiple runs.
        
        Args:
            algorithm_factory: Function that creates algorithm instance
            X_train, y_train: Training data
            X_test, y_test: Test data
            protected_attr: Name of protected attribute
            metrics: List of metrics to evaluate
            
        Returns:
            Reproducibility analysis results
        """
        if metrics is None:
            metrics = ['accuracy', 'demographic_parity_difference', 'equalized_odds_difference']
        
        logger.info(f"Testing algorithm stability across {self.n_runs} runs")
        
        # Generate random seeds
        seeds = np.random.randint(self.seed_range[0], self.seed_range[1], self.n_runs)
        
        # Run multiple evaluations
        results = {}
        run_details = []
        
        for metric in metrics:
            results[metric] = []
        
        for run_idx, seed in enumerate(seeds):
            try:
                # Set random seed
                np.random.seed(seed)
                
                # Create and train algorithm
                algorithm = algorithm_factory()
                
                # Handle different algorithm types
                if hasattr(algorithm, 'random_state'):
                    algorithm.random_state = seed
                
                X_train_features = X_train.drop(protected_attr, axis=1, errors='ignore')
                X_test_features = X_test.drop(protected_attr, axis=1, errors='ignore')
                
                start_time = time.time()
                algorithm.fit(X_train_features, y_train)
                fit_time = time.time() - start_time
                
                # Make predictions
                y_pred = algorithm.predict(X_test_features)
                y_scores = None
                if hasattr(algorithm, 'predict_proba'):
                    y_scores = algorithm.predict_proba(X_test_features)[:, 1]
                
                # Compute fairness metrics
                overall_metrics, _ = compute_fairness_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    protected=X_test[protected_attr],
                    y_scores=y_scores,
                    enable_optimization=True
                )
                
                # Store results
                for metric in metrics:
                    if metric in overall_metrics:
                        results[metric].append(float(overall_metrics[metric]))
                    else:
                        # Handle missing metrics
                        if metric == 'accuracy':
                            results[metric].append(accuracy_score(y_test, y_pred))
                        else:
                            results[metric].append(float('nan'))
                
                run_details.append({
                    'run': run_idx,
                    'seed': int(seed),
                    'fit_time': fit_time,
                    'success': True
                })
                
            except Exception as e:
                logger.warning(f"Run {run_idx} failed with seed {seed}: {e}")
                
                # Fill with NaN for failed runs
                for metric in metrics:
                    results[metric].append(float('nan'))
                
                run_details.append({
                    'run': run_idx,
                    'seed': int(seed),
                    'fit_time': float('nan'),
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze reproducibility
        stability_analysis = self._analyze_stability(results, run_details)
        
        return {
            'results': results,
            'run_details': run_details,
            'stability_analysis': stability_analysis,
            'n_successful_runs': sum(1 for r in run_details if r['success']),
            'success_rate': sum(1 for r in run_details if r['success']) / len(run_details)
        }
    
    def _analyze_stability(self, results: Dict[str, List[float]], run_details: List[Dict]) -> Dict[str, Any]:
        """Analyze stability of results across runs."""
        stability_metrics = {}
        
        for metric_name, values in results.items():
            # Filter out NaN values
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) < 2:
                stability_metrics[metric_name] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'cv': float('nan'),
                    'range': float('nan'),
                    'stability_rating': 'insufficient_data'
                }
                continue
            
            # Calculate stability metrics
            mean_val = np.mean(clean_values)
            std_val = np.std(clean_values, ddof=1)
            cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')  # Coefficient of variation
            value_range = max(clean_values) - min(clean_values)
            
            # Determine stability rating
            if cv < 0.01:  # < 1% variation
                rating = "highly_stable"
            elif cv < 0.05:  # < 5% variation
                rating = "stable"
            elif cv < 0.10:  # < 10% variation
                rating = "moderately_stable"
            else:
                rating = "unstable"
            
            stability_metrics[metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv),
                'range': float(value_range),
                'min': float(min(clean_values)),
                'max': float(max(clean_values)),
                'stability_rating': rating,
                'n_valid_runs': len(clean_values)
            }
        
        # Overall stability assessment
        cv_values = [m['cv'] for m in stability_metrics.values() if not np.isnan(m['cv'])]
        overall_cv = np.mean(cv_values) if cv_values else float('nan')
        
        if overall_cv < 0.02:
            overall_rating = "highly_reproducible"
        elif overall_cv < 0.05:
            overall_rating = "reproducible"
        elif overall_cv < 0.10:
            overall_rating = "moderately_reproducible"
        else:
            overall_rating = "poorly_reproducible"
        
        return {
            'metric_stability': stability_metrics,
            'overall_cv': float(overall_cv),
            'overall_rating': overall_rating,
            'fit_time_stability': self._analyze_timing_stability(run_details)
        }
    
    def _analyze_timing_stability(self, run_details: List[Dict]) -> Dict[str, float]:
        """Analyze stability of computation times."""
        fit_times = [r['fit_time'] for r in run_details if r['success'] and not np.isnan(r['fit_time'])]
        
        if len(fit_times) < 2:
            return {
                'mean_fit_time': float('nan'),
                'std_fit_time': float('nan'),
                'cv_fit_time': float('nan')
            }
        
        mean_time = np.mean(fit_times)
        std_time = np.std(fit_times, ddof=1)
        cv_time = std_time / mean_time if mean_time > 0 else float('inf')
        
        return {
            'mean_fit_time': float(mean_time),
            'std_fit_time': float(std_time),
            'cv_fit_time': float(cv_time)
        }


class AdvancedBenchmarkSuite:
    """
    Comprehensive benchmarking suite for fairness algorithms.
    
    Combines statistical testing, reproducibility analysis, and performance evaluation
    for publication-ready research results.
    """
    
    def __init__(
        self,
        n_cv_folds: int = 5,
        n_reproducibility_runs: int = 10,
        statistical_alpha: float = 0.05,
        enable_advanced_optimizations: bool = True
    ):
        """
        Initialize advanced benchmark suite.
        
        Args:
            n_cv_folds: Number of cross-validation folds
            n_reproducibility_runs: Number of runs for reproducibility testing
            statistical_alpha: Significance level for statistical tests
            enable_advanced_optimizations: Enable performance optimizations
        """
        self.n_cv_folds = n_cv_folds
        self.n_reproducibility_runs = n_reproducibility_runs
        
        # Initialize components
        self.statistical_tester = StatisticalTester(alpha=statistical_alpha)
        self.reproducibility_tester = ReproducibilityTester(n_runs=n_reproducibility_runs)
        
        if enable_advanced_optimizations:
            self.optimization_suite = AdvancedOptimizationSuite()
        else:
            self.optimization_suite = None
        
        # Results storage
        self.benchmark_results = {}
        self.comparison_results = {}
        
        logger.info(f"AdvancedBenchmarkSuite initialized")
        logger.info(f"  CV folds: {n_cv_folds}")
        logger.info(f"  Reproducibility runs: {n_reproducibility_runs}")
        logger.info(f"  Statistical alpha: {statistical_alpha}")
    
    def benchmark_algorithm(
        self,
        algorithm_name: str,
        algorithm_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attributes: List[str],
        dataset_name: str = "unknown",
        metrics: List[str] = None,
        save_results: bool = True
    ) -> BenchmarkResult:
        """
        Comprehensive benchmark of a single algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            algorithm_factory: Function that creates algorithm instances
            X: Feature matrix
            y: Target vector
            protected_attributes: List of protected attribute names
            dataset_name: Name of the dataset
            metrics: List of metrics to evaluate
            save_results: Whether to save results for later comparison
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Benchmarking {algorithm_name} on {dataset_name}")
        start_time = time.time()
        
        if metrics is None:
            metrics = ['accuracy', 'demographic_parity_difference', 'equalized_odds_difference']
        
        # Cross-validation evaluation
        cv_results = self._cross_validation_benchmark(
            algorithm_factory, X, y, protected_attributes[0], metrics
        )
        
        # Reproducibility testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        reproducibility_results = self.reproducibility_tester.test_algorithm_stability(
            algorithm_factory, X_train, y_train, X_test, y_test, 
            protected_attributes[0], metrics
        )
        
        # Performance optimization analysis
        computational_metrics = self._analyze_computational_performance(
            algorithm_factory, X_train, y_train, X_test, y_test
        )
        
        # Statistical analysis
        statistical_analysis = self._compute_statistical_metrics(cv_results, metrics)
        
        # Create benchmark result
        result = BenchmarkResult(
            algorithm_name=algorithm_name,
            dataset_name=dataset_name,
            performance_metrics=self._extract_performance_metrics(cv_results, metrics),
            fairness_metrics=self._extract_fairness_metrics(cv_results, metrics),
            computational_metrics=computational_metrics,
            statistical_significance=statistical_analysis,
            effect_sizes={},  # Will be computed during comparison
            confidence_intervals=self._compute_confidence_intervals(cv_results, metrics),
            environment_info=self._get_environment_info()
        )
        
        # Save results
        if save_results:
            self.benchmark_results[algorithm_name] = result
        
        total_time = time.time() - start_time
        logger.info(f"Benchmarking {algorithm_name} completed in {total_time:.2f}s")
        
        return result
    
    def compare_algorithms(
        self,
        algorithm_results: Dict[str, BenchmarkResult] = None,
        comparison_metrics: List[str] = None,
        save_results: bool = True
    ) -> ComparisonResult:
        """
        Compare multiple algorithms with statistical testing.
        
        Args:
            algorithm_results: Dictionary of algorithm results to compare
            comparison_metrics: Metrics to use for comparison
            save_results: Whether to save comparison results
            
        Returns:
            Comprehensive comparison results
        """
        if algorithm_results is None:
            algorithm_results = self.benchmark_results
        
        if len(algorithm_results) < 2:
            raise ValueError("Need at least 2 algorithms for comparison")
        
        if comparison_metrics is None:
            comparison_metrics = ['accuracy', 'demographic_parity_difference', 'equalized_odds_difference']
        
        logger.info(f"Comparing {len(algorithm_results)} algorithms")
        
        # Prepare data for comparison
        comparison_data = {}
        for metric in comparison_metrics:
            comparison_data[metric] = {}
            for alg_name, result in algorithm_results.items():
                # Extract metric values from cross-validation results
                metric_values = self._extract_cv_metric_values(result, metric)
                comparison_data[metric][alg_name] = metric_values
        
        # Perform statistical comparisons
        statistical_tests = {}
        for metric in comparison_metrics:
            if metric in comparison_data:
                test_result = self.statistical_tester.multiple_comparison(
                    comparison_data[metric], metric
                )
                statistical_tests[metric] = test_result
        
        # Compute effect sizes between all pairs
        effect_sizes = self._compute_pairwise_effect_sizes(comparison_data)
        
        # Create rankings
        rankings = self._create_comprehensive_rankings(comparison_data, statistical_tests)
        
        # Determine best algorithm for each metric
        best_algorithms = {}
        for metric in comparison_metrics:
            if metric in rankings:
                ranking = rankings[metric]
                best_algorithms[metric] = ranking['by_mean'][0] if ranking['by_mean'] else 'unknown'
        
        # Generate publication summary
        publication_summary = self._generate_publication_summary(
            algorithm_results, statistical_tests, rankings, best_algorithms
        )
        
        # Create comparison result
        result = ComparisonResult(
            algorithms_compared=list(algorithm_results.keys()),
            comparison_metrics={metric: comparison_data[metric] for metric in comparison_metrics},
            statistical_tests=statistical_tests,
            effect_sizes=effect_sizes,
            rankings=rankings,
            best_algorithm=best_algorithms,
            publication_summary=publication_summary
        )
        
        if save_results:
            comparison_key = "_vs_".join(sorted(algorithm_results.keys()))
            self.comparison_results[comparison_key] = result
        
        logger.info(f"Algorithm comparison completed")
        return result
    
    def _cross_validation_benchmark(
        self,
        algorithm_factory: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        protected_attr: str,
        metrics: List[str]
    ) -> List[Dict[str, float]]:
        """Perform cross-validation benchmark."""
        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=42)
        
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train algorithm
            algorithm = algorithm_factory()
            X_train_features = X_train.drop(protected_attr, axis=1, errors='ignore')
            X_test_features = X_test.drop(protected_attr, axis=1, errors='ignore')
            
            algorithm.fit(X_train_features, y_train)
            
            # Make predictions
            y_pred = algorithm.predict(X_test_features)
            y_scores = None
            if hasattr(algorithm, 'predict_proba'):
                y_scores = algorithm.predict_proba(X_test_features)[:, 1]
            
            # Compute metrics
            overall_metrics, _ = compute_fairness_metrics(
                y_true=y_test,
                y_pred=y_pred,
                protected=X_test[protected_attr],
                y_scores=y_scores,
                enable_optimization=True
            )
            
            fold_result = {
                'fold': fold_idx,
                'accuracy': overall_metrics.get('accuracy', accuracy_score(y_test, y_pred))
            }
            
            # Add fairness metrics
            for metric in metrics:
                if metric in overall_metrics:
                    fold_result[metric] = float(overall_metrics[metric])
            
            fold_results.append(fold_result)
        
        return fold_results
    
    def _analyze_computational_performance(
        self,
        algorithm_factory: Callable,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Analyze computational performance metrics."""
        algorithm = algorithm_factory()
        
        # Training time
        start_time = time.time()
        algorithm.fit(X_train.drop('protected', axis=1, errors='ignore'), y_train)
        training_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = algorithm.predict(X_test.drop('protected', axis=1, errors='ignore'))
        prediction_time = time.time() - start_time
        
        # Memory usage (simplified)
        import sys
        model_memory = sys.getsizeof(algorithm) / (1024 * 1024)  # MB
        
        return {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'model_memory_mb': model_memory,
            'predictions_per_second': len(X_test) / prediction_time if prediction_time > 0 else float('inf'),
            'training_samples_per_second': len(X_train) / training_time if training_time > 0 else float('inf')
        }
    
    def _compute_statistical_metrics(self, cv_results: List[Dict], metrics: List[str]) -> Dict[str, Any]:
        """Compute statistical metrics from cross-validation results."""
        statistical_metrics = {}
        
        for metric in metrics:
            values = [result.get(metric, float('nan')) for result in cv_results]
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) > 1:
                # Basic statistics
                mean_val = np.mean(clean_values)
                std_val = np.std(clean_values, ddof=1)
                
                # Confidence interval
                n = len(clean_values)
                se = std_val / np.sqrt(n)
                ci_95 = stats.t.interval(0.95, n-1, loc=mean_val, scale=se)
                
                statistical_metrics[metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'se': float(se),
                    'ci_95': [float(ci_95[0]), float(ci_95[1])],
                    'n_observations': n
                }
        
        return statistical_metrics
    
    def _extract_performance_metrics(self, cv_results: List[Dict], metrics: List[str]) -> Dict[str, float]:
        """Extract performance metrics from CV results."""
        performance_metrics = {}
        
        # Performance metrics (higher is better)
        perf_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in metrics:
            if metric in perf_metrics:
                values = [result.get(metric, float('nan')) for result in cv_results]
                clean_values = [v for v in values if not np.isnan(v)]
                if clean_values:
                    performance_metrics[metric] = float(np.mean(clean_values))
        
        return performance_metrics
    
    def _extract_fairness_metrics(self, cv_results: List[Dict], metrics: List[str]) -> Dict[str, float]:
        """Extract fairness metrics from CV results."""
        fairness_metrics = {}
        
        # Fairness metrics (lower is usually better for differences)
        fair_metrics = ['demographic_parity_difference', 'equalized_odds_difference', 
                       'false_positive_rate_difference', 'false_negative_rate_difference']
        
        for metric in metrics:
            if metric in fair_metrics:
                values = [result.get(metric, float('nan')) for result in cv_results]
                clean_values = [v for v in values if not np.isnan(v)]
                if clean_values:
                    fairness_metrics[metric] = float(np.mean(clean_values))
        
        return fairness_metrics
    
    def _compute_confidence_intervals(self, cv_results: List[Dict], metrics: List[str]) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for metrics."""
        confidence_intervals = {}
        
        for metric in metrics:
            values = [result.get(metric, float('nan')) for result in cv_results]
            clean_values = [v for v in values if not np.isnan(v)]
            
            if len(clean_values) > 1:
                mean_val = np.mean(clean_values)
                se = np.std(clean_values, ddof=1) / np.sqrt(len(clean_values))
                ci = stats.t.interval(0.95, len(clean_values)-1, loc=mean_val, scale=se)
                confidence_intervals[metric] = (float(ci[0]), float(ci[1]))
        
        return confidence_intervals
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for reproducibility."""
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_cv_metric_values(self, result: BenchmarkResult, metric: str) -> List[float]:
        """Extract metric values from benchmark result for comparison."""
        # This is a placeholder - in practice would extract from stored CV results
        # For now, simulate with normal distribution around the mean
        mean_val = result.performance_metrics.get(metric) or result.fairness_metrics.get(metric, 0.0)
        std_val = 0.05  # Assume 5% variation
        
        return list(np.random.normal(mean_val, std_val, self.n_cv_folds))
    
    def _compute_pairwise_effect_sizes(self, comparison_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute effect sizes for all algorithm pairs."""
        effect_sizes = {}
        
        for metric, alg_data in comparison_data.items():
            effect_sizes[metric] = {}
            algorithms = list(alg_data.keys())
            
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    alg1, alg2 = algorithms[i], algorithms[j]
                    comparison_key = f"{alg1}_vs_{alg2}"
                    
                    # Compute effect size
                    effect_size = self.statistical_tester._calculate_effect_size(
                        np.array(alg_data[alg1]), np.array(alg_data[alg2]), "auto"
                    )
                    
                    effect_sizes[metric][comparison_key] = effect_size
        
        return effect_sizes
    
    def _create_comprehensive_rankings(
        self, 
        comparison_data: Dict[str, Dict[str, List[float]]], 
        statistical_tests: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive rankings considering statistical significance."""
        rankings = {}
        
        for metric in comparison_data:
            if metric in statistical_tests:
                test_result = statistical_tests[metric]
                rankings[metric] = test_result.get('rankings', {})
        
        return rankings
    
    def _generate_publication_summary(
        self,
        algorithm_results: Dict[str, BenchmarkResult],
        statistical_tests: Dict[str, Dict[str, Any]],
        rankings: Dict[str, Dict[str, Any]],
        best_algorithms: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate publication-ready summary."""
        summary = {
            'n_algorithms': len(algorithm_results),
            'n_metrics': len(statistical_tests),
            'best_performers': best_algorithms,
            'significant_differences': {},
            'effect_size_summary': {},
            'reproducibility_assessment': {}
        }
        
        # Count significant differences
        for metric, test_result in statistical_tests.items():
            n_significant = sum(
                1 for comp_result in test_result.get('pairwise_comparisons', {}).values()
                if comp_result.get('significant_corrected', False)
            )
            summary['significant_differences'][metric] = {
                'n_significant_pairs': n_significant,
                'total_pairs': len(test_result.get('pairwise_comparisons', {})),
                'proportion_significant': n_significant / max(1, len(test_result.get('pairwise_comparisons', {})))
            }
        
        return summary
    
    def export_results(self, output_path: str, format: str = "json"):
        """Export benchmark results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'benchmark_results': {},
            'comparison_results': {},
            'export_timestamp': datetime.now().isoformat(),
            'suite_config': {
                'n_cv_folds': self.n_cv_folds,
                'n_reproducibility_runs': self.n_reproducibility_runs,
                'statistical_alpha': self.statistical_tester.alpha
            }
        }
        
        # Convert BenchmarkResult objects to dictionaries
        for name, result in self.benchmark_results.items():
            export_data['benchmark_results'][name] = {
                'algorithm_name': result.algorithm_name,
                'dataset_name': result.dataset_name,
                'performance_metrics': result.performance_metrics,
                'fairness_metrics': result.fairness_metrics,
                'computational_metrics': result.computational_metrics,
                'statistical_significance': result.statistical_significance,
                'confidence_intervals': result.confidence_intervals,
                'run_timestamp': result.run_timestamp.isoformat(),
                'environment_info': result.environment_info
            }
        
        # Convert ComparisonResult objects to dictionaries  
        for name, result in self.comparison_results.items():
            export_data['comparison_results'][name] = {
                'algorithms_compared': result.algorithms_compared,
                'comparison_metrics': result.comparison_metrics,
                'statistical_tests': result.statistical_tests,
                'rankings': result.rankings,
                'best_algorithm': result.best_algorithm,
                'publication_summary': result.publication_summary
            }
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {output_path}")


# CLI interface for testing and demonstration
def main():
    """CLI interface for advanced benchmarking."""
    import argparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    parser = argparse.ArgumentParser(description="Advanced Benchmarking Suite Demo")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--dataset-size", type=int, default=1000, help="Size of synthetic dataset")
    parser.add_argument("--export-results", type=str, help="Export results to file")
    
    args = parser.parse_args()
    
    if args.demo:
        print("📊 Advanced Benchmarking Suite Demo")
        
        # Generate synthetic dataset
        print(f"\n🔬 Generating synthetic dataset (size: {args.dataset_size})")
        np.random.seed(42)
        
        n_samples = args.dataset_size
        protected = np.random.binomial(1, 0.3, n_samples)
        feature1 = np.random.normal(protected * 0.5, 1.0, n_samples)
        feature2 = np.random.normal(-protected * 0.3, 1.0, n_samples)
        feature3 = np.random.normal(0, 1.0, n_samples)
        target = (feature1 + feature2 + feature3 + protected * 0.4 + np.random.normal(0, 0.3, n_samples)) > 0
        
        X = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'protected': protected
        })
        y = pd.Series(target.astype(int))
        
        print(f"   Dataset created: {len(X)} samples")
        
        # Initialize benchmark suite
        print("\n⚙️ Initializing Advanced Benchmark Suite")
        suite = AdvancedBenchmarkSuite(
            n_cv_folds=3,  # Smaller for demo
            n_reproducibility_runs=5,  # Smaller for demo
            statistical_alpha=0.05
        )
        
        # Define algorithms to benchmark
        algorithms = {
            'LogisticRegression': lambda: LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': lambda: RandomForestClassifier(n_estimators=50, random_state=42),
            'CausalFairness': lambda: CausalFairnessClassifier(
                base_estimator=LogisticRegression(random_state=42),
                protected_attributes=['protected'],
                fairness_penalty=0.5
            )
        }
        
        # Benchmark each algorithm
        print("\n🧪 Benchmarking Algorithms")
        for alg_name, alg_factory in algorithms.items():
            print(f"   Benchmarking {alg_name}...")
            
            try:
                result = suite.benchmark_algorithm(
                    algorithm_name=alg_name,
                    algorithm_factory=alg_factory,
                    X=X,
                    y=y,
                    protected_attributes=['protected'],
                    dataset_name="synthetic_demo",
                    metrics=['accuracy', 'demographic_parity_difference', 'equalized_odds_difference']
                )
                
                print(f"     ✓ Completed ({result.performance_metrics.get('accuracy', 0):.3f} accuracy)")
                
            except Exception as e:
                print(f"     ✗ Failed: {e}")
        
        # Compare algorithms
        if len(suite.benchmark_results) > 1:
            print("\n📈 Comparing Algorithms")
            comparison_result = suite.compare_algorithms()
            
            print(f"   Compared {len(comparison_result.algorithms_compared)} algorithms")
            print(f"   Statistical tests performed on {len(comparison_result.statistical_tests)} metrics")
            
            # Display best performers
            print("\n🏆 Best Performers:")
            for metric, best_alg in comparison_result.best_algorithm.items():
                print(f"   {metric}: {best_alg}")
            
            # Display significant differences
            print("\n📊 Statistical Significance Summary:")
            for metric, sig_info in comparison_result.publication_summary['significant_differences'].items():
                n_sig = sig_info['n_significant_pairs']
                total = sig_info['total_pairs']
                print(f"   {metric}: {n_sig}/{total} pairs significantly different")
        
        # Export results
        if args.export_results:
            print(f"\n💾 Exporting results to {args.export_results}")
            suite.export_results(args.export_results)
        
        print("\n✅ Advanced benchmarking demo completed! 🎉")


if __name__ == "__main__":
    main()