"""
Advanced feature engineering service for fair credit scoring.

This module provides sophisticated feature engineering capabilities with
built-in fairness considerations and bias prevention techniques.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..logging_config import get_logger

logger = get_logger(__name__)


class FairnessAwareFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that considers fairness implications.
    
    Removes features that are highly correlated with protected attributes
    to prevent proxy discrimination.
    """

    def __init__(self, protected_attributes: List[str], correlation_threshold: float = 0.7):
        """
        Initialize fairness-aware feature selector.
        
        Args:
            protected_attributes: List of protected attribute column names
            correlation_threshold: Threshold for removing correlated features
        """
        self.protected_attributes = protected_attributes
        self.correlation_threshold = correlation_threshold
        self.selected_features_ = None
        self.correlation_scores_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the feature selector.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored)
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Calculate correlations with protected attributes
        correlations = {}
        selected_features = []

        for feature in X.columns:
            if feature in self.protected_attributes:
                continue

            max_correlation = 0.0
            for protected_attr in self.protected_attributes:
                if protected_attr in X.columns:
                    try:
                        # Handle both numeric and categorical correlations
                        if X[feature].dtype in ['object', 'category'] or X[protected_attr].dtype in ['object', 'category']:
                            # Use Cramér's V for categorical associations
                            correlation = self._cramers_v(X[feature], X[protected_attr])
                        else:
                            # Use Pearson correlation for numeric features
                            correlation = abs(X[feature].corr(X[protected_attr]))

                        max_correlation = max(max_correlation, correlation)
                    except Exception as e:
                        logger.warning(f"Could not compute correlation for {feature} and {protected_attr}: {e}")
                        correlation = 0.0

            correlations[feature] = max_correlation

            # Select feature if correlation is below threshold
            if max_correlation < self.correlation_threshold:
                selected_features.append(feature)
            else:
                logger.info(f"Removing feature {feature} due to high correlation ({max_correlation:.3f}) with protected attributes")

        self.selected_features_ = selected_features
        self.correlation_scores_ = correlations

        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns) - len(self.protected_attributes)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted")

        # Keep protected attributes and selected features
        columns_to_keep = self.protected_attributes + self.selected_features_
        available_columns = [col for col in columns_to_keep if col in X.columns]

        return X[available_columns]

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Cramér's V statistic for categorical association.
        
        Args:
            x: First categorical variable
            y: Second categorical variable
            
        Returns:
            Cramér's V statistic (0-1)
        """
        try:
            contingency_table = pd.crosstab(x, y)
            chi2, _, _, _ = stats.chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1

            if min_dim == 0:
                return 0.0

            cramers_v = np.sqrt(chi2 / (n * min_dim))
            return min(1.0, cramers_v)  # Cap at 1.0
        except Exception:
            return 0.0


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Outlier detection and handling with fairness considerations.
    """

    def __init__(self, method: str = "iqr", threshold: float = 1.5, protected_attributes: List[str] = None):
        """
        Initialize outlier detector.
        
        Args:
            method: Detection method ('iqr', 'zscore', 'isolation')
            threshold: Threshold for outlier detection
            protected_attributes: Protected attributes to consider
        """
        self.method = method
        self.threshold = threshold
        self.protected_attributes = protected_attributes or []
        self.outlier_bounds_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the outlier detector.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored)
            
        Returns:
            Self
        """
        numeric_features = X.select_dtypes(include=[np.number]).columns
        self.outlier_bounds_ = {}

        for feature in numeric_features:
            if feature in self.protected_attributes:
                continue

            if self.method == "iqr":
                Q1 = X[feature].quantile(0.25)
                Q3 = X[feature].quantile(0.75)
                IQR = Q3 - Q1

                self.outlier_bounds_[feature] = {
                    'lower': Q1 - self.threshold * IQR,
                    'upper': Q3 + self.threshold * IQR
                }

            elif self.method == "zscore":
                mean = X[feature].mean()
                std = X[feature].std()

                self.outlier_bounds_[feature] = {
                    'lower': mean - self.threshold * std,
                    'upper': mean + self.threshold * std
                }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the feature matrix by handling outliers.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.outlier_bounds_ is None:
            raise ValueError("Transformer has not been fitted")

        X_transformed = X.copy()

        for feature, bounds in self.outlier_bounds_.items():
            if feature in X_transformed.columns:
                # Cap outliers at bounds
                X_transformed[feature] = X_transformed[feature].clip(
                    lower=bounds['lower'],
                    upper=bounds['upper']
                )

        return X_transformed


class FeatureEngineeringService:
    """
    Comprehensive feature engineering service with fairness considerations.
    
    Provides advanced feature engineering capabilities including:
    - Automated feature selection with bias prevention
    - Outlier detection and handling
    - Feature scaling and normalization
    - Derived feature creation
    - Fairness-aware preprocessing
    """

    def __init__(
        self,
        protected_attributes: List[str],
        feature_selection_method: str = "mutual_info",
        max_features: Optional[int] = None,
        correlation_threshold: float = 0.7,
        handle_outliers: bool = True,
        outlier_method: str = "iqr"
    ):
        """
        Initialize feature engineering service.
        
        Args:
            protected_attributes: List of protected attribute names
            feature_selection_method: Method for feature selection
            max_features: Maximum number of features to select
            correlation_threshold: Threshold for removing correlated features
            handle_outliers: Whether to detect and handle outliers
            outlier_method: Method for outlier detection
        """
        self.protected_attributes = protected_attributes
        self.feature_selection_method = feature_selection_method
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method

        self.pipeline_ = None
        self.feature_names_ = None
        self.feature_importance_ = None
        self.preprocessing_stats_ = {}

        logger.info("FeatureEngineeringService initialized")

    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Create preprocessing pipeline based on data characteristics.
        
        Args:
            X: Training feature matrix
            
        Returns:
            Configured preprocessing pipeline
        """
        # Identify feature types
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove protected attributes from preprocessing (they're handled separately)
        numeric_features = [f for f in numeric_features if f not in self.protected_attributes]
        categorical_features = [f for f in categorical_features if f not in self.protected_attributes]

        # Create preprocessing steps
        preprocessing_steps = []

        # 1. Outlier handling
        if self.handle_outliers and len(numeric_features) > 0:
            preprocessing_steps.append((
                'outlier_detector',
                OutlierDetector(
                    method=self.outlier_method,
                    protected_attributes=self.protected_attributes
                )
            ))

        # 2. Create derived features
        preprocessing_steps.append((
            'feature_creator',
            DerivedFeatureCreator(numeric_features=numeric_features)
        ))

        # 3. Column transformation (scaling, encoding)
        transformers = []

        if len(numeric_features) > 0:
            transformers.append((
                'numeric',
                StandardScaler(),
                numeric_features
            ))

        if len(categorical_features) > 0:
            transformers.append((
                'categorical',
                OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                categorical_features
            ))

        if transformers:
            preprocessing_steps.append((
                'preprocessor',
                ColumnTransformer(transformers, remainder='passthrough')
            ))

        # 4. Fairness-aware feature selection
        preprocessing_steps.append((
            'fairness_selector',
            FairnessAwareFeatureSelector(
                protected_attributes=self.protected_attributes,
                correlation_threshold=self.correlation_threshold
            )
        ))

        # 5. Statistical feature selection
        if self.max_features is not None:
            if self.feature_selection_method == "mutual_info":
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(self.max_features, len(numeric_features) + len(categorical_features))
                )
            else:
                selector = SelectKBest(
                    score_func=f_classif,
                    k=min(self.max_features, len(numeric_features) + len(categorical_features))
                )

            preprocessing_steps.append(('feature_selector', selector))

        return Pipeline(preprocessing_steps)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureEngineeringService':
        """
        Fit the feature engineering pipeline.
        
        Args:
            X: Training feature matrix
            y: Target variable
            
        Returns:
            Self
        """
        logger.info("Fitting feature engineering pipeline")

        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # Check for protected attributes
        missing_protected = [attr for attr in self.protected_attributes if attr not in X.columns]
        if missing_protected:
            raise ValueError(f"Protected attributes not found in data: {missing_protected}")

        # Create and fit pipeline
        self.pipeline_ = self.create_preprocessing_pipeline(X)

        try:
            self.pipeline_.fit(X, y)

            # Store feature names after transformation
            X_transformed = self.pipeline_.transform(X)
            if hasattr(X_transformed, 'columns'):
                self.feature_names_ = list(X_transformed.columns)
            else:
                # Handle numpy array output
                self.feature_names_ = [f"feature_{i}" for i in range(X_transformed.shape[1])]

            # Compute feature importance if possible
            self._compute_feature_importance(X, y)

            # Store preprocessing statistics
            self._compute_preprocessing_stats(X)

            logger.info(f"Feature engineering pipeline fitted with {len(self.feature_names_)} features")

        except Exception as e:
            logger.error(f"Failed to fit feature engineering pipeline: {e}")
            raise

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature matrix using fitted pipeline.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        if self.pipeline_ is None:
            raise ValueError("Pipeline has not been fitted")

        try:
            X_transformed = self.pipeline_.transform(X)

            # Convert to DataFrame if needed
            if not isinstance(X_transformed, pd.DataFrame):
                X_transformed = pd.DataFrame(
                    X_transformed,
                    columns=self.feature_names_,
                    index=X.index
                )

            logger.debug(f"Transformed {len(X)} samples with {X_transformed.shape[1]} features")
            return X_transformed

        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            raise

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit pipeline and transform data in one step.
        
        Args:
            X: Training feature matrix
            y: Target variable
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names and importance scores
        """
        return self.feature_importance_

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about features.
        
        Returns:
            Dictionary with feature information
        """
        return {
            "feature_names": self.feature_names_,
            "n_features": len(self.feature_names_) if self.feature_names_ else 0,
            "protected_attributes": self.protected_attributes,
            "feature_importance": self.feature_importance_,
            "preprocessing_stats": self.preprocessing_stats_
        }

    def validate_fairness(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate fairness properties of engineered features.
        
        Args:
            X: Transformed feature matrix
            
        Returns:
            Fairness validation results
        """
        results = {
            "protected_attribute_leakage": {},
            "feature_correlation_matrix": {},
            "recommendations": []
        }

        try:
            if not isinstance(X, pd.DataFrame):
                results["error"] = "Input must be a pandas DataFrame"
                return results

            # Check for protected attribute leakage
            for protected_attr in self.protected_attributes:
                if protected_attr in X.columns:
                    correlations = {}
                    for feature in X.columns:
                        if feature != protected_attr:
                            try:
                                corr = abs(X[feature].corr(X[protected_attr]))
                                if not np.isnan(corr):
                                    correlations[feature] = corr
                            except Exception:
                                pass

                    results["protected_attribute_leakage"][protected_attr] = correlations

                    # Generate recommendations
                    high_corr_features = [
                        f for f, corr in correlations.items()
                        if corr > self.correlation_threshold
                    ]

                    if high_corr_features:
                        results["recommendations"].append(
                            f"High correlation detected between {protected_attr} and features: {high_corr_features}"
                        )

            # Compute feature correlation matrix for investigation
            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 1:
                corr_matrix = X[numeric_features].corr()
                results["feature_correlation_matrix"] = corr_matrix.to_dict()

        except Exception as e:
            logger.error(f"Fairness validation failed: {e}")
            results["error"] = str(e)

        return results

    def _compute_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Compute feature importance using mutual information."""
        try:
            # Get feature selector if available
            if hasattr(self.pipeline_, 'named_steps') and 'feature_selector' in self.pipeline_.named_steps:
                selector = self.pipeline_.named_steps['feature_selector']
                if hasattr(selector, 'scores_'):
                    scores = selector.scores_
                    selected_features = selector.get_support()

                    # Map scores to feature names
                    importance_dict = {}
                    feature_idx = 0

                    for i, selected in enumerate(selected_features):
                        if selected and feature_idx < len(self.feature_names_):
                            importance_dict[self.feature_names_[feature_idx]] = scores[i]
                            feature_idx += 1

                    self.feature_importance_ = importance_dict

        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")

    def _compute_preprocessing_stats(self, X: pd.DataFrame):
        """Compute preprocessing statistics."""
        try:
            stats = {
                "original_features": len(X.columns),
                "protected_attributes": len(self.protected_attributes),
                "numeric_features": len(X.select_dtypes(include=[np.number]).columns),
                "categorical_features": len(X.select_dtypes(include=['object', 'category']).columns),
                "missing_values": X.isnull().sum().sum(),
                "final_features": len(self.feature_names_) if self.feature_names_ else 0
            }

            self.preprocessing_stats_ = stats

        except Exception as e:
            logger.warning(f"Could not compute preprocessing stats: {e}")


class DerivedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create derived features from existing numeric features.
    """

    def __init__(self, numeric_features: List[str]):
        """
        Initialize derived feature creator.
        
        Args:
            numeric_features: List of numeric feature names
        """
        self.numeric_features = numeric_features
        self.derived_features_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the derived feature creator.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored)
            
        Returns:
            Self
        """
        # Define derived features to create
        self.derived_features_ = []

        # Create interaction features for pairs of numeric features
        for i, feat1 in enumerate(self.numeric_features):
            for j, feat2 in enumerate(self.numeric_features[i+1:], i+1):
                if feat1 in X.columns and feat2 in X.columns:
                    # Avoid creating too many features
                    if len(self.derived_features_) < 10:
                        self.derived_features_.append({
                            'name': f"{feat1}_x_{feat2}",
                            'type': 'interaction',
                            'features': [feat1, feat2]
                        })

        # Create polynomial features for important numeric features
        for feat in self.numeric_features[:5]:  # Limit to first 5 features
            if feat in X.columns:
                self.derived_features_.append({
                    'name': f"{feat}_squared",
                    'type': 'polynomial',
                    'feature': feat,
                    'power': 2
                })

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the feature matrix by adding derived features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix with derived features
        """
        X_transformed = X.copy()

        for derived_feature in self.derived_features_:
            try:
                if derived_feature['type'] == 'interaction':
                    feat1, feat2 = derived_feature['features']
                    if feat1 in X.columns and feat2 in X.columns:
                        X_transformed[derived_feature['name']] = X[feat1] * X[feat2]

                elif derived_feature['type'] == 'polynomial':
                    feat = derived_feature['feature']
                    power = derived_feature['power']
                    if feat in X.columns:
                        X_transformed[derived_feature['name']] = X[feat] ** power

            except Exception as e:
                logger.warning(f"Could not create derived feature {derived_feature['name']}: {e}")

        return X_transformed


# CLI interface
def main():
    """CLI interface for feature engineering operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Service CLI")
    parser.add_argument("command", choices=["process", "validate", "info"])
    parser.add_argument("--data-path", required=True, help="Path to data file")
    parser.add_argument("--target", default="target", help="Target column name")
    parser.add_argument("--protected", nargs="+", default=["protected"], help="Protected attribute names")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # Load data
    try:
        data = pd.read_csv(args.data_path)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Split features and target
    if args.target not in data.columns:
        print(f"Error: Target column '{args.target}' not found")
        return

    X = data.drop(columns=[args.target])
    y = data[args.target]

    # Create feature engineering service
    service = FeatureEngineeringService(
        protected_attributes=args.protected,
        max_features=50
    )

    if args.command == "process":
        # Process features
        print("Processing features...")
        X_processed = service.fit_transform(X, y)
        print(f"Processed features shape: {X_processed.shape}")

        # Save results
        if args.output:
            result_data = X_processed.copy()
            result_data[args.target] = y
            result_data.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")

        # Show feature info
        info = service.get_feature_info()
        print(f"Final features: {info['n_features']}")
        print(f"Protected attributes: {info['protected_attributes']}")

    elif args.command == "validate":
        # Validate fairness
        print("Validating fairness...")
        X_processed = service.fit_transform(X, y)
        validation = service.validate_fairness(X_processed)

        print("Fairness validation results:")
        if validation.get("recommendations"):
            for rec in validation["recommendations"]:
                print(f"  - {rec}")
        else:
            print("  No fairness issues detected")

    elif args.command == "info":
        # Show feature information
        service.fit(X, y)
        info = service.get_feature_info()

        print("Feature Engineering Information:")
        print(f"  Original features: {info['preprocessing_stats'].get('original_features', 'N/A')}")
        print(f"  Final features: {info['n_features']}")
        print(f"  Protected attributes: {info['protected_attributes']}")

        if info.get('feature_importance'):
            print("  Top important features:")
            sorted_features = sorted(
                info['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feat, score in sorted_features[:10]:
                print(f"    {feat}: {score:.4f}")


if __name__ == "__main__":
    main()
