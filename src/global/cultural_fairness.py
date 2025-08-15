"""
Cross-Cultural Fairness Framework.

Provides comprehensive support for evaluating fairness across different
cultural contexts, addressing varying definitions of fairness and
protected attributes across global communities.

Research contributions:
- Cultural context-aware fairness evaluation
- Cross-cultural bias detection and mitigation
- Intersectional fairness across multiple cultural dimensions
- Adaptive fairness metrics based on cultural values
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..fairness_metrics import compute_fairness_metrics
from ..logging_config import get_logger

logger = get_logger(__name__)


class CulturalRegion(Enum):
    """Major cultural regions with distinct fairness perspectives."""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    EAST_ASIA = "east_asia"
    SOUTH_ASIA = "south_asia"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    LATIN_AMERICA = "latin_america"
    OCEANIA = "oceania"


class FairnessPhilosophy(Enum):
    """Different philosophical approaches to fairness."""
    INDIVIDUALISTIC = "individualistic"    # Western emphasis on individual rights
    COLLECTIVISTIC = "collectivistic"      # Eastern emphasis on group harmony
    EGALITARIAN = "egalitarian"            # Equal treatment and outcomes
    MERITOCRATIC = "meritocratic"          # Achievement-based fairness
    COMMUNITARIAN = "communitarian"        # Community-centered fairness
    CONTEXTUAL = "contextual"              # Context-dependent fairness


class ProtectedAttributeType(Enum):
    """Types of protected attributes with varying cultural significance."""
    RACE_ETHNICITY = "race_ethnicity"
    GENDER = "gender"
    AGE = "age"
    RELIGION = "religion"
    SOCIOECONOMIC_STATUS = "socioeconomic_status"
    CASTE = "caste"
    TRIBAL_AFFILIATION = "tribal_affiliation"
    POLITICAL_AFFILIATION = "political_affiliation"
    SEXUAL_ORIENTATION = "sexual_orientation"
    DISABILITY_STATUS = "disability_status"


@dataclass
class CulturalContext:
    """Represents a cultural context with its fairness characteristics."""
    region: CulturalRegion
    country_codes: List[str]
    primary_philosophy: FairnessPhilosophy
    secondary_philosophies: List[FairnessPhilosophy]
    protected_attributes: List[ProtectedAttributeType]
    attribute_priorities: Dict[ProtectedAttributeType, float]
    fairness_metric_weights: Dict[str, float]
    cultural_values: Dict[str, float]
    legal_framework: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'region': self.region.value,
            'country_codes': self.country_codes,
            'primary_philosophy': self.primary_philosophy.value,
            'secondary_philosophies': [p.value for p in self.secondary_philosophies],
            'protected_attributes': [a.value for a in self.protected_attributes],
            'attribute_priorities': {a.value: w for a, w in self.attribute_priorities.items()},
            'fairness_metric_weights': self.fairness_metric_weights,
            'cultural_values': self.cultural_values,
            'legal_framework': self.legal_framework
        }


@dataclass
class CrossCulturalFairnessResult:
    """Result from cross-cultural fairness analysis."""
    primary_context: CulturalContext
    comparison_contexts: List[CulturalContext]
    fairness_scores: Dict[str, Dict[str, float]]
    cultural_differences: List[Dict[str, Any]]
    recommendations: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'primary_context': self.primary_context.to_dict(),
            'comparison_contexts': [ctx.to_dict() for ctx in self.comparison_contexts],
            'fairness_scores': self.fairness_scores,
            'cultural_differences': self.cultural_differences,
            'recommendations': self.recommendations,
            'confidence_intervals': {k: list(v) for k, v in self.confidence_intervals.items()},
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }


class CulturalContextManager:
    """
    Manages cultural contexts and their fairness characteristics.
    
    Provides a comprehensive database of cultural contexts with
    their associated fairness values, legal frameworks, and
    protected attribute priorities.
    """

    def __init__(self, contexts_file: Optional[str] = None):
        """
        Initialize cultural context manager.
        
        Args:
            contexts_file: Optional file path for custom cultural contexts
        """
        self.contexts: Dict[str, CulturalContext] = {}
        self._initialize_default_contexts()

        if contexts_file:
            self._load_contexts_from_file(contexts_file)

        logger.info(f"CulturalContextManager initialized with {len(self.contexts)} contexts")

    def _initialize_default_contexts(self):
        """Initialize default cultural contexts based on research."""

        # North American context
        north_america = CulturalContext(
            region=CulturalRegion.NORTH_AMERICA,
            country_codes=['US', 'CA'],
            primary_philosophy=FairnessPhilosophy.INDIVIDUALISTIC,
            secondary_philosophies=[FairnessPhilosophy.EGALITARIAN, FairnessPhilosophy.MERITOCRATIC],
            protected_attributes=[
                ProtectedAttributeType.RACE_ETHNICITY,
                ProtectedAttributeType.GENDER,
                ProtectedAttributeType.AGE,
                ProtectedAttributeType.RELIGION,
                ProtectedAttributeType.DISABILITY_STATUS
            ],
            attribute_priorities={
                ProtectedAttributeType.RACE_ETHNICITY: 0.9,
                ProtectedAttributeType.GENDER: 0.8,
                ProtectedAttributeType.AGE: 0.6,
                ProtectedAttributeType.RELIGION: 0.5,
                ProtectedAttributeType.DISABILITY_STATUS: 0.7
            },
            fairness_metric_weights={
                'demographic_parity': 0.8,
                'equalized_odds': 0.9,
                'individual_fairness': 0.7,
                'calibration': 0.6
            },
            cultural_values={
                'individualism': 0.9,
                'power_distance': 0.3,
                'uncertainty_avoidance': 0.5,
                'long_term_orientation': 0.6
            },
            legal_framework={
                'anti_discrimination_laws': True,
                'data_protection_level': 'moderate',
                'algorithmic_accountability': True
            }
        )

        # European context
        europe = CulturalContext(
            region=CulturalRegion.EUROPE,
            country_codes=['DE', 'FR', 'UK', 'NL', 'SE', 'IT', 'ES'],
            primary_philosophy=FairnessPhilosophy.EGALITARIAN,
            secondary_philosophies=[FairnessPhilosophy.COMMUNITARIAN, FairnessPhilosophy.INDIVIDUALISTIC],
            protected_attributes=[
                ProtectedAttributeType.RACE_ETHNICITY,
                ProtectedAttributeType.GENDER,
                ProtectedAttributeType.AGE,
                ProtectedAttributeType.RELIGION,
                ProtectedAttributeType.DISABILITY_STATUS,
                ProtectedAttributeType.SEXUAL_ORIENTATION
            ],
            attribute_priorities={
                ProtectedAttributeType.RACE_ETHNICITY: 0.8,
                ProtectedAttributeType.GENDER: 0.9,
                ProtectedAttributeType.AGE: 0.7,
                ProtectedAttributeType.RELIGION: 0.6,
                ProtectedAttributeType.DISABILITY_STATUS: 0.8,
                ProtectedAttributeType.SEXUAL_ORIENTATION: 0.7
            },
            fairness_metric_weights={
                'demographic_parity': 0.9,
                'equalized_odds': 0.8,
                'individual_fairness': 0.8,
                'calibration': 0.7
            },
            cultural_values={
                'individualism': 0.6,
                'power_distance': 0.4,
                'uncertainty_avoidance': 0.7,
                'long_term_orientation': 0.7
            },
            legal_framework={
                'anti_discrimination_laws': True,
                'data_protection_level': 'high',  # GDPR
                'algorithmic_accountability': True
            }
        )

        # East Asian context
        east_asia = CulturalContext(
            region=CulturalRegion.EAST_ASIA,
            country_codes=['CN', 'JP', 'KR', 'TW', 'HK'],
            primary_philosophy=FairnessPhilosophy.COLLECTIVISTIC,
            secondary_philosophies=[FairnessPhilosophy.MERITOCRATIC, FairnessPhilosophy.CONTEXTUAL],
            protected_attributes=[
                ProtectedAttributeType.AGE,
                ProtectedAttributeType.GENDER,
                ProtectedAttributeType.SOCIOECONOMIC_STATUS,
                ProtectedAttributeType.RACE_ETHNICITY
            ],
            attribute_priorities={
                ProtectedAttributeType.AGE: 0.9,
                ProtectedAttributeType.GENDER: 0.7,
                ProtectedAttributeType.SOCIOECONOMIC_STATUS: 0.8,
                ProtectedAttributeType.RACE_ETHNICITY: 0.6
            },
            fairness_metric_weights={
                'demographic_parity': 0.6,
                'equalized_odds': 0.7,
                'individual_fairness': 0.5,
                'calibration': 0.8
            },
            cultural_values={
                'individualism': 0.2,
                'power_distance': 0.8,
                'uncertainty_avoidance': 0.6,
                'long_term_orientation': 0.9
            },
            legal_framework={
                'anti_discrimination_laws': False,
                'data_protection_level': 'moderate',
                'algorithmic_accountability': False
            }
        )

        # South Asian context
        south_asia = CulturalContext(
            region=CulturalRegion.SOUTH_ASIA,
            country_codes=['IN', 'BD', 'PK', 'LK', 'NP'],
            primary_philosophy=FairnessPhilosophy.CONTEXTUAL,
            secondary_philosophies=[FairnessPhilosophy.COMMUNITARIAN, FairnessPhilosophy.COLLECTIVISTIC],
            protected_attributes=[
                ProtectedAttributeType.CASTE,
                ProtectedAttributeType.RELIGION,
                ProtectedAttributeType.GENDER,
                ProtectedAttributeType.TRIBAL_AFFILIATION,
                ProtectedAttributeType.SOCIOECONOMIC_STATUS
            ],
            attribute_priorities={
                ProtectedAttributeType.CASTE: 0.9,
                ProtectedAttributeType.RELIGION: 0.8,
                ProtectedAttributeType.GENDER: 0.7,
                ProtectedAttributeType.TRIBAL_AFFILIATION: 0.8,
                ProtectedAttributeType.SOCIOECONOMIC_STATUS: 0.8
            },
            fairness_metric_weights={
                'demographic_parity': 0.9,
                'equalized_odds': 0.8,
                'individual_fairness': 0.6,
                'calibration': 0.7
            },
            cultural_values={
                'individualism': 0.3,
                'power_distance': 0.9,
                'uncertainty_avoidance': 0.4,
                'long_term_orientation': 0.6
            },
            legal_framework={
                'anti_discrimination_laws': True,  # Constitutional provisions
                'data_protection_level': 'moderate',
                'algorithmic_accountability': False
            }
        )

        # Store contexts
        self.contexts['north_america'] = north_america
        self.contexts['europe'] = europe
        self.contexts['east_asia'] = east_asia
        self.contexts['south_asia'] = south_asia

    def _load_contexts_from_file(self, contexts_file: str):
        """Load additional cultural contexts from file."""
        try:
            with open(contexts_file) as f:
                contexts_data = json.load(f)

            for context_name, context_dict in contexts_data.items():
                context = self._dict_to_context(context_dict)
                self.contexts[context_name] = context

            logger.info(f"Loaded {len(contexts_data)} contexts from file")

        except Exception as e:
            logger.error(f"Failed to load contexts from file {contexts_file}: {e}")

    def _dict_to_context(self, context_dict: Dict[str, Any]) -> CulturalContext:
        """Convert dictionary to CulturalContext object."""
        return CulturalContext(
            region=CulturalRegion(context_dict['region']),
            country_codes=context_dict['country_codes'],
            primary_philosophy=FairnessPhilosophy(context_dict['primary_philosophy']),
            secondary_philosophies=[
                FairnessPhilosophy(p) for p in context_dict['secondary_philosophies']
            ],
            protected_attributes=[
                ProtectedAttributeType(a) for a in context_dict['protected_attributes']
            ],
            attribute_priorities={
                ProtectedAttributeType(a): w
                for a, w in context_dict['attribute_priorities'].items()
            },
            fairness_metric_weights=context_dict['fairness_metric_weights'],
            cultural_values=context_dict['cultural_values'],
            legal_framework=context_dict['legal_framework']
        )

    def get_context(self, context_name: str) -> Optional[CulturalContext]:
        """Get cultural context by name."""
        return self.contexts.get(context_name)

    def get_context_by_country(self, country_code: str) -> Optional[CulturalContext]:
        """Get cultural context by country code."""
        for context in self.contexts.values():
            if country_code.upper() in context.country_codes:
                return context
        return None

    def list_contexts(self) -> List[str]:
        """List available cultural context names."""
        return list(self.contexts.keys())

    def add_custom_context(self, name: str, context: CulturalContext):
        """Add a custom cultural context."""
        self.contexts[name] = context
        logger.info(f"Added custom cultural context: {name}")

    def get_context_similarity(self, context1: str, context2: str) -> float:
        """Calculate similarity between two cultural contexts."""
        ctx1 = self.get_context(context1)
        ctx2 = self.get_context(context2)

        if not ctx1 or not ctx2:
            return 0.0

        # Calculate similarity based on cultural values
        value_similarity = 0.0
        common_values = set(ctx1.cultural_values.keys()) & set(ctx2.cultural_values.keys())

        if common_values:
            for value in common_values:
                diff = abs(ctx1.cultural_values[value] - ctx2.cultural_values[value])
                value_similarity += 1.0 - diff  # Inverted difference

            value_similarity /= len(common_values)

        # Calculate similarity based on protected attributes
        attr_similarity = 0.0
        all_attrs = set(ctx1.protected_attributes) | set(ctx2.protected_attributes)
        common_attrs = set(ctx1.protected_attributes) & set(ctx2.protected_attributes)

        if all_attrs:
            attr_similarity = len(common_attrs) / len(all_attrs)

        # Calculate similarity based on fairness philosophy
        phil_similarity = 1.0 if ctx1.primary_philosophy == ctx2.primary_philosophy else 0.5

        # Weighted average
        total_similarity = (0.4 * value_similarity + 0.4 * attr_similarity + 0.2 * phil_similarity)

        return min(1.0, max(0.0, total_similarity))


class CrossCulturalValidator:
    """
    Validates fairness across different cultural contexts.
    
    Evaluates how fairness metrics and bias detection results
    vary across different cultural perspectives and value systems.
    """

    def __init__(self, context_manager: CulturalContextManager):
        """
        Initialize cross-cultural validator.
        
        Args:
            context_manager: Cultural context manager
        """
        self.context_manager = context_manager
        self.validation_history: List[CrossCulturalFairnessResult] = []

        logger.info("CrossCulturalValidator initialized")

    def validate_across_cultures(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive_attrs: pd.DataFrame,
        primary_context: str,
        comparison_contexts: List[str],
        confidence_level: float = 0.95
    ) -> CrossCulturalFairnessResult:
        """
        Validate fairness across multiple cultural contexts.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attrs: Sensitive attributes
            primary_context: Primary cultural context
            comparison_contexts: List of contexts to compare against
            confidence_level: Confidence level for intervals
            
        Returns:
            Cross-cultural fairness analysis result
        """
        logger.info(f"Validating fairness across {len(comparison_contexts) + 1} cultural contexts")

        # Get cultural contexts
        primary_ctx = self.context_manager.get_context(primary_context)
        comparison_ctxs = [
            self.context_manager.get_context(ctx) for ctx in comparison_contexts
            if self.context_manager.get_context(ctx) is not None
        ]

        if not primary_ctx:
            raise ValueError(f"Primary context '{primary_context}' not found")

        # Evaluate fairness for each context
        fairness_scores = {}
        cultural_differences = []

        # Primary context evaluation
        primary_scores = self._evaluate_context_fairness(
            y_true, y_pred, sensitive_attrs, primary_ctx
        )
        fairness_scores[primary_context] = primary_scores

        # Comparison contexts evaluation
        for i, ctx in enumerate(comparison_ctxs):
            ctx_name = comparison_contexts[i]
            ctx_scores = self._evaluate_context_fairness(
                y_true, y_pred, sensitive_attrs, ctx
            )
            fairness_scores[ctx_name] = ctx_scores

            # Analyze differences
            differences = self._analyze_cultural_differences(
                primary_ctx, ctx, primary_scores, ctx_scores
            )
            cultural_differences.extend(differences)

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            fairness_scores, confidence_level
        )

        # Generate recommendations
        recommendations = self._generate_cultural_recommendations(
            primary_ctx, comparison_ctxs, cultural_differences
        )

        result = CrossCulturalFairnessResult(
            primary_context=primary_ctx,
            comparison_contexts=comparison_ctxs,
            fairness_scores=fairness_scores,
            cultural_differences=cultural_differences,
            recommendations=recommendations,
            confidence_intervals=confidence_intervals
        )

        self.validation_history.append(result)

        logger.info(f"Cross-cultural validation completed with {len(cultural_differences)} differences identified")
        return result

    def _evaluate_context_fairness(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive_attrs: pd.DataFrame,
        context: CulturalContext
    ) -> Dict[str, float]:
        """Evaluate fairness metrics weighted by cultural context."""

        # Base fairness metrics
        base_scores = {}

        for attr_col in sensitive_attrs.columns:
            # Map to protected attribute type
            attr_type = self._map_to_protected_attribute(attr_col)

            if attr_type in context.protected_attributes:
                # Get attribute priority weight
                priority_weight = context.attribute_priorities.get(attr_type, 1.0)

                # Compute standard fairness metrics
                overall, by_group = compute_fairness_metrics(y_true, y_pred, sensitive_attrs[attr_col])

                # Apply cultural weighting
                for metric_name, metric_value in overall.items():
                    if metric_name in context.fairness_metric_weights:
                        cultural_weight = context.fairness_metric_weights[metric_name]
                        weighted_score = metric_value * cultural_weight * priority_weight

                        key = f"{attr_col}_{metric_name}"
                        base_scores[key] = weighted_score

        # Compute context-specific aggregate scores
        aggregate_scores = self._compute_aggregate_scores(base_scores, context)
        base_scores.update(aggregate_scores)

        return base_scores

    def _map_to_protected_attribute(self, attr_name: str) -> ProtectedAttributeType:
        """Map attribute name to protected attribute type."""
        attr_mapping = {
            'race': ProtectedAttributeType.RACE_ETHNICITY,
            'ethnicity': ProtectedAttributeType.RACE_ETHNICITY,
            'gender': ProtectedAttributeType.GENDER,
            'sex': ProtectedAttributeType.GENDER,
            'age': ProtectedAttributeType.AGE,
            'religion': ProtectedAttributeType.RELIGION,
            'caste': ProtectedAttributeType.CASTE,
            'income': ProtectedAttributeType.SOCIOECONOMIC_STATUS,
            'education': ProtectedAttributeType.SOCIOECONOMIC_STATUS,
            'disability': ProtectedAttributeType.DISABILITY_STATUS,
            'sexual_orientation': ProtectedAttributeType.SEXUAL_ORIENTATION,
            'political': ProtectedAttributeType.POLITICAL_AFFILIATION
        }

        attr_lower = attr_name.lower()
        for key, attr_type in attr_mapping.items():
            if key in attr_lower:
                return attr_type

        # Default to race/ethnicity if unknown
        return ProtectedAttributeType.RACE_ETHNICITY

    def _compute_aggregate_scores(
        self,
        base_scores: Dict[str, float],
        context: CulturalContext
    ) -> Dict[str, float]:
        """Compute context-specific aggregate fairness scores."""

        aggregate_scores = {}

        # Overall fairness score based on cultural philosophy
        if context.primary_philosophy == FairnessPhilosophy.INDIVIDUALISTIC:
            # Emphasize individual fairness metrics
            individual_scores = [v for k, v in base_scores.items() if 'individual' in k.lower()]
            if individual_scores:
                aggregate_scores['cultural_fairness_score'] = np.mean(individual_scores)

        elif context.primary_philosophy == FairnessPhilosophy.COLLECTIVISTIC:
            # Emphasize group fairness metrics
            group_scores = [v for k, v in base_scores.items() if 'demographic' in k.lower() or 'equalized' in k.lower()]
            if group_scores:
                aggregate_scores['cultural_fairness_score'] = np.mean(group_scores)

        elif context.primary_philosophy == FairnessPhilosophy.EGALITARIAN:
            # Emphasize equal treatment metrics
            equality_scores = [v for k, v in base_scores.items() if 'parity' in k.lower()]
            if equality_scores:
                aggregate_scores['cultural_fairness_score'] = np.mean(equality_scores)

        elif context.primary_philosophy == FairnessPhilosophy.MERITOCRATIC:
            # Emphasize calibration and predictive fairness
            merit_scores = [v for k, v in base_scores.items() if 'calibration' in k.lower()]
            if merit_scores:
                aggregate_scores['cultural_fairness_score'] = np.mean(merit_scores)

        else:
            # Default: average of all scores
            if base_scores:
                aggregate_scores['cultural_fairness_score'] = np.mean(list(base_scores.values()))

        # Context-specific weighted score
        if base_scores and context.fairness_metric_weights:
            weighted_total = 0.0
            total_weight = 0.0

            for score_name, score_value in base_scores.items():
                for metric_name, weight in context.fairness_metric_weights.items():
                    if metric_name in score_name:
                        weighted_total += score_value * weight
                        total_weight += weight
                        break

            if total_weight > 0:
                aggregate_scores['weighted_fairness_score'] = weighted_total / total_weight

        return aggregate_scores

    def _analyze_cultural_differences(
        self,
        ctx1: CulturalContext,
        ctx2: CulturalContext,
        scores1: Dict[str, float],
        scores2: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Analyze differences between cultural contexts."""

        differences = []

        # Compare fairness scores
        common_metrics = set(scores1.keys()) & set(scores2.keys())

        for metric in common_metrics:
            score_diff = abs(scores1[metric] - scores2[metric])

            if score_diff > 0.1:  # Significant difference threshold
                differences.append({
                    'type': 'fairness_score_difference',
                    'metric': metric,
                    'context1_score': scores1[metric],
                    'context2_score': scores2[metric],
                    'difference': score_diff,
                    'significance': 'high' if score_diff > 0.2 else 'moderate'
                })

        # Compare protected attribute priorities
        common_attrs = set(ctx1.protected_attributes) & set(ctx2.protected_attributes)

        for attr in common_attrs:
            priority1 = ctx1.attribute_priorities.get(attr, 0.5)
            priority2 = ctx2.attribute_priorities.get(attr, 0.5)
            priority_diff = abs(priority1 - priority2)

            if priority_diff > 0.2:
                differences.append({
                    'type': 'attribute_priority_difference',
                    'attribute': attr.value,
                    'context1_priority': priority1,
                    'context2_priority': priority2,
                    'difference': priority_diff
                })

        # Compare cultural values
        common_values = set(ctx1.cultural_values.keys()) & set(ctx2.cultural_values.keys())

        for value in common_values:
            value_diff = abs(ctx1.cultural_values[value] - ctx2.cultural_values[value])

            if value_diff > 0.3:
                differences.append({
                    'type': 'cultural_value_difference',
                    'value': value,
                    'context1_value': ctx1.cultural_values[value],
                    'context2_value': ctx2.cultural_values[value],
                    'difference': value_diff
                })

        return differences

    def _calculate_confidence_intervals(
        self,
        fairness_scores: Dict[str, Dict[str, float]],
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for fairness scores."""

        confidence_intervals = {}

        # Simplified confidence interval calculation
        # In practice, would use bootstrap or other statistical methods

        for context_name, scores in fairness_scores.items():
            for metric_name, score in scores.items():
                # Assume normal distribution with estimated standard error
                # This is a simplification - real implementation would use proper statistics
                standard_error = 0.05  # Assumed standard error

                from scipy import stats as scipy_stats

                # Calculate confidence interval
                z_score = scipy_stats.norm.ppf((1 + confidence_level) / 2)
                margin_error = z_score * standard_error

                ci_key = f"{context_name}_{metric_name}"
                confidence_intervals[ci_key] = (
                    max(0.0, score - margin_error),
                    min(1.0, score + margin_error)
                )

        return confidence_intervals

    def _generate_cultural_recommendations(
        self,
        primary_ctx: CulturalContext,
        comparison_ctxs: List[CulturalContext],
        differences: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for cross-cultural fairness."""

        recommendations = []

        # Analyze types of differences
        score_differences = [d for d in differences if d['type'] == 'fairness_score_difference']
        priority_differences = [d for d in differences if d['type'] == 'attribute_priority_difference']
        value_differences = [d for d in differences if d['type'] == 'cultural_value_difference']

        if score_differences:
            recommendations.append(
                "Consider context-specific fairness thresholds when deploying across cultures"
            )

            high_diff_metrics = [d['metric'] for d in score_differences if d['significance'] == 'high']
            if high_diff_metrics:
                recommendations.append(
                    f"Pay special attention to {', '.join(high_diff_metrics[:3])} metrics which show significant cultural variation"
                )

        if priority_differences:
            recommendations.append(
                "Adjust protected attribute priorities based on cultural context"
            )

            varying_attrs = list(set(d['attribute'] for d in priority_differences))
            recommendations.append(
                f"Attributes showing cultural variation: {', '.join(varying_attrs[:3])}"
            )

        if value_differences:
            recommendations.append(
                "Consider cultural values when designing fairness interventions"
            )

            # Specific recommendations based on primary context philosophy
            if primary_ctx.primary_philosophy == FairnessPhilosophy.INDIVIDUALISTIC:
                recommendations.append(
                    "Emphasize individual fairness metrics and personal autonomy"
                )
            elif primary_ctx.primary_philosophy == FairnessPhilosophy.COLLECTIVISTIC:
                recommendations.append(
                    "Focus on group fairness and community harmony"
                )
            elif primary_ctx.primary_philosophy == FairnessPhilosophy.CONTEXTUAL:
                recommendations.append(
                    "Implement adaptive fairness based on specific situational context"
                )

        # General recommendations
        recommendations.extend([
            "Conduct stakeholder consultations in each cultural context",
            "Validate fairness metrics with local cultural experts",
            "Consider implementing multiple fairness definitions simultaneously",
            "Monitor fairness outcomes across different cultural groups",
            "Provide culturally appropriate explanations of fairness decisions"
        ])

        return recommendations[:10]  # Return top 10 recommendations

    def generate_cultural_fairness_report(
        self,
        validation_results: List[CrossCulturalFairnessResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive cross-cultural fairness report."""

        if not validation_results:
            validation_results = self.validation_history

        if not validation_results:
            return {'error': 'No validation results available'}

        # Aggregate analysis
        total_validations = len(validation_results)
        total_contexts = len(set(
            [result.primary_context.region.value for result in validation_results] +
            [ctx.region.value for result in validation_results for ctx in result.comparison_contexts]
        ))

        # Common differences
        all_differences = []
        for result in validation_results:
            all_differences.extend(result.cultural_differences)

        difference_types = {}
        for diff in all_differences:
            diff_type = diff['type']
            difference_types[diff_type] = difference_types.get(diff_type, 0) + 1

        # Most affected metrics
        metric_impacts = {}
        for diff in all_differences:
            if 'metric' in diff:
                metric = diff['metric']
                metric_impacts[metric] = metric_impacts.get(metric, 0) + 1

        # Most variable attributes
        attribute_impacts = {}
        for diff in all_differences:
            if 'attribute' in diff:
                attr = diff['attribute']
                attribute_impacts[attr] = attribute_impacts.get(attr, 0) + 1

        # Aggregate recommendations
        all_recommendations = []
        for result in validation_results:
            all_recommendations.extend(result.recommendations)

        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        top_recommendations = sorted(
            recommendation_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            'report_summary': {
                'total_validations': total_validations,
                'cultural_contexts_analyzed': total_contexts,
                'total_differences_identified': len(all_differences),
                'most_common_difference_type': max(difference_types.items(), key=lambda x: x[1])[0] if difference_types else None
            },
            'difference_analysis': {
                'difference_types': difference_types,
                'most_affected_metrics': dict(sorted(metric_impacts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'most_variable_attributes': dict(sorted(attribute_impacts.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            'recommendations': {
                'top_recommendations': [rec for rec, count in top_recommendations],
                'recommendation_frequencies': dict(top_recommendations)
            },
            'cultural_insights': {
                'contexts_requiring_attention': [
                    result.primary_context.region.value
                    for result in validation_results
                    if len(result.cultural_differences) > 5
                ],
                'philosophy_conflicts': [
                    (result.primary_context.primary_philosophy.value,
                     [ctx.primary_philosophy.value for ctx in result.comparison_contexts])
                    for result in validation_results
                    if any(ctx.primary_philosophy != result.primary_context.primary_philosophy
                          for ctx in result.comparison_contexts)
                ]
            }
        }


class CulturalFairnessFramework:
    """
    Comprehensive framework for cultural fairness research.
    
    Integrates cultural context management with cross-cultural
    validation to provide comprehensive fairness evaluation
    across different cultural perspectives.
    """

    def __init__(self, contexts_file: Optional[str] = None):
        """
        Initialize cultural fairness framework.
        
        Args:
            contexts_file: Optional file for custom cultural contexts
        """
        self.context_manager = CulturalContextManager(contexts_file)
        self.validator = CrossCulturalValidator(self.context_manager)

        # Framework state
        self.active_contexts: List[str] = []
        self.evaluation_history: List[Dict[str, Any]] = []

        logger.info("CulturalFairnessFramework initialized")

    def set_active_contexts(self, context_names: List[str]):
        """Set active cultural contexts for evaluation."""
        valid_contexts = []

        for context_name in context_names:
            if self.context_manager.get_context(context_name):
                valid_contexts.append(context_name)
            else:
                logger.warning(f"Context '{context_name}' not found")

        self.active_contexts = valid_contexts
        logger.info(f"Set {len(valid_contexts)} active contexts: {valid_contexts}")

    def evaluate_cultural_fairness(
        self,
        algorithm_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive_attrs: pd.DataFrame,
        primary_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate fairness across all active cultural contexts.
        
        Args:
            algorithm_name: Name of the algorithm being evaluated
            y_true: True labels
            y_pred: Predicted labels  
            sensitive_attrs: Sensitive attributes
            primary_context: Primary cultural context (uses first active if None)
            
        Returns:
            Comprehensive cultural fairness evaluation
        """
        if not self.active_contexts:
            raise ValueError("No active contexts set. Use set_active_contexts() first.")

        if primary_context is None:
            primary_context = self.active_contexts[0]

        comparison_contexts = [ctx for ctx in self.active_contexts if ctx != primary_context]

        logger.info(f"Evaluating cultural fairness for {algorithm_name} across {len(self.active_contexts)} contexts")

        # Perform cross-cultural validation
        validation_result = self.validator.validate_across_cultures(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_attrs=sensitive_attrs,
            primary_context=primary_context,
            comparison_contexts=comparison_contexts
        )

        # Additional cultural analysis
        cultural_analysis = self._perform_cultural_analysis(
            validation_result, sensitive_attrs
        )

        # Compile results
        evaluation_result = {
            'algorithm_name': algorithm_name,
            'primary_context': primary_context,
            'comparison_contexts': comparison_contexts,
            'validation_result': validation_result.to_dict(),
            'cultural_analysis': cultural_analysis,
            'evaluation_timestamp': datetime.now().isoformat()
        }

        self.evaluation_history.append(evaluation_result)

        logger.info(f"Cultural fairness evaluation completed for {algorithm_name}")
        return evaluation_result

    def _perform_cultural_analysis(
        self,
        validation_result: CrossCulturalFairnessResult,
        sensitive_attrs: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform additional cultural analysis."""

        analysis = {
            'cultural_sensitivity_score': 0.0,
            'context_compatibility': {},
            'attribute_cultural_impact': {},
            'fairness_philosophy_alignment': {},
            'recommendations_by_context': {}
        }

        # Calculate cultural sensitivity score
        total_differences = len(validation_result.cultural_differences)
        significant_differences = len([
            d for d in validation_result.cultural_differences
            if d.get('significance') == 'high'
        ])

        if total_differences > 0:
            analysis['cultural_sensitivity_score'] = 1.0 - (significant_differences / total_differences)
        else:
            analysis['cultural_sensitivity_score'] = 1.0

        # Analyze context compatibility
        primary_ctx = validation_result.primary_context

        for comparison_ctx in validation_result.comparison_contexts:
            compatibility = self.context_manager.get_context_similarity(
                primary_ctx.region.value, comparison_ctx.region.value
            )
            analysis['context_compatibility'][comparison_ctx.region.value] = compatibility

        # Analyze attribute cultural impact
        for attr_col in sensitive_attrs.columns:
            attr_differences = [
                d for d in validation_result.cultural_differences
                if d.get('attribute') == attr_col or attr_col in str(d.get('metric', ''))
            ]

            impact_score = len(attr_differences) / max(1, len(validation_result.comparison_contexts))
            analysis['attribute_cultural_impact'][attr_col] = impact_score

        # Analyze fairness philosophy alignment
        for ctx in validation_result.comparison_contexts:
            philosophy_alignment = (
                1.0 if ctx.primary_philosophy == primary_ctx.primary_philosophy
                else 0.5 if ctx.primary_philosophy in primary_ctx.secondary_philosophies
                else 0.0
            )
            analysis['fairness_philosophy_alignment'][ctx.region.value] = philosophy_alignment

        return analysis

    def compare_algorithms_culturally(
        self,
        algorithm_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple algorithms from cultural fairness perspective.
        
        Args:
            algorithm_results: Dictionary mapping algorithm names to their cultural evaluation results
            
        Returns:
            Comprehensive cross-algorithm cultural comparison
        """
        logger.info(f"Comparing {len(algorithm_results)} algorithms culturally")

        comparison = {
            'algorithms': list(algorithm_results.keys()),
            'cultural_sensitivity_ranking': {},
            'context_specific_rankings': {},
            'philosophy_alignment_scores': {},
            'overall_cultural_fairness': {},
            'recommendations': []
        }

        # Rank algorithms by cultural sensitivity
        sensitivity_scores = {}
        for alg_name, result in algorithm_results.items():
            if 'cultural_analysis' in result:
                score = result['cultural_analysis'].get('cultural_sensitivity_score', 0.0)
                sensitivity_scores[alg_name] = score

        comparison['cultural_sensitivity_ranking'] = dict(
            sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        )

        # Context-specific rankings
        all_contexts = set()
        for result in algorithm_results.values():
            if 'comparison_contexts' in result:
                all_contexts.update(result['comparison_contexts'])
                if 'primary_context' in result:
                    all_contexts.add(result['primary_context'])

        for context in all_contexts:
            context_scores = {}

            for alg_name, result in algorithm_results.items():
                # Get fairness scores for this context
                if 'validation_result' in result:
                    fairness_scores = result['validation_result'].get('fairness_scores', {})
                    if context in fairness_scores:
                        # Use cultural fairness score if available, otherwise use average
                        scores = fairness_scores[context]
                        context_score = scores.get('cultural_fairness_score',
                                                 np.mean(list(scores.values())))
                        context_scores[alg_name] = context_score

            if context_scores:
                comparison['context_specific_rankings'][context] = dict(
                    sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
                )

        # Generate cross-algorithm recommendations
        comparison['recommendations'] = self._generate_cross_algorithm_recommendations(
            algorithm_results, comparison
        )

        return comparison

    def _generate_cross_algorithm_recommendations(
        self,
        algorithm_results: Dict[str, Dict[str, Any]],
        comparison: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for cross-algorithm cultural fairness."""

        recommendations = []

        # Best algorithm by cultural sensitivity
        if comparison['cultural_sensitivity_ranking']:
            best_alg = list(comparison['cultural_sensitivity_ranking'].keys())[0]
            recommendations.append(
                f"Consider {best_alg} for culturally diverse deployments due to highest cultural sensitivity"
            )

        # Context-specific recommendations
        for context, rankings in comparison['context_specific_rankings'].items():
            if rankings:
                best_for_context = list(rankings.keys())[0]
                recommendations.append(
                    f"For {context} context, {best_for_context} shows best cultural alignment"
                )

        # General cultural fairness recommendations
        recommendations.extend([
            "Conduct stakeholder engagement in each target cultural context",
            "Implement context-aware fairness thresholds",
            "Monitor fairness metrics across all cultural groups",
            "Provide culturally appropriate model explanations",
            "Regular cross-cultural fairness audits"
        ])

        return recommendations[:15]  # Return top 15

    def get_cultural_fairness_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive cultural fairness dashboard data."""

        if not self.evaluation_history:
            return {
                'message': 'No cultural fairness evaluations performed yet',
                'active_contexts': self.active_contexts
            }

        # Aggregate statistics
        total_evaluations = len(self.evaluation_history)
        contexts_used = set()
        algorithms_evaluated = set()

        for evaluation in self.evaluation_history:
            contexts_used.add(evaluation['primary_context'])
            contexts_used.update(evaluation['comparison_contexts'])
            algorithms_evaluated.add(evaluation['algorithm_name'])

        # Cultural sensitivity trends
        sensitivity_scores = []
        for evaluation in self.evaluation_history:
            if 'cultural_analysis' in evaluation:
                score = evaluation['cultural_analysis'].get('cultural_sensitivity_score', 0.0)
                sensitivity_scores.append(score)

        avg_sensitivity = np.mean(sensitivity_scores) if sensitivity_scores else 0.0

        # Generate comprehensive report
        cultural_report = self.validator.generate_cultural_fairness_report(
            self.validator.validation_history
        )

        dashboard = {
            'summary_statistics': {
                'total_evaluations': total_evaluations,
                'contexts_evaluated': len(contexts_used),
                'algorithms_evaluated': len(algorithms_evaluated),
                'average_cultural_sensitivity': avg_sensitivity,
                'active_contexts': self.active_contexts
            },
            'cultural_contexts': {
                'available_contexts': self.context_manager.list_contexts(),
                'contexts_used': list(contexts_used),
                'context_similarities': self._get_context_similarity_matrix()
            },
            'cultural_report': cultural_report,
            'recent_evaluations': self.evaluation_history[-5:] if self.evaluation_history else []
        }

        return dashboard

    def _get_context_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get similarity matrix between all contexts."""
        contexts = self.context_manager.list_contexts()
        similarity_matrix = {}

        for ctx1 in contexts:
            similarity_matrix[ctx1] = {}
            for ctx2 in contexts:
                if ctx1 == ctx2:
                    similarity_matrix[ctx1][ctx2] = 1.0
                else:
                    similarity = self.context_manager.get_context_similarity(ctx1, ctx2)
                    similarity_matrix[ctx1][ctx2] = similarity

        return similarity_matrix


# Example usage and CLI interface
def main():
    """CLI interface for cultural fairness framework."""
    import argparse

    parser = argparse.ArgumentParser(description="Cultural Fairness Framework")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--contexts", nargs='+',
                       choices=['north_america', 'europe', 'east_asia', 'south_asia'],
                       default=['north_america', 'europe'],
                       help="Cultural contexts to compare")

    args = parser.parse_args()

    if args.demo:
        print("🌍 Starting Cultural Fairness Framework Demo")

        # Initialize framework
        framework = CulturalFairnessFramework()

        print("✅ Framework initialized with default cultural contexts")

        # Show available contexts
        available_contexts = framework.context_manager.list_contexts()
        print(f"\n📋 Available Cultural Contexts: {available_contexts}")

        # Set active contexts
        framework.set_active_contexts(args.contexts)
        print(f"   Active contexts: {args.contexts}")

        # Show context details
        print("\n🏛️ Cultural Context Details:")
        for context_name in args.contexts:
            context = framework.context_manager.get_context(context_name)
            if context:
                print(f"\n   {context_name.upper()}:")
                print(f"     Philosophy: {context.primary_philosophy.value}")
                print(f"     Key protected attributes: {[a.value for a in context.protected_attributes[:3]]}")
                print(f"     Cultural values sample: {dict(list(context.cultural_values.items())[:2])}")

        # Context similarity analysis
        print("\n📊 Context Similarity Analysis:")
        if len(args.contexts) >= 2:
            similarity = framework.context_manager.get_context_similarity(
                args.contexts[0], args.contexts[1]
            )
            print(f"   Similarity between {args.contexts[0]} and {args.contexts[1]}: {similarity:.2f}")

        # Simulate fairness evaluation
        print("\n⚖️ Simulating Cross-Cultural Fairness Evaluation:")

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000

        y_true = np.random.binomial(1, 0.3, n_samples)
        y_pred = np.random.binomial(1, 0.3 + 0.1 * np.random.randn(n_samples))

        # Create sensitive attributes with different cultural relevance
        sensitive_attrs = pd.DataFrame({
            'gender': np.random.choice(['M', 'F'], n_samples),
            'age_group': np.random.choice(['young', 'middle', 'senior'], n_samples),
            'religion': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        })

        print(f"   Dataset: {n_samples} samples with {len(sensitive_attrs.columns)} sensitive attributes")

        # Perform cultural fairness evaluation
        evaluation_result = framework.evaluate_cultural_fairness(
            algorithm_name="DemoAlgorithm",
            y_true=pd.Series(y_true),
            y_pred=y_pred,
            sensitive_attrs=sensitive_attrs,
            primary_context=args.contexts[0]
        )

        print("\n📈 Cultural Fairness Results:")
        print(f"   Primary context: {evaluation_result['primary_context']}")
        print(f"   Comparison contexts: {evaluation_result['comparison_contexts']}")

        # Show cultural differences
        validation_result = evaluation_result['validation_result']
        differences = validation_result['cultural_differences']

        print(f"\n🔍 Cultural Differences Identified: {len(differences)}")
        for i, diff in enumerate(differences[:3], 1):  # Show first 3
            print(f"   {i}. {diff['type']}: {diff.get('metric', diff.get('attribute', 'N/A'))}")
            if 'difference' in diff:
                print(f"      Magnitude: {diff['difference']:.3f}")

        # Show recommendations
        recommendations = validation_result['recommendations']
        print("\n💡 Cross-Cultural Recommendations (top 5):")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec}")

        # Cultural analysis
        cultural_analysis = evaluation_result['cultural_analysis']
        sensitivity_score = cultural_analysis['cultural_sensitivity_score']

        print("\n📊 Cultural Analysis:")
        print(f"   Cultural sensitivity score: {sensitivity_score:.3f}")
        print(f"   Context compatibility: {cultural_analysis['context_compatibility']}")

        # Dashboard overview
        print("\n🎛️ Cultural Fairness Dashboard:")
        dashboard = framework.get_cultural_fairness_dashboard()

        summary = dashboard['summary_statistics']
        print(f"   Total evaluations: {summary['total_evaluations']}")
        print(f"   Contexts evaluated: {summary['contexts_evaluated']}")
        print(f"   Average cultural sensitivity: {summary['average_cultural_sensitivity']:.3f}")

        # Simulate multi-algorithm comparison
        print("\n🔄 Multi-Algorithm Cultural Comparison:")

        # Create second algorithm results
        y_pred2 = np.random.binomial(1, 0.35 + 0.05 * np.random.randn(n_samples))

        evaluation_result2 = framework.evaluate_cultural_fairness(
            algorithm_name="DemoAlgorithm2",
            y_true=pd.Series(y_true),
            y_pred=y_pred2,
            sensitive_attrs=sensitive_attrs,
            primary_context=args.contexts[0]
        )

        # Compare algorithms
        algorithm_results = {
            'DemoAlgorithm': evaluation_result,
            'DemoAlgorithm2': evaluation_result2
        }

        comparison = framework.compare_algorithms_culturally(algorithm_results)

        print("   Cultural sensitivity ranking:")
        for i, (alg, score) in enumerate(comparison['cultural_sensitivity_ranking'].items(), 1):
            print(f"     {i}. {alg}: {score:.3f}")

        print("\n   Cross-algorithm recommendations (top 3):")
        for i, rec in enumerate(comparison['recommendations'][:3], 1):
            print(f"     {i}. {rec}")

        print("\n✅ Cultural Fairness Framework demo completed! 🎉")


if __name__ == "__main__":
    main()
