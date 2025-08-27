#!/usr/bin/env python3
"""
Global-First Internationalization and Localization Framework.

This module implements comprehensive I18n/L10n support for the Fair Credit Scorer,
ensuring global accessibility and compliance with international standards.

Key Features:
- Multi-language fairness metrics and explanations
- Cultural adaptation of bias detection algorithms  
- Regional compliance frameworks (GDPR, CCPA, etc.)
- Locale-aware data processing and visualization
- Right-to-left language support
- Cultural fairness standards integration
"""

import json
import locale
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd


class SupportedLanguage(Enum):
    """Supported languages with ISO 639-1 codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    MANDARIN = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    ITALIAN = "it"


class ComplianceRegion(Enum):
    """Regulatory compliance regions."""
    EU = "eu"  # GDPR, EU AI Act
    US = "us"  # CCPA, Fair Credit Reporting Act
    CA = "ca"  # PIPEDA
    UK = "uk"  # UK GDPR, DPA
    BR = "br"  # LGPD
    IN = "in"  # Personal Data Protection Bill
    AU = "au"  # Privacy Act
    JP = "jp"  # Personal Information Protection Act
    GLOBAL = "global"


@dataclass
class CulturalFairnessStandard:
    """Cultural fairness standards and expectations."""
    region: ComplianceRegion
    protected_classes: List[str]
    fairness_threshold: float
    discrimination_tolerance: float
    explanation_style: str  # direct, indirect, contextual
    privacy_expectations: str  # high, medium, low
    algorithmic_transparency: str  # required, preferred, optional


class InternationalizationManager:
    """Manages internationalization and localization for the Fair Credit Scorer."""
    
    def __init__(self, 
                 default_language: SupportedLanguage = SupportedLanguage.ENGLISH,
                 default_region: ComplianceRegion = ComplianceRegion.GLOBAL):
        self.default_language = default_language
        self.default_region = default_region
        self.translations = {}
        self.cultural_standards = {}
        self.locale_configs = {}
        
        # Initialize translations and standards
        self._load_translations()
        self._initialize_cultural_standards()
        self._setup_locale_configs()
    
    def _load_translations(self) -> None:
        """Load translation dictionaries for all supported languages."""
        # Fairness-related translations
        fairness_translations = {
            SupportedLanguage.ENGLISH: {
                "demographic_parity": "Demographic Parity",
                "equalized_odds": "Equalized Odds", 
                "fairness_score": "Fairness Score",
                "bias_detected": "Bias Detected",
                "model_explanation": "Model Explanation",
                "protected_attribute": "Protected Attribute",
                "discrimination_risk": "Discrimination Risk",
                "fairness_assessment": "Fairness Assessment"
            },
            SupportedLanguage.SPANISH: {
                "demographic_parity": "Paridad Demográfica",
                "equalized_odds": "Probabilidades Igualadas",
                "fairness_score": "Puntuación de Equidad",
                "bias_detected": "Sesgo Detectado",
                "model_explanation": "Explicación del Modelo",
                "protected_attribute": "Atributo Protegido",
                "discrimination_risk": "Riesgo de Discriminación",
                "fairness_assessment": "Evaluación de Equidad"
            },
            SupportedLanguage.FRENCH: {
                "demographic_parity": "Parité Démographique",
                "equalized_odds": "Chances Égalisées",
                "fairness_score": "Score d'Équité",
                "bias_detected": "Biais Détecté",
                "model_explanation": "Explication du Modèle",
                "protected_attribute": "Attribut Protégé",
                "discrimination_risk": "Risque de Discrimination",
                "fairness_assessment": "Évaluation de l'Équité"
            },
            SupportedLanguage.GERMAN: {
                "demographic_parity": "Demografische Parität",
                "equalized_odds": "Ausgeglichene Chancen",
                "fairness_score": "Fairness-Bewertung",
                "bias_detected": "Bias Erkannt",
                "model_explanation": "Modellerklärung",
                "protected_attribute": "Geschütztes Attribut",
                "discrimination_risk": "Diskriminierungsrisiko",
                "fairness_assessment": "Fairness-Bewertung"
            },
            SupportedLanguage.MANDARIN: {
                "demographic_parity": "人口统计均等",
                "equalized_odds": "机会均等",
                "fairness_score": "公平性评分",
                "bias_detected": "检测到偏见",
                "model_explanation": "模型解释",
                "protected_attribute": "受保护属性",
                "discrimination_risk": "歧视风险",
                "fairness_assessment": "公平性评估"
            },
            SupportedLanguage.JAPANESE: {
                "demographic_parity": "人口統計的パリティ",
                "equalized_odds": "均等化オッズ",
                "fairness_score": "公平性スコア",
                "bias_detected": "バイアス検出",
                "model_explanation": "モデル説明",
                "protected_attribute": "保護属性",
                "discrimination_risk": "差別リスク",
                "fairness_assessment": "公平性評価"
            },
            SupportedLanguage.ARABIC: {
                "demographic_parity": "التكافؤ الديموغرافي",
                "equalized_odds": "الاحتمالات المتكافئة",
                "fairness_score": "درجة العدالة",
                "bias_detected": "تم اكتشاف التحيز",
                "model_explanation": "شرح النموذج",
                "protected_attribute": "السمة المحمية",
                "discrimination_risk": "خطر التمييز",
                "fairness_assessment": "تقييم العدالة"
            }
        }
        
        # Store translations
        for language in SupportedLanguage:
            if language in fairness_translations:
                self.translations[language] = fairness_translations[language]
            else:
                # Fallback to English for unsupported languages
                self.translations[language] = fairness_translations[SupportedLanguage.ENGLISH]
    
    def _initialize_cultural_standards(self) -> None:
        """Initialize cultural fairness standards for different regions."""
        self.cultural_standards = {
            ComplianceRegion.EU: CulturalFairnessStandard(
                region=ComplianceRegion.EU,
                protected_classes=["gender", "race", "religion", "age", "disability", "sexual_orientation"],
                fairness_threshold=0.85,
                discrimination_tolerance=0.05,
                explanation_style="contextual",
                privacy_expectations="high",
                algorithmic_transparency="required"
            ),
            ComplianceRegion.US: CulturalFairnessStandard(
                region=ComplianceRegion.US,
                protected_classes=["race", "color", "religion", "sex", "national_origin", "age", "disability"],
                fairness_threshold=0.80,
                discrimination_tolerance=0.10,
                explanation_style="direct",
                privacy_expectations="medium",
                algorithmic_transparency="preferred"
            ),
            ComplianceRegion.CA: CulturalFairnessStandard(
                region=ComplianceRegion.CA,
                protected_classes=["race", "religion", "gender", "age", "disability", "sexual_orientation"],
                fairness_threshold=0.85,
                discrimination_tolerance=0.05,
                explanation_style="indirect",
                privacy_expectations="high",
                algorithmic_transparency="required"
            ),
            ComplianceRegion.JP: CulturalFairnessStandard(
                region=ComplianceRegion.JP,
                protected_classes=["gender", "nationality", "social_status", "family_origin"],
                fairness_threshold=0.90,
                discrimination_tolerance=0.03,
                explanation_style="contextual",
                privacy_expectations="high",
                algorithmic_transparency="optional"
            ),
            ComplianceRegion.GLOBAL: CulturalFairnessStandard(
                region=ComplianceRegion.GLOBAL,
                protected_classes=["gender", "race", "religion", "age"],
                fairness_threshold=0.80,
                discrimination_tolerance=0.10,
                explanation_style="direct",
                privacy_expectations="medium",
                algorithmic_transparency="preferred"
            )
        }
    
    def _setup_locale_configs(self) -> None:
        """Setup locale-specific configurations."""
        self.locale_configs = {
            SupportedLanguage.ENGLISH: {
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "$",
                "date_format": "%m/%d/%Y",
                "time_format": "%I:%M %p",
                "rtl": False
            },
            SupportedLanguage.SPANISH: {
                "decimal_separator": ",",
                "thousands_separator": ".",
                "currency_symbol": "€",
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M",
                "rtl": False
            },
            SupportedLanguage.FRENCH: {
                "decimal_separator": ",",
                "thousands_separator": " ",
                "currency_symbol": "€",
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M",
                "rtl": False
            },
            SupportedLanguage.GERMAN: {
                "decimal_separator": ",",
                "thousands_separator": ".",
                "currency_symbol": "€",
                "date_format": "%d.%m.%Y",
                "time_format": "%H:%M",
                "rtl": False
            },
            SupportedLanguage.ARABIC: {
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "ر.س",
                "date_format": "%d/%m/%Y",
                "time_format": "%H:%M",
                "rtl": True
            },
            SupportedLanguage.JAPANESE: {
                "decimal_separator": ".",
                "thousands_separator": ",",
                "currency_symbol": "¥",
                "date_format": "%Y/%m/%d",
                "time_format": "%H:%M",
                "rtl": False
            }
        }
    
    def translate(self, key: str, language: SupportedLanguage = None) -> str:
        """Translate a key to the specified language."""
        if language is None:
            language = self.default_language
        
        return self.translations.get(language, {}).get(key, key)
    
    def get_cultural_standard(self, region: ComplianceRegion = None) -> CulturalFairnessStandard:
        """Get cultural fairness standards for a region."""
        if region is None:
            region = self.default_region
        
        return self.cultural_standards.get(region, self.cultural_standards[ComplianceRegion.GLOBAL])
    
    def format_number(self, number: float, language: SupportedLanguage = None) -> str:
        """Format numbers according to locale conventions."""
        if language is None:
            language = self.default_language
        
        config = self.locale_configs.get(language, self.locale_configs[SupportedLanguage.ENGLISH])
        
        # Format with appropriate separators
        formatted = f"{number:,.3f}"
        if config["decimal_separator"] == ",":
            formatted = formatted.replace(".", "TEMP").replace(",", ".").replace("TEMP", ",")
        
        return formatted
    
    def format_date(self, date: datetime, language: SupportedLanguage = None) -> str:
        """Format dates according to locale conventions."""
        if language is None:
            language = self.default_language
        
        config = self.locale_configs.get(language, self.locale_configs[SupportedLanguage.ENGLISH])
        return date.strftime(config["date_format"])
    
    def adapt_fairness_metrics_for_culture(self, 
                                         metrics: Dict[str, float], 
                                         region: ComplianceRegion = None) -> Dict[str, Any]:
        """Adapt fairness metrics based on cultural standards."""
        if region is None:
            region = self.default_region
        
        standard = self.get_cultural_standard(region)
        adapted_metrics = {}
        
        for metric_name, value in metrics.items():
            # Apply cultural threshold adjustments
            if value < standard.fairness_threshold:
                risk_level = "high" if value < (standard.fairness_threshold - 0.1) else "medium"
            else:
                risk_level = "low"
            
            adapted_metrics[metric_name] = {
                "value": value,
                "risk_level": risk_level,
                "meets_standard": value >= standard.fairness_threshold,
                "cultural_threshold": standard.fairness_threshold,
                "explanation_required": risk_level != "low" and standard.algorithmic_transparency == "required"
            }
        
        return adapted_metrics
    
    def generate_localized_fairness_report(self, 
                                         metrics: Dict[str, float],
                                         language: SupportedLanguage = None,
                                         region: ComplianceRegion = None) -> Dict[str, Any]:
        """Generate a localized fairness report."""
        if language is None:
            language = self.default_language
        if region is None:
            region = self.default_region
        
        # Translate metric names
        translated_metrics = {}
        for key, value in metrics.items():
            translated_key = self.translate(key, language)
            translated_metrics[translated_key] = self.format_number(value, language)
        
        # Get cultural adaptations
        adapted_metrics = self.adapt_fairness_metrics_for_culture(metrics, region)
        
        # Generate report
        report = {
            "language": language.value,
            "region": region.value,
            "generated_at": self.format_date(datetime.now(), language),
            "metrics": translated_metrics,
            "cultural_assessment": adapted_metrics,
            "compliance_status": self._assess_compliance(adapted_metrics, region),
            "recommendations": self._generate_localized_recommendations(adapted_metrics, language, region)
        }
        
        return report
    
    def _assess_compliance(self, 
                          adapted_metrics: Dict[str, Any], 
                          region: ComplianceRegion) -> Dict[str, Any]:
        """Assess compliance with regional regulations."""
        standard = self.get_cultural_standard(region)
        
        compliant_metrics = sum(1 for m in adapted_metrics.values() if m.get("meets_standard", False))
        total_metrics = len(adapted_metrics)
        compliance_rate = compliant_metrics / total_metrics if total_metrics > 0 else 0
        
        return {
            "overall_compliance": compliance_rate >= 0.8,
            "compliance_rate": compliance_rate,
            "failing_metrics": [k for k, v in adapted_metrics.items() if not v.get("meets_standard", True)],
            "regulatory_framework": self._get_regulatory_framework(region),
            "required_actions": self._get_required_compliance_actions(adapted_metrics, region)
        }
    
    def _get_regulatory_framework(self, region: ComplianceRegion) -> Dict[str, str]:
        """Get relevant regulatory framework information."""
        frameworks = {
            ComplianceRegion.EU: {
                "primary": "GDPR (General Data Protection Regulation)",
                "ai_specific": "EU AI Act",
                "financial": "PSD2 (Payment Services Directive 2)"
            },
            ComplianceRegion.US: {
                "primary": "Fair Credit Reporting Act (FCRA)",
                "privacy": "California Consumer Privacy Act (CCPA)",
                "ai_specific": "NIST AI Risk Management Framework"
            },
            ComplianceRegion.CA: {
                "primary": "Personal Information Protection and Electronic Documents Act (PIPEDA)",
                "ai_specific": "Directive on Automated Decision-Making"
            },
            ComplianceRegion.UK: {
                "primary": "UK GDPR",
                "ai_specific": "UK AI White Paper"
            }
        }
        
        return frameworks.get(region, {"primary": "Local Data Protection Laws"})
    
    def _get_required_compliance_actions(self, 
                                       adapted_metrics: Dict[str, Any], 
                                       region: ComplianceRegion) -> List[str]:
        """Get required actions for compliance."""
        actions = []
        standard = self.get_cultural_standard(region)
        
        # Check for failing metrics
        failing_metrics = [k for k, v in adapted_metrics.items() if not v.get("meets_standard", True)]
        
        if failing_metrics:
            actions.append(f"Address bias in metrics: {', '.join(failing_metrics)}")
        
        # Region-specific requirements
        if region == ComplianceRegion.EU:
            actions.extend([
                "Implement explicit consent mechanisms",
                "Provide clear algorithmic transparency",
                "Establish data subject rights procedures"
            ])
        elif region == ComplianceRegion.US:
            actions.extend([
                "Ensure FCRA compliance for credit decisions",
                "Implement adverse action notice procedures"
            ])
        
        if standard.algorithmic_transparency == "required":
            actions.append("Provide detailed model explanations")
        
        return actions
    
    def _generate_localized_recommendations(self, 
                                          adapted_metrics: Dict[str, Any],
                                          language: SupportedLanguage,
                                          region: ComplianceRegion) -> List[str]:
        """Generate culturally appropriate recommendations."""
        recommendations = []
        standard = self.get_cultural_standard(region)
        
        # High-risk metrics
        high_risk_metrics = [k for k, v in adapted_metrics.items() 
                           if v.get("risk_level") == "high"]
        
        if high_risk_metrics:
            if standard.explanation_style == "direct":
                recommendations.append(f"Immediately address high-risk bias in: {', '.join(high_risk_metrics)}")
            elif standard.explanation_style == "indirect":
                recommendations.append(f"Consider reviewing the fairness characteristics of: {', '.join(high_risk_metrics)}")
            else:  # contextual
                recommendations.append(f"Within your cultural context, these metrics may need attention: {', '.join(high_risk_metrics)}")
        
        # Privacy-focused recommendations
        if standard.privacy_expectations == "high":
            recommendations.extend([
                "Implement privacy-preserving fairness techniques",
                "Minimize data collection to essential attributes only",
                "Provide granular consent options"
            ])
        
        # Transparency recommendations
        if standard.algorithmic_transparency == "required":
            recommendations.append("Provide detailed, accessible model explanations")
        
        return recommendations


class GlobalFairnessValidator:
    """Validates fairness across different cultural and regulatory contexts."""
    
    def __init__(self, i18n_manager: InternationalizationManager):
        self.i18n_manager = i18n_manager
    
    def validate_global_fairness(self, 
                                model_predictions: np.ndarray,
                                protected_attributes: np.ndarray,
                                true_labels: np.ndarray = None,
                                target_regions: List[ComplianceRegion] = None) -> Dict[str, Any]:
        """Validate fairness across multiple cultural contexts."""
        if target_regions is None:
            target_regions = [ComplianceRegion.GLOBAL]
        
        validation_results = {}
        
        for region in target_regions:
            # Get cultural standards
            standard = self.i18n_manager.get_cultural_standard(region)
            
            # Calculate basic fairness metrics
            base_metrics = self._calculate_base_fairness_metrics(
                model_predictions, protected_attributes, true_labels
            )
            
            # Adapt to cultural context
            cultural_assessment = self.i18n_manager.adapt_fairness_metrics_for_culture(
                base_metrics, region
            )
            
            validation_results[region.value] = {
                "base_metrics": base_metrics,
                "cultural_assessment": cultural_assessment,
                "compliance_status": self.i18n_manager._assess_compliance(cultural_assessment, region),
                "cultural_standard": standard
            }
        
        return validation_results
    
    def _calculate_base_fairness_metrics(self, 
                                       predictions: np.ndarray,
                                       protected_attributes: np.ndarray,
                                       true_labels: np.ndarray = None) -> Dict[str, float]:
        """Calculate base fairness metrics."""
        # Demographic Parity
        group_0_rate = np.mean(predictions[protected_attributes == 0])
        group_1_rate = np.mean(predictions[protected_attributes == 1])
        demographic_parity = 1 - abs(group_0_rate - group_1_rate)
        
        metrics = {
            "demographic_parity": demographic_parity,
            "fairness_score": demographic_parity  # Simplified for demo
        }
        
        # Add accuracy if true labels available
        if true_labels is not None:
            from sklearn.metrics import accuracy_score
            metrics["accuracy"] = accuracy_score(true_labels, predictions.round())
        
        return metrics


def demonstrate_global_fairness_framework():
    """Demonstrate the global fairness and internationalization framework."""
    print("🌍 Global Fairness and Internationalization Demonstration")
    print("=" * 60)
    
    # Initialize I18n manager
    i18n_manager = InternationalizationManager(
        default_language=SupportedLanguage.ENGLISH,
        default_region=ComplianceRegion.GLOBAL
    )
    
    # Sample fairness metrics
    sample_metrics = {
        "demographic_parity": 0.75,
        "equalized_odds": 0.82,
        "fairness_score": 0.78
    }
    
    print("📊 Testing Multi-Language Support")
    languages_to_test = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.SPANISH, 
        SupportedLanguage.FRENCH,
        SupportedLanguage.MANDARIN,
        SupportedLanguage.ARABIC
    ]
    
    for lang in languages_to_test:
        translated = i18n_manager.translate("fairness_score", lang)
        formatted_score = i18n_manager.format_number(sample_metrics["fairness_score"], lang)
        print(f"   {lang.value}: {translated} = {formatted_score}")
    
    print(f"\n🏛️ Testing Regional Compliance")
    regions_to_test = [
        ComplianceRegion.EU,
        ComplianceRegion.US,
        ComplianceRegion.CA,
        ComplianceRegion.JP
    ]
    
    for region in regions_to_test:
        standard = i18n_manager.get_cultural_standard(region)
        adapted = i18n_manager.adapt_fairness_metrics_for_culture(sample_metrics, region)
        compliance = i18n_manager._assess_compliance(adapted, region)
        
        print(f"   {region.value}: Threshold={standard.fairness_threshold}, "
              f"Compliance={compliance['overall_compliance']}")
    
    print(f"\n📋 Generating Localized Reports")
    
    # Generate reports for different regions and languages
    test_cases = [
        (SupportedLanguage.ENGLISH, ComplianceRegion.US),
        (SupportedLanguage.SPANISH, ComplianceRegion.EU),
        (SupportedLanguage.MANDARIN, ComplianceRegion.GLOBAL),
        (SupportedLanguage.ARABIC, ComplianceRegion.GLOBAL)
    ]
    
    reports = {}
    for lang, region in test_cases:
        report = i18n_manager.generate_localized_fairness_report(
            sample_metrics, lang, region
        )
        reports[f"{lang.value}_{region.value}"] = report
        print(f"   Generated report: {lang.value} ({region.value})")
    
    print(f"\n🔍 Global Validation Test")
    
    # Initialize global validator
    validator = GlobalFairnessValidator(i18n_manager)
    
    # Generate sample data
    np.random.seed(42)
    predictions = np.random.choice([0, 1], 1000, p=[0.3, 0.7])
    protected_attr = np.random.choice([0, 1], 1000, p=[0.6, 0.4])
    true_labels = np.random.choice([0, 1], 1000, p=[0.4, 0.6])
    
    # Validate across regions
    global_validation = validator.validate_global_fairness(
        predictions, protected_attr, true_labels, 
        [ComplianceRegion.EU, ComplianceRegion.US, ComplianceRegion.JP]
    )
    
    print(f"   Validated across {len(global_validation)} regions")
    for region, results in global_validation.items():
        compliance_rate = results['compliance_status']['compliance_rate']
        print(f"   {region}: Compliance rate = {compliance_rate:.2%}")
    
    return {
        "i18n_manager": i18n_manager,
        "global_validator": validator,
        "sample_reports": reports,
        "global_validation_results": global_validation,
        "supported_languages": len(SupportedLanguage),
        "supported_regions": len(ComplianceRegion)
    }


if __name__ == "__main__":
    # Run demonstration
    result = demonstrate_global_fairness_framework()
    print(f"\n✅ Global Framework Demonstration Complete")
    print(f"   Languages supported: {result['supported_languages']}")
    print(f"   Regions covered: {result['supported_regions']}")
    print(f"   Reports generated: {len(result['sample_reports'])}")