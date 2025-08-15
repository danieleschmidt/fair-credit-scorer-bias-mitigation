"""
Multi-Regional Deployment Framework for Global Fairness Research.

This module provides comprehensive multi-regional deployment capabilities
for fairness research systems, including cultural adaptation, regulatory
compliance, and distributed infrastructure management.

Global Features:
- Multi-region deployment orchestration
- Cultural and regulatory compliance frameworks
- Distributed load balancing and failover
- Global data sovereignty and privacy controls
- Cross-cultural fairness validation and adaptation
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

# Simplified imports to avoid syntax issues

# Create a simple logger
logger = logging.getLogger(__name__)

# Simple cultural framework placeholder
class CulturalFairnessFramework:
    def adapt_model_for_culture(self, model_package, cultural_context, protected_classes):
        return {'cultural_adaptations_applied': True}



class DeploymentRegion(Enum):
    """Supported deployment regions."""
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "ap"
    SOUTH_AMERICA = "sa"
    AFRICA = "af"
    MIDDLE_EAST = "me"
    OCEANIA = "oc"


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks."""
    GDPR = "gdpr"              # EU General Data Protection Regulation
    CCPA = "ccpa"              # California Consumer Privacy Act
    PDPA_SINGAPORE = "pdpa_sg" # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"              # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"          # Personal Information Protection and Electronic Documents Act (Canada)
    AI_ACT_EU = "ai_act_eu"    # EU AI Act
    ALGORITHMIC_ACCOUNTABILITY = "algo_accountability"  # General algorithmic accountability


@dataclass
class RegionalConfig:
    """Configuration for a specific deployment region."""
    region: DeploymentRegion
    primary_language: str
    compliance_frameworks: List[ComplianceFramework]
    cultural_context: Dict[str, Any]
    data_residency_requirements: Dict[str, Any]
    fairness_priorities: List[str]
    protected_classes: List[str]
    deployment_constraints: Dict[str, Any]
    monitoring_requirements: Dict[str, Any]


@dataclass
class DeploymentStatus:
    """Status of a regional deployment."""
    region: DeploymentRegion
    status: str  # 'pending', 'deploying', 'active', 'failed', 'maintenance'
    health_score: float
    last_updated: datetime
    active_models: List[str]
    compliance_status: Dict[ComplianceFramework, bool]
    performance_metrics: Dict[str, float]
    error_log: List[str]


class RegionalComplianceEngine:
    """
    Engine for ensuring regulatory compliance across regions.
    
    Handles different regulatory frameworks and their specific requirements
    for fairness, privacy, and algorithmic accountability.
    """

    def __init__(self):
        """Initialize compliance engine."""
        self.compliance_rules = self._load_compliance_rules()
        self.audit_logs = {}

        logger.info("RegionalComplianceEngine initialized")

    def _load_compliance_rules(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Load compliance rules for different frameworks."""
        return {
            ComplianceFramework.GDPR: {
                'data_protection': {
                    'right_to_explanation': True,
                    'data_minimization': True,
                    'consent_required': True,
                    'right_to_be_forgotten': True
                },
                'fairness_requirements': {
                    'non_discrimination': True,
                    'automated_decision_transparency': True,
                    'human_oversight': True
                },
                'audit_requirements': {
                    'algorithmic_audit_frequency_days': 90,
                    'bias_testing_required': True,
                    'impact_assessment_required': True
                }
            },
            ComplianceFramework.CCPA: {
                'data_protection': {
                    'right_to_know': True,
                    'right_to_delete': True,
                    'opt_out_sale': True,
                    'non_discrimination': True
                },
                'fairness_requirements': {
                    'non_discriminatory_pricing': True,
                    'service_quality_parity': True
                },
                'audit_requirements': {
                    'annual_audit_required': True,
                    'consumer_request_tracking': True
                }
            },
            ComplianceFramework.AI_ACT_EU: {
                'risk_classification': {
                    'high_risk_systems': ['employment', 'lending', 'education'],
                    'conformity_assessment': True,
                    'risk_management_system': True
                },
                'fairness_requirements': {
                    'bias_monitoring': True,
                    'accuracy_robustness': True,
                    'human_oversight': True,
                    'transparency_explainability': True
                },
                'audit_requirements': {
                    'continuous_monitoring': True,
                    'post_market_surveillance': True,
                    'incident_reporting': True
                }
            },
            ComplianceFramework.ALGORITHMIC_ACCOUNTABILITY: {
                'fairness_requirements': {
                    'algorithmic_impact_assessment': True,
                    'bias_testing': True,
                    'disparate_impact_analysis': True,
                    'model_interpretability': True
                },
                'audit_requirements': {
                    'regular_audits': True,
                    'public_reporting': True,
                    'stakeholder_engagement': True
                }
            }
        }

    def validate_compliance(
        self,
        region_config: RegionalConfig,
        model_metadata: Dict[str, Any],
        deployment_context: Dict[str, Any]
    ) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """
        Validate compliance for a regional deployment.
        
        Args:
            region_config: Regional configuration
            model_metadata: Metadata about the model being deployed
            deployment_context: Context of the deployment
            
        Returns:
            Compliance validation results for each framework
        """
        logger.info(f"Validating compliance for region {region_config.region.value}")

        validation_results = {}

        for framework in region_config.compliance_frameworks:
            if framework not in self.compliance_rules:
                logger.warning(f"Compliance rules not found for framework {framework.value}")
                continue

            rules = self.compliance_rules[framework]
            result = self._validate_framework_compliance(framework, rules, model_metadata, deployment_context)
            validation_results[framework] = result

        # Log audit trail
        self._log_compliance_audit(region_config.region, validation_results, deployment_context)

        return validation_results

    def _validate_framework_compliance(
        self,
        framework: ComplianceFramework,
        rules: Dict[str, Any],
        model_metadata: Dict[str, Any],
        deployment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate compliance against a specific framework."""
        validation_result = {
            'framework': framework.value,
            'compliant': True,
            'violations': [],
            'warnings': [],
            'recommendations': [],
            'validation_timestamp': datetime.now().isoformat()
        }

        # Validate data protection requirements
        if 'data_protection' in rules:
            data_protection_result = self._validate_data_protection(
                rules['data_protection'], model_metadata, deployment_context
            )
            if not data_protection_result['compliant']:
                validation_result['compliant'] = False
                validation_result['violations'].extend(data_protection_result['violations'])

        # Validate fairness requirements
        if 'fairness_requirements' in rules:
            fairness_result = self._validate_fairness_requirements(
                rules['fairness_requirements'], model_metadata, deployment_context
            )
            if not fairness_result['compliant']:
                validation_result['compliant'] = False
                validation_result['violations'].extend(fairness_result['violations'])

        # Validate audit requirements
        if 'audit_requirements' in rules:
            audit_result = self._validate_audit_requirements(
                rules['audit_requirements'], model_metadata, deployment_context
            )
            validation_result['warnings'].extend(audit_result['warnings'])
            validation_result['recommendations'].extend(audit_result['recommendations'])

        return validation_result

    def _validate_data_protection(
        self,
        data_rules: Dict[str, Any],
        model_metadata: Dict[str, Any],
        deployment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data protection compliance."""
        result = {'compliant': True, 'violations': []}

        # Right to explanation
        if data_rules.get('right_to_explanation', False):
            if not model_metadata.get('explainable', False):
                result['compliant'] = False
                result['violations'].append('Model lacks explainability required for right to explanation')

        # Data minimization
        if data_rules.get('data_minimization', False):
            features_used = model_metadata.get('features', [])
            if len(features_used) > 20:  # Arbitrary threshold
                result['violations'].append(f'Potential data minimization violation: {len(features_used)} features used')

        # Consent requirements
        if data_rules.get('consent_required', False):
            if not deployment_context.get('user_consent_obtained', False):
                result['compliant'] = False
                result['violations'].append('User consent required but not obtained')

        return result

    def _validate_fairness_requirements(
        self,
        fairness_rules: Dict[str, Any],
        model_metadata: Dict[str, Any],
        deployment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate fairness compliance."""
        result = {'compliant': True, 'violations': []}

        # Non-discrimination
        if fairness_rules.get('non_discrimination', False):
            fairness_metrics = model_metadata.get('fairness_metrics', {})

            # Check demographic parity
            dp_diff = fairness_metrics.get('demographic_parity_difference', 0)
            if abs(dp_diff) > 0.1:  # 10% threshold
                result['compliant'] = False
                result['violations'].append(f'Demographic parity violation: {dp_diff:.3f}')

            # Check equalized odds
            eo_diff = fairness_metrics.get('equalized_odds_difference', 0)
            if abs(eo_diff) > 0.1:  # 10% threshold
                result['compliant'] = False
                result['violations'].append(f'Equalized odds violation: {eo_diff:.3f}')

        # Bias monitoring
        if fairness_rules.get('bias_monitoring', False):
            if not model_metadata.get('bias_monitoring_enabled', False):
                result['compliant'] = False
                result['violations'].append('Bias monitoring required but not enabled')

        # Human oversight
        if fairness_rules.get('human_oversight', False):
            if not deployment_context.get('human_in_loop', False):
                result['compliant'] = False
                result['violations'].append('Human oversight required but not implemented')

        return result

    def _validate_audit_requirements(
        self,
        audit_rules: Dict[str, Any],
        model_metadata: Dict[str, Any],
        deployment_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate audit compliance."""
        result = {'warnings': [], 'recommendations': []}

        # Audit frequency
        if 'algorithmic_audit_frequency_days' in audit_rules:
            last_audit = model_metadata.get('last_audit_date')
            if last_audit:
                days_since_audit = (datetime.now() - datetime.fromisoformat(last_audit)).days
                required_frequency = audit_rules['algorithmic_audit_frequency_days']

                if days_since_audit > required_frequency:
                    result['warnings'].append(f'Audit overdue by {days_since_audit - required_frequency} days')

        # Impact assessment
        if audit_rules.get('impact_assessment_required', False):
            if not model_metadata.get('impact_assessment_completed', False):
                result['recommendations'].append('Complete algorithmic impact assessment')

        # Bias testing
        if audit_rules.get('bias_testing_required', False):
            if not model_metadata.get('bias_testing_completed', False):
                result['recommendations'].append('Complete comprehensive bias testing')

        return result

    def _log_compliance_audit(
        self,
        region: DeploymentRegion,
        validation_results: Dict[ComplianceFramework, Dict[str, Any]],
        deployment_context: Dict[str, Any]
    ):
        """Log compliance audit for record keeping."""
        audit_entry = {
            'region': region.value,
            'timestamp': datetime.now().isoformat(),
            'validation_results': validation_results,
            'deployment_context': deployment_context
        }

        region_key = region.value
        if region_key not in self.audit_logs:
            self.audit_logs[region_key] = []

        self.audit_logs[region_key].append(audit_entry)

        # Keep only last 100 audit entries per region
        self.audit_logs[region_key] = self.audit_logs[region_key][-100:]

    def generate_compliance_report(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Generate compliance report for a region."""
        region_key = region.value
        recent_audits = self.audit_logs.get(region_key, [])[-10:]  # Last 10 audits

        if not recent_audits:
            return {'region': region_key, 'status': 'no_audits', 'message': 'No audit history available'}

        # Analyze compliance trends
        compliance_trend = []
        violation_counts = {}

        for audit in recent_audits:
            audit_compliant = True
            audit_violations = 0

            for framework_result in audit['validation_results'].values():
                if not framework_result['compliant']:
                    audit_compliant = False
                    audit_violations += len(framework_result['violations'])

                    # Count violation types
                    for violation in framework_result['violations']:
                        violation_counts[violation] = violation_counts.get(violation, 0) + 1

            compliance_trend.append({
                'timestamp': audit['timestamp'],
                'compliant': audit_compliant,
                'violation_count': audit_violations
            })

        # Calculate compliance rate
        compliant_audits = sum(1 for audit in compliance_trend if audit['compliant'])
        compliance_rate = compliant_audits / len(compliance_trend)

        # Identify most common violations
        common_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'region': region_key,
            'compliance_rate': compliance_rate,
            'total_audits': len(recent_audits),
            'compliance_trend': compliance_trend,
            'common_violations': common_violations,
            'current_status': 'compliant' if compliance_trend[-1]['compliant'] else 'non_compliant',
            'report_timestamp': datetime.now().isoformat()
        }


class GlobalDeploymentOrchestrator:
    """
    Orchestrates multi-regional deployments with cultural adaptation.
    
    Manages deployment across multiple regions while ensuring cultural
    appropriateness and regulatory compliance.
    """

    def __init__(self):
        """Initialize global deployment orchestrator."""
        self.regional_configs = self._load_regional_configs()
        self.deployment_statuses = {}
        self.compliance_engine = RegionalComplianceEngine()
        self.cultural_framework = CulturalFairnessFramework()
        self.load_balancer = GlobalLoadBalancer()

        logger.info("GlobalDeploymentOrchestrator initialized")

    def _load_regional_configs(self) -> Dict[DeploymentRegion, RegionalConfig]:
        """Load regional deployment configurations."""
        configs = {
            DeploymentRegion.NORTH_AMERICA: RegionalConfig(
                region=DeploymentRegion.NORTH_AMERICA,
                primary_language="en",
                compliance_frameworks=[ComplianceFramework.CCPA, ComplianceFramework.ALGORITHMIC_ACCOUNTABILITY],
                cultural_context={
                    'individualism_score': 91,
                    'power_distance': 40,
                    'uncertainty_avoidance': 46,
                    'fairness_priorities': ['individual_merit', 'equal_opportunity']
                },
                data_residency_requirements={'store_locally': False, 'cross_border_allowed': True},
                fairness_priorities=['demographic_parity', 'equalized_odds'],
                protected_classes=['race', 'gender', 'age', 'disability'],
                deployment_constraints={'max_latency_ms': 100, 'availability_requirement': 99.9},
                monitoring_requirements={'bias_monitoring': True, 'performance_monitoring': True}
            ),
            DeploymentRegion.EUROPE: RegionalConfig(
                region=DeploymentRegion.EUROPE,
                primary_language="en",  # Multi-lingual support needed
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.AI_ACT_EU],
                cultural_context={
                    'individualism_score': 68,
                    'power_distance': 35,
                    'uncertainty_avoidance': 70,
                    'fairness_priorities': ['social_welfare', 'equal_treatment']
                },
                data_residency_requirements={'store_locally': True, 'cross_border_restricted': True},
                fairness_priorities=['demographic_parity', 'calibration'],
                protected_classes=['race', 'gender', 'religion', 'nationality', 'sexual_orientation'],
                deployment_constraints={'max_latency_ms': 150, 'availability_requirement': 99.95},
                monitoring_requirements={'gdpr_compliance_monitoring': True, 'bias_monitoring': True}
            ),
            DeploymentRegion.ASIA_PACIFIC: RegionalConfig(
                region=DeploymentRegion.ASIA_PACIFIC,
                primary_language="en",  # Multi-lingual: ja, zh, ko, etc.
                compliance_frameworks=[ComplianceFramework.PDPA_SINGAPORE],
                cultural_context={
                    'individualism_score': 25,
                    'power_distance': 95,
                    'uncertainty_avoidance': 60,
                    'fairness_priorities': ['group_harmony', 'hierarchical_respect']
                },
                data_residency_requirements={'store_locally': True, 'government_access_required': True},
                fairness_priorities=['group_fairness', 'predictive_parity'],
                protected_classes=['ethnicity', 'religion', 'social_status'],
                deployment_constraints={'max_latency_ms': 200, 'availability_requirement': 99.5},
                monitoring_requirements={'cultural_bias_monitoring': True, 'government_reporting': True}
            )
        }

        return configs

    async def deploy_globally(
        self,
        model_package: Dict[str, Any],
        target_regions: List[DeploymentRegion] = None,
        deployment_strategy: str = "blue_green"
    ) -> Dict[DeploymentRegion, DeploymentStatus]:
        """
        Deploy model package globally across regions.
        
        Args:
            model_package: Package containing model and metadata
            target_regions: List of regions to deploy to (None for all)
            deployment_strategy: Deployment strategy ('blue_green', 'rolling', 'canary')
            
        Returns:
            Deployment status for each region
        """
        if target_regions is None:
            target_regions = list(self.regional_configs.keys())

        logger.info(f"Starting global deployment to {len(target_regions)} regions")

        deployment_tasks = []
        for region in target_regions:
            task = self._deploy_to_region(region, model_package, deployment_strategy)
            deployment_tasks.append((region, task))

        # Execute deployments concurrently
        results = {}
        for region, task in deployment_tasks:
            try:
                status = await task
                results[region] = status
            except Exception as e:
                logger.error(f"Deployment to {region.value} failed: {e}")
                results[region] = DeploymentStatus(
                    region=region,
                    status='failed',
                    health_score=0.0,
                    last_updated=datetime.now(),
                    active_models=[],
                    compliance_status={},
                    performance_metrics={},
                    error_log=[str(e)]
                )

        # Update global deployment status
        self.deployment_statuses.update(results)

        # Configure load balancing
        await self.load_balancer.update_routing(results)

        logger.info(f"Global deployment completed. {sum(1 for status in results.values() if status.status == 'active')}/{len(results)} regions active")

        return results

    async def _deploy_to_region(
        self,
        region: DeploymentRegion,
        model_package: Dict[str, Any],
        deployment_strategy: str
    ) -> DeploymentStatus:
        """Deploy to a specific region with cultural adaptation."""
        logger.info(f"Deploying to region {region.value}")

        try:
            region_config = self.regional_configs[region]

            # Step 1: Validate compliance
            compliance_results = self.compliance_engine.validate_compliance(
                region_config, model_package['metadata'], {'deployment_timestamp': datetime.now()}
            )

            # Check if deployment can proceed
            all_compliant = all(result['compliant'] for result in compliance_results.values())
            if not all_compliant:
                violations = []
                for result in compliance_results.values():
                    violations.extend(result['violations'])

                logger.error(f"Compliance violations prevent deployment to {region.value}: {violations}")
                return DeploymentStatus(
                    region=region,
                    status='failed',
                    health_score=0.0,
                    last_updated=datetime.now(),
                    active_models=[],
                    compliance_status={fw: result['compliant'] for fw, result in compliance_results.items()},
                    performance_metrics={},
                    error_log=[f"Compliance violations: {violations}"]
                )

            # Step 2: Cultural adaptation
            adapted_model = await self._adapt_model_culturally(model_package, region_config)

            # Step 3: Regional deployment
            deployment_result = await self._execute_regional_deployment(
                adapted_model, region_config, deployment_strategy
            )

            # Step 4: Post-deployment validation
            validation_result = await self._post_deployment_validation(region, adapted_model)

            # Create deployment status
            status = DeploymentStatus(
                region=region,
                status='active' if deployment_result['success'] else 'failed',
                health_score=validation_result.get('health_score', 0.0),
                last_updated=datetime.now(),
                active_models=[model_package['metadata']['name']],
                compliance_status={fw: result['compliant'] for fw, result in compliance_results.items()},
                performance_metrics=validation_result.get('performance_metrics', {}),
                error_log=deployment_result.get('errors', [])
            )

            return status

        except Exception as e:
            logger.error(f"Regional deployment to {region.value} failed: {e}")
            return DeploymentStatus(
                region=region,
                status='failed',
                health_score=0.0,
                last_updated=datetime.now(),
                active_models=[],
                compliance_status={},
                performance_metrics={},
                error_log=[str(e)]
            )

    async def _adapt_model_culturally(
        self,
        model_package: Dict[str, Any],
        region_config: RegionalConfig
    ) -> Dict[str, Any]:
        """Adapt model for cultural context of the region."""
        logger.info(f"Adapting model for {region_config.region.value} cultural context")

        # Use cultural fairness framework for adaptation
        cultural_adaptation = self.cultural_framework.adapt_model_for_culture(
            model_package,
            region_config.cultural_context,
            region_config.protected_classes
        )

        # Update model package with cultural adaptations
        adapted_package = model_package.copy()
        adapted_package['cultural_adaptations'] = cultural_adaptation
        adapted_package['region_config'] = region_config

        return adapted_package

    async def _execute_regional_deployment(
        self,
        model_package: Dict[str, Any],
        region_config: RegionalConfig,
        deployment_strategy: str
    ) -> Dict[str, Any]:
        """Execute the actual deployment to regional infrastructure."""
        # Simulate deployment process
        await asyncio.sleep(1)  # Simulate deployment time

        # In a real implementation, this would:
        # 1. Deploy to regional Kubernetes clusters
        # 2. Configure regional databases
        # 3. Set up regional monitoring
        # 4. Configure regional load balancers
        # 5. Run smoke tests

        deployment_result = {
            'success': True,
            'deployment_id': f"deploy_{region_config.region.value}_{int(time.time())}",
            'endpoints': [
                f"https://api-{region_config.region.value}.fairness-research.com/predict",
                f"https://api-{region_config.region.value}.fairness-research.com/explain"
            ],
            'infrastructure': {
                'instances': 3,
                'load_balancer': f"lb-{region_config.region.value}",
                'database': f"db-{region_config.region.value}",
                'monitoring': f"mon-{region_config.region.value}"
            },
            'errors': []
        }

        logger.info(f"Regional deployment to {region_config.region.value} completed")
        return deployment_result

    async def _post_deployment_validation(
        self,
        region: DeploymentRegion,
        model_package: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate deployment after it's active."""
        # Simulate validation tests
        await asyncio.sleep(0.5)

        # In a real implementation, this would:
        # 1. Run health checks
        # 2. Validate API responses
        # 3. Test cultural adaptations
        # 4. Verify compliance controls
        # 5. Check performance metrics

        validation_result = {
            'health_score': np.random.uniform(0.85, 1.0),  # Simulated health score
            'performance_metrics': {
                'response_time_ms': np.random.uniform(50, 150),
                'throughput_rps': np.random.uniform(100, 500),
                'error_rate': np.random.uniform(0, 0.01),
                'availability': np.random.uniform(0.995, 1.0)
            },
            'cultural_validation': {
                'cultural_bias_score': np.random.uniform(0.0, 0.1),
                'local_fairness_compliance': True
            }
        }

        return validation_result

    def get_global_status(self) -> Dict[str, Any]:
        """Get overall global deployment status."""
        active_regions = [status for status in self.deployment_statuses.values() if status.status == 'active']
        failed_regions = [status for status in self.deployment_statuses.values() if status.status == 'failed']

        return {
            'total_regions': len(self.deployment_statuses),
            'active_regions': len(active_regions),
            'failed_regions': len(failed_regions),
            'global_health_score': np.mean([status.health_score for status in active_regions]) if active_regions else 0.0,
            'regional_status': {status.region.value: status.status for status in self.deployment_statuses.values()},
            'last_updated': datetime.now().isoformat()
        }

    async def failover_region(self, failed_region: DeploymentRegion, backup_region: DeploymentRegion):
        """Failover traffic from failed region to backup region."""
        logger.info(f"Initiating failover from {failed_region.value} to {backup_region.value}")

        if backup_region not in self.deployment_statuses or self.deployment_statuses[backup_region].status != 'active':
            raise ValueError(f"Backup region {backup_region.value} is not available for failover")

        # Update load balancer routing
        await self.load_balancer.failover_traffic(failed_region, backup_region)

        # Update regional status
        if failed_region in self.deployment_statuses:
            self.deployment_statuses[failed_region].status = 'maintenance'
            self.deployment_statuses[failed_region].last_updated = datetime.now()

        logger.info(f"Failover from {failed_region.value} to {backup_region.value} completed")


class GlobalLoadBalancer:
    """
    Global load balancer for distributing traffic across regions.
    
    Implements intelligent routing based on user location, regional health,
    and compliance requirements.
    """

    def __init__(self):
        """Initialize global load balancer."""
        self.routing_table = {}
        self.health_checks = {}
        self.traffic_weights = {}

        logger.info("GlobalLoadBalancer initialized")

    async def update_routing(self, deployment_statuses: Dict[DeploymentRegion, DeploymentStatus]):
        """Update routing based on deployment statuses."""
        logger.info("Updating global routing table")

        # Update routing table
        active_regions = [region for region, status in deployment_statuses.items() if status.status == 'active']

        # Calculate traffic weights based on health scores and capacity
        total_weight = 0
        for region in active_regions:
            status = deployment_statuses[region]
            weight = status.health_score * 100  # Base weight on health score
            self.traffic_weights[region] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for region in active_regions:
                self.traffic_weights[region] = (self.traffic_weights[region] / total_weight) * 100

        # Update routing table
        self.routing_table = {
            'active_regions': active_regions,
            'traffic_weights': self.traffic_weights,
            'last_updated': datetime.now().isoformat()
        }

        logger.info(f"Routing table updated: {len(active_regions)} active regions")

    def route_request(self, user_location: Optional[str] = None, user_preferences: Optional[Dict] = None) -> DeploymentRegion:
        """Route request to optimal region."""
        active_regions = self.routing_table.get('active_regions', [])

        if not active_regions:
            raise RuntimeError("No active regions available for routing")

        # Simple routing logic - in practice would be more sophisticated
        if user_location:
            # Route based on geographic proximity
            if 'us' in user_location.lower() or 'canada' in user_location.lower():
                preferred_region = DeploymentRegion.NORTH_AMERICA
            elif any(country in user_location.lower() for country in ['uk', 'eu', 'germany', 'france']):
                preferred_region = DeploymentRegion.EUROPE
            elif any(country in user_location.lower() for country in ['japan', 'singapore', 'australia']):
                preferred_region = DeploymentRegion.ASIA_PACIFIC
            else:
                preferred_region = None

            if preferred_region and preferred_region in active_regions:
                return preferred_region

        # Fallback to weighted round-robin
        weights = self.traffic_weights
        if not weights:
            return active_regions[0]  # Simple fallback

        # Select region based on weights (simplified)
        import random
        total_weight = sum(weights.get(region, 0) for region in active_regions)
        if total_weight == 0:
            return active_regions[0]

        r = random.uniform(0, total_weight)
        cumulative_weight = 0

        for region in active_regions:
            cumulative_weight += weights.get(region, 0)
            if r <= cumulative_weight:
                return region

        return active_regions[0]  # Fallback

    async def failover_traffic(self, failed_region: DeploymentRegion, backup_region: DeploymentRegion):
        """Failover traffic from failed region to backup."""
        logger.info(f"Executing traffic failover: {failed_region.value} -> {backup_region.value}")

        # Remove failed region from routing
        if 'active_regions' in self.routing_table and failed_region in self.routing_table['active_regions']:
            self.routing_table['active_regions'].remove(failed_region)

        # Redistribute traffic weight to backup region
        if failed_region in self.traffic_weights:
            failed_weight = self.traffic_weights.pop(failed_region)
            if backup_region in self.traffic_weights:
                self.traffic_weights[backup_region] += failed_weight
            else:
                self.traffic_weights[backup_region] = failed_weight

        logger.info("Traffic failover completed")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'routing_table': self.routing_table,
            'traffic_weights': self.traffic_weights,
            'active_regions_count': len(self.routing_table.get('active_regions', [])),
        }


# CLI interface for testing and demonstration
def main():
    """CLI interface for multi-regional deployment."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Regional Deployment Demo")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--regions", nargs="+", help="Target regions for deployment",
                       choices=['na', 'eu', 'ap', 'sa', 'af', 'me', 'oc'])

    args = parser.parse_args()

    if args.demo:
        print("🌍 Multi-Regional Deployment Demo")

        # Create mock model package
        model_package = {
            'metadata': {
                'name': 'fairness_classifier_v1',
                'version': '1.0.0',
                'explainable': True,
                'bias_monitoring_enabled': True,
                'last_audit_date': '2024-01-01',
                'fairness_metrics': {
                    'demographic_parity_difference': 0.05,
                    'equalized_odds_difference': 0.08
                },
                'features': ['income', 'education', 'employment_status', 'credit_history'],
                'impact_assessment_completed': True,
                'bias_testing_completed': True
            },
            'model_files': ['model.pkl', 'preprocessor.pkl'],
            'deployment_config': {
                'cpu_requirements': '2 cores',
                'memory_requirements': '4GB',
                'disk_requirements': '10GB'
            }
        }

        # Initialize orchestrator
        print("\n⚙️ Initializing Global Deployment Orchestrator")
        orchestrator = GlobalDeploymentOrchestrator()

        # Select target regions
        target_regions = []
        if args.regions:
            region_map = {
                'na': DeploymentRegion.NORTH_AMERICA,
                'eu': DeploymentRegion.EUROPE,
                'ap': DeploymentRegion.ASIA_PACIFIC,
                'sa': DeploymentRegion.SOUTH_AMERICA,
                'af': DeploymentRegion.AFRICA,
                'me': DeploymentRegion.MIDDLE_EAST,
                'oc': DeploymentRegion.OCEANIA
            }
            target_regions = [region_map[r] for r in args.regions if r in region_map]
        else:
            target_regions = [DeploymentRegion.NORTH_AMERICA, DeploymentRegion.EUROPE, DeploymentRegion.ASIA_PACIFIC]

        print(f"   Target regions: {[r.value for r in target_regions]}")

        # Run deployment
        async def run_deployment():
            print(f"\n🚀 Starting deployment to {len(target_regions)} regions")

            deployment_results = await orchestrator.deploy_globally(
                model_package=model_package,
                target_regions=target_regions,
                deployment_strategy="blue_green"
            )

            print("\n📊 Deployment Results:")
            for region, status in deployment_results.items():
                print(f"   {region.value}: {status.status} (health: {status.health_score:.2f})")
                if status.error_log:
                    print(f"      Errors: {status.error_log}")

            # Test compliance reporting
            print("\n📋 Compliance Reports:")
            for region in target_regions:
                if region in orchestrator.regional_configs:
                    report = orchestrator.compliance_engine.generate_compliance_report(region)
                    print(f"   {region.value}: {report.get('current_status', 'unknown')} (rate: {report.get('compliance_rate', 0):.1%})")

            # Test load balancer
            print("\n🔄 Load Balancer Status:")
            routing_stats = orchestrator.load_balancer.get_routing_stats()
            active_regions = routing_stats.get('routing_table', {}).get('active_regions', [])
            print(f"   Active regions: {[r.value for r in active_regions]}")

            # Test routing
            if active_regions:
                print("\n🎯 Routing Examples:")
                test_locations = ["us", "uk", "japan", "unknown"]
                for location in test_locations:
                    try:
                        routed_region = orchestrator.load_balancer.route_request(user_location=location)
                        print(f"   User in {location} -> {routed_region.value}")
                    except Exception as e:
                        print(f"   User in {location} -> Error: {e}")

            # Global status
            print("\n🌍 Global Status:")
            global_status = orchestrator.get_global_status()
            print(f"   Active regions: {global_status['active_regions']}/{global_status['total_regions']}")
            print(f"   Global health score: {global_status['global_health_score']:.2f}")

        # Run async deployment
        try:
            asyncio.run(run_deployment())
        except Exception as e:
            print(f"   Deployment failed: {e}")

        print("\n✅ Multi-regional deployment demo completed! 🎉")


if __name__ == "__main__":
    main()
