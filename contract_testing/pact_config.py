"""
Contract Testing Configuration for MATURING repositories
Advanced API validation and consumer-driven contract testing with Pact
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pact
from pact import Consumer, Provider


class ContractTestConfig:
    """Configuration for contract testing setup."""
    
    def __init__(self, service_name: str = "fair-credit-scorer"):
        self.service_name = service_name
        self.pact_dir = Path("contract_tests/pacts")
        self.pact_dir.mkdir(parents=True, exist_ok=True)
        
        # Pact Broker configuration
        self.broker_url = os.getenv("PACT_BROKER_URL", "http://localhost:9292")
        self.broker_username = os.getenv("PACT_BROKER_USERNAME")
        self.broker_password = os.getenv("PACT_BROKER_PASSWORD")
        
        # Version and tags
        self.consumer_version = os.getenv("CONSUMER_VERSION", "1.0.0")
        self.provider_version = os.getenv("PROVIDER_VERSION", "1.0.0")
        
    def create_consumer_pact(self, provider_name: str) -> pact.Pact:
        """Create a Pact for consumer testing."""
        return pact.Pact(
            consumer=Consumer(self.service_name),
            provider=Provider(provider_name),
            pact_dir=str(self.pact_dir),
            publish_to_broker=bool(self.broker_url),
            broker_base_url=self.broker_url,
            broker_username=self.broker_username,
            broker_password=self.broker_password,
            consumer_version=self.consumer_version
        )
    
    def get_verification_config(self) -> Dict[str, Any]:
        """Get provider verification configuration."""
        return {
            "provider": self.service_name,
            "provider_base_url": os.getenv("PROVIDER_BASE_URL", "http://localhost:8000"),
            "pact_urls": [str(self.pact_dir / "*.json")],
            "consumer_version_selectors": [
                {"latest": True},
                {"deployed": True},
                {"released": True}
            ],
            "provider_version": self.provider_version,
            "publish_verification_results": True,
            "broker_base_url": self.broker_url,
            "broker_username": self.broker_username,
            "broker_password": self.broker_password,
        }


class FairnessScoringContractTests:
    """Contract tests for fairness scoring API."""
    
    def __init__(self):
        self.config = ContractTestConfig()
        self.scoring_pact = self.config.create_consumer_pact("fairness-scoring-service")
    
    def test_score_prediction_contract(self):
        """Test contract for score prediction endpoint."""
        expected_request = {
            "method": "POST",
            "path": "/api/v1/score",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "body": {
                "applicant_data": {
                    "age": pact.Like(30),
                    "income": pact.Like(50000.0),
                    "credit_history_length": pact.Like(10),
                    "education_level": pact.Like("bachelor"),
                    "employment_status": pact.Like("employed")
                },
                "model_version": pact.Like("v1.2.0")
            }
        }
        
        expected_response = {
            "status": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "credit_score": pact.Like(750),
                "risk_category": pact.Like("low"),
                "confidence": pact.Like(0.85),
                "fairness_metrics": {
                    "demographic_parity": pact.Like(0.02),
                    "equalized_odds": pact.Like(0.03),
                    "calibration": pact.Like(0.01)
                },
                "explanation": {
                    "top_factors": pact.EachLike({
                        "feature": pact.Like("income"),
                        "impact": pact.Like(0.15),
                        "direction": pact.Like("positive")
                    }, minimum=3),
                    "counterfactual": pact.Like("Increasing income by $10K would improve score by 25 points")
                },
                "request_id": pact.Like("req_123456789"),
                "timestamp": pact.Like("2024-01-15T10:30:00Z"),
                "processing_time_ms": pact.Like(150)
            }
        }
        
        return self.scoring_pact.given("Valid applicant data provided").upon_receiving(
            "A request for credit score prediction"
        ).with_request(**expected_request).will_respond_with(**expected_response)
    
    def test_batch_scoring_contract(self):
        """Test contract for batch scoring endpoint."""
        expected_request = {
            "method": "POST",
            "path": "/api/v1/score/batch",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "body": {
                "applicants": pact.EachLike({
                    "id": pact.Like("app_001"),
                    "applicant_data": {
                        "age": pact.Like(30),
                        "income": pact.Like(50000.0),
                        "credit_history_length": pact.Like(10)
                    }
                }, minimum=1, maximum=100),
                "model_version": pact.Like("v1.2.0"),
                "include_explanations": pact.Like(True)
            }
        }
        
        expected_response = {
            "status": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "results": pact.EachLike({
                    "id": pact.Like("app_001"),
                    "credit_score": pact.Like(750),
                    "risk_category": pact.Like("low"),
                    "confidence": pact.Like(0.85),
                    "fairness_metrics": {
                        "demographic_parity": pact.Like(0.02)
                    }
                }, minimum=1),
                "batch_id": pact.Like("batch_789"),
                "processed_count": pact.Like(10),
                "failed_count": pact.Like(0),
                "processing_time_ms": pact.Like(1200)
            }
        }
        
        return self.scoring_pact.given("Valid batch of applicant data").upon_receiving(
            "A request for batch scoring"
        ).will_respond_with(**expected_response)
    
    def test_model_info_contract(self):
        """Test contract for model information endpoint."""
        expected_request = {
            "method": "GET",
            "path": "/api/v1/model/info",
            "headers": {
                "Accept": "application/json"
            }
        }
        
        expected_response = {
            "status": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "model_version": pact.Like("v1.2.0"),
                "model_name": pact.Like("FairCreditScorer"),
                "training_date": pact.Like("2024-01-10T00:00:00Z"),
                "performance_metrics": {
                    "accuracy": pact.Like(0.89),
                    "precision": pact.Like(0.87),
                    "recall": pact.Like(0.91),
                    "f1_score": pact.Like(0.89)
                },
                "fairness_metrics": {
                    "overall_demographic_parity": pact.Like(0.025),
                    "overall_equalized_odds": pact.Like(0.031)
                },
                "feature_importance": pact.EachLike({
                    "feature": pact.Like("income"),
                    "importance": pact.Like(0.25)
                }, minimum=5),
                "supported_features": pact.EachLike(
                    pact.Like("age"), minimum=5
                )
            }
        }
        
        return self.scoring_pact.given("Model service is available").upon_receiving(
            "A request for model information"
        ).will_respond_with(**expected_response)


class BiasDetectionContractTests:
    """Contract tests for bias detection API."""
    
    def __init__(self):
        self.config = ContractTestConfig()
        self.bias_pact = self.config.create_consumer_pact("bias-detection-service")
    
    def test_bias_analysis_contract(self):
        """Test contract for bias analysis endpoint."""
        expected_request = {
            "method": "POST",
            "path": "/api/v1/bias/analyze",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            "body": {
                "predictions": pact.EachLike({
                    "applicant_id": pact.Like("app_001"),
                    "predicted_score": pact.Like(750),
                    "actual_outcome": pact.Like(1),
                    "protected_attributes": {
                        "race": pact.Like("white"),
                        "gender": pact.Like("female"),
                        "age_group": pact.Like("25-35")
                    }
                }, minimum=100),
                "analysis_type": pact.Like("demographic_parity"),
                "protected_attribute": pact.Like("race")
            }
        }
        
        expected_response = {
            "status": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "bias_metrics": {
                    "demographic_parity_difference": pact.Like(0.05),
                    "equalized_odds_difference": pact.Like(0.03),
                    "statistical_significance": pact.Like(0.001)
                },
                "group_metrics": pact.EachLike({
                    "group": pact.Like("white"),
                    "positive_rate": pact.Like(0.75),
                    "sample_size": pact.Like(150),
                    "confidence_interval": {
                        "lower": pact.Like(0.70),
                        "upper": pact.Like(0.80)
                    }
                }, minimum=2),
                "recommendations": pact.EachLike({
                    "type": pact.Like("threshold_adjustment"),
                    "description": pact.Like("Adjust decision threshold for better fairness"),
                    "expected_impact": pact.Like(0.02)
                }, minimum=1),
                "analysis_id": pact.Like("analysis_456"),
                "timestamp": pact.Like("2024-01-15T10:30:00Z")
            }
        }
        
        return self.bias_pact.given("Sufficient prediction data available").upon_receiving(
            "A request for bias analysis"
        ).will_respond_with(**expected_response)