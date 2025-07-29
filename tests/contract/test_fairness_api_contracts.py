"""
Contract tests for Fairness API
Advanced API validation using consumer-driven contract testing
"""

import pytest
import requests
from contract_testing.pact_config import FairnessScoringContractTests, BiasDetectionContractTests


class TestFairnessAPIPacts:
    """Consumer contract tests for fairness API endpoints."""
    
    def setup_method(self):
        """Setup contract testing environment."""
        self.fairness_contracts = FairnessScoringContractTests()
        self.bias_contracts = BiasDetectionContractTests()
    
    def test_score_prediction_pact(self):
        """Test the score prediction API contract."""
        pact = self.fairness_contracts.test_score_prediction_contract()
        
        with pact:
            # Make the actual request to the mock service
            response = requests.post(
                f"{pact.uri}/api/v1/score",
                json={
                    "applicant_data": {
                        "age": 30,
                        "income": 50000.0,
                        "credit_history_length": 10,
                        "education_level": "bachelor",
                        "employment_status": "employed"
                    },
                    "model_version": "v1.2.0"
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            # Verify the response matches the contract
            assert response.status_code == 200
            response_data = response.json()
            
            # Validate response structure
            assert "credit_score" in response_data
            assert "risk_category" in response_data
            assert "confidence" in response_data
            assert "fairness_metrics" in response_data
            assert "explanation" in response_data
            assert "request_id" in response_data
            assert "timestamp" in response_data
            assert "processing_time_ms" in response_data
            
            # Validate fairness metrics structure
            fairness_metrics = response_data["fairness_metrics"]
            assert "demographic_parity" in fairness_metrics
            assert "equalized_odds" in fairness_metrics
            assert "calibration" in fairness_metrics
            
            # Validate explanation structure
            explanation = response_data["explanation"]
            assert "top_factors" in explanation
            assert "counterfactual" in explanation
            assert len(explanation["top_factors"]) >= 3
            
            for factor in explanation["top_factors"]:
                assert "feature" in factor
                assert "impact" in factor
                assert "direction" in factor
    
    def test_batch_scoring_pact(self):
        """Test the batch scoring API contract."""
        pact = self.fairness_contracts.test_batch_scoring_contract()
        
        with pact:
            response = requests.post(
                f"{pact.uri}/api/v1/score/batch",
                json={
                    "applicants": [
                        {
                            "id": "app_001",
                            "applicant_data": {
                                "age": 30,
                                "income": 50000.0,
                                "credit_history_length": 10
                            }
                        }
                    ],
                    "model_version": "v1.2.0",
                    "include_explanations": True
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Validate batch response structure
            assert "results" in response_data
            assert "batch_id" in response_data
            assert "processed_count" in response_data
            assert "failed_count" in response_data
            assert "processing_time_ms" in response_data
            
            # Validate individual results
            results = response_data["results"]
            assert len(results) >= 1
            
            for result in results:
                assert "id" in result
                assert "credit_score" in result
                assert "risk_category" in result
                assert "confidence" in result
                assert "fairness_metrics" in result
    
    def test_model_info_pact(self):
        """Test the model information API contract."""
        pact = self.fairness_contracts.test_model_info_contract()
        
        with pact:
            response = requests.get(
                f"{pact.uri}/api/v1/model/info",
                headers={"Accept": "application/json"}
            )
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Validate model info structure
            assert "model_version" in response_data
            assert "model_name" in response_data
            assert "training_date" in response_data
            assert "performance_metrics" in response_data
            assert "fairness_metrics" in response_data
            assert "feature_importance" in response_data
            assert "supported_features" in response_data
            
            # Validate performance metrics
            perf_metrics = response_data["performance_metrics"]
            assert "accuracy" in perf_metrics
            assert "precision" in perf_metrics
            assert "recall" in perf_metrics
            assert "f1_score" in perf_metrics
            
            # Validate fairness metrics
            fairness_metrics = response_data["fairness_metrics"]
            assert "overall_demographic_parity" in fairness_metrics
            assert "overall_equalized_odds" in fairness_metrics
            
            # Validate feature importance
            feature_importance = response_data["feature_importance"]
            assert len(feature_importance) >= 5
            
            for feature in feature_importance:
                assert "feature" in feature
                assert "importance" in feature
    
    def test_bias_analysis_pact(self):
        """Test the bias analysis API contract."""
        pact = self.bias_contracts.test_bias_analysis_contract()
        
        with pact:
            response = requests.post(
                f"{pact.uri}/api/v1/bias/analyze",
                json={
                    "predictions": [
                        {
                            "applicant_id": "app_001",
                            "predicted_score": 750,
                            "actual_outcome": 1,
                            "protected_attributes": {
                                "race": "white",
                                "gender": "female",
                                "age_group": "25-35"
                            }
                        }
                    ] * 100,  # Minimum 100 samples
                    "analysis_type": "demographic_parity",
                    "protected_attribute": "race"
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Validate bias analysis structure
            assert "bias_metrics" in response_data
            assert "group_metrics" in response_data
            assert "recommendations" in response_data
            assert "analysis_id" in response_data
            assert "timestamp" in response_data
            
            # Validate bias metrics
            bias_metrics = response_data["bias_metrics"]
            assert "demographic_parity_difference" in bias_metrics
            assert "equalized_odds_difference" in bias_metrics
            assert "statistical_significance" in bias_metrics
            
            # Validate group metrics
            group_metrics = response_data["group_metrics"]
            assert len(group_metrics) >= 2
            
            for group in group_metrics:
                assert "group" in group
                assert "positive_rate" in group
                assert "sample_size" in group
                assert "confidence_interval" in group
                assert "lower" in group["confidence_interval"]
                assert "upper" in group["confidence_interval"]
            
            # Validate recommendations
            recommendations = response_data["recommendations"]
            assert len(recommendations) >= 1
            
            for rec in recommendations:
                assert "type" in rec
                assert "description" in rec
                assert "expected_impact" in rec


@pytest.mark.integration
class TestContractIntegration:
    """Integration tests to verify contract compatibility."""
    
    def test_pact_generation(self):
        """Test that pact files are generated correctly."""
        fairness_contracts = FairnessScoringContractTests()
        
        # Create all pacts
        score_pact = fairness_contracts.test_score_prediction_contract()
        batch_pact = fairness_contracts.test_batch_scoring_contract()
        info_pact = fairness_contracts.test_model_info_contract()
        
        # Verify pact files exist (they should be created during teardown)
        pact_dir = fairness_contracts.config.pact_dir
        assert pact_dir.exists()
        
    def test_contract_backwards_compatibility(self):
        """Test that contracts maintain backwards compatibility."""
        # This would typically involve loading previous pact versions
        # and verifying they still work with current implementation
        pass
    
    def test_cross_service_contracts(self):
        """Test contracts between different services."""
        # Test that fairness service contracts work with bias detection
        fairness_contracts = FairnessScoringContractTests()
        bias_contracts = BiasDetectionContractTests()
        
        # Verify both services can be tested together
        assert fairness_contracts.config.service_name
        assert bias_contracts.config.service_name