"""
Advanced Load Testing Configuration for MATURING repositories
Comprehensive performance validation using Locust for fairness scoring API
"""

import json
import random
import time
from typing import Dict, Any, List
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
from faker import Faker


# Global test data
fake = Faker()

# Sample applicant data templates for realistic load testing
APPLICANT_TEMPLATES = [
    {
        "age": lambda: random.randint(18, 80),
        "income": lambda: random.randint(20000, 200000),
        "credit_history_length": lambda: random.randint(0, 40),
        "education_level": lambda: random.choice(["high_school", "bachelor", "master", "phd"]),
        "employment_status": lambda: random.choice(["employed", "unemployed", "self_employed", "retired"]),
        "debt_to_income_ratio": lambda: round(random.uniform(0.1, 0.8), 2),
        "number_of_accounts": lambda: random.randint(1, 20),
        "payment_history": lambda: round(random.uniform(0.7, 1.0), 2)
    }
]

# Performance thresholds for quality gates
PERFORMANCE_THRESHOLDS = {
    "max_response_time": 2000,  # 2 seconds
    "p95_response_time": 1000,  # 1 second
    "p99_response_time": 1500,  # 1.5 seconds
    "error_rate": 0.01,         # 1% error rate
    "min_throughput": 100       # 100 requests per second
}


class MetricsCollector:
    """Collect and analyze performance metrics during load testing."""
    
    def __init__(self):
        self.response_times = []
        self.error_count = 0
        self.total_requests = 0
        self.throughput_data = []
        self.start_time = time.time()
    
    def record_request(self, response_time: float, success: bool):
        """Record a request's performance metrics."""
        self.response_times.append(response_time)
        self.total_requests += 1
        if not success:
            self.error_count += 1
    
    def calculate_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        count = len(sorted_times)
        
        return {
            "p50": sorted_times[int(count * 0.50)],
            "p75": sorted_times[int(count * 0.75)],
            "p90": sorted_times[int(count * 0.90)],
            "p95": sorted_times[int(count * 0.95)],
            "p99": sorted_times[int(count * 0.99)],
            "max": sorted_times[-1],
            "min": sorted_times[0],
            "avg": sum(sorted_times) / count
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        duration = time.time() - self.start_time
        throughput = self.total_requests / duration if duration > 0 else 0
        error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "duration_seconds": duration,
            "throughput_rps": throughput,
            "percentiles": self.calculate_percentiles(),
            "quality_gates": self.check_quality_gates()
        }
    
    def check_quality_gates(self) -> Dict[str, Dict[str, Any]]:
        """Check performance against quality gates."""
        percentiles = self.calculate_percentiles()
        error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0
        duration = time.time() - self.start_time
        throughput = self.total_requests / duration if duration > 0 else 0
        
        gates = {
            "max_response_time": {
                "threshold": PERFORMANCE_THRESHOLDS["max_response_time"],
                "actual": percentiles.get("max", 0),
                "passed": percentiles.get("max", 0) <= PERFORMANCE_THRESHOLDS["max_response_time"]
            },
            "p95_response_time": {
                "threshold": PERFORMANCE_THRESHOLDS["p95_response_time"],
                "actual": percentiles.get("p95", 0),
                "passed": percentiles.get("p95", 0) <= PERFORMANCE_THRESHOLDS["p95_response_time"]
            },
            "p99_response_time": {
                "threshold": PERFORMANCE_THRESHOLDS["p99_response_time"],
                "actual": percentiles.get("p99", 0),
                "passed": percentiles.get("p99", 0) <= PERFORMANCE_THRESHOLDS["p99_response_time"]
            },
            "error_rate": {
                "threshold": PERFORMANCE_THRESHOLDS["error_rate"],
                "actual": error_rate,
                "passed": error_rate <= PERFORMANCE_THRESHOLDS["error_rate"]
            },
            "throughput": {
                "threshold": PERFORMANCE_THRESHOLDS["min_throughput"],
                "actual": throughput,
                "passed": throughput >= PERFORMANCE_THRESHOLDS["min_throughput"]
            }
        }
        
        return gates


# Global metrics collector
metrics_collector = MetricsCollector()


@events.request.add_listener
def record_request_metrics(request_type, name, response_time, response_length, exception, **kwargs):
    """Event listener to record request metrics."""
    success = exception is None
    metrics_collector.record_request(response_time, success)


@events.test_stop.add_listener
def print_performance_summary(environment, **kwargs):
    """Print performance summary at the end of the test."""
    summary = metrics_collector.get_summary()
    
    print("\n" + "="*80)
    print("LOAD TESTING PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Error Count: {summary['error_count']}")
    print(f"Error Rate: {summary['error_rate']:.2%}")
    print(f"Duration: {summary['duration_seconds']:.2f} seconds")
    print(f"Throughput: {summary['throughput_rps']:.2f} requests/second")
    
    print("\nResponse Time Percentiles (ms):")
    percentiles = summary['percentiles']
    for key, value in percentiles.items():
        print(f"  {key.upper()}: {value:.2f} ms")
    
    print("\nQuality Gates:")
    gates = summary['quality_gates']
    all_passed = True
    
    for gate_name, gate_data in gates.items():
        status = "✅ PASS" if gate_data['passed'] else "❌ FAIL"
        if not gate_data['passed']:
            all_passed = False
        
        print(f"  {gate_name}: {gate_data['actual']:.2f} (threshold: {gate_data['threshold']}) {status}")
    
    print("\nOverall Result:", "✅ ALL GATES PASSED" if all_passed else "❌ QUALITY GATES FAILED")
    
    # Save detailed results to file
    with open('load_test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: load_test_results.json")
    print("="*80)


def generate_applicant_data() -> Dict[str, Any]:
    """Generate realistic applicant data for testing."""
    template = random.choice(APPLICANT_TEMPLATES)
    return {key: func() if callable(func) else func for key, func in template.items()}


class FairnessScoringUser(FastHttpUser):
    """High-performance load testing user for fairness scoring API."""
    
    # Wait between requests (1-5 seconds to simulate realistic usage)
    wait_time = between(1, 5)
    
    def on_start(self):
        """Setup executed when user starts."""
        self.client.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LoadTest/1.0"
        }
    
    @task(weight=10)
    def score_single_applicant(self):
        """Test single applicant scoring endpoint (most common operation)."""
        applicant_data = generate_applicant_data()
        
        payload = {
            "applicant_data": applicant_data,
            "model_version": "v1.2.0"
        }
        
        with self.client.post(
            "/api/v1/score",
            json=payload,
            catch_response=True,
            name="score_single"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Validate response structure
                    required_fields = ["credit_score", "risk_category", "confidence", "fairness_metrics"]
                    if all(field in data for field in required_fields):
                        response.success()
                    else:
                        response.failure(f"Invalid response structure: missing fields")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(weight=3)
    def batch_score_applicants(self):
        """Test batch scoring endpoint (less frequent, higher load)."""
        batch_size = random.randint(5, 20)
        applicants = []
        
        for i in range(batch_size):
            applicants.append({
                "id": f"app_{i:03d}",
                "applicant_data": generate_applicant_data()
            })
        
        payload = {
            "applicants": applicants,
            "model_version": "v1.2.0",
            "include_explanations": random.choice([True, False])
        }
        
        with self.client.post(
            "/api/v1/score/batch",
            json=payload,
            catch_response=True,
            name="batch_score"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "results" in data and len(data["results"]) == batch_size:
                        response.success()
                    else:
                        response.failure(f"Batch size mismatch: expected {batch_size}, got {len(data.get('results', []))}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(weight=2)
    def get_model_info(self):
        """Test model information endpoint (occasional operation)."""
        with self.client.get(
            "/api/v1/model/info",
            catch_response=True,
            name="model_info"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    required_fields = ["model_version", "model_name", "performance_metrics"]
                    if all(field in data for field in required_fields):
                        response.success()
                    else:
                        response.failure("Invalid model info structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(weight=2)
    def analyze_bias(self):
        """Test bias analysis endpoint (analytical operation)."""
        # Generate sample prediction data
        predictions = []
        for i in range(50):  # Smaller batch for bias analysis
            predictions.append({
                "applicant_id": f"app_{i:03d}",
                "predicted_score": random.randint(300, 850),
                "actual_outcome": random.choice([0, 1]),
                "protected_attributes": {
                    "race": random.choice(["white", "black", "hispanic", "asian", "other"]),
                    "gender": random.choice(["male", "female"]),
                    "age_group": random.choice(["18-25", "26-35", "36-50", "51-65", "65+"])
                }
            })
        
        payload = {
            "predictions": predictions,
            "analysis_type": "demographic_parity",
            "protected_attribute": "race"
        }
        
        with self.client.post(
            "/api/v1/bias/analyze",
            json=payload,
            catch_response=True,
            name="bias_analysis"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "bias_metrics" in data and "group_metrics" in data:
                        response.success()
                    else:
                        response.failure("Invalid bias analysis structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(weight=1)
    def health_check(self):
        """Test health check endpoint (monitoring operation)."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="health_check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: HTTP {response.status_code}")


class StressTestUser(FairnessScoringUser):
    """Stress testing user with more aggressive patterns."""
    
    # Shorter wait times for stress testing
    wait_time = between(0.1, 1.0)
    
    @task(weight=15)
    def high_frequency_scoring(self):
        """High-frequency scoring requests for stress testing."""
        self.score_single_applicant()
    
    @task(weight=8)
    def concurrent_batch_requests(self):
        """Multiple concurrent batch requests."""
        self.batch_score_applicants()


class SpikeTestUser(FairnessScoringUser):
    """Spike testing user with burst patterns."""
    
    wait_time = between(0, 0.5)
    
    def on_start(self):
        """Setup for spike testing."""
        super().on_start()
        # Simulate sudden user arrival
        time.sleep(random.uniform(0, 10))  # Stagger arrivals over 10 seconds
    
    @task
    def burst_requests(self):
        """Send burst of requests to simulate spike load."""
        # Send 3-7 requests in quick succession
        burst_size = random.randint(3, 7)
        
        for _ in range(burst_size):
            self.score_single_applicant()
            time.sleep(0.1)  # Very short delay between burst requests
        
        # Then wait longer before next burst
        time.sleep(random.uniform(5, 15))


# Load testing scenarios
class LoadTestScenarios:
    """Predefined load testing scenarios for different test objectives."""
    
    @staticmethod
    def baseline_load():
        """Baseline load: 50 users, 5 minute ramp-up, 10 minutes steady."""
        return {
            "users": 50,
            "spawn_rate": 10,
            "run_time": "15m"
        }
    
    @staticmethod
    def stress_test():
        """Stress test: 200 users, rapid ramp-up, 5 minutes steady."""
        return {
            "users": 200,
            "spawn_rate": 50,
            "run_time": "10m"
        }
    
    @staticmethod
    def spike_test():
        """Spike test: sudden load increase simulation."""
        return {
            "users": 100,
            "spawn_rate": 100,  # Very rapid spawn
            "run_time": "5m"
        }
    
    @staticmethod
    def endurance_test():
        """Endurance test: moderate load over extended period."""
        return {
            "users": 30,
            "spawn_rate": 5,
            "run_time": "30m"
        }