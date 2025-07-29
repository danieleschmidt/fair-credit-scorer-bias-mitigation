#!/usr/bin/env python3
"""
Advanced Load Testing Runner for MATURING repositories
Comprehensive performance validation and quality gate enforcement
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests


class LoadTestRunner:
    """Advanced load testing orchestration with quality gates."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "load_test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Performance thresholds
        self.thresholds = {
            "max_response_time": 2000,  # 2 seconds
            "p95_response_time": 1000,  # 1 second  
            "p99_response_time": 1500,  # 1.5 seconds
            "error_rate": 0.01,         # 1% error rate
            "min_throughput": 100       # 100 requests per second
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load load testing configuration."""
        default_config = {
            "host": "http://localhost:8000",
            "scenarios": {
                "baseline": {"users": 50, "spawn_rate": 10, "run_time": "15m"},
                "stress": {"users": 200, "spawn_rate": 50, "run_time": "10m"},
                "spike": {"users": 100, "spawn_rate": 100, "run_time": "5m"},
                "endurance": {"users": 30, "spawn_rate": 5, "run_time": "30m"}
            },
            "warmup_time": 30,
            "cooldown_time": 10
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                custom_config = json.load(f)
                # Merge configurations
                default_config.update(custom_config)
        
        return default_config
    
    def run_scenario(self, scenario_name: str, **overrides) -> Dict[str, Any]:
        """Run a specific load testing scenario."""
        print(f"🚀 Starting load test scenario: {scenario_name}")
        
        # Get scenario configuration
        scenario_config = self.config["scenarios"].get(scenario_name)
        if not scenario_config:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Apply overrides
        scenario_config.update(overrides)
        
        # Check if target service is available
        if not self._check_service_health():
            print("❌ Target service is not healthy. Aborting load test.")
            return {"success": False, "error": "Service unavailable"}
        
        # Run warmup if configured
        if self.config.get("warmup_time", 0) > 0:
            self._run_warmup()
        
        # Execute load test
        results = self._execute_locust_test(scenario_name, scenario_config)
        
        # Run cooldown if configured
        if self.config.get("cooldown_time", 0) > 0:
            self._run_cooldown()
        
        # Analyze results and check quality gates
        analysis = self._analyze_results(results)
        
        # Generate comprehensive report
        report = self._generate_report(scenario_name, scenario_config, results, analysis)
        
        return {
            "success": analysis["quality_gates"]["all_passed"],
            "scenario": scenario_name,
            "results": results,
            "analysis": analysis,
            "report_path": report["file_path"]
        }
    
    def _check_service_health(self) -> bool:
        """Check if the target service is healthy and responsive."""
        print("🔍 Checking service health...")
        
        try:
            response = requests.get(
                f"{self.config['host']}/health",
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Service is healthy")
                return True
            else:
                print(f"⚠️ Service returned status {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Service health check failed: {e}")
            return False
    
    def _run_warmup(self):
        """Run service warmup before load testing."""
        warmup_time = self.config["warmup_time"]
        print(f"🔥 Running service warmup for {warmup_time} seconds...")
        
        # Send light load to warm up the service
        warmup_cmd = [
            "locust",
            "-f", str(self.project_root / "load_testing" / "locustfile.py"),
            "--headless",
            "-u", "5",  # 5 concurrent users
            "-r", "1",  # 1 user per second spawn rate
            "-t", f"{warmup_time}s",
            "-H", self.config["host"],
            "--only-summary"
        ]
        
        try:
            subprocess.run(warmup_cmd, check=True, capture_output=True)
            print("✅ Warmup completed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warmup failed: {e}")
    
    def _run_cooldown(self):
        """Run cooldown period after load testing."""
        cooldown_time = self.config["cooldown_time"]
        print(f"❄️ Cooling down for {cooldown_time} seconds...")
        time.sleep(cooldown_time)
    
    def _execute_locust_test(self, scenario_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual Locust load test."""
        print(f"⚡ Executing load test: {config['users']} users, {config['run_time']} duration")
        
        # Prepare output files
        stats_file = self.results_dir / f"{scenario_name}_stats.csv"
        failures_file = self.results_dir / f"{scenario_name}_failures.csv"
        
        # Build Locust command
        locust_cmd = [
            "locust",
            "-f", str(self.project_root / "load_testing" / "locustfile.py"),
            "--headless",
            "-u", str(config["users"]),
            "-r", str(config["spawn_rate"]),
            "-t", config["run_time"],
            "-H", self.config["host"],
            "--csv", str(self.results_dir / scenario_name),
            "--html", str(self.results_dir / f"{scenario_name}_report.html"),
            "--logfile", str(self.results_dir / f"{scenario_name}.log")
        ]
        
        # Add user class if specified
        if "user_class" in config:
            locust_cmd.extend(["--class-picker", config["user_class"]])
        
        print(f"Running command: {' '.join(locust_cmd)}")
        
        try:
            # Execute Locust
            result = subprocess.run(
                locust_cmd,
                capture_output=True,
                text=True,
                timeout=self._parse_time_to_seconds(config["run_time"]) + 300  # Add 5 min buffer
            )
            
            if result.returncode == 0:
                print("✅ Load test completed successfully")
                
                # Parse results from generated files
                return self._parse_locust_results(scenario_name)
            else:
                print(f"❌ Load test failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            print("⏰ Load test timed out")
            return {"success": False, "error": "Test timed out"}
        except Exception as e:
            print(f"💥 Load test execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_time_to_seconds(self, time_str: str) -> int:
        """Parse time string (e.g., '15m', '30s') to seconds."""
        if time_str.endswith('s'):
            return int(time_str[:-1])
        elif time_str.endswith('m'):
            return int(time_str[:-1]) * 60
        elif time_str.endswith('h'):
            return int(time_str[:-1]) * 3600
        else:
            return int(time_str)  # Assume seconds
    
    def _parse_locust_results(self, scenario_name: str) -> Dict[str, Any]:
        """Parse Locust results from CSV files."""
        stats_file = self.results_dir / f"{scenario_name}_stats.csv"
        failures_file = self.results_dir / f"{scenario_name}_failures.csv"
        
        results = {
            "stats": [],
            "failures": [],
            "summary": {}
        }
        
        # Parse stats
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Has data beyond header
                    headers = lines[0].strip().split(',')
                    for line in lines[1:]:
                        if line.strip():
                            values = line.strip().split(',')
                            results["stats"].append(dict(zip(headers, values)))
        
        # Parse failures
        if failures_file.exists():
            with open(failures_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Has data beyond header
                    headers = lines[0].strip().split(',')
                    for line in lines[1:]:
                        if line.strip():
                            values = line.strip().split(',')
                            results["failures"].append(dict(zip(headers, values)))
        
        # Calculate summary metrics
        if results["stats"]:
            total_stats = [s for s in results["stats"] if s.get("Name") == "Aggregated"]
            if total_stats:
                stat = total_stats[0]
                results["summary"] = {
                    "total_requests": int(stat.get("Request Count", 0)),
                    "failure_count": int(stat.get("Failure Count", 0)),
                    "median_response_time": float(stat.get("Median Response Time", 0)),
                    "p95_response_time": float(stat.get("95%", 0)),
                    "p99_response_time": float(stat.get("99%", 0)),
                    "max_response_time": float(stat.get("Max Response Time", 0)),
                    "avg_response_time": float(stat.get("Average Response Time", 0)),
                    "min_response_time": float(stat.get("Min Response Time", 0)),
                    "requests_per_second": float(stat.get("Requests/s", 0)),
                    "error_rate": float(stat.get("Failure Count", 0)) / float(stat.get("Request Count", 1))
                }
        
        return results
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results and check quality gates."""
        if not results.get("summary"):
            return {"quality_gates": {"all_passed": False, "gates": {}}}
        
        summary = results["summary"]
        
        # Check quality gates
        gates = {
            "max_response_time": {
                "threshold": self.thresholds["max_response_time"],
                "actual": summary["max_response_time"],
                "passed": summary["max_response_time"] <= self.thresholds["max_response_time"]
            },
            "p95_response_time": {
                "threshold": self.thresholds["p95_response_time"],
                "actual": summary["p95_response_time"],
                "passed": summary["p95_response_time"] <= self.thresholds["p95_response_time"]
            },
            "p99_response_time": {
                "threshold": self.thresholds["p99_response_time"],
                "actual": summary["p99_response_time"],
                "passed": summary["p99_response_time"] <= self.thresholds["p99_response_time"]
            },
            "error_rate": {
                "threshold": self.thresholds["error_rate"],
                "actual": summary["error_rate"],
                "passed": summary["error_rate"] <= self.thresholds["error_rate"]
            },
            "throughput": {
                "threshold": self.thresholds["min_throughput"],
                "actual": summary["requests_per_second"],
                "passed": summary["requests_per_second"] >= self.thresholds["min_throughput"]
            }
        }
        
        all_passed = all(gate["passed"] for gate in gates.values())
        
        return {
            "quality_gates": {
                "all_passed": all_passed,
                "gates": gates
            },
            "performance_score": self._calculate_performance_score(summary, gates),
            "recommendations": self._generate_recommendations(summary, gates)
        }
    
    def _calculate_performance_score(self, summary: Dict[str, Any], gates: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        # Base score from quality gates
        passed_gates = sum(1 for gate in gates.values() if gate["passed"])
        gate_score = (passed_gates / len(gates)) * 70  # Up to 70 points from gates
        
        # Bonus points for exceptional performance
        bonus_score = 0
        
        # Throughput bonus (up to 15 points)
        if summary["requests_per_second"] > self.thresholds["min_throughput"] * 2:
            bonus_score += 15
        elif summary["requests_per_second"] > self.thresholds["min_throughput"] * 1.5:
            bonus_score += 10
        elif summary["requests_per_second"] > self.thresholds["min_throughput"]:
            bonus_score += 5
        
        # Response time bonus (up to 15 points)
        if summary["p95_response_time"] < self.thresholds["p95_response_time"] * 0.5:
            bonus_score += 15
        elif summary["p95_response_time"] < self.thresholds["p95_response_time"] * 0.75:
            bonus_score += 10
        elif summary["p95_response_time"] < self.thresholds["p95_response_time"]:
            bonus_score += 5
        
        return min(100, gate_score + bonus_score)
    
    def _generate_recommendations(self, summary: Dict[str, Any], gates: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Response time recommendations
        if not gates["p95_response_time"]["passed"]:
            recommendations.append(
                f"P95 response time ({summary['p95_response_time']:.1f}ms) exceeds threshold. "
                "Consider optimizing database queries, adding caching, or scaling horizontally."
            )
        
        if not gates["max_response_time"]["passed"]:
            recommendations.append(
                f"Maximum response time ({summary['max_response_time']:.1f}ms) is too high. "
                "Investigate outliers and implement request timeouts."
            )
        
        # Throughput recommendations
        if not gates["throughput"]["passed"]:
            recommendations.append(
                f"Throughput ({summary['requests_per_second']:.1f} RPS) is below target. "
                "Consider scaling up resources or optimizing application performance."
            )
        
        # Error rate recommendations
        if not gates["error_rate"]["passed"]:
            recommendations.append(
                f"Error rate ({summary['error_rate']:.1%}) exceeds threshold. "
                "Review application logs and fix underlying issues."
            )
        
        # General recommendations
        if summary["p99_response_time"] > summary["p95_response_time"] * 3:
            recommendations.append(
                "Large gap between P95 and P99 response times suggests inconsistent performance. "
                "Investigate periodic spikes or resource contention."
            )
        
        if not recommendations:
            recommendations.append("Performance is within acceptable limits. Consider stress testing with higher loads.")
        
        return recommendations
    
    def _generate_report(self, scenario_name: str, config: Dict[str, Any], 
                        results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive load testing report."""
        report_file = self.results_dir / f"{scenario_name}_detailed_report.json"
        
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scenario": {
                "name": scenario_name,
                "configuration": config
            },
            "results": results,
            "analysis": analysis,
            "thresholds": self.thresholds,
            "environment": {
                "host": self.config["host"],
                "test_duration": config["run_time"],
                "concurrent_users": config["users"],
                "spawn_rate": config["spawn_rate"]
            }
        }
        
        # Save detailed JSON report
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate markdown summary
        markdown_file = self.results_dir / f"{scenario_name}_summary.md"
        self._generate_markdown_report(markdown_file, report_data)
        
        return {
            "file_path": str(report_file),
            "markdown_path": str(markdown_file)
        }
    
    def _generate_markdown_report(self, file_path: Path, report_data: Dict[str, Any]):
        """Generate markdown summary report."""
        with open(file_path, 'w') as f:
            f.write(f"# Load Testing Report: {report_data['scenario']['name']}\n\n")
            f.write(f"**Generated:** {report_data['timestamp']}\n\n")
            
            # Configuration
            f.write("## Test Configuration\n\n")
            config = report_data['scenario']['configuration']
            f.write(f"- **Users:** {config['users']}\n")
            f.write(f"- **Spawn Rate:** {config['spawn_rate']} users/second\n")
            f.write(f"- **Duration:** {config['run_time']}\n")
            f.write(f"- **Target Host:** {report_data['environment']['host']}\n\n")
            
            # Results Summary
            if 'summary' in report_data['results']:
                summary = report_data['results']['summary']
                f.write("## Performance Summary\n\n")
                f.write(f"- **Total Requests:** {summary.get('total_requests', 'N/A')}\n")
                f.write(f"- **Failed Requests:** {summary.get('failure_count', 'N/A')}\n")
                f.write(f"- **Error Rate:** {summary.get('error_rate', 0):.2%}\n")
                f.write(f"- **Requests/Second:** {summary.get('requests_per_second', 'N/A'):.2f}\n")
                f.write(f"- **Avg Response Time:** {summary.get('avg_response_time', 'N/A'):.2f} ms\n")
                f.write(f"- **P95 Response Time:** {summary.get('p95_response_time', 'N/A'):.2f} ms\n")
                f.write(f"- **P99 Response Time:** {summary.get('p99_response_time', 'N/A'):.2f} ms\n")
                f.write(f"- **Max Response Time:** {summary.get('max_response_time', 'N/A'):.2f} ms\n\n")
            
            # Quality Gates
            f.write("## Quality Gates\n\n")
            gates = report_data['analysis']['quality_gates']['gates']
            all_passed = report_data['analysis']['quality_gates']['all_passed']
            
            f.write(f"**Overall Result:** {'✅ PASSED' if all_passed else '❌ FAILED'}\n\n")
            
            f.write("| Gate | Threshold | Actual | Result |\n")
            f.write("|------|-----------|--------|--------|\n")
            
            for gate_name, gate_data in gates.items():
                status = "✅ PASS" if gate_data['passed'] else "❌ FAIL"
                f.write(f"| {gate_name} | {gate_data['threshold']} | {gate_data['actual']:.2f} | {status} |\n")
            
            f.write("\n")
            
            # Performance Score
            f.write(f"**Performance Score:** {report_data['analysis']['performance_score']:.1f}/100\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report_data['analysis']['recommendations'], 1):
                f.write(f"{i}. {rec}\n")


def main():
    """Main entry point for load testing."""
    parser = argparse.ArgumentParser(description="Advanced Load Testing Runner")
    parser.add_argument("scenario", choices=["baseline", "stress", "spike", "endurance"], 
                       help="Load testing scenario to run")
    parser.add_argument("--host", help="Target host URL")
    parser.add_argument("--users", type=int, help="Number of concurrent users")
    parser.add_argument("--spawn-rate", type=int, help="User spawn rate per second")
    parser.add_argument("--run-time", help="Test duration (e.g., '10m', '30s')")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup phase")
    parser.add_argument("--no-cooldown", action="store_true", help="Skip cooldown phase")
    
    args = parser.parse_args()
    
    # Create runner
    runner = LoadTestRunner(args.config)
    
    # Override configuration with command line arguments
    overrides = {}
    if args.host:
        runner.config["host"] = args.host
    if args.users:
        overrides["users"] = args.users
    if args.spawn_rate:
        overrides["spawn_rate"] = args.spawn_rate
    if args.run_time:
        overrides["run_time"] = args.run_time
    if args.no_warmup:
        runner.config["warmup_time"] = 0
    if args.no_cooldown:
        runner.config["cooldown_time"] = 0
    
    # Run the scenario
    try:
        result = runner.run_scenario(args.scenario, **overrides)
        
        if result["success"]:
            print(f"\n🎉 Load test completed successfully!")
            print(f"   Scenario: {result['scenario']}")
            print(f"   Report: {result['report_path']}")
            sys.exit(0)
        else:
            print(f"\n💥 Load test failed!")
            print(f"   Error: {result.get('error', 'Quality gates not met')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Load test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()