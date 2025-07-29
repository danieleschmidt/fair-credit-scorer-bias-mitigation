#!/usr/bin/env python3
"""
Advanced SonarQube Integration for MATURING repositories
Comprehensive code quality analysis with custom quality gates
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests


class SonarQubeRunner:
    """Advanced SonarQube analysis orchestration with quality gates."""
    
    def __init__(self, project_key: str = "fair-credit-scorer-bias-mitigation"):
        self.project_key = project_key
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "quality_reports"
        self.results_dir.mkdir(exist_ok=True)
        
        # SonarQube configuration
        self.sonar_host = os.getenv("SONAR_HOST_URL", "http://localhost:9000")
        self.sonar_token = os.getenv("SONAR_TOKEN")
        self.sonar_scanner = os.getenv("SONAR_SCANNER_PATH", "sonar-scanner")
        
        # Quality gate thresholds for MATURING repositories
        self.quality_thresholds = {
            "coverage": 80.0,
            "duplicated_lines_density": 3.0,
            "maintainability_rating": "A",
            "reliability_rating": "A", 
            "security_rating": "A",
            "security_hotspots_reviewed": 100.0,
            "new_coverage": 85.0,
            "new_duplicated_lines_density": 3.0,
            "new_maintainability_rating": "A",
            "new_reliability_rating": "A",
            "new_security_rating": "A",
            "new_security_hotspots_reviewed": 100.0
        }
    
    def run_full_analysis(self, branch: Optional[str] = None, 
                         pull_request: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run complete SonarQube analysis with quality gates."""
        print("🔍 Starting comprehensive code quality analysis...")
        
        # Step 1: Prepare analysis environment
        self._prepare_analysis_environment()
        
        # Step 2: Generate required reports
        reports_generated = self._generate_prerequisite_reports()
        if not reports_generated:
            return {"success": False, "error": "Failed to generate prerequisite reports"}
        
        # Step 3: Run SonarQube analysis
        analysis_result = self._run_sonar_analysis(branch, pull_request)
        if not analysis_result["success"]:
            return analysis_result
        
        # Step 4: Wait for analysis to complete and get quality gate status
        quality_gate_result = self._wait_for_quality_gate(analysis_result["task_id"])
        
        # Step 5: Generate comprehensive report
        report = self._generate_quality_report(quality_gate_result)
        
        return {
            "success": quality_gate_result["passed"],
            "quality_gate": quality_gate_result,
            "analysis_id": analysis_result.get("analysis_id"),
            "report_path": report["file_path"],
            "sonar_url": f"{self.sonar_host}/dashboard?id={self.project_key}"
        }
    
    def _prepare_analysis_environment(self):
        """Prepare the environment for SonarQube analysis."""
        print("🛠️ Preparing analysis environment...")
        
        # Ensure all necessary directories exist
        required_dirs = ["htmlcov", "reports", "quality_reports"]
        for dir_name in required_dirs:
            (self.project_root / dir_name).mkdir(exist_ok=True)
        
        # Check SonarQube connectivity
        if self.sonar_token:
            try:
                response = requests.get(
                    f"{self.sonar_host}/api/system/status",
                    auth=(self.sonar_token, ""),
                    timeout=10
                )
                if response.status_code == 200:
                    print("✅ SonarQube server is accessible")
                else:
                    print(f"⚠️ SonarQube server returned {response.status_code}")
            except requests.RequestException as e:
                print(f"⚠️ Cannot connect to SonarQube server: {e}")
    
    def _generate_prerequisite_reports(self) -> bool:
        """Generate all reports required for SonarQube analysis."""
        print("📊 Generating prerequisite reports...")
        
        reports_to_generate = [
            ("coverage", self._generate_coverage_report),
            ("pylint", self._generate_pylint_report),
            ("bandit", self._generate_bandit_report),
            ("mypy", self._generate_mypy_report)
        ]
        
        success_count = 0
        for report_name, generator_func in reports_to_generate:
            try:
                print(f"  Generating {report_name} report...")
                if generator_func():
                    print(f"  ✅ {report_name} report generated")
                    success_count += 1
                else:
                    print(f"  ❌ {report_name} report failed")
            except Exception as e:
                print(f"  💥 {report_name} report error: {e}")
        
        # Require at least coverage and bandit reports
        required_reports = 2
        if success_count >= required_reports:
            print(f"✅ Generated {success_count}/{len(reports_to_generate)} reports")
            return True
        else:
            print(f"❌ Only generated {success_count}/{len(reports_to_generate)} reports")
            return False
    
    def _generate_coverage_report(self) -> bool:
        """Generate coverage report in XML format for SonarQube."""
        try:
            cmd = [
                "python", "-m", "pytest", "tests/",
                "--cov=src",
                "--cov-report=xml:coverage.xml",
                "--cov-report=html:htmlcov",
                "-q"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return result.returncode == 0 and Path("coverage.xml").exists()
        except Exception:
            return False
    
    def _generate_pylint_report(self) -> bool:
        """Generate pylint report for SonarQube."""
        try:
            cmd = [
                "python", "-m", "pylint", "src",
                "--output-format=text",
                "--reports=yes",
                "--msg-template='{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}'"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Pylint returns non-zero even with minor issues, so check output instead
            if result.stdout:
                with open("pylint-report.txt", "w") as f:
                    f.write(result.stdout)
                return True
            return False
        except Exception:
            return False
    
    def _generate_bandit_report(self) -> bool:
        """Generate bandit security report for SonarQube."""
        try:
            cmd = [
                "python", "-m", "bandit", "-r", "src",
                "-f", "json",
                "-o", "bandit-report.json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return Path("bandit-report.json").exists()
        except Exception:
            return False
    
    def _generate_mypy_report(self) -> bool:
        """Generate mypy type checking report for SonarQube."""
        try:
            cmd = [
                "python", "-m", "mypy", "src",
                "--ignore-missing-imports",
                "--txt-report", ".",
                "--cobertura-xml-report", "."
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Save mypy output
            with open("mypy-report.txt", "w") as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)
            
            return Path("mypy-report.txt").exists()
        except Exception:
            return False
    
    def _run_sonar_analysis(self, branch: Optional[str] = None, 
                           pull_request: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute SonarQube analysis."""
        print("🔬 Running SonarQube analysis...")
        
        # Build sonar-scanner command
        cmd = [self.sonar_scanner]
        
        # Add branch analysis parameters
        if branch and not pull_request:
            cmd.extend([f"-Dsonar.branch.name={branch}"])
        
        # Add pull request analysis parameters
        if pull_request:
            cmd.extend([
                f"-Dsonar.pullrequest.key={pull_request['key']}",
                f"-Dsonar.pullrequest.branch={pull_request['branch']}",
                f"-Dsonar.pullrequest.base={pull_request['base']}"
            ])
        
        # Add authentication if available
        if self.sonar_token:
            cmd.extend([f"-Dsonar.login={self.sonar_token}"])
        
        try:
            # Execute analysis
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                # Extract task ID from output
                task_id = self._extract_task_id(result.stdout)
                analysis_id = self._extract_analysis_id(result.stdout)
                
                print("✅ SonarQube analysis completed")
                return {
                    "success": True,
                    "task_id": task_id,
                    "analysis_id": analysis_id,
                    "output": result.stdout
                }
            else:
                print(f"❌ SonarQube analysis failed")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "output": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            print("⏰ SonarQube analysis timed out")
            return {"success": False, "error": "Analysis timed out"}
        except Exception as e:
            print(f"💥 SonarQube analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_task_id(self, output: str) -> Optional[str]:
        """Extract task ID from SonarQube scanner output."""
        for line in output.split('\n'):
            if "ANALYSIS SUCCESSFUL" in line and "task id" in line.lower():
                # Extract task ID from line like "ANALYSIS SUCCESSFUL, you can browse http://localhost:9000/dashboard?id=project (task id: ABC123)"
                if "task id:" in line:
                    return line.split("task id:")[-1].strip().rstrip(")")
        return None
    
    def _extract_analysis_id(self, output: str) -> Optional[str]:
        """Extract analysis ID from SonarQube scanner output."""
        for line in output.split('\n'):
            if "Analysis id:" in line:
                return line.split("Analysis id:")[-1].strip()
        return None
    
    def _wait_for_quality_gate(self, task_id: Optional[str], 
                              max_wait_time: int = 300) -> Dict[str, Any]:
        """Wait for quality gate evaluation and return results."""
        if not task_id or not self.sonar_token:
            print("⚠️ Cannot check quality gate status (missing task ID or token)")
            return {"passed": False, "error": "Cannot check quality gate"}
        
        print(f"⏳ Waiting for quality gate evaluation (task: {task_id})...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                # Check task status
                task_response = requests.get(
                    f"{self.sonar_host}/api/ce/task",
                    params={"id": task_id},
                    auth=(self.sonar_token, ""),
                    timeout=10
                )
                
                if task_response.status_code == 200:
                    task_data = task_response.json()
                    task_status = task_data.get("task", {}).get("status")
                    
                    if task_status == "SUCCESS":
                        # Get quality gate status
                        return self._get_quality_gate_status()
                    elif task_status == "FAILED":
                        return {"passed": False, "error": "Analysis task failed"}
                    elif task_status in ["PENDING", "IN_PROGRESS"]:
                        print(f"  Task status: {task_status}, waiting...")
                        time.sleep(10)
                        continue
                    else:
                        return {"passed": False, "error": f"Unknown task status: {task_status}"}
                
                time.sleep(10)
                
            except requests.RequestException as e:
                print(f"⚠️ Error checking task status: {e}")
                time.sleep(10)
        
        return {"passed": False, "error": "Quality gate evaluation timeout"}
    
    def _get_quality_gate_status(self) -> Dict[str, Any]:
        """Get detailed quality gate status."""
        try:
            response = requests.get(
                f"{self.sonar_host}/api/qualitygates/project_status",
                params={"projectKey": self.project_key},
                auth=(self.sonar_token, ""),
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                project_status = data.get("projectStatus", {})
                
                overall_status = project_status.get("status", "ERROR")
                conditions = project_status.get("conditions", [])
                
                # Analyze individual conditions
                failed_conditions = [c for c in conditions if c.get("status") == "ERROR"]
                warning_conditions = [c for c in conditions if c.get("status") == "WARN"]
                
                print(f"📊 Quality Gate Status: {overall_status}")
                
                if failed_conditions:
                    print("❌ Failed conditions:")
                    for condition in failed_conditions:
                        metric = condition.get("metricKey", "unknown")
                        actual = condition.get("actualValue", "N/A")
                        threshold = condition.get("errorThreshold", "N/A")
                        print(f"  {metric}: {actual} (threshold: {threshold})")
                
                if warning_conditions:
                    print("⚠️ Warning conditions:")
                    for condition in warning_conditions:
                        metric = condition.get("metricKey", "unknown")
                        actual = condition.get("actualValue", "N/A")
                        threshold = condition.get("warningThreshold", "N/A")
                        print(f"  {metric}: {actual} (threshold: {threshold})")
                
                return {
                    "passed": overall_status == "OK",
                    "status": overall_status,
                    "conditions": conditions,
                    "failed_conditions": failed_conditions,
                    "warning_conditions": warning_conditions,
                    "metrics": self._extract_metrics(conditions)
                }
            else:
                return {"passed": False, "error": f"API returned {response.status_code}"}
                
        except requests.RequestException as e:
            return {"passed": False, "error": f"Quality gate check failed: {e}"}
    
    def _extract_metrics(self, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key metrics from quality gate conditions."""
        metrics = {}
        
        for condition in conditions:
            metric_key = condition.get("metricKey")
            actual_value = condition.get("actualValue")
            
            if metric_key and actual_value is not None:
                # Convert to appropriate type
                try:
                    if "." in str(actual_value):
                        metrics[metric_key] = float(actual_value)
                    else:
                        metrics[metric_key] = int(actual_value)
                except (ValueError, TypeError):
                    metrics[metric_key] = actual_value
        
        return metrics
    
    def _generate_quality_report(self, quality_gate_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive quality analysis report."""
        report_file = self.results_dir / "sonarqube_quality_report.json"
        
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "project_key": self.project_key,
            "sonar_host": self.sonar_host,
            "quality_gate": quality_gate_result,
            "thresholds": self.quality_thresholds,
            "recommendations": self._generate_quality_recommendations(quality_gate_result)
        }
        
        # Save detailed JSON report
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate markdown summary
        markdown_file = self.results_dir / "sonarqube_summary.md"
        self._generate_markdown_quality_report(markdown_file, report_data)
        
        print(f"📄 Quality report generated: {report_file}")
        
        return {
            "file_path": str(report_file),
            "markdown_path": str(markdown_file)
        }
    
    def _generate_quality_recommendations(self, quality_gate_result: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if not quality_gate_result.get("passed", False):
            failed_conditions = quality_gate_result.get("failed_conditions", [])
            
            for condition in failed_conditions:
                metric = condition.get("metricKey", "")
                actual = condition.get("actualValue", "")
                threshold = condition.get("errorThreshold", "")
                
                if "coverage" in metric:
                    recommendations.append(
                        f"Increase test coverage from {actual}% to at least {threshold}%. "
                        "Focus on testing critical business logic and edge cases."
                    )
                elif "duplicated_lines" in metric:
                    recommendations.append(
                        f"Reduce code duplication from {actual}% to below {threshold}%. "
                        "Extract common functionality into shared modules."
                    )
                elif "maintainability" in metric:
                    recommendations.append(
                        "Improve code maintainability by reducing complexity, "
                        "improving naming, and adding documentation."
                    )
                elif "reliability" in metric:
                    recommendations.append(
                        "Fix reliability issues including potential bugs, "
                        "error handling, and resource management problems."
                    )
                elif "security" in metric:
                    recommendations.append(
                        "Address security vulnerabilities and hotspots. "
                        "Review authentication, input validation, and data handling."
                    )
        
        if not recommendations:
            recommendations.append(
                "Quality gate passed! Consider implementing additional "
                "quality improvements for enhanced code excellence."
            )
        
        return recommendations
    
    def _generate_markdown_quality_report(self, file_path: Path, report_data: Dict[str, Any]):
        """Generate markdown quality report."""
        with open(file_path, 'w') as f:
            f.write(f"# SonarQube Quality Report\n\n")
            f.write(f"**Generated:** {report_data['timestamp']}\n")
            f.write(f"**Project:** {report_data['project_key']}\n")
            f.write(f"**SonarQube URL:** [{report_data['sonar_host']}]({report_data['sonar_host']}/dashboard?id={report_data['project_key']})\n\n")
            
            # Quality Gate Status
            quality_gate = report_data['quality_gate']
            status = quality_gate.get('status', 'UNKNOWN')
            f.write(f"## Quality Gate Status: {status}\n\n")
            
            if quality_gate.get('passed', False):
                f.write("✅ **PASSED** - All quality criteria met\n\n")
            else:
                f.write("❌ **FAILED** - Quality criteria not met\n\n")
            
            # Metrics Summary
            metrics = quality_gate.get('metrics', {})
            if metrics:
                f.write("## Metrics Summary\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                for metric_key, value in metrics.items():
                    display_name = metric_key.replace('_', ' ').title()
                    f.write(f"| {display_name} | {value} |\n")
                
                f.write("\n")
            
            # Failed Conditions
            failed_conditions = quality_gate.get('failed_conditions', [])
            if failed_conditions:
                f.write("## Failed Conditions\n\n")
                for condition in failed_conditions:
                    metric = condition.get('metricKey', 'Unknown')
                    actual = condition.get('actualValue', 'N/A')
                    threshold = condition.get('errorThreshold', 'N/A')
                    f.write(f"- **{metric}**: {actual} (threshold: {threshold})\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, rec in enumerate(report_data['recommendations'], 1):
                f.write(f"{i}. {rec}\n")


def main():
    """Main entry point for SonarQube analysis."""
    parser = argparse.ArgumentParser(description="Advanced SonarQube Analysis Runner")
    parser.add_argument("--project-key", help="SonarQube project key")
    parser.add_argument("--branch", help="Branch name for analysis")
    parser.add_argument("--pull-request-key", help="Pull request key")
    parser.add_argument("--pull-request-branch", help="Pull request branch")
    parser.add_argument("--pull-request-base", help="Pull request base branch")
    parser.add_argument("--sonar-host", help="SonarQube server URL")
    parser.add_argument("--sonar-token", help="SonarQube authentication token")
    parser.add_argument("--wait-for-quality-gate", action="store_true", 
                       help="Wait for quality gate evaluation")
    
    args = parser.parse_args()
    
    # Create runner
    project_key = args.project_key or "fair-credit-scorer-bias-mitigation"
    runner = SonarQubeRunner(project_key)
    
    # Override configuration with command line arguments
    if args.sonar_host:
        runner.sonar_host = args.sonar_host
    if args.sonar_token:
        runner.sonar_token = args.sonar_token
    
    # Prepare pull request info
    pull_request = None
    if args.pull_request_key:
        pull_request = {
            "key": args.pull_request_key,
            "branch": args.pull_request_branch or "feature-branch",
            "base": args.pull_request_base or "main"
        }
    
    # Run analysis
    try:
        result = runner.run_full_analysis(args.branch, pull_request)
        
        if result["success"]:
            print(f"\n🎉 Quality analysis completed successfully!")
            print(f"   Quality Gate: PASSED")
            print(f"   Report: {result['report_path']}")
            print(f"   SonarQube: {result['sonar_url']}")
            sys.exit(0)
        else:
            print(f"\n💥 Quality analysis failed!")
            print(f"   Quality Gate: FAILED")
            print(f"   Error: {result.get('error', 'Quality gate criteria not met')}")
            if 'report_path' in result:
                print(f"   Report: {result['report_path']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()