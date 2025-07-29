#!/usr/bin/env python3
"""
Advanced Mutation Testing Runner for MATURING repositories
Provides comprehensive mutation testing with quality gates and reporting
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class MutationTestRunner:
    """Advanced mutation testing orchestration."""
    
    def __init__(self, config_path: str = "mutation_testing.toml"):
        self.config_path = config_path
        self.results_dir = Path("mutation_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_full_suite(self, module: Optional[str] = None) -> Dict:
        """Run complete mutation testing suite."""
        print("🧬 Starting mutation testing suite...")
        
        # Step 1: Run baseline tests
        if not self._run_baseline_tests():
            return {"success": False, "error": "Baseline tests failed"}
        
        # Step 2: Generate coverage report
        coverage_data = self._generate_coverage()
        
        # Step 3: Run mutation testing
        mutation_results = self._run_mutations(module)
        
        # Step 4: Generate reports
        report = self._generate_report(mutation_results, coverage_data)
        
        # Step 5: Check quality gates
        gate_result = self._check_quality_gates(report)
        
        return {
            "success": gate_result["passed"],
            "mutation_score": report.get("mutation_score", 0),
            "survived_mutants": report.get("survived_mutants", 0),
            "killed_mutants": report.get("killed_mutants", 0),
            "report_path": str(self.results_dir / "mutation_report.json"),
            "quality_gates": gate_result
        }
    
    def _run_baseline_tests(self) -> bool:
        """Ensure all tests pass before mutation testing."""
        print("📋 Running baseline test suite...")
        try:
            result = subprocess.run([
                "python", "-m", "pytest", "tests/", 
                "--tb=short", "-q"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Baseline tests passed")
                return True
            else:
                print(f"❌ Baseline tests failed:\n{result.stdout}\n{result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⏰ Baseline tests timed out")
            return False
        except Exception as e:
            print(f"💥 Error running baseline tests: {e}")
            return False
    
    def _generate_coverage(self) -> Dict:
        """Generate coverage report for mutation focus."""
        print("📊 Generating coverage data...")
        try:
            # Run coverage
            subprocess.run([
                "python", "-m", "pytest", "tests/", 
                "--cov=src", "--cov-report=json:mutation_results/coverage.json",
                "-q"
            ], capture_output=True, timeout=300)
            
            # Load coverage data
            coverage_file = self.results_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"⚠️  Could not generate coverage: {e}")
            return {}
    
    def _run_mutations(self, module: Optional[str] = None) -> Dict:
        """Execute mutation testing with mutmut."""
        print("🔬 Running mutation testing...")
        
        try:
            # Build mutmut command
            cmd = ["python", "-m", "mutmut", "run"]
            
            if module:
                cmd.extend(["--paths-to-mutate", f"src/{module}"])
            
            # Add configuration
            cmd.extend([
                "--runner", "python -m pytest tests/ -x -q",
                "--use-coverage",
                "--show-times"
            ])
            
            # Run mutations
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=1800  # 30 minutes max
            )
            
            # Get results
            results_cmd = ["python", "-m", "mutmut", "results"]
            results = subprocess.run(
                results_cmd, capture_output=True, text=True
            )
            
            # Parse results
            mutation_data = self._parse_mutmut_output(results.stdout)
            
            # Export detailed results
            subprocess.run([
                "python", "-m", "mutmut", "export", "json",
                str(self.results_dir / "mutants.json")
            ], capture_output=True)
            
            return mutation_data
            
        except subprocess.TimeoutExpired:
            print("⏰ Mutation testing timed out")
            return {"error": "timeout"}
        except Exception as e:
            print(f"💥 Error running mutations: {e}")
            return {"error": str(e)}
    
    def _parse_mutmut_output(self, output: str) -> Dict:
        """Parse mutmut results output."""
        lines = output.strip().split('\n')
        
        killed = survived = 0
        for line in lines:
            if "killed mutants" in line:
                killed = int(line.split()[0])
            elif "survived mutants" in line:
                survived = int(line.split()[0])
        
        total = killed + survived
        score = (killed / total * 100) if total > 0 else 0
        
        return {
            "killed_mutants": killed,
            "survived_mutants": survived,
            "total_mutants": total,
            "mutation_score": score
        }
    
    def _generate_report(self, mutation_results: Dict, coverage_data: Dict) -> Dict:
        """Generate comprehensive mutation testing report."""
        report = {
            "timestamp": subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], 
                capture_output=True, text=True
            ).stdout.strip(),
            "mutation_results": mutation_results,
            "coverage_summary": {
                "line_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                "branch_coverage": coverage_data.get("totals", {}).get("percent_covered_display", "N/A")
            },
            "quality_metrics": {
                "mutation_score": mutation_results.get("mutation_score", 0),
                "test_effectiveness": self._calculate_test_effectiveness(mutation_results, coverage_data),
                "code_quality_index": self._calculate_quality_index(mutation_results, coverage_data)
            }
        }
        
        # Save report
        report_file = self.results_dir / "mutation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
        return report
    
    def _calculate_test_effectiveness(self, mutation_results: Dict, coverage_data: Dict) -> float:
        """Calculate test effectiveness score (0-100)."""
        mutation_score = mutation_results.get("mutation_score", 0)
        coverage_score = coverage_data.get("totals", {}).get("percent_covered", 0)
        
        # Weighted combination: 70% mutation score, 30% coverage
        return (mutation_score * 0.7 + coverage_score * 0.3)
    
    def _calculate_quality_index(self, mutation_results: Dict, coverage_data: Dict) -> float:
        """Calculate overall code quality index."""
        test_effectiveness = self._calculate_test_effectiveness(mutation_results, coverage_data)
        
        # Penalize for low mutation scores or high survived mutants
        survived_ratio = mutation_results.get("survived_mutants", 0) / max(mutation_results.get("total_mutants", 1), 1)
        penalty = survived_ratio * 10  # Up to 10 point penalty
        
        return max(0, test_effectiveness - penalty)
    
    def _check_quality_gates(self, report: Dict) -> Dict:
        """Check mutation testing quality gates."""
        gates = {
            "mutation_score": {
                "threshold": 80,
                "actual": report["quality_metrics"]["mutation_score"],
                "passed": report["quality_metrics"]["mutation_score"] >= 80
            },
            "test_effectiveness": {
                "threshold": 75,
                "actual": report["quality_metrics"]["test_effectiveness"],
                "passed": report["quality_metrics"]["test_effectiveness"] >= 75
            },
            "survived_mutants": {
                "threshold": 10,  # Max 10 survived mutants
                "actual": report["mutation_results"].get("survived_mutants", 0),
                "passed": report["mutation_results"].get("survived_mutants", 0) <= 10
            }
        }
        
        all_passed = all(gate["passed"] for gate in gates.values())
        
        print("\n🚪 Quality Gate Results:")
        for gate_name, gate_data in gates.items():
            status = "✅ PASS" if gate_data["passed"] else "❌ FAIL"
            print(f"  {gate_name}: {gate_data['actual']:.1f} (threshold: {gate_data['threshold']}) {status}")
        
        return {"passed": all_passed, "gates": gates}


def main():
    """Main entry point for mutation testing."""
    parser = argparse.ArgumentParser(description="Advanced Mutation Testing Runner")
    parser.add_argument("--module", help="Specific module to test")
    parser.add_argument("--config", default="mutation_testing.toml", help="Configuration file")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    
    args = parser.parse_args()
    
    runner = MutationTestRunner(args.config)
    
    if args.report_only:
        print("📊 Generating report from existing results...")
        # Logic to generate report from existing data
        return
    
    result = runner.run_full_suite(args.module)
    
    if result["success"]:
        print(f"\n🎉 Mutation testing completed successfully!")
        print(f"   Mutation Score: {result['mutation_score']:.1f}%")
        print(f"   Killed Mutants: {result['killed_mutants']}")
        print(f"   Survived Mutants: {result['survived_mutants']}")
        sys.exit(0)
    else:
        print(f"\n💥 Mutation testing failed: {result.get('error', 'Quality gates not met')}")
        sys.exit(1)


if __name__ == "__main__":
    main()