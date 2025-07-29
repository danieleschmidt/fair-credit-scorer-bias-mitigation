#!/usr/bin/env python3
"""
Automated benchmark runner for performance regression testing.

This script runs comprehensive benchmarks and compares results against baselines
to detect performance regressions in the fair credit scorer system.
"""

import json
import time
import argparse
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from performance_monitoring import PerformanceMonitor, monitor_performance
from baseline_model import train_baseline_model
from fairness_metrics import calculate_fairness_metrics
from data_loader_preprocessor import load_and_preprocess_data


class BenchmarkRunner:
    """Automated benchmark execution and analysis."""
    
    def __init__(self, baseline_file: Optional[str] = None):
        """Initialize benchmark runner."""
        self.baseline_file = baseline_file or "benchmarks/baseline.json"
        self.results: Dict[str, Any] = {}
        self.monitor = PerformanceMonitor()
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 Starting comprehensive benchmark suite...")
        
        benchmarks = {
            "data_loading": self._benchmark_data_loading,
            "model_training": self._benchmark_model_training,
            "fairness_computation": self._benchmark_fairness_metrics,
            "end_to_end": self._benchmark_end_to_end_pipeline
        }
        
        results = {}
        for name, benchmark_func in benchmarks.items():
            print(f"\n📊 Running {name} benchmark...")
            try:
                results[name] = self._run_benchmark_iterations(benchmark_func, name)
                print(f"✅ {name}: {results[name]['avg_time']:.3f}s")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = {"error": str(e), "status": "failed"}
        
        return results
    
    def _run_benchmark_iterations(self, benchmark_func, name: str, iterations: int = 5) -> Dict[str, Any]:
        """Run benchmark function multiple times and collect statistics."""
        times = []
        memory_usage = []
        
        for i in range(iterations):
            with self.monitor.monitor_execution(f"{name}_iteration_{i}"):
                start_time = time.time()
                try:
                    benchmark_func()
                    execution_time = time.time() - start_time
                    times.append(execution_time)
                    
                    # Get memory info from last monitoring entry
                    if self.monitor.metrics_history:
                        memory_usage.append(self.monitor.metrics_history[-1].memory_mb)
                        
                except Exception as e:
                    raise RuntimeError(f"Benchmark iteration {i+1} failed: {e}")
        
        return {
            "status": "success",
            "iterations": iterations,
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time": min(times),
            "max_time": max(times),
            "avg_memory_mb": statistics.mean(memory_usage) if memory_usage else 0,
            "raw_times": times
        }
    
    @monitor_performance("data_loading_benchmark")
    def _benchmark_data_loading(self):
        """Benchmark data loading and preprocessing."""
        data, X_train, X_test, y_train, y_test = load_and_preprocess_data()
        return len(data)
    
    @monitor_performance("model_training_benchmark")
    def _benchmark_model_training(self):
        """Benchmark model training performance."""
        data, X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model = train_baseline_model(X_train, y_train)
        return model
    
    @monitor_performance("fairness_metrics_benchmark")
    def _benchmark_fairness_metrics(self):
        """Benchmark fairness metrics calculation."""
        data, X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model = train_baseline_model(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_fairness_metrics(
            y_test, y_pred, y_pred_proba, X_test['age']
        )
        return metrics
    
    @monitor_performance("end_to_end_benchmark")
    def _benchmark_end_to_end_pipeline(self):
        """Benchmark complete end-to-end pipeline."""
        # Full pipeline simulation
        data, X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model = train_baseline_model(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_fairness_metrics(
            y_test, y_pred, y_pred_proba, X_test['age']
        )
        
        return {
            "model_accuracy": metrics.get("accuracy", 0),
            "fairness_score": metrics.get("demographic_parity_difference", 0)
        }
    
    def compare_with_baseline(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results with saved baseline."""
        baseline = self._load_baseline()
        if not baseline:
            print("⚠️  No baseline found, saving current results as baseline")
            self._save_baseline(current_results)
            return {"status": "baseline_created", "comparison": {}}
        
        comparison = {}
        for benchmark_name, current in current_results.items():
            if benchmark_name not in baseline:
                continue
                
            baseline_data = baseline[benchmark_name]
            if current.get("status") != "success" or baseline_data.get("status") != "success":
                continue
            
            current_time = current["avg_time"]
            baseline_time = baseline_data["avg_time"]
            
            change_percent = ((current_time - baseline_time) / baseline_time) * 100
            
            comparison[benchmark_name] = {
                "current_time": current_time,
                "baseline_time": baseline_time,
                "change_percent": change_percent,
                "status": self._get_performance_status(change_percent)
            }
        
        return {
            "status": "compared",
            "comparison": comparison,
            "overall_status": self._get_overall_status(comparison)
        }
    
    def _get_performance_status(self, change_percent: float) -> str:
        """Determine performance status based on change percentage."""
        if change_percent < -10:
            return "improved"
        elif change_percent > 20:
            return "degraded"
        else:
            return "stable"
    
    def _get_overall_status(self, comparison: Dict[str, Any]) -> str:
        """Determine overall performance status."""
        statuses = [data["status"] for data in comparison.values()]
        
        if "degraded" in statuses:
            return "degraded"
        elif "improved" in statuses:
            return "improved"
        else:
            return "stable"
    
    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline benchmark results."""
        baseline_path = Path(self.baseline_file)
        if not baseline_path.exists():
            return None
        
        try:
            with open(baseline_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Error loading baseline: {e}")
            return None
    
    def _save_baseline(self, results: Dict[str, Any]):
        """Save results as new baseline."""
        baseline_path = Path(self.baseline_file)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(baseline_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Baseline saved to {baseline_path}")
        except IOError as e:
            print(f"❌ Error saving baseline: {e}")
    
    def generate_report(self, results: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("# 📊 Benchmark Report")
        report.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Overall Status**: {comparison.get('overall_status', 'unknown').upper()}")
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("| Benchmark | Current (s) | Baseline (s) | Change | Status |")
        report.append("|-----------|-------------|--------------|--------|--------|")
        
        for name, data in comparison.get("comparison", {}).items():
            current = f"{data['current_time']:.3f}"
            baseline = f"{data['baseline_time']:.3f}"
            change = f"{data['change_percent']:+.1f}%"
            status = data['status'].upper()
            
            status_emoji = {
                "IMPROVED": "🟢",
                "STABLE": "🟡", 
                "DEGRADED": "🔴"
            }.get(status, "⚪")
            
            report.append(f"| {name} | {current} | {baseline} | {change} | {status_emoji} {status} |")
        
        # Detailed results
        report.append("\n## Detailed Results")
        for name, data in results.items():
            if data.get("status") == "success":
                report.append(f"\n### {name.replace('_', ' ').title()}")
                report.append(f"- **Average**: {data['avg_time']:.3f}s")
                report.append(f"- **Median**: {data['median_time']:.3f}s")
                report.append(f"- **Min/Max**: {data['min_time']:.3f}s / {data['max_time']:.3f}s")
                report.append(f"- **Std Dev**: {data['std_dev']:.3f}s")
                if data.get('avg_memory_mb'):
                    report.append(f"- **Avg Memory**: {data['avg_memory_mb']:.1f} MB")
        
        return "\n".join(report)


def main():
    """Main benchmark runner entry point."""
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--baseline", help="Baseline file path")
    parser.add_argument("--save-baseline", action="store_true", 
                       help="Save results as new baseline")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Output format")
    parser.add_argument("--ci", action="store_true", help="CI mode (exit 1 on degradation)")
    
    args = parser.parse_args()
    
    # Run benchmarks
    runner = BenchmarkRunner(args.baseline)
    results = runner.run_all_benchmarks()
    
    # Compare with baseline
    comparison = runner.compare_with_baseline(results)
    
    # Save new baseline if requested
    if args.save_baseline:
        runner._save_baseline(results)
    
    # Generate output
    if args.format == "json":
        output = json.dumps({
            "results": results,
            "comparison": comparison,
            "timestamp": time.time()
        }, indent=2)
    else:
        output = runner.generate_report(results, comparison)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"📄 Report saved to {args.output}")
    else:
        print("\n" + output)
    
    # CI mode: exit with error on degradation
    if args.ci and comparison.get("overall_status") == "degraded":
        print("\n❌ Performance degradation detected in CI mode")
        sys.exit(1)
    
    print(f"\n✅ Benchmarks completed - Status: {comparison.get('overall_status', 'unknown').upper()}")


if __name__ == "__main__":
    main()