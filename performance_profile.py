#!/usr/bin/env python3
"""
Performance profiling automation for continuous optimization.
Part of advanced SDLC enhancement suite.
"""

import cProfile
import pstats
import io
import json
import time
import psutil
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    execution_time: float
    peak_memory_mb: float
    cpu_percent: float
    function_calls: int
    top_functions: List[Dict[str, Any]]
    memory_timeline: List[Dict[str, Any]]
    timestamp: str

class PerformanceProfiler:
    """Advanced performance profiler with automated optimization insights."""
    
    def __init__(self, output_dir: str = "profiling-results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def profile_function(self, func, *args, **kwargs) -> PerformanceMetrics:
        """Profile a single function with comprehensive metrics."""
        # Start memory tracing
        tracemalloc.start()
        
        # Start CPU profiling
        profiler = cProfile.Profile()
        process = psutil.Process()
        
        # Measure execution
        start_time = time.time()
        start_cpu = process.cpu_percent()
        
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        end_time = time.time()
        end_cpu = process.cpu_percent()
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Analyze profiling data
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        top_functions = []
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, func_name = func_info
            top_functions.append({
                'function': f"{filename}:{line}({func_name})",
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'per_call': tt/nc if nc > 0 else 0
            })
        
        top_functions = sorted(top_functions, key=lambda x: x['cumulative_time'], reverse=True)[:10]
        
        return PerformanceMetrics(
            execution_time=end_time - start_time,
            peak_memory_mb=peak / 1024 / 1024,
            cpu_percent=(start_cpu + end_cpu) / 2,
            function_calls=sum(stat[1] for stat in stats.stats.values()),
            top_functions=top_functions,
            memory_timeline=[{'timestamp': time.time(), 'memory_mb': current / 1024 / 1024}],
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def benchmark_module(self, module_name: str, test_functions: List[str]) -> Dict[str, PerformanceMetrics]:
        """Benchmark multiple functions in a module."""
        import importlib
        
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise ImportError(f"Could not import module: {module_name}")
        
        results = {}
        for func_name in test_functions:
            if not hasattr(module, func_name):
                continue
                
            func = getattr(module, func_name)
            if callable(func):
                try:
                    # Create sample data for testing
                    sample_args = self._generate_sample_args(func)
                    metrics = self.profile_function(func, *sample_args)
                    results[func_name] = metrics
                except Exception as e:
                    print(f"Error profiling {func_name}: {e}")
                    
        return results
    
    def _generate_sample_args(self, func) -> tuple:
        """Generate sample arguments for function testing."""
        import inspect
        
        sig = inspect.signature(func)
        args = []
        
        for param in sig.parameters.values():
            if param.annotation == int:
                args.append(1000)
            elif param.annotation == float:
                args.append(100.0)
            elif param.annotation == str:
                args.append("test_data")
            elif param.annotation == list:
                args.append(list(range(100)))
            else:
                # Default test data
                args.append(None)
                
        return tuple(args)
    
    def generate_report(self, results: Dict[str, PerformanceMetrics], output_file: str = "performance_report"):
        """Generate comprehensive performance report."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # JSON report
        json_data = {
            'timestamp': timestamp,
            'results': {name: asdict(metrics) for name, metrics in results.items()},
            'summary': self._generate_summary(results)
        }
        
        json_file = self.output_dir / f"{output_file}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # HTML report
        html_file = self.output_dir / f"{output_file}_{timestamp}.html"
        self._generate_html_report(results, html_file)
        
        # Performance graphs
        self._generate_performance_graphs(results, timestamp)
        
        print(f"Performance report generated: {json_file}")
        print(f"HTML report: {html_file}")
        
    def _generate_summary(self, results: Dict[str, PerformanceMetrics]) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        if not results:
            return {}
            
        times = [m.execution_time for m in results.values()]
        memories = [m.peak_memory_mb for m in results.values()]
        
        return {
            'total_functions': len(results),
            'avg_execution_time': np.mean(times),
            'max_execution_time': np.max(times),
            'avg_memory_usage': np.mean(memories),
            'max_memory_usage': np.max(memories),
            'performance_bottlenecks': [
                name for name, metrics in results.items() 
                if metrics.execution_time > np.mean(times) + 2 * np.std(times)
            ]
        }
    
    def _generate_html_report(self, results: Dict[str, PerformanceMetrics], output_file: Path):
        """Generate HTML performance report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {time.strftime('%Y-%m-%d %H:%M:%S')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .bottleneck {{ background-color: #ffebee; }}
                .good {{ background-color: #e8f5e8; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Performance Report</h1>
            <h2>Summary</h2>
            <table>
                <tr><th>Function</th><th>Execution Time (s)</th><th>Memory (MB)</th><th>Function Calls</th></tr>
        """
        
        for name, metrics in results.items():
            css_class = "bottleneck" if metrics.execution_time > 1.0 else "good"
            html_content += f"""
                <tr class="{css_class}">
                    <td>{name}</td>
                    <td>{metrics.execution_time:.4f}</td>
                    <td>{metrics.peak_memory_mb:.2f}</td>
                    <td>{metrics.function_calls}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_performance_graphs(self, results: Dict[str, PerformanceMetrics], timestamp: str):
        """Generate performance visualization graphs."""
        if not results:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        names = list(results.keys())
        times = [m.execution_time for m in results.values()]
        memories = [m.peak_memory_mb for m in results.values()]
        calls = [m.function_calls for m in results.values()]
        
        # Execution time chart
        ax1.bar(names, times)
        ax1.set_title('Execution Time by Function')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage chart
        ax2.bar(names, memories, color='orange')
        ax2.set_title('Peak Memory Usage by Function')
        ax2.set_ylabel('Memory (MB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Function calls chart
        ax3.bar(names, calls, color='green')
        ax3.set_title('Function Calls by Function')
        ax3.set_ylabel('Number of Calls')
        ax3.tick_params(axis='x', rotation=45)
        
        # Performance correlation
        ax4.scatter(times, memories, alpha=0.7)
        ax4.set_xlabel('Execution Time (seconds)')
        ax4.set_ylabel('Peak Memory (MB)')
        ax4.set_title('Time vs Memory Usage')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"performance_charts_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main profiling automation."""
    profiler = PerformanceProfiler()
    
    # Profile key modules
    modules_to_profile = [
        ('src.fairness_metrics', ['calculate_demographic_parity', 'calculate_equalized_odds']),
        ('src.baseline_model', ['train_model', 'predict']),
        ('src.bias_mitigator', ['apply_reweighting', 'apply_postprocessing']),
        ('src.data_loader_preprocessor', ['load_data', 'preprocess_features'])
    ]
    
    all_results = {}
    
    for module_name, functions in modules_to_profile:
        try:
            results = profiler.benchmark_module(module_name, functions)
            all_results.update(results)
            print(f"Profiled {len(results)} functions from {module_name}")
        except Exception as e:
            print(f"Error profiling {module_name}: {e}")
    
    if all_results:
        profiler.generate_report(all_results)
        
        # Performance optimization suggestions
        print("\n=== Performance Optimization Suggestions ===")
        summary = profiler._generate_summary(all_results)
        
        if summary.get('performance_bottlenecks'):
            print("⚠️  Performance bottlenecks detected:")
            for func in summary['performance_bottlenecks']:
                print(f"   - {func}")
                
        print(f"📊 Average execution time: {summary.get('avg_execution_time', 0):.4f}s")
        print(f"💾 Average memory usage: {summary.get('avg_memory_usage', 0):.2f}MB")
    else:
        print("No profiling results generated.")

if __name__ == "__main__":
    main()