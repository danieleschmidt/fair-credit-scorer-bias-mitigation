"""
Comprehensive Metrics and Reporting System

This module provides detailed metrics collection, analysis, and reporting
for the autonomous backlog management system. It tracks:

- Backlog health metrics (WSJF distribution, aging, completion rates)
- Development velocity (cycle time, throughput, lead time)
- Quality metrics (test coverage, bug rates, security findings)
- Process efficiency (discovery rate, automation success rate)
- Trend analysis and predictive insights

Reports are generated in multiple formats and stored for historical analysis.
"""

import datetime
import json
import logging
import os
import statistics
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yaml

from src.backlog_manager import BacklogItem, TaskStatus, TaskType
from src.security_quality_gates import GateResult, QualityGate

logger = logging.getLogger(__name__)


@dataclass
class VelocityMetrics:
    """Development velocity measurements"""
    cycle_time_avg: float  # Average time from READY to DONE
    cycle_time_p90: float  # 90th percentile cycle time
    throughput_weekly: float  # Items completed per week
    lead_time_avg: float  # Average time from NEW to DONE
    completion_rate: float  # Percentage of started items completed
    blocking_rate: float  # Percentage of items that get blocked


@dataclass
class QualityMetrics:
    """Quality measurements"""
    test_coverage: float
    bug_rate: float  # Bugs per feature implemented
    security_findings_avg: float  # Average security findings per cycle
    defect_escape_rate: float  # Bugs found in production vs caught in dev
    technical_debt_ratio: float  # Ratio of tech debt items to features


@dataclass
class BacklogHealthMetrics:
    """Backlog health indicators"""
    total_items: int
    wsjf_distribution: Dict[str, int]  # Distribution across score ranges
    aging_items_count: int  # Items older than 30 days
    blocked_items_count: int
    ready_items_count: int
    completion_ratio: float  # Completed vs total items
    discovery_rate: float  # New items discovered per cycle


@dataclass
class CycleMetrics:
    """Metrics for a single execution cycle"""
    cycle_id: int
    timestamp: datetime.datetime
    duration_seconds: float
    items_completed: int
    items_discovered: int
    items_blocked: int
    quality_score: float
    security_score: float
    errors_count: int
    backlog_size_start: int
    backlog_size_end: int


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    current_value: float
    trend_direction: str  # "improving", "degrading", "stable"
    trend_strength: float  # 0-1, strength of trend
    prediction_30d: Optional[float]  # Predicted value in 30 days
    confidence: float  # 0-1, confidence in prediction


class MetricsCollector:
    """Collects metrics from various sources"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.metrics_dir = os.path.join(repo_path, "DOCS", "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def collect_velocity_metrics(self, backlog: List[BacklogItem], days: int = 30) -> VelocityMetrics:
        """Collect development velocity metrics"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Filter to items updated in the specified period
        recent_items = [
            item for item in backlog 
            if item.last_updated >= cutoff_date
        ]
        
        completed_items = [
            item for item in recent_items 
            if item.status in [TaskStatus.DONE, TaskStatus.MERGED]
        ]
        
        # Calculate cycle times (READY to DONE)
        cycle_times = []
        lead_times = []
        
        for item in completed_items:
            # Simulate cycle time calculation (in real system, track status transitions)
            cycle_time = max(1, (item.last_updated - item.created_date).total_seconds() / 3600)  # Hours
            cycle_times.append(cycle_time)
            
            # Lead time is full lifecycle
            lead_time = (item.last_updated - item.created_date).total_seconds() / 3600
            lead_times.append(lead_time)
        
        # Calculate metrics
        if cycle_times:
            cycle_time_avg = statistics.mean(cycle_times)
            cycle_time_p90 = statistics.quantiles(cycle_times, n=10)[8] if len(cycle_times) >= 10 else max(cycle_times)
        else:
            cycle_time_avg = cycle_time_p90 = 0.0
        
        if lead_times:
            lead_time_avg = statistics.mean(lead_times)
        else:
            lead_time_avg = 0.0
        
        # Throughput (items per week)
        throughput_weekly = len(completed_items) * (7 / days)
        
        # Completion rate
        in_progress_items = [
            item for item in recent_items 
            if item.status in [TaskStatus.DOING, TaskStatus.PR]
        ]
        
        total_started = len(completed_items) + len(in_progress_items)
        completion_rate = (len(completed_items) / max(total_started, 1)) * 100
        
        # Blocking rate
        blocked_items = [item for item in recent_items if item.is_blocked()]
        blocking_rate = (len(blocked_items) / max(len(recent_items), 1)) * 100
        
        return VelocityMetrics(
            cycle_time_avg=cycle_time_avg,
            cycle_time_p90=cycle_time_p90,
            throughput_weekly=throughput_weekly,
            lead_time_avg=lead_time_avg,
            completion_rate=completion_rate,
            blocking_rate=blocking_rate
        )
    
    def collect_quality_metrics(self, backlog: List[BacklogItem], gate_results: List[GateResult]) -> QualityMetrics:
        """Collect quality metrics from backlog and gate results"""
        
        # Test coverage from gate results
        test_coverage = 0.0
        security_findings_avg = 0.0
        
        for result in gate_results:
            if result.gate_type == QualityGate.TESTING:
                test_coverage = result.score
            elif result.gate_type == QualityGate.SECURITY:
                security_findings_avg = len(result.findings)
        
        # Bug rate calculation
        features = [item for item in backlog if item.task_type == TaskType.FEATURE]
        bugs = [item for item in backlog if item.task_type == TaskType.BUG]
        
        bug_rate = len(bugs) / max(len(features), 1)
        
        # Technical debt ratio
        tech_debt_items = [item for item in backlog if item.task_type == TaskType.TECH_DEBT]
        technical_debt_ratio = len(tech_debt_items) / max(len(backlog), 1)
        
        # Defect escape rate (simplified - would need production data)
        defect_escape_rate = 0.05  # Assume 5% escape rate
        
        return QualityMetrics(
            test_coverage=test_coverage,
            bug_rate=bug_rate,
            security_findings_avg=security_findings_avg,
            defect_escape_rate=defect_escape_rate,
            technical_debt_ratio=technical_debt_ratio
        )
    
    def collect_backlog_health_metrics(self, backlog: List[BacklogItem]) -> BacklogHealthMetrics:
        """Collect backlog health indicators"""
        
        total_items = len(backlog)
        
        # WSJF distribution
        wsjf_ranges = {
            "0-2": 0, "2-5": 0, "5-10": 0, "10-20": 0, "20+": 0
        }
        
        for item in backlog:
            score = item.wsjf_score
            if score <= 2:
                wsjf_ranges["0-2"] += 1
            elif score <= 5:
                wsjf_ranges["2-5"] += 1
            elif score <= 10:
                wsjf_ranges["5-10"] += 1
            elif score <= 20:
                wsjf_ranges["10-20"] += 1
            else:
                wsjf_ranges["20+"] += 1
        
        # Aging items (older than 30 days)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
        aging_items = [item for item in backlog if item.created_date < cutoff_date]
        
        # Status counts
        blocked_items = [item for item in backlog if item.is_blocked()]
        ready_items = [item for item in backlog if item.is_ready()]
        completed_items = [item for item in backlog if item.status == TaskStatus.DONE]
        
        # Completion ratio
        completion_ratio = len(completed_items) / max(total_items, 1)
        
        # Discovery rate (items created in last 7 days)
        week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
        new_items = [item for item in backlog if item.created_date >= week_ago]
        discovery_rate = len(new_items) / 7  # Items per day
        
        return BacklogHealthMetrics(
            total_items=total_items,
            wsjf_distribution=wsjf_ranges,
            aging_items_count=len(aging_items),
            blocked_items_count=len(blocked_items),
            ready_items_count=len(ready_items),
            completion_ratio=completion_ratio,
            discovery_rate=discovery_rate
        )
    
    def record_cycle_metrics(self, cycle_data: Dict) -> CycleMetrics:
        """Record metrics for a completed cycle"""
        cycle_metrics = CycleMetrics(
            cycle_id=cycle_data.get('cycle_id', 0),
            timestamp=datetime.datetime.now(),
            duration_seconds=cycle_data.get('duration', 0),
            items_completed=cycle_data.get('items_completed', 0),
            items_discovered=cycle_data.get('items_discovered', 0),
            items_blocked=cycle_data.get('items_blocked', 0),
            quality_score=cycle_data.get('quality_score', 0),
            security_score=cycle_data.get('security_score', 0),
            errors_count=cycle_data.get('errors_count', 0),
            backlog_size_start=cycle_data.get('backlog_size_start', 0),
            backlog_size_end=cycle_data.get('backlog_size_end', 0)
        )
        
        # Save cycle metrics
        self._save_cycle_metrics(cycle_metrics)
        
        return cycle_metrics
    
    def _save_cycle_metrics(self, metrics: CycleMetrics):
        """Save cycle metrics to file"""
        metrics_file = os.path.join(
            self.metrics_dir, 
            f"cycle_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cycle metrics: {e}")


class TrendAnalyzer:
    """Analyzes trends in metrics over time"""
    
    def __init__(self, metrics_dir: str):
        self.metrics_dir = metrics_dir
    
    def analyze_metric_trend(self, metric_name: str, days: int = 30) -> TrendAnalysis:
        """Analyze trend for a specific metric"""
        
        # Load historical data
        data_points = self._load_metric_history(metric_name, days)
        
        if len(data_points) < 3:
            return TrendAnalysis(
                metric_name=metric_name,
                current_value=data_points[-1][1] if data_points else 0.0,
                trend_direction="stable",
                trend_strength=0.0,
                prediction_30d=None,
                confidence=0.0
            )
        
        # Extract values and timestamps
        timestamps = [point[0] for point in data_points]
        values = [point[1] for point in data_points]
        
        current_value = values[-1]
        
        # Calculate trend
        trend_direction, trend_strength = self._calculate_trend(values)
        
        # Simple linear prediction
        prediction_30d, confidence = self._predict_future_value(timestamps, values, 30)
        
        return TrendAnalysis(
            metric_name=metric_name,
            current_value=current_value,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            prediction_30d=prediction_30d,
            confidence=confidence
        )
    
    def _load_metric_history(self, metric_name: str, days: int) -> List[Tuple[datetime.datetime, float]]:
        """Load historical data for a metric"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        data_points = []
        
        try:
            # Load all cycle metrics files
            for filename in os.listdir(self.metrics_dir):
                if filename.startswith("cycle_") and filename.endswith(".json"):
                    file_path = os.path.join(self.metrics_dir, filename)
                    
                    try:
                        with open(file_path, 'r') as f:
                            cycle_data = json.load(f)
                        
                        # Parse timestamp
                        timestamp = datetime.datetime.fromisoformat(cycle_data['timestamp'])
                        
                        if timestamp >= cutoff_date:
                            # Extract the requested metric
                            value = cycle_data.get(metric_name, 0)
                            if isinstance(value, (int, float)):
                                data_points.append((timestamp, float(value)))
                    
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        
        except OSError:
            pass
        
        # Sort by timestamp
        data_points.sort(key=lambda x: x[0])
        return data_points
    
    def _calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return "stable", 0.0
        
        # Simple linear regression slope
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return "stable", 0.0
        
        slope = numerator / denominator
        
        # Determine trend direction and strength
        if abs(slope) < 0.01:
            return "stable", abs(slope)
        elif slope > 0:
            return "improving", min(abs(slope), 1.0)
        else:
            return "degrading", min(abs(slope), 1.0)
    
    def _predict_future_value(self, timestamps: List[datetime.datetime], 
                             values: List[float], days_ahead: int) -> Tuple[Optional[float], float]:
        """Simple linear prediction of future value"""
        if len(values) < 3:
            return None, 0.0
        
        # Convert timestamps to numeric values (days since first timestamp)
        base_time = timestamps[0]
        x_values = [(ts - base_time).days for ts in timestamps]
        
        # Linear regression
        n = len(values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return values[-1], 0.5  # Return current value with low confidence
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict future value
        future_x = x_values[-1] + days_ahead
        prediction = slope * future_x + intercept
        
        # Calculate confidence based on R-squared
        y_pred = [slope * x + intercept for x in x_values]
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        ss_res = sum((y - y_pred) ** 2 for y, y_pred in zip(values, y_pred))
        
        r_squared = 1 - (ss_res / max(ss_tot, 1e-10))
        confidence = max(0.0, min(1.0, r_squared))
        
        return prediction, confidence


class ReportGenerator:
    """Generates comprehensive reports in multiple formats"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.reports_dir = os.path.join(repo_path, "DOCS", "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.collector = MetricsCollector(repo_path)
        self.analyzer = TrendAnalyzer(self.collector.metrics_dir)
    
    def generate_comprehensive_report(self, backlog: List[BacklogItem], 
                                    gate_results: List[GateResult]) -> Dict:
        """Generate comprehensive metrics report"""
        
        report_timestamp = datetime.datetime.now()
        
        # Collect all metrics
        velocity = self.collector.collect_velocity_metrics(backlog)
        quality = self.collector.collect_quality_metrics(backlog, gate_results)
        health = self.collector.collect_backlog_health_metrics(backlog)
        
        # Analyze trends for key metrics
        trends = {
            'throughput': self.analyzer.analyze_metric_trend('items_completed'),
            'quality_score': self.analyzer.analyze_metric_trend('quality_score'),
            'cycle_time': self.analyzer.analyze_metric_trend('duration_seconds')
        }
        
        # Generate insights and recommendations
        insights = self._generate_insights(velocity, quality, health, trends)
        
        comprehensive_report = {
            'metadata': {
                'generated_at': report_timestamp.isoformat(),
                'report_type': 'comprehensive',
                'backlog_snapshot_size': len(backlog),
                'analysis_period_days': 30
            },
            'velocity_metrics': asdict(velocity),
            'quality_metrics': asdict(quality),
            'backlog_health': asdict(health),
            'trends': {name: asdict(trend) for name, trend in trends.items()},
            'insights': insights,
            'executive_summary': self._generate_executive_summary(velocity, quality, health, trends)
        }
        
        # Save report
        self._save_report(comprehensive_report, 'comprehensive')
        
        # Generate visualizations
        self._generate_charts(comprehensive_report)
        
        return comprehensive_report
    
    def _generate_insights(self, velocity: VelocityMetrics, quality: QualityMetrics,
                          health: BacklogHealthMetrics, trends: Dict[str, TrendAnalysis]) -> List[Dict]:
        """Generate actionable insights from metrics"""
        insights = []
        
        # Velocity insights
        if velocity.throughput_weekly < 1:
            insights.append({
                'type': 'warning',
                'category': 'velocity',
                'message': 'Low throughput detected - completing less than 1 item per week',
                'recommendation': 'Review blocking factors and consider reducing item complexity',
                'priority': 'high'
            })
        
        if velocity.blocking_rate > 20:
            insights.append({
                'type': 'warning',
                'category': 'velocity',
                'message': f'High blocking rate ({velocity.blocking_rate:.1f}%)',
                'recommendation': 'Identify and address common blocking factors',
                'priority': 'medium'
            })
        
        # Quality insights
        if quality.test_coverage < 80:
            insights.append({
                'type': 'warning',
                'category': 'quality',
                'message': f'Test coverage below target ({quality.test_coverage:.1f}%)',
                'recommendation': 'Increase test coverage before implementing new features',
                'priority': 'medium'
            })
        
        if quality.bug_rate > 0.3:
            insights.append({
                'type': 'warning',
                'category': 'quality',
                'message': f'High bug rate ({quality.bug_rate:.2f} bugs per feature)',
                'recommendation': 'Review development process and add more testing',
                'priority': 'high'
            })
        
        # Backlog health insights
        if health.aging_items_count > health.total_items * 0.2:
            insights.append({
                'type': 'warning',
                'category': 'backlog_health',
                'message': f'{health.aging_items_count} items are aging (>30 days)',
                'recommendation': 'Review and prioritize or remove stale items',
                'priority': 'medium'
            })
        
        if health.blocked_items_count > health.total_items * 0.1:
            insights.append({
                'type': 'warning',
                'category': 'backlog_health',
                'message': f'{health.blocked_items_count} items are blocked',
                'recommendation': 'Address blocking factors to improve flow',
                'priority': 'high'
            })
        
        # Trend insights
        for metric_name, trend in trends.items():
            if trend.trend_direction == 'degrading' and trend.trend_strength > 0.5:
                insights.append({
                    'type': 'warning',
                    'category': 'trends',
                    'message': f'{metric_name} is degrading (strength: {trend.trend_strength:.2f})',
                    'recommendation': f'Investigate causes of {metric_name} degradation',
                    'priority': 'medium'
                })
        
        # Positive insights
        if velocity.completion_rate > 90:
            insights.append({
                'type': 'success',
                'category': 'velocity',
                'message': f'Excellent completion rate ({velocity.completion_rate:.1f}%)',
                'recommendation': 'Maintain current practices',
                'priority': 'low'
            })
        
        if quality.test_coverage > 90:
            insights.append({
                'type': 'success',
                'category': 'quality',
                'message': f'Excellent test coverage ({quality.test_coverage:.1f}%)',
                'recommendation': 'Continue maintaining high coverage standards',
                'priority': 'low'
            })
        
        return insights
    
    def _generate_executive_summary(self, velocity: VelocityMetrics, quality: QualityMetrics,
                                   health: BacklogHealthMetrics, trends: Dict[str, TrendAnalysis]) -> Dict:
        """Generate executive summary of key metrics"""
        
        # Calculate overall health score
        velocity_score = min(100, (velocity.throughput_weekly * 25) + (100 - velocity.blocking_rate))
        quality_score = (quality.test_coverage + (100 - quality.bug_rate * 100)) / 2
        health_score = (health.completion_ratio * 100 + (100 - (health.blocked_items_count / max(health.total_items, 1)) * 100)) / 2
        
        overall_score = (velocity_score + quality_score + health_score) / 3
        
        # Determine status
        if overall_score >= 80:
            status = "Excellent"
            status_color = "green"
        elif overall_score >= 65:
            status = "Good"
            status_color = "yellow"
        elif overall_score >= 50:
            status = "Needs Attention"
            status_color = "orange"
        else:
            status = "Critical"
            status_color = "red"
        
        return {
            'overall_score': overall_score,
            'status': status,
            'status_color': status_color,
            'key_metrics': {
                'weekly_throughput': velocity.throughput_weekly,
                'test_coverage': quality.test_coverage,
                'completion_rate': velocity.completion_rate,
                'blocked_items': health.blocked_items_count
            },
            'improvement_areas': [
                trend.metric_name for trend in trends.values()
                if trend.trend_direction == 'degrading'
            ]
        }
    
    def _save_report(self, report: Dict, report_type: str):
        """Save report to file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_report_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report saved to {filepath}")
            
            # Also save as latest
            latest_path = os.path.join(self.reports_dir, f"latest_{report_type}_report.json")
            with open(latest_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def _generate_charts(self, report: Dict):
        """Generate visualization charts"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. WSJF Distribution Chart
            self._create_wsjf_distribution_chart(
                report['backlog_health']['wsjf_distribution'],
                f"wsjf_distribution_{timestamp}.png"
            )
            
            # 2. Quality Metrics Chart
            self._create_quality_metrics_chart(
                report['quality_metrics'],
                f"quality_metrics_{timestamp}.png"
            )
            
            # 3. Trend Analysis Chart
            self._create_trends_chart(
                report['trends'],
                f"trends_analysis_{timestamp}.png"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate charts: {e}")
    
    def _create_wsjf_distribution_chart(self, distribution: Dict[str, int], filename: str):
        """Create WSJF distribution pie chart"""
        plt.figure(figsize=(10, 6))
        
        # Filter out zero values
        non_zero_dist = {k: v for k, v in distribution.items() if v > 0}
        
        if non_zero_dist:
            plt.pie(non_zero_dist.values(), labels=non_zero_dist.keys(), autopct='%1.1f%%')
            plt.title('WSJF Score Distribution')
        else:
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
            plt.title('WSJF Score Distribution - No Data')
        
        plt.savefig(os.path.join(self.reports_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quality_metrics_chart(self, metrics: Dict, filename: str):
        """Create quality metrics bar chart"""
        plt.figure(figsize=(12, 6))
        
        metric_names = ['Test Coverage', 'Bug Rate', 'Security Findings', 'Tech Debt Ratio']
        metric_values = [
            metrics['test_coverage'],
            metrics['bug_rate'] * 100,  # Convert to percentage
            metrics['security_findings_avg'],
            metrics['technical_debt_ratio'] * 100  # Convert to percentage
        ]
        
        colors = ['green', 'red', 'orange', 'blue']
        
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
        plt.title('Quality Metrics Overview')
        plt.ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trends_chart(self, trends: Dict[str, Dict], filename: str):
        """Create trends analysis chart"""
        plt.figure(figsize=(12, 8))
        
        metric_names = list(trends.keys())
        trend_strengths = [trends[name]['trend_strength'] for name in metric_names]
        trend_directions = [trends[name]['trend_direction'] for name in metric_names]
        
        # Color based on trend direction
        colors = []
        for direction in trend_directions:
            if direction == 'improving':
                colors.append('green')
            elif direction == 'degrading':
                colors.append('red')
            else:
                colors.append('gray')
        
        bars = plt.bar(metric_names, trend_strengths, color=colors, alpha=0.7)
        plt.title('Trend Analysis - Strength and Direction')
        plt.ylabel('Trend Strength')
        plt.xlabel('Metrics')
        
        # Add direction labels
        for bar, direction, strength in zip(bars, trend_directions, trend_strengths):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    direction, ha='center', va='bottom', fontsize=8)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_status_dashboard(self, backlog: List[BacklogItem]) -> str:
        """Generate a simple text-based status dashboard"""
        health = self.collector.collect_backlog_health_metrics(backlog)
        
        dashboard = f"""
🎯 AUTONOMOUS BACKLOG STATUS DASHBOARD
{'=' * 50}

📊 BACKLOG OVERVIEW
  Total Items: {health.total_items}
  Ready to Work: {health.ready_items_count}
  Blocked: {health.blocked_items_count}
  Aging (>30d): {health.aging_items_count}

🎲 WSJF DISTRIBUTION
  High Priority (10+): {health.wsjf_distribution.get('10-20', 0) + health.wsjf_distribution.get('20+', 0)}
  Medium Priority (5-10): {health.wsjf_distribution.get('5-10', 0)}
  Low Priority (<5): {health.wsjf_distribution.get('0-2', 0) + health.wsjf_distribution.get('2-5', 0)}

📈 COMPLETION STATUS
  Completion Ratio: {health.completion_ratio:.1%}
  Discovery Rate: {health.discovery_rate:.1f} items/day

⚡ NEXT ACTIONS
  • {'Execute ready items' if health.ready_items_count > 0 else 'Refine blocked items'}
  • {'Address aging backlog' if health.aging_items_count > 5 else 'Maintain current pace'}
  • {'Unblock items' if health.blocked_items_count > 3 else 'Continue execution'}

Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return dashboard