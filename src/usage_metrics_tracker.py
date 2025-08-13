"""
Usage Metrics Tracking System v2.0
Advanced metrics collection, analysis, and export for AI fairness systems.

This module provides comprehensive usage tracking with real-time analytics,
bias detection, and multi-format export capabilities.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from enum import Enum
import sqlite3
import pickle
import csv
import uuid

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics collected by the system."""
    PREDICTION = "prediction"
    FAIRNESS = "fairness"
    PERFORMANCE = "performance"
    USER_INTERACTION = "user_interaction"
    SYSTEM = "system"
    BIAS_DETECTION = "bias_detection"

class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "xlsx"
    PDF = "pdf"
    HTML = "html"

@dataclass
class MetricEntry:
    """Individual metric entry with metadata."""
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: MetricType = MetricType.SYSTEM
    name: str = ""
    value: Union[float, int, str, Dict] = 0
    tags: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedMetrics:
    """Aggregated metrics for reporting."""
    total_predictions: int = 0
    fairness_score: float = 0.0
    bias_alerts: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    user_activity: Dict[str, int] = field(default_factory=dict)
    time_period: str = ""
    generated_at: datetime = field(default_factory=datetime.now)

class UsageMetricsTracker:
    """
    Advanced usage metrics tracking system with real-time analytics.
    
    Features:
    - Real-time metric collection and aggregation
    - Bias detection and alerting  
    - Performance monitoring
    - Multi-format export capabilities
    - Adaptive caching and self-optimization
    """
    
    def __init__(self, 
                 storage_path: str = "data/metrics.db",
                 cache_size: int = 10000,
                 auto_export_interval: int = 3600):
        self.storage_path = Path(storage_path)
        self.cache_size = cache_size
        self.auto_export_interval = auto_export_interval
        
        # In-memory storage for real-time access
        self.metrics_cache = deque(maxlen=cache_size)
        self.aggregation_cache = {}
        self.bias_alerts = deque(maxlen=1000)
        
        # Threading for async operations
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.auto_export_timer = None
        
        # Performance tracking
        self.performance_stats = {
            "total_metrics": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "export_count": 0,
            "last_export": None
        }
        
        # Initialize storage
        self._initialize_storage()
        self._start_auto_export()
        
        logger.info(f"📊 Usage Metrics Tracker initialized with cache size: {cache_size}")
    
    def _initialize_storage(self) -> None:
        """Initialize persistent storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    metric_type TEXT,
                    name TEXT,
                    value TEXT,
                    tags TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_type ON metrics(metric_type);
            """)
    
    def track_metric(self, 
                    name: str,
                    value: Union[float, int, str, Dict],
                    metric_type: MetricType = MetricType.SYSTEM,
                    tags: Optional[Dict[str, str]] = None,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Track a single metric entry.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for categorization
            user_id: User identifier
            session_id: Session identifier  
            metadata: Additional metadata
            
        Returns:
            Metric ID for the tracked entry
        """
        
        entry = MetricEntry(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        with self.lock:
            # Add to cache
            self.metrics_cache.append(entry)
            
            # Update performance stats
            self.performance_stats["total_metrics"] += 1
            
            # Async persist to storage
            self.executor.submit(self._persist_metric, entry)
            
            # Check for bias alerts
            if metric_type == MetricType.FAIRNESS:
                self._check_bias_alert(entry)
            
            # Invalidate relevant aggregation cache
            self._invalidate_cache(entry)
        
        logger.debug(f"Tracked metric: {name} = {value}")
        return entry.metric_id
    
    def track_prediction(self,
                        model_name: str,
                        prediction: float,
                        actual: Optional[float] = None,
                        protected_attributes: Optional[Dict[str, Any]] = None,
                        features: Optional[Dict[str, Any]] = None,
                        user_id: Optional[str] = None) -> str:
        """Track a model prediction with fairness monitoring."""
        
        metadata = {
            "model_name": model_name,
            "prediction": prediction,
            "protected_attributes": protected_attributes or {},
            "features": features or {}
        }
        
        if actual is not None:
            metadata["actual"] = actual
            metadata["error"] = abs(prediction - actual)
        
        return self.track_metric(
            name="model_prediction",
            value=prediction,
            metric_type=MetricType.PREDICTION,
            tags={"model": model_name},
            user_id=user_id,
            metadata=metadata
        )
    
    def track_fairness_metric(self,
                             metric_name: str,
                             value: float,
                             protected_group: str,
                             threshold: Optional[float] = None) -> str:
        """Track fairness metrics with bias detection."""
        
        tags = {"protected_group": protected_group}
        metadata = {"threshold": threshold}
        
        # Check if value exceeds bias threshold
        if threshold is not None and value > threshold:
            self._trigger_bias_alert(metric_name, value, protected_group, threshold)
        
        return self.track_metric(
            name=metric_name,
            value=value,
            metric_type=MetricType.FAIRNESS,
            tags=tags,
            metadata=metadata
        )
    
    def get_metrics(self,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   metric_types: Optional[List[MetricType]] = None,
                   limit: Optional[int] = None) -> List[MetricEntry]:
        """Retrieve metrics with filtering."""
        
        # Try cache first for recent metrics
        if start_time is None and end_time is None and limit and limit <= len(self.metrics_cache):
            with self.lock:
                self.performance_stats["cache_hits"] += 1
                recent_metrics = list(self.metrics_cache)[-limit:]
                if metric_types:
                    recent_metrics = [m for m in recent_metrics if m.metric_type in metric_types]
                return recent_metrics
        
        # Query persistent storage
        self.performance_stats["cache_misses"] += 1
        return self._query_storage(start_time, end_time, metric_types, limit)
    
    def get_aggregated_metrics(self,
                              time_period: str = "hour",
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> AggregatedMetrics:
        """Get aggregated metrics for a time period."""
        
        cache_key = f"{time_period}_{start_time}_{end_time}"
        
        # Check aggregation cache
        with self.lock:
            if cache_key in self.aggregation_cache:
                self.performance_stats["cache_hits"] += 1
                return self.aggregation_cache[cache_key]
        
        self.performance_stats["cache_misses"] += 1
        
        # Calculate aggregations
        metrics = self.get_metrics(start_time, end_time)
        aggregated = self._calculate_aggregations(metrics, time_period)
        
        # Cache result
        with self.lock:
            self.aggregation_cache[cache_key] = aggregated
        
        return aggregated
    
    def export_metrics(self,
                      format: ExportFormat,
                      output_path: str,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      include_aggregations: bool = True) -> Path:
        """
        Export metrics in various formats.
        
        Args:
            format: Export format
            output_path: Output file path
            start_time: Start time for filtering
            end_time: End time for filtering
            include_aggregations: Include aggregated metrics
            
        Returns:
            Path to exported file
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get metrics data
        metrics = self.get_metrics(start_time, end_time)
        
        if format == ExportFormat.JSON:
            self._export_json(metrics, output_path, include_aggregations)
        elif format == ExportFormat.CSV:
            self._export_csv(metrics, output_path)
        elif format == ExportFormat.PARQUET:
            self._export_parquet(metrics, output_path)
        elif format == ExportFormat.EXCEL:
            self._export_excel(metrics, output_path, include_aggregations)
        elif format == ExportFormat.PDF:
            self._export_pdf(metrics, output_path, include_aggregations)
        elif format == ExportFormat.HTML:
            self._export_html(metrics, output_path, include_aggregations)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Update performance stats
        self.performance_stats["export_count"] += 1
        self.performance_stats["last_export"] = datetime.now()
        
        logger.info(f"📤 Exported {len(metrics)} metrics to {output_path}")
        return output_path
    
    def get_bias_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent bias alerts."""
        with self.lock:
            return list(self.bias_alerts)[-limit:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get tracker performance statistics."""
        with self.lock:
            stats = self.performance_stats.copy()
            stats["cache_size"] = len(self.metrics_cache)
            stats["cache_utilization"] = len(self.metrics_cache) / self.cache_size
            return stats
    
    def _persist_metric(self, entry: MetricEntry) -> None:
        """Persist metric to storage."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO metrics 
                    (metric_id, timestamp, metric_type, name, value, tags, user_id, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.metric_id,
                    entry.timestamp.isoformat(),
                    entry.metric_type.value,
                    entry.name,
                    json.dumps(entry.value) if isinstance(entry.value, (dict, list)) else str(entry.value),
                    json.dumps(entry.tags),
                    entry.user_id,
                    entry.session_id,
                    json.dumps(entry.metadata)
                ))
        except Exception as e:
            logger.error(f"Failed to persist metric {entry.metric_id}: {e}")
    
    def _query_storage(self,
                      start_time: Optional[datetime],
                      end_time: Optional[datetime], 
                      metric_types: Optional[List[MetricType]],
                      limit: Optional[int]) -> List[MetricEntry]:
        """Query metrics from persistent storage."""
        
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if metric_types:
            query += f" AND metric_type IN ({','.join(['?' for _ in metric_types])})"
            params.extend([mt.value for mt in metric_types])
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_metric_entry(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query storage: {e}")
            return []
    
    def _row_to_metric_entry(self, row) -> MetricEntry:
        """Convert database row to MetricEntry."""
        try:
            value = row[4]
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Keep as string if not JSON
                pass
            
            return MetricEntry(
                metric_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                metric_type=MetricType(row[2]),
                name=row[3],
                value=value,
                tags=json.loads(row[5]) if row[5] else {},
                user_id=row[6],
                session_id=row[7],
                metadata=json.loads(row[8]) if row[8] else {}
            )
        except Exception as e:
            logger.error(f"Failed to parse metric row: {e}")
            # Return minimal entry to avoid breaking
            return MetricEntry(name="parse_error", value=str(e))
    
    def _check_bias_alert(self, entry: MetricEntry) -> None:
        """Check if a fairness metric triggers a bias alert."""
        if entry.metric_type != MetricType.FAIRNESS:
            return
        
        threshold = entry.metadata.get("threshold")
        if threshold and isinstance(entry.value, (int, float)) and entry.value > threshold:
            self._trigger_bias_alert(entry.name, entry.value, 
                                   entry.tags.get("protected_group", "unknown"), threshold)
    
    def _trigger_bias_alert(self, metric_name: str, value: float, 
                           protected_group: str, threshold: float) -> None:
        """Trigger a bias alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": metric_name,
            "value": value,
            "threshold": threshold,
            "protected_group": protected_group,
            "severity": "high" if value > threshold * 1.5 else "medium"
        }
        
        with self.lock:
            self.bias_alerts.append(alert)
        
        logger.warning(f"🚨 Bias alert: {metric_name}={value:.3f} > {threshold:.3f} for group {protected_group}")
    
    def _invalidate_cache(self, entry: MetricEntry) -> None:
        """Invalidate relevant aggregation cache entries."""
        # Simple implementation - clear all cache
        # Could be more sophisticated to only clear relevant entries
        with self.lock:
            self.aggregation_cache.clear()
    
    def _calculate_aggregations(self, metrics: List[MetricEntry], time_period: str) -> AggregatedMetrics:
        """Calculate aggregated metrics."""
        
        # Group metrics by type
        predictions = [m for m in metrics if m.metric_type == MetricType.PREDICTION]
        fairness = [m for m in metrics if m.metric_type == MetricType.FAIRNESS]
        performance = [m for m in metrics if m.metric_type == MetricType.PERFORMANCE]
        
        # Calculate aggregations
        total_predictions = len(predictions)
        
        fairness_score = 0.0
        if fairness:
            fairness_values = [m.value for m in fairness if isinstance(m.value, (int, float))]
            fairness_score = np.mean(fairness_values) if fairness_values else 0.0
        
        bias_alerts = len([a for a in self.get_bias_alerts() 
                          if datetime.fromisoformat(a["timestamp"]) >= 
                          (datetime.now() - timedelta(hours=1))])
        
        performance_metrics = {}
        if performance:
            for metric in performance:
                if isinstance(metric.value, (int, float)):
                    performance_metrics[metric.name] = metric.value
        
        # User activity
        user_activity = defaultdict(int)
        for metric in metrics:
            if metric.user_id:
                user_activity[metric.user_id] += 1
        
        return AggregatedMetrics(
            total_predictions=total_predictions,
            fairness_score=fairness_score,
            bias_alerts=bias_alerts,
            performance_metrics=performance_metrics,
            user_activity=dict(user_activity),
            time_period=time_period
        )
    
    def _export_json(self, metrics: List[MetricEntry], output_path: Path, include_aggregations: bool) -> None:
        """Export metrics to JSON format."""
        data = {
            "metrics": [asdict(m) for m in metrics],
            "export_info": {
                "total_count": len(metrics),
                "exported_at": datetime.now().isoformat(),
                "format": "json"
            }
        }
        
        if include_aggregations:
            data["aggregations"] = asdict(self.get_aggregated_metrics())
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_csv(self, metrics: List[MetricEntry], output_path: Path) -> None:
        """Export metrics to CSV format."""
        if not metrics:
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'metric_id', 'timestamp', 'metric_type', 'name', 'value',
                'tags', 'user_id', 'session_id', 'metadata'
            ])
            
            # Data
            for metric in metrics:
                writer.writerow([
                    metric.metric_id,
                    metric.timestamp.isoformat(),
                    metric.metric_type.value,
                    metric.name,
                    str(metric.value),
                    json.dumps(metric.tags),
                    metric.user_id,
                    metric.session_id,
                    json.dumps(metric.metadata)
                ])
    
    def _export_parquet(self, metrics: List[MetricEntry], output_path: Path) -> None:
        """Export metrics to Parquet format."""
        if not metrics:
            return
            
        # Convert to DataFrame
        data = []
        for metric in metrics:
            data.append({
                'metric_id': metric.metric_id,
                'timestamp': metric.timestamp,
                'metric_type': metric.metric_type.value,
                'name': metric.name,
                'value': str(metric.value),
                'tags': json.dumps(metric.tags),
                'user_id': metric.user_id,
                'session_id': metric.session_id,
                'metadata': json.dumps(metric.metadata)
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
    
    def _export_excel(self, metrics: List[MetricEntry], output_path: Path, include_aggregations: bool) -> None:
        """Export metrics to Excel format."""
        if not metrics:
            return
            
        # Convert to DataFrame
        data = []
        for metric in metrics:
            data.append({
                'Metric ID': metric.metric_id,
                'Timestamp': metric.timestamp,
                'Type': metric.metric_type.value,
                'Name': metric.name,
                'Value': str(metric.value),
                'User ID': metric.user_id,
                'Session ID': metric.session_id
            })
        
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Metrics', index=False)
            
            if include_aggregations:
                agg = self.get_aggregated_metrics()
                agg_data = pd.DataFrame([asdict(agg)])
                agg_data.to_excel(writer, sheet_name='Aggregations', index=False)
    
    def _export_pdf(self, metrics: List[MetricEntry], output_path: Path, include_aggregations: bool) -> None:
        """Export metrics to PDF format (simplified)."""
        # This would require a PDF library like reportlab
        # For now, export as JSON and log
        json_path = output_path.with_suffix('.json')
        self._export_json(metrics, json_path, include_aggregations)
        logger.info(f"PDF export not implemented, exported as JSON to {json_path}")
    
    def _export_html(self, metrics: List[MetricEntry], output_path: Path, include_aggregations: bool) -> None:
        """Export metrics to HTML format."""
        if not metrics:
            return
            
        # Convert to DataFrame for easy HTML export
        data = []
        for metric in metrics:
            data.append({
                'Metric ID': metric.metric_id,
                'Timestamp': metric.timestamp,
                'Type': metric.metric_type.value,
                'Name': metric.name,
                'Value': str(metric.value),
                'User ID': metric.user_id
            })
        
        df = pd.DataFrame(data)
        
        html_content = f"""
        <html>
        <head><title>Usage Metrics Report</title></head>
        <body>
        <h1>Usage Metrics Report</h1>
        <p>Generated at: {datetime.now()}</p>
        <p>Total metrics: {len(metrics)}</p>
        
        <h2>Metrics Data</h2>
        {df.to_html(index=False)}
        
        """
        
        if include_aggregations:
            agg = self.get_aggregated_metrics()
            html_content += f"""
            <h2>Aggregated Metrics</h2>
            <ul>
            <li>Total Predictions: {agg.total_predictions}</li>
            <li>Fairness Score: {agg.fairness_score:.3f}</li>
            <li>Bias Alerts: {agg.bias_alerts}</li>
            </ul>
            """
        
        html_content += "</body></html>"
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _start_auto_export(self) -> None:
        """Start automatic export timer."""
        if self.auto_export_interval > 0:
            self.auto_export_timer = threading.Timer(
                self.auto_export_interval,
                self._auto_export
            )
            self.auto_export_timer.daemon = True
            self.auto_export_timer.start()
    
    def _auto_export(self) -> None:
        """Perform automatic export."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"data/metrics_auto_export_{timestamp}.json"
            self.export_metrics(ExportFormat.JSON, export_path)
            
            # Restart timer
            self._start_auto_export()
            
        except Exception as e:
            logger.error(f"Auto export failed: {e}")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'auto_export_timer') and self.auto_export_timer:
            self.auto_export_timer.cancel()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Global instance for easy access
_global_tracker: Optional[UsageMetricsTracker] = None

def get_tracker() -> UsageMetricsTracker:
    """Get or create the global metrics tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = UsageMetricsTracker()
    return _global_tracker

def track_metric(name: str, value: Union[float, int, str, Dict], **kwargs) -> str:
    """Convenience function to track a metric using the global tracker."""
    return get_tracker().track_metric(name, value, **kwargs)

def track_prediction(model_name: str, prediction: float, **kwargs) -> str:
    """Convenience function to track a prediction using the global tracker.""" 
    return get_tracker().track_prediction(model_name, prediction, **kwargs)

def export_metrics(format: ExportFormat, output_path: str, **kwargs) -> Path:
    """Convenience function to export metrics using the global tracker."""
    return get_tracker().export_metrics(format, output_path, **kwargs)