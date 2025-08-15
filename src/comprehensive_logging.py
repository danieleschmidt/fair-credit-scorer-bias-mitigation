"""
Comprehensive Logging System v2.0
Advanced logging with structured formats, multiple outputs, and analytics.

This module provides enterprise-grade logging with correlation IDs,
structured logging, log aggregation, real-time monitoring, and
automated log analysis capabilities.
"""

import json
import logging
import logging.handlers
import os
import queue
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """Extended log levels for fine-grained control."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60

class LogCategory(Enum):
    """Log categories for better organization."""
    SYSTEM = "system"
    APPLICATION = "application"
    SECURITY = "security"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    INTEGRATION = "integration"
    USER_ACTION = "user_action"
    DATA_FLOW = "data_flow"
    MODEL_OPERATIONS = "model_operations"

@dataclass
class LogContext:
    """Context information for structured logging."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: str = ""
    operation: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

@dataclass
class StructuredLogEntry:
    """Structured log entry with comprehensive metadata."""
    timestamp: datetime = field(default_factory=datetime.now)
    level: str = ""
    category: LogCategory = LogCategory.APPLICATION
    message: str = ""
    context: LogContext = field(default_factory=LogContext)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_line: int = 0
    function_name: str = ""
    thread_id: str = ""
    process_id: int = field(default_factory=os.getpid)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "category": self.category.value,
            "message": self.message,
            "context": asdict(self.context),
            "metadata": self.metadata,
            "source": {
                "file": self.source_file,
                "line": self.source_line,
                "function": self.function_name
            },
            "runtime": {
                "thread_id": self.thread_id,
                "process_id": self.process_id
            }
        }

class LogFormatter:
    """Advanced log formatters for different output formats."""

    @staticmethod
    def json_formatter(entry: StructuredLogEntry) -> str:
        """Format log entry as JSON."""
        return json.dumps(entry.to_dict(), default=str, ensure_ascii=False)

    @staticmethod
    def human_readable_formatter(entry: StructuredLogEntry) -> str:
        """Format log entry for human reading."""
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = entry.level.ljust(8)
        category = entry.category.value.ljust(12)

        context_str = ""
        if entry.context.correlation_id:
            context_str = f"[{entry.context.correlation_id[:8]}]"

        if entry.context.user_id:
            context_str += f"[user:{entry.context.user_id}]"

        location = f"{entry.source_file}:{entry.source_line}"

        return (f"{timestamp} {level} {category} {context_str} "
                f"{entry.message} ({location})")

    @staticmethod
    def structured_formatter(entry: StructuredLogEntry) -> str:
        """Format log entry with structured key-value pairs."""
        fields = [
            f"timestamp={entry.timestamp.isoformat()}",
            f"level={entry.level}",
            f"category={entry.category.value}",
            f"message=\"{entry.message}\"",
            f"correlation_id={entry.context.correlation_id[:8]}",
            f"source={entry.source_file}:{entry.source_line}"
        ]

        # Add metadata fields
        for key, value in entry.metadata.items():
            if isinstance(value, str):
                fields.append(f"{key}=\"{value}\"")
            else:
                fields.append(f"{key}={value}")

        return " ".join(fields)

class LogAggregator:
    """Aggregates and analyzes log entries for insights."""

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.log_entries = deque(maxlen=max_entries)
        self.metrics = defaultdict(int)
        self.error_patterns = defaultdict(int)
        self.performance_metrics = deque(maxlen=1000)

        self.lock = threading.RLock()

        # Analysis intervals
        self.last_analysis = time.time()
        self.analysis_interval = 300  # 5 minutes

    def add_entry(self, entry: StructuredLogEntry) -> None:
        """Add a log entry for aggregation."""
        with self.lock:
            self.log_entries.append(entry)

            # Update metrics
            self.metrics[f"level_{entry.level}"] += 1
            self.metrics[f"category_{entry.category.value}"] += 1

            # Track error patterns
            if entry.level in ["ERROR", "CRITICAL"]:
                pattern_key = f"{entry.category.value}:{entry.source_file}"
                self.error_patterns[pattern_key] += 1

            # Performance tracking
            if entry.category == LogCategory.PERFORMANCE:
                if "response_time" in entry.metadata:
                    self.performance_metrics.append(entry.metadata["response_time"])

            # Trigger analysis if needed
            if time.time() - self.last_analysis > self.analysis_interval:
                self._analyze_logs()

    def _analyze_logs(self) -> None:
        """Analyze recent logs for patterns and insights."""
        self.last_analysis = time.time()

        # Calculate error rates
        recent_entries = [e for e in self.log_entries
                         if (datetime.now() - e.timestamp).total_seconds() < 300]

        if recent_entries:
            error_count = sum(1 for e in recent_entries if e.level in ["ERROR", "CRITICAL"])
            error_rate = error_count / len(recent_entries)

            if error_rate > 0.1:  # 10% error rate
                self._trigger_alert("high_error_rate", {
                    "error_rate": error_rate,
                    "error_count": error_count,
                    "total_logs": len(recent_entries)
                })

        # Analyze performance trends
        if self.performance_metrics and len(self.performance_metrics) > 10:
            recent_perf = list(self.performance_metrics)[-50:]
            avg_response_time = statistics.mean(recent_perf)

            if avg_response_time > 1000:  # 1 second
                self._trigger_alert("slow_response_time", {
                    "avg_response_time": avg_response_time,
                    "sample_size": len(recent_perf)
                })

    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger an alert based on log analysis."""
        logger = logging.getLogger(__name__)
        logger.warning(f"🚨 Log analysis alert: {alert_type} - {data}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated log statistics."""
        with self.lock:
            stats = {
                "total_entries": len(self.log_entries),
                "metrics": dict(self.metrics),
                "top_error_patterns": sorted(
                    self.error_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

            if self.performance_metrics:
                perf_data = list(self.performance_metrics)
                stats["performance"] = {
                    "avg_response_time": statistics.mean(perf_data),
                    "min_response_time": min(perf_data),
                    "max_response_time": max(perf_data),
                    "p95_response_time": statistics.quantiles(perf_data, n=20)[18] if len(perf_data) > 20 else 0
                }

            return stats

class AsyncLogHandler:
    """Asynchronous log handler for high-performance logging."""

    def __init__(self, output_handlers: List[logging.Handler], queue_size: int = 10000):
        self.output_handlers = output_handlers
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.running = True

        # Start background thread
        self.worker_thread = threading.Thread(target=self._process_logs, daemon=True)
        self.worker_thread.start()

    def emit(self, entry: StructuredLogEntry) -> None:
        """Emit a log entry asynchronously."""
        try:
            self.log_queue.put_nowait(entry)
        except queue.Full:
            # Drop logs if queue is full to prevent blocking
            pass

    def _process_logs(self) -> None:
        """Process logs in background thread."""
        while self.running:
            try:
                entry = self.log_queue.get(timeout=1)
                if entry is None:  # Shutdown signal
                    break

                # Send to all output handlers
                for handler in self.output_handlers:
                    try:
                        self._send_to_handler(handler, entry)
                    except Exception as e:
                        # Log handler error to stderr to avoid recursion
                        print(f"Log handler error: {e}", file=sys.stderr)

                self.log_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Log processing error: {e}", file=sys.stderr)

    def _send_to_handler(self, handler: logging.Handler, entry: StructuredLogEntry) -> None:
        """Send log entry to a specific handler."""
        # Convert to LogRecord for standard handlers
        record = logging.LogRecord(
            name=entry.context.component or "app",
            level=getattr(logging, entry.level),
            pathname=entry.source_file,
            lineno=entry.source_line,
            msg=entry.message,
            args=(),
            exc_info=None,
            func=entry.function_name
        )

        # Add structured data as extra fields
        record.structured_data = entry.to_dict()

        handler.emit(record)

    def shutdown(self) -> None:
        """Shutdown the async handler."""
        self.running = False
        self.log_queue.put(None)  # Shutdown signal
        self.worker_thread.join(timeout=5)

class ComprehensiveLogger:
    """
    Main comprehensive logging system.
    
    Features:
    - Structured logging with correlation IDs
    - Multiple output formats and destinations
    - Asynchronous processing for performance
    - Log aggregation and analysis
    - Automatic log rotation and compression
    - Real-time monitoring and alerting
    """

    def __init__(self,
                 app_name: str = "fair_credit_scorer",
                 log_level: LogLevel = LogLevel.INFO,
                 log_dir: str = "logs"):

        self.app_name = app_name
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Context tracking
        self.current_context = LogContext()
        self._context_stack = []
        self._context_lock = threading.local()

        # Components
        self.aggregator = LogAggregator()
        self.async_handler = None

        # Setup logging infrastructure
        self._setup_logging()

        # Performance tracking
        self.start_time = time.time()
        self.log_count = 0

        self.info("🏁 Comprehensive logging system initialized",
                 category=LogCategory.SYSTEM,
                 metadata={"app_name": app_name, "log_level": log_level.name})

    def _setup_logging(self) -> None:
        """Setup the logging infrastructure."""

        # Create output handlers
        handlers = []

        # Console handler with human-readable format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._create_custom_formatter("human"))
        handlers.append(console_handler)

        # File handler with JSON format
        json_log_file = self.log_dir / f"{self.app_name}.json"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        json_handler.setFormatter(self._create_custom_formatter("json"))
        handlers.append(json_handler)

        # Error-only file handler
        error_log_file = self.log_dir / f"{self.app_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self._create_custom_formatter("structured"))
        handlers.append(error_handler)

        # Audit log handler
        audit_log_file = self.log_dir / f"{self.app_name}_audit.log"
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=10
        )
        audit_handler.setFormatter(self._create_custom_formatter("json"))
        handlers.append(audit_handler)

        # Setup async handler
        self.async_handler = AsyncLogHandler(handlers)

    def _create_custom_formatter(self, format_type: str) -> logging.Formatter:
        """Create a custom formatter that handles structured data."""

        class StructuredFormatter(logging.Formatter):
            def __init__(self, fmt_type):
                super().__init__()
                self.fmt_type = fmt_type

            def format(self, record):
                # If record has structured data, use it
                if hasattr(record, 'structured_data'):
                    entry_dict = record.structured_data
                    entry = StructuredLogEntry(**{
                        k: v for k, v in entry_dict.items()
                        if k in StructuredLogEntry.__dataclass_fields__
                    })

                    if self.fmt_type == "json":
                        return LogFormatter.json_formatter(entry)
                    elif self.fmt_type == "human":
                        return LogFormatter.human_readable_formatter(entry)
                    else:
                        return LogFormatter.structured_formatter(entry)
                else:
                    # Fallback to standard formatting
                    return super().format(record)

        return StructuredFormatter(format_type)

    def set_context(self, **kwargs) -> None:
        """Set logging context for current thread."""
        if not hasattr(self._context_lock, 'context'):
            self._context_lock.context = LogContext()

        for key, value in kwargs.items():
            if hasattr(self._context_lock.context, key):
                setattr(self._context_lock.context, key, value)

    def push_context(self, **kwargs) -> None:
        """Push a new context frame."""
        if not hasattr(self._context_lock, 'context_stack'):
            self._context_lock.context_stack = []
            self._context_lock.context = LogContext()

        # Save current context
        self._context_lock.context_stack.append(self._context_lock.context)

        # Create new context with inheritance
        new_context = LogContext(**asdict(self._context_lock.context))
        for key, value in kwargs.items():
            if hasattr(new_context, key):
                setattr(new_context, key, value)

        self._context_lock.context = new_context

    def pop_context(self) -> None:
        """Pop the current context frame."""
        if hasattr(self._context_lock, 'context_stack') and self._context_lock.context_stack:
            self._context_lock.context = self._context_lock.context_stack.pop()

    def _get_current_context(self) -> LogContext:
        """Get the current context for this thread."""
        if hasattr(self._context_lock, 'context'):
            return self._context_lock.context
        else:
            # Create default context
            self._context_lock.context = LogContext()
            return self._context_lock.context

    def _log(self,
            level: LogLevel,
            message: str,
            category: LogCategory = LogCategory.APPLICATION,
            metadata: Optional[Dict[str, Any]] = None,
            context_override: Optional[LogContext] = None) -> None:
        """Internal logging method."""

        # Get source information
        frame = sys._getframe(2)  # Skip this method and the public method
        source_file = frame.f_code.co_filename
        source_line = frame.f_lineno
        function_name = frame.f_code.co_name

        # Create log entry
        entry = StructuredLogEntry(
            level=level.name,
            category=category,
            message=message,
            context=context_override or self._get_current_context(),
            metadata=metadata or {},
            source_file=os.path.basename(source_file),
            source_line=source_line,
            function_name=function_name,
            thread_id=str(threading.get_ident())
        )

        # Add to aggregator
        self.aggregator.add_entry(entry)

        # Emit asynchronously
        if self.async_handler:
            self.async_handler.emit(entry)

        # Update stats
        self.log_count += 1

    # Public logging methods
    def trace(self, message: str, **kwargs) -> None:
        """Log trace level message."""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info level message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def success(self, message: str, **kwargs) -> None:
        """Log success level message."""
        self._log(LogLevel.SUCCESS, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error level message."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical level message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def security(self, message: str, **kwargs) -> None:
        """Log security level message."""
        self._log(LogLevel.SECURITY, message, **kwargs)

    def audit(self, action: str, **kwargs) -> None:
        """Log audit message."""
        metadata = kwargs.get("metadata", {})
        metadata["action"] = action
        kwargs["metadata"] = metadata
        kwargs["category"] = LogCategory.AUDIT
        self._log(LogLevel.INFO, f"Audit: {action}", **kwargs)

    def performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        metadata = kwargs.get("metadata", {})
        metadata.update({
            "operation": operation,
            "duration_ms": duration * 1000,
            "response_time": duration * 1000
        })
        kwargs["metadata"] = metadata
        kwargs["category"] = LogCategory.PERFORMANCE

        self._log(LogLevel.INFO, f"Performance: {operation} took {duration:.3f}s", **kwargs)

    def model_operation(self, operation: str, model_name: str, **kwargs) -> None:
        """Log model-specific operations."""
        metadata = kwargs.get("metadata", {})
        metadata.update({
            "operation": operation,
            "model_name": model_name
        })
        kwargs["metadata"] = metadata
        kwargs["category"] = LogCategory.MODEL_OPERATIONS

        self._log(LogLevel.INFO, f"Model {operation}: {model_name}", **kwargs)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        uptime = time.time() - self.start_time

        stats = {
            "uptime_seconds": uptime,
            "total_logs": self.log_count,
            "logs_per_second": self.log_count / uptime if uptime > 0 else 0,
            "aggregator_stats": self.aggregator.get_statistics()
        }

        return stats

    def shutdown(self) -> None:
        """Shutdown the logging system."""
        self.info("🔚 Shutting down comprehensive logging system",
                 category=LogCategory.SYSTEM)

        if self.async_handler:
            self.async_handler.shutdown()

# Context manager for temporary logging context
class LoggingContext:
    """Context manager for temporary logging context."""

    def __init__(self, logger: ComprehensiveLogger, **kwargs):
        self.logger = logger
        self.context_data = kwargs

    def __enter__(self):
        self.logger.push_context(**self.context_data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.pop_context()

# Performance timing decorator
def log_performance(operation: str, category: LogCategory = LogCategory.PERFORMANCE):
    """Decorator to automatically log function performance."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.performance(
                    operation or func.__name__,
                    duration,
                    metadata={
                        "function": func.__name__,
                        "module": func.__module__,
                        "success": True
                    }
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                logger.performance(
                    operation or func.__name__,
                    duration,
                    metadata={
                        "function": func.__name__,
                        "module": func.__module__,
                        "success": False,
                        "error": str(e)
                    }
                )
                raise

        return wrapper
    return decorator

# Global logger instance
_global_logger: Optional[ComprehensiveLogger] = None

def get_logger() -> ComprehensiveLogger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ComprehensiveLogger()
    return _global_logger

def set_logging_context(**kwargs) -> None:
    """Set logging context globally."""
    get_logger().set_context(**kwargs)

def logging_context(**kwargs) -> LoggingContext:
    """Create a logging context manager."""
    return LoggingContext(get_logger(), **kwargs)
