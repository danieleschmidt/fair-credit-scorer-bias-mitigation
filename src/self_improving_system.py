"""
Self-Improving System Patterns v2.0
Implements adaptive and evolving system behaviors for autonomous optimization.

This module provides self-improving patterns including adaptive caching,
auto-scaling triggers, self-healing mechanisms, and performance optimization
based on real-time metrics and usage patterns.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np
import psutil

logger = logging.getLogger(__name__)

class AdaptationTrigger(Enum):
    """Triggers for adaptive system behavior."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_PRESSURE = "resource_pressure"
    LOAD_INCREASE = "load_increase"
    ERROR_RATE_HIGH = "error_rate_high"
    BIAS_DETECTED = "bias_detected"
    PATTERN_CHANGE = "pattern_change"

class OptimizationAction(Enum):
    """Available optimization actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CACHE_ADJUST = "cache_adjust"
    CIRCUIT_BREAK = "circuit_break"
    LOAD_BALANCE = "load_balance"
    MEMORY_OPTIMIZE = "memory_optimize"
    MODEL_RETRAIN = "model_retrain"

@dataclass
class SystemMetrics:
    """Current system performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    cache_hit_rate: float = 0.0
    active_connections: int = 0

@dataclass
class AdaptiveRule:
    """Rule for adaptive system behavior."""
    name: str
    trigger: AdaptationTrigger
    condition: Callable[[SystemMetrics], bool]
    action: OptimizationAction
    priority: int = 1
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None

class AdaptiveCache:
    """
    Self-adapting cache with usage pattern learning.
    
    Features:
    - Access pattern analysis
    - Dynamic size adjustment
    - TTL optimization based on usage
    - Predictive preloading
    """

    def __init__(self, initial_size: int = 1000, max_size: int = 10000):
        self.cache = {}
        self.access_patterns = defaultdict(list)
        self.hit_rates = deque(maxlen=100)
        self.size_history = deque(maxlen=50)

        self.current_size = initial_size
        self.max_size = max_size
        self.min_size = max(100, initial_size // 10)

        self.lock = threading.RLock()
        self.total_requests = 0
        self.total_hits = 0

        # Adaptation parameters
        self.target_hit_rate = 0.8
        self.adaptation_interval = 60  # seconds
        self.last_adaptation = time.time()

        logger.info(f"🧠 Adaptive cache initialized with size {initial_size}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with pattern tracking."""
        with self.lock:
            self.total_requests += 1

            # Track access pattern
            self.access_patterns[key].append(time.time())

            if key in self.cache:
                self.total_hits += 1
                value, timestamp, ttl = self.cache[key]

                # Check TTL
                if time.time() - timestamp < ttl:
                    return value
                else:
                    del self.cache[key]

            # Trigger adaptation check
            if time.time() - self.last_adaptation > self.adaptation_interval:
                self._adapt_cache()

            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache with adaptive TTL."""
        with self.lock:
            # Calculate adaptive TTL
            if ttl is None:
                ttl = self._calculate_adaptive_ttl(key)

            # Check if we need to evict
            if len(self.cache) >= self.current_size:
                self._evict_items()

            self.cache[key] = (value, time.time(), ttl)

    def _calculate_adaptive_ttl(self, key: str) -> float:
        """Calculate TTL based on access patterns."""
        accesses = self.access_patterns.get(key, [])

        if len(accesses) < 2:
            return 300  # Default 5 minutes

        # Calculate access frequency
        recent_accesses = [a for a in accesses if time.time() - a < 3600]  # Last hour

        if len(recent_accesses) < 2:
            return 600  # 10 minutes for infrequent access

        # High frequency → longer TTL
        access_interval = np.mean(np.diff(sorted(recent_accesses)))

        if access_interval < 60:  # Very frequent
            return 1800  # 30 minutes
        elif access_interval < 300:  # Frequent
            return 900   # 15 minutes
        else:  # Infrequent
            return 300   # 5 minutes

    def _evict_items(self) -> None:
        """Evict least recently used items."""
        if not self.cache:
            return

        # Sort by timestamp (LRU)
        items = [(k, v[1]) for k, v in self.cache.items()]
        items.sort(key=lambda x: x[1])

        # Remove oldest 10% of items
        remove_count = max(1, len(items) // 10)
        for i in range(remove_count):
            key = items[i][0]
            del self.cache[key]

    def _adapt_cache(self) -> None:
        """Adapt cache size based on performance."""
        current_hit_rate = self.total_hits / max(1, self.total_requests)
        self.hit_rates.append(current_hit_rate)

        # Calculate recent trend
        if len(self.hit_rates) >= 10:
            recent_rate = np.mean(list(self.hit_rates)[-10:])

            if recent_rate < self.target_hit_rate * 0.9:
                # Poor hit rate - increase cache size
                new_size = min(self.max_size, int(self.current_size * 1.2))
                if new_size != self.current_size:
                    self.current_size = new_size
                    logger.info(f"🔄 Cache size increased to {new_size} (hit rate: {recent_rate:.3f})")

            elif recent_rate > self.target_hit_rate * 1.1 and self.current_size > self.min_size:
                # Excellent hit rate - can reduce size
                new_size = max(self.min_size, int(self.current_size * 0.9))
                if new_size != self.current_size:
                    self.current_size = new_size
                    logger.info(f"🔄 Cache size reduced to {new_size} (hit rate: {recent_rate:.3f})")

        self.size_history.append(self.current_size)
        self.last_adaptation = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            hit_rate = self.total_hits / max(1, self.total_requests)
            return {
                "current_size": self.current_size,
                "items_count": len(self.cache),
                "hit_rate": hit_rate,
                "total_requests": self.total_requests,
                "total_hits": self.total_hits,
                "avg_size": np.mean(self.size_history) if self.size_history else self.current_size
            }

class CircuitBreaker:
    """Circuit breaker for self-healing system behavior."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        self.lock = threading.RLock()

        logger.info(f"🔌 Circuit breaker initialized (threshold: {failure_threshold})")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception:
                self._on_failure()
                raise

    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("🔌 Circuit breaker reset to CLOSED")
        elif self.state == "CLOSED":
            self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"🔌 Circuit breaker opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time and
                time.time() - self.last_failure_time >= self.recovery_timeout)

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self.lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time
            }

class AutoScaler:
    """Auto-scaling based on system metrics and load patterns."""

    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances

        self.metrics_history = deque(maxlen=50)
        self.scaling_history = deque(maxlen=20)

        self.lock = threading.RLock()
        self.last_scaling = time.time()
        self.scaling_cooldown = 300  # 5 minutes

        logger.info(f"📈 Auto-scaler initialized ({min_instances}-{max_instances} instances)")

    def evaluate_scaling(self, metrics: SystemMetrics) -> Optional[OptimizationAction]:
        """Evaluate if scaling is needed based on metrics."""
        with self.lock:
            self.metrics_history.append(metrics)

            # Need enough history for decision
            if len(self.metrics_history) < 5:
                return None

            # Check cooldown
            if time.time() - self.last_scaling < self.scaling_cooldown:
                return None

            # Calculate recent averages
            recent_metrics = list(self.metrics_history)[-5:]
            avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
            avg_memory = np.mean([m.memory_usage for m in recent_metrics])
            avg_response_time = np.mean([m.response_time for m in recent_metrics])

            # Scale up conditions
            if (avg_cpu > 80 or avg_memory > 85 or avg_response_time > 1000) and \
               self.current_instances < self.max_instances:
                return self._scale_up()

            # Scale down conditions
            elif (avg_cpu < 30 and avg_memory < 40 and avg_response_time < 200) and \
                 self.current_instances > self.min_instances:
                return self._scale_down()

            return None

    def _scale_up(self) -> OptimizationAction:
        """Scale up instances."""
        self.current_instances = min(self.max_instances, self.current_instances + 1)
        self.last_scaling = time.time()
        self.scaling_history.append(("scale_up", time.time(), self.current_instances))

        logger.info(f"📈 Scaled up to {self.current_instances} instances")
        return OptimizationAction.SCALE_UP

    def _scale_down(self) -> OptimizationAction:
        """Scale down instances."""
        self.current_instances = max(self.min_instances, self.current_instances - 1)
        self.last_scaling = time.time()
        self.scaling_history.append(("scale_down", time.time(), self.current_instances))

        logger.info(f"📉 Scaled down to {self.current_instances} instances")
        return OptimizationAction.SCALE_DOWN

    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        with self.lock:
            return {
                "current_instances": self.current_instances,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "scaling_events": len(self.scaling_history),
                "last_scaling": self.last_scaling
            }

class PerformanceOptimizer:
    """Performance optimizer based on metrics feedback."""

    def __init__(self):
        self.optimization_history = deque(maxlen=100)
        self.performance_baseline = None
        self.optimization_params = {
            "batch_size": 32,
            "worker_count": 4,
            "cache_size": 1000,
            "timeout": 30
        }

        self.lock = threading.RLock()

        logger.info("⚡ Performance optimizer initialized")

    def optimize_parameters(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Optimize system parameters based on current metrics."""
        with self.lock:
            if self.performance_baseline is None:
                self.performance_baseline = self._calculate_performance_score(metrics)
                return {}

            current_score = self._calculate_performance_score(metrics)

            # If performance degraded, try optimization
            if current_score < self.performance_baseline * 0.9:
                return self._suggest_optimizations(metrics)

            # If performance improved significantly, update baseline
            elif current_score > self.performance_baseline * 1.1:
                self.performance_baseline = current_score
                logger.info(f"⚡ Performance baseline updated: {current_score:.3f}")

            return {}

    def _calculate_performance_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall performance score."""
        # Weighted performance score
        score = (
            (100 - metrics.cpu_usage) * 0.3 +
            (100 - metrics.memory_usage) * 0.2 +
            max(0, 1000 - metrics.response_time) / 10 * 0.3 +
            (100 - metrics.error_rate * 100) * 0.2
        )
        return max(0, score)

    def _suggest_optimizations(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Suggest parameter optimizations."""
        suggestions = {}

        # High CPU usage → reduce worker count or batch size
        if metrics.cpu_usage > 80:
            if self.optimization_params["worker_count"] > 2:
                suggestions["worker_count"] = self.optimization_params["worker_count"] - 1
            elif self.optimization_params["batch_size"] > 16:
                suggestions["batch_size"] = self.optimization_params["batch_size"] // 2

        # High memory usage → reduce cache size or batch size
        elif metrics.memory_usage > 85:
            if self.optimization_params["cache_size"] > 500:
                suggestions["cache_size"] = int(self.optimization_params["cache_size"] * 0.8)
            elif self.optimization_params["batch_size"] > 16:
                suggestions["batch_size"] = self.optimization_params["batch_size"] // 2

        # High response time → increase timeout or reduce load
        elif metrics.response_time > 1000:
            suggestions["timeout"] = min(60, self.optimization_params["timeout"] + 10)

        # Low resource usage → can increase performance
        elif metrics.cpu_usage < 30 and metrics.memory_usage < 40:
            if self.optimization_params["worker_count"] < 8:
                suggestions["worker_count"] = self.optimization_params["worker_count"] + 1
            elif self.optimization_params["batch_size"] < 128:
                suggestions["batch_size"] = self.optimization_params["batch_size"] * 2

        # Apply suggestions
        if suggestions:
            self.optimization_params.update(suggestions)
            self.optimization_history.append((time.time(), suggestions))
            logger.info(f"⚡ Applied optimizations: {suggestions}")

        return suggestions

class SelfImprovingSystem:
    """
    Main orchestrator for self-improving system patterns.
    
    Coordinates adaptive caching, auto-scaling, circuit breaking,
    and performance optimization based on real-time metrics.
    """

    def __init__(self):
        self.adaptive_cache = AdaptiveCache()
        self.circuit_breaker = CircuitBreaker()
        self.auto_scaler = AutoScaler()
        self.performance_optimizer = PerformanceOptimizer()

        self.adaptive_rules = []
        self.system_metrics = deque(maxlen=1000)
        self.improvement_history = deque(maxlen=500)

        self.running = False
        self.monitor_thread = None
        self.lock = threading.RLock()

        # Initialize default adaptive rules
        self._setup_default_rules()

        logger.info("🧬 Self-improving system initialized")

    def start_monitoring(self) -> None:
        """Start continuous system monitoring and adaptation."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("🧬 Started self-improving system monitoring")

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🧬 Stopped self-improving system monitoring")

    def add_adaptive_rule(self, rule: AdaptiveRule) -> None:
        """Add a custom adaptive rule."""
        with self.lock:
            self.adaptive_rules.append(rule)
            logger.info(f"🧬 Added adaptive rule: {rule.name}")

    def get_cache(self) -> AdaptiveCache:
        """Get the adaptive cache instance."""
        return self.adaptive_cache

    def execute_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        return self.circuit_breaker.call(func, *args, **kwargs)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self.lock:
            recent_metrics = list(self.system_metrics)[-10:] if self.system_metrics else []

            stats = {
                "cache_stats": self.adaptive_cache.get_stats(),
                "circuit_breaker": self.circuit_breaker.get_state(),
                "auto_scaler": self.auto_scaler.get_stats(),
                "improvement_events": len(self.improvement_history),
                "active_rules": len(self.adaptive_rules),
                "monitoring": self.running
            }

            if recent_metrics:
                stats["avg_cpu"] = np.mean([m.cpu_usage for m in recent_metrics])
                stats["avg_memory"] = np.mean([m.memory_usage for m in recent_metrics])
                stats["avg_response_time"] = np.mean([m.response_time for m in recent_metrics])

            return stats

    def _setup_default_rules(self) -> None:
        """Setup default adaptive rules."""

        # High error rate rule
        self.add_adaptive_rule(AdaptiveRule(
            name="high_error_rate",
            trigger=AdaptationTrigger.ERROR_RATE_HIGH,
            condition=lambda m: m.error_rate > 0.05,  # 5% error rate
            action=OptimizationAction.CIRCUIT_BREAK,
            priority=1,
            cooldown=300
        ))

        # High CPU rule
        self.add_adaptive_rule(AdaptiveRule(
            name="high_cpu_usage",
            trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
            condition=lambda m: m.cpu_usage > 80,
            action=OptimizationAction.SCALE_UP,
            priority=2,
            cooldown=300
        ))

        # Low cache hit rate rule
        self.add_adaptive_rule(AdaptiveRule(
            name="low_cache_hit_rate",
            trigger=AdaptationTrigger.PERFORMANCE_DEGRADATION,
            condition=lambda m: m.cache_hit_rate < 0.6,
            action=OptimizationAction.CACHE_ADJUST,
            priority=3,
            cooldown=180
        ))

    def _monitoring_loop(self) -> None:
        """Main monitoring loop for continuous adaptation."""
        while self.running:
            try:
                # Collect current system metrics
                metrics = self._collect_system_metrics()

                with self.lock:
                    self.system_metrics.append(metrics)

                # Evaluate adaptive rules
                self._evaluate_adaptive_rules(metrics)

                # Auto-scaling evaluation
                scaling_action = self.auto_scaler.evaluate_scaling(metrics)
                if scaling_action:
                    self._record_improvement("auto_scaling", scaling_action.value)

                # Performance optimization
                optimizations = self.performance_optimizer.optimize_parameters(metrics)
                if optimizations:
                    self._record_improvement("performance_optimization", optimizations)

                # Sleep before next iteration
                time.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Get cache hit rate
            cache_stats = self.adaptive_cache.get_stats()
            cache_hit_rate = cache_stats.get("hit_rate", 0.0)

            return SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                response_time=50.0,  # Would be measured from actual requests
                error_rate=0.01,     # Would be measured from actual error tracking
                throughput=100.0,    # Would be measured from actual request counting
                cache_hit_rate=cache_hit_rate,
                active_connections=10  # Would be measured from actual connection tracking
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()  # Return default metrics

    def _evaluate_adaptive_rules(self, metrics: SystemMetrics) -> None:
        """Evaluate all adaptive rules against current metrics."""
        now = datetime.now()

        for rule in self.adaptive_rules:
            # Check cooldown
            if rule.last_triggered and \
               (now - rule.last_triggered).total_seconds() < rule.cooldown:
                continue

            # Evaluate condition
            try:
                if rule.condition(metrics):
                    self._trigger_adaptive_action(rule, metrics)
                    rule.last_triggered = now
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

    def _trigger_adaptive_action(self, rule: AdaptiveRule, metrics: SystemMetrics) -> None:
        """Trigger an adaptive action based on rule."""
        logger.info(f"🧬 Triggering adaptive action: {rule.action.value} (rule: {rule.name})")

        try:
            if rule.action == OptimizationAction.CIRCUIT_BREAK:
                # Circuit breaker is already active, just log
                pass
            elif rule.action == OptimizationAction.SCALE_UP:
                self.auto_scaler._scale_up()
            elif rule.action == OptimizationAction.CACHE_ADJUST:
                # Force cache adaptation
                self.adaptive_cache._adapt_cache()

            self._record_improvement(rule.name, rule.action.value)

        except Exception as e:
            logger.error(f"Failed to trigger action {rule.action.value}: {e}")

    def _record_improvement(self, source: str, action: str) -> None:
        """Record a system improvement event."""
        improvement = {
            "timestamp": time.time(),
            "source": source,
            "action": action
        }

        with self.lock:
            self.improvement_history.append(improvement)

        logger.info(f"🧬 Recorded improvement: {source} → {action}")

# Global instance for easy access
_global_system: Optional[SelfImprovingSystem] = None

def get_self_improving_system() -> SelfImprovingSystem:
    """Get or create the global self-improving system instance."""
    global _global_system
    if _global_system is None:
        _global_system = SelfImprovingSystem()
        _global_system.start_monitoring()
    return _global_system

def get_adaptive_cache() -> AdaptiveCache:
    """Get the adaptive cache from the global system."""
    return get_self_improving_system().get_cache()

def execute_with_protection(func: Callable, *args, **kwargs) -> Any:
    """Execute function with circuit breaker protection."""
    return get_self_improving_system().execute_with_circuit_breaker(func, *args, **kwargs)
