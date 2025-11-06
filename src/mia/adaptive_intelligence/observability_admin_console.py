"""
Observability & Admin Console
"""

import logging
import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import statistics

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

import platform

from ..providers import provider_registry

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for observability."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Represents a log entry."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class Metric:
    """Represents a metric."""
    name: str
    type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    condition: str
    value: Union[int, float]
    threshold: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """System health information."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_connections: int
    active_threads: int
    uptime: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComponentStatus:
    """Status of a system component."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time: float
    last_check: datetime
    error_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and manages metrics from various sources.
    """

    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, Union[int, float]] = {}
        self.histograms: Dict[str, List[Union[int, float]]] = {}
        self.lock = threading.Lock()

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self.lock:
            key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            self.counters[key] = self.counters.get(key, 0) + value

            metric = Metric(
                name=name,
                type=MetricType.COUNTER,
                value=self.counters[key],
                labels=labels or {}
            )
            self._store_metric(metric)

    def set_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self.lock:
            key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            self.gauges[key] = value

            metric = Metric(
                name=name,
                type=MetricType.GAUGE,
                value=value,
                labels=labels or {}
            )
            self._store_metric(metric)

    def record_histogram(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self.lock:
            key = f"{name}_{json.dumps(labels or {}, sort_keys=True)}"
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)

            # Keep only last 1000 values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

            metric = Metric(
                name=name,
                type=MetricType.HISTOGRAM,
                value=value,
                labels=labels or {}
            )
            self._store_metric(metric)

    def _store_metric(self, metric: Metric):
        """Store a metric in the metrics history."""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
        self.metrics[metric.name].append(metric)

        # Keep only last 10000 metrics per name
        if len(self.metrics[metric.name]) > 10000:
            self.metrics[metric.name] = self.metrics[metric.name][-10000:]

    def get_metric_stats(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a metric over the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self.lock:
            if name not in self.metrics:
                return {}

            recent_metrics = [m for m in self.metrics[name] if m.timestamp > cutoff]
            if not recent_metrics:
                return {}

            values = [m.value for m in recent_metrics]

            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                "latest": values[-1],
                "latest_timestamp": recent_metrics[-1].timestamp.isoformat()
            }

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all current metric values."""
        result = []

        with self.lock:
            # Counters
            for key, value in self.counters.items():
                name = key.split('_', 1)[0]
                labels_str = key.split('_', 1)[1] if '_' in key else '{}'
                try:
                    labels = json.loads(labels_str)
                except:
                    labels = {}

                result.append({
                    "name": name,
                    "type": "counter",
                    "value": value,
                    "labels": labels
                })

            # Gauges
            for key, value in self.gauges.items():
                name = key.split('_', 1)[0]
                labels_str = key.split('_', 1)[1] if '_' in key else '{}'
                try:
                    labels = json.loads(labels_str)
                except:
                    labels = {}

                result.append({
                    "name": name,
                    "type": "gauge",
                    "value": value,
                    "labels": labels
                })

            # Histograms
            for key, values in self.histograms.items():
                name = key.split('_', 1)[0]
                labels_str = key.split('_', 1)[1] if '_' in key else '{}'
                try:
                    labels = json.loads(labels_str)
                except:
                    labels = {}

                if values:
                    result.append({
                        "name": name,
                        "type": "histogram",
                        "value": values[-1],
                        "count": len(values),
                        "labels": labels
                    })

        return result


class AlertManager:
    """
    Manages alerts and notifications.
    """

    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_callbacks: List[Callable] = []
        self.lock = threading.Lock()

    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules.append(rule)

    def check_alerts(self, metrics_collector: MetricsCollector):
        """Check all alert rules and generate alerts if needed."""
        with self.lock:
            for rule in self.alert_rules:
                self._check_rule(rule, metrics_collector)

    def _check_rule(self, rule: Dict[str, Any], metrics_collector: MetricsCollector):
        """Check a single alert rule."""
        metric_name = rule.get("metric")
        condition = rule.get("condition")  # "gt", "lt", "eq"
        threshold = rule.get("threshold")
        severity = rule.get("severity", "medium")
        name = rule.get("name", f"Alert on {metric_name}")

        if not metric_name or threshold is None:
            return

        # Get current metric value
        stats = metrics_collector.get_metric_stats(metric_name, hours=1)
        if not stats:
            return

        current_value = stats["latest"]

        # Check condition
        triggered = False
        if condition == "gt" and current_value > threshold:
            triggered = True
        elif condition == "lt" and current_value < threshold:
            triggered = True
        elif condition == "eq" and current_value == threshold:
            triggered = True

        if triggered:
            # Check if alert already exists and is unresolved
            alert_key = f"{name}_{metric_name}"
            if alert_key not in self.alerts or self.alerts[alert_key].resolved:
                alert = Alert(
                    id=alert_key,
                    name=name,
                    severity=AlertSeverity(severity),
                    message=f"{metric_name} {condition} {threshold} (current: {current_value})",
                    condition=f"{metric_name} {condition} {threshold}",
                    value=current_value,
                    threshold=threshold
                )

                self.alerts[alert_key] = alert

                # Notify
                self._notify_alert(alert)

    def _notify_alert(self, alert: Alert):
        """Notify about an alert."""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as exc:
                logger.error(f"Alert notification failed: {exc}")

    def add_notification_callback(self, callback: Callable):
        """Add a notification callback."""
        with self.lock:
            self.notification_callbacks.append(callback)

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.now()
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]

    def get_all_alerts(self, hours: int = 24) -> List[Alert]:
        """Get all alerts from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.timestamp > cutoff]


class LogAggregator:
    """
    Aggregates and manages logs from various components.
    """

    def __init__(self, max_entries: int = 10000):
        self.logs: List[LogEntry] = []
        self.max_entries = max_entries
        self.lock = threading.Lock()

    def add_log(self, entry: LogEntry):
        """Add a log entry."""
        with self.lock:
            self.logs.append(entry)

            # Maintain max entries
            if len(self.logs) > self.max_entries:
                self.logs = self.logs[-self.max_entries:]

    def get_logs(
        self,
        component: Optional[str] = None,
        level: Optional[LogLevel] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[LogEntry]:
        """Get filtered logs."""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self.lock:
            filtered_logs = [
                log for log in self.logs
                if log.timestamp > cutoff
                and (component is None or log.component == component)
                and (level is None or log.level == level)
            ]

            return filtered_logs[-limit:]

    def get_log_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get log statistics."""
        logs = self.get_logs(hours=hours, limit=self.max_entries)

        stats = {
            "total_logs": len(logs),
            "by_level": {},
            "by_component": {},
            "time_range": {
                "start": logs[0].timestamp.isoformat() if logs else None,
                "end": logs[-1].timestamp.isoformat() if logs else None
            }
        }

        for log in logs:
            level_str = log.level.value
            component_str = log.component

            stats["by_level"][level_str] = stats["by_level"].get(level_str, 0) + 1
            stats["by_component"][component_str] = stats["by_component"].get(component_str, 0) + 1

        return stats


class ObservabilityAdminConsole:
    """
    Main observability and admin console for the Adaptive Intelligence system.
    """

    def __init__(
        self,
        config_manager=None,
        *,
        metrics_interval: int = 60,
        health_check_interval: int = 30,
        log_retention_hours: int = 168  # 7 days
    ):
        self.config_manager = config_manager
        self.metrics_interval = metrics_interval
        self.health_check_interval = health_check_interval
        self.log_retention_hours = log_retention_hours

        # Core components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.log_aggregator = LogAggregator()

        # System monitoring
        self.system_health_history: List[SystemHealth] = []
        self.component_statuses: Dict[str, ComponentStatus] = {}

        # Background tasks
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False

        # Admin interface
        self.admin_commands: Dict[str, Callable] = {}
        self._register_admin_commands()

        logger.info("Observability & Admin Console initialized")

    def _register_admin_commands(self):
        """Register admin commands."""
        self.admin_commands = {
            "status": self._cmd_status,
            "metrics": self._cmd_metrics,
            "alerts": self._cmd_alerts,
            "logs": self._cmd_logs,
            "health": self._cmd_health,
            "components": self._cmd_components,
            "config": self._cmd_config,
            "restart": self._cmd_restart,
            "shutdown": self._cmd_shutdown
        }

    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started monitoring thread")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_metrics_time = 0
        last_health_time = 0

        while self.running:
            current_time = time.time()

            # Collect metrics
            if current_time - last_metrics_time >= self.metrics_interval:
                try:
                    self._collect_system_metrics()
                    last_metrics_time = current_time
                except Exception as exc:
                    logger.error(f"Metrics collection failed: {exc}")

            # Health checks
            if current_time - last_health_time >= self.health_check_interval:
                try:
                    self._perform_health_checks()
                    last_health_time = current_time
                except Exception as exc:
                    logger.error(f"Health check failed: {exc}")

            # Check alerts
            try:
                self.alert_manager.check_alerts(self.metrics_collector)
            except Exception as exc:
                logger.error(f"Alert checking failed: {exc}")

            time.sleep(1)

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        if not HAS_PSUTIL:
            return

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)  # type: ignore
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()  # type: ignore
            self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
            self.metrics_collector.set_gauge("system_memory_used_mb", memory.used / 1024 / 1024)

            # Disk usage
            disk = psutil.disk_usage('/')  # type: ignore
            self.metrics_collector.set_gauge("system_disk_percent", disk.percent)

            # Network connections
            net_connections = len(psutil.net_connections())  # type: ignore
            self.metrics_collector.set_gauge("system_net_connections", net_connections)

            # Thread count
            thread_count = threading.active_count()
            self.metrics_collector.set_gauge("system_active_threads", thread_count)

            # System health snapshot
            health = SystemHealth(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_connections=net_connections,
                active_threads=thread_count,
                uptime=time.time() - psutil.boot_time()  # type: ignore
            )
            self.system_health_history.append(health)

            # Keep only last 1000 health snapshots
            if len(self.system_health_history) > 1000:
                self.system_health_history = self.system_health_history[-1000:]

        except Exception as exc:
            logger.error(f"System metrics collection failed: {exc}")

    def _perform_health_checks(self):
        """Perform health checks on system components."""
        # Check core components
        components_to_check = [
            "metrics_collector",
            "alert_manager",
            "log_aggregator",
            "provider_registry"
        ]

        for component_name in components_to_check:
            self._check_component_health(component_name)

    def _check_component_health(self, component_name: str):
        """Check health of a specific component."""
        start_time = time.time()

        try:
            # Simple health check - in real implementation would be more sophisticated
            if component_name == "metrics_collector":
                # Check if we can collect a metric
                self.metrics_collector.increment_counter("health_check")
                status = "healthy"
                error_count = 0
            elif component_name == "alert_manager":
                # Check if we can get alerts
                alerts = self.alert_manager.get_active_alerts()
                status = "healthy"
                error_count = 0
            elif component_name == "log_aggregator":
                # Check if we can get logs
                logs = self.log_aggregator.get_logs(limit=1)
                status = "healthy"
                error_count = 0
            elif component_name == "provider_registry":
                # Check provider registry
                try:
                    # Try to get providers info - adjust based on actual API
                    providers = getattr(provider_registry, 'providers', {})
                    status = "healthy"
                    error_count = 0
                except Exception:
                    status = "unhealthy"
                    error_count = 1
            else:
                status = "unknown"
                error_count = 0

        except Exception as exc:
            status = "unhealthy"
            error_count = 1
            logger.error(f"Health check failed for {component_name}: {exc}")

        response_time = (time.time() - start_time) * 1000  # ms

        component_status = ComponentStatus(
            name=component_name,
            status=status,
            response_time=response_time,
            last_check=datetime.now(),
            error_count=error_count
        )

        self.component_statuses[component_name] = component_status

    def log_event(
        self,
        level: LogLevel,
        component: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log an event."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            metadata=metadata or {},
            trace_id=trace_id,
            user_id=user_id
        )

        self.log_aggregator.add_log(entry)

        # Also log to standard logger
        log_method = getattr(logger, level.value.lower(), logger.info)
        log_method(f"[{component}] {message}")

    def add_metric(self, metric: Metric):
        """Add a custom metric."""
        if metric.type == MetricType.COUNTER:
            self.metrics_collector.increment_counter(
                metric.name,
                int(metric.value),
                metric.labels
            )
        elif metric.type == MetricType.GAUGE:
            self.metrics_collector.set_gauge(
                metric.name,
                metric.value,
                metric.labels
            )
        elif metric.type == MetricType.HISTOGRAM:
            self.metrics_collector.record_histogram(
                metric.name,
                metric.value,
                metric.labels
            )

    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add an alert rule."""
        self.alert_manager.add_alert_rule(rule)

    def add_notification_callback(self, callback: Callable):
        """Add a notification callback for alerts."""
        self.alert_manager.add_notification_callback(callback)

    def execute_admin_command(self, command: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute an admin command."""
        args = args or []

        if command not in self.admin_commands:
            return {"error": f"Unknown command: {command}"}

        try:
            return self.admin_commands[command](args)
        except Exception as exc:
            return {"error": f"Command execution failed: {str(exc)}"}

    def _cmd_status(self, args: List[str]) -> Dict[str, Any]:
        """Get system status."""
        return {
            "status": "running",
            "uptime": self._get_uptime(),
            "monitoring_active": self.running,
            "components_checked": len(self.component_statuses)
        }

    def _cmd_metrics(self, args: List[str]) -> Dict[str, Any]:
        """Get metrics."""
        metric_name = args[0] if args else None
        if metric_name:
            return self.metrics_collector.get_metric_stats(metric_name)
        else:
            return {"metrics": self.metrics_collector.get_all_metrics()}

    def _cmd_alerts(self, args: List[str]) -> Dict[str, Any]:
        """Get alerts."""
        active_only = "--active" in args
        if active_only:
            alerts = self.alert_manager.get_active_alerts()
        else:
            alerts = self.alert_manager.get_all_alerts()

        return {
            "alerts": [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved
                }
                for alert in alerts
            ]
        }

    def _cmd_logs(self, args: List[str]) -> Dict[str, Any]:
        """Get logs."""
        component = None
        level = None
        hours = 24
        limit = 100

        i = 0
        while i < len(args):
            if args[i] == "--component" and i + 1 < len(args):
                component = args[i + 1]
                i += 2
            elif args[i] == "--level" and i + 1 < len(args):
                level = LogLevel(args[i + 1].upper())
                i += 2
            elif args[i] == "--hours" and i + 1 < len(args):
                hours = int(args[i + 1])
                i += 2
            elif args[i] == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            else:
                i += 1

        logs = self.log_aggregator.get_logs(component, level, hours, limit)

        return {
            "logs": [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level.value,
                    "component": log.component,
                    "message": log.message,
                    "metadata": log.metadata
                }
                for log in logs
            ]
        }

    def _cmd_health(self, args: List[str]) -> Dict[str, Any]:
        """Get health information."""
        return {
            "system": self._get_current_system_health(),
            "components": {
                name: {
                    "status": status.status,
                    "response_time": status.response_time,
                    "last_check": status.last_check.isoformat(),
                    "error_count": status.error_count
                }
                for name, status in self.component_statuses.items()
            }
        }

    def _cmd_components(self, args: List[str]) -> Dict[str, Any]:
        """Get component information."""
        return {
            "components": list(self.component_statuses.keys()),
            "statuses": {
                name: status.status
                for name, status in self.component_statuses.items()
            }
        }

    def _cmd_config(self, args: List[str]) -> Dict[str, Any]:
        """Get configuration information."""
        return {
            "metrics_interval": self.metrics_interval,
            "health_check_interval": self.health_check_interval,
            "log_retention_hours": self.log_retention_hours
        }

    def _cmd_restart(self, args: List[str]) -> Dict[str, Any]:
        """Restart monitoring."""
        self.stop_monitoring()
        self.start_monitoring()
        return {"status": "restarted"}

    def _cmd_shutdown(self, args: List[str]) -> Dict[str, Any]:
        """Shutdown monitoring."""
        self.stop_monitoring()
        return {"status": "shutdown"}

    def _get_uptime(self) -> float:
        """Get system uptime."""
        if not HAS_PSUTIL:
            return 0.0
        try:
            return time.time() - psutil.boot_time()  # type: ignore
        except:
            return 0.0

    def _get_current_system_health(self) -> Dict[str, Any]:
        """Get current system health."""
        if not self.system_health_history:
            return {}

        latest = self.system_health_history[-1]
        return {
            "cpu_usage": latest.cpu_usage,
            "memory_usage": latest.memory_usage,
            "disk_usage": latest.disk_usage,
            "network_connections": latest.network_connections,
            "active_threads": latest.active_threads,
            "uptime": latest.uptime,
            "timestamp": latest.timestamp.isoformat()
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "status": self._cmd_status([]),
            "health": self._cmd_health([]),
            "metrics": self._cmd_metrics([]),
            "alerts": self._cmd_alerts(["--active"]),
            "logs": self._cmd_logs(["--limit", "50"]),
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count() if HAS_PSUTIL else None,  # type: ignore
                "memory_total": (psutil.virtual_memory().total / 1024 / 1024 / 1024) if HAS_PSUTIL else None  # type: ignore
            }
        }


# Register with provider registry
def create_observability_admin_console(config_manager=None, **kwargs):
    """Factory function for ObservabilityAdminConsole."""
    console = ObservabilityAdminConsole(config_manager=config_manager, **kwargs)
    console.start_monitoring()  # Auto-start monitoring
    return console


provider_registry.register_lazy(
    'adaptive_intelligence', 'observability_console',
    'mia.adaptive_intelligence.observability_admin_console', 'create_observability_admin_console',
    default=True
)