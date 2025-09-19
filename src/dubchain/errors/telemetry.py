"""Error telemetry and monitoring for DubChain.

This module provides comprehensive error telemetry, metrics collection,
reporting, and dashboard capabilities for the DubChain error handling system.
"""

import json
import logging
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .exceptions import DubChainError, ErrorCategory, ErrorSeverity


@dataclass
class ErrorMetrics:
    """Error metrics data structure."""

    # Counts
    total_errors: int = 0
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    # Rates
    error_rate: float = 0.0
    error_rate_by_severity: Dict[str, float] = field(default_factory=dict)
    error_rate_by_category: Dict[str, float] = field(default_factory=dict)

    # Time series
    error_timeline: List[Dict[str, Any]] = field(default_factory=list)
    error_frequency: Dict[str, int] = field(default_factory=dict)

    # Performance impact
    average_recovery_time: float = 0.0
    recovery_success_rate: float = 0.0
    system_impact_score: float = 0.0

    def update_rates(self, time_window: float = 3600.0) -> None:
        """Update error rates based on time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window

        # Filter timeline to time window
        recent_errors = [
            error
            for error in self.error_timeline
            if error.get("timestamp", 0) >= cutoff_time
        ]

        # Calculate rates
        self.error_rate = len(recent_errors) / (time_window / 3600.0)  # errors per hour

        # Calculate rates by severity
        for severity in ErrorSeverity:
            severity_errors = [
                error
                for error in recent_errors
                if error.get("severity") == severity.value
            ]
            self.error_rate_by_severity[severity.value] = len(severity_errors) / (
                time_window / 3600.0
            )

        # Calculate rates by category
        for category in ErrorCategory:
            category_errors = [
                error
                for error in recent_errors
                if error.get("category") == category.value
            ]
            self.error_rate_by_category[category.value] = len(category_errors) / (
                time_window / 3600.0
            )


@dataclass
class ErrorReport:
    """Error report data structure."""

    report_id: str
    timestamp: float
    time_range: Dict[str, float]  # start_time, end_time
    summary: Dict[str, Any]
    top_errors: List[Dict[str, Any]]
    trends: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorAggregator:
    """Error aggregator for collecting and processing error data."""

    def __init__(self, max_errors: int = 10000, time_window: float = 3600.0):
        self.max_errors = max_errors
        self.time_window = time_window
        self._errors: deque = deque(maxlen=max_errors)
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # Metrics
        self._metrics = ErrorMetrics()
        self._last_metrics_update = 0.0
        self._metrics_update_interval = 60.0  # 1 minute

    def add_error(self, error: DubChainError) -> None:
        """Add an error to the aggregator."""
        with self._lock:
            error_data = error.to_dict()
            self._errors.append(error_data)

            # Update metrics
            self._update_metrics(error_data)

            # Check if we need to update rates
            current_time = time.time()
            if (
                current_time - self._last_metrics_update
                >= self._metrics_update_interval
            ):
                self._metrics.update_rates(self.time_window)
                self._last_metrics_update = current_time

    def _update_metrics(self, error_data: Dict[str, Any]) -> None:
        """Update metrics with new error data."""
        # Update counts
        self._metrics.total_errors += 1

        # Update by severity
        severity = error_data.get("severity", "medium")
        self._metrics.errors_by_severity[severity] = (
            self._metrics.errors_by_severity.get(severity, 0) + 1
        )

        # Update by category
        category = error_data.get("category", "system")
        self._metrics.errors_by_category[category] = (
            self._metrics.errors_by_category.get(category, 0) + 1
        )

        # Update by type
        error_type = error_data.get("type", "DubChainError")
        self._metrics.errors_by_type[error_type] = (
            self._metrics.errors_by_type.get(error_type, 0) + 1
        )

        # Add to timeline
        self._metrics.error_timeline.append(
            {
                "timestamp": error_data.get("timestamp", time.time()),
                "severity": severity,
                "category": category,
                "type": error_type,
                "message": error_data.get("message", ""),
                "error_code": error_data.get("error_code"),
            }
        )

        # Update frequency
        error_key = f"{error_type}:{error_data.get('message', '')}"
        self._metrics.error_frequency[error_key] = (
            self._metrics.error_frequency.get(error_key, 0) + 1
        )

    def get_metrics(self) -> ErrorMetrics:
        """Get current error metrics."""
        with self._lock:
            # Update rates if needed
            current_time = time.time()
            if (
                current_time - self._last_metrics_update
                >= self._metrics_update_interval
            ):
                self._metrics.update_rates(self.time_window)
                self._last_metrics_update = current_time

            return self._metrics

    def get_errors(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        error_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get filtered errors."""
        with self._lock:
            filtered_errors = []

            for error in self._errors:
                if severity and error.get("severity") != severity:
                    continue
                if category and error.get("category") != category:
                    continue
                if error_type and error.get("type") != error_type:
                    continue

                filtered_errors.append(error)

                if len(filtered_errors) >= limit:
                    break

            return filtered_errors

    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top errors by frequency."""
        with self._lock:
            sorted_errors = sorted(
                self._metrics.error_frequency.items(), key=lambda x: x[1], reverse=True
            )

            return [
                {
                    "error": error_key,
                    "count": count,
                    "percentage": (count / self._metrics.total_errors) * 100
                    if self._metrics.total_errors > 0
                    else 0,
                }
                for error_key, count in sorted_errors[:limit]
            ]

    def get_error_trends(self, time_window: float = 3600.0) -> Dict[str, Any]:
        """Get error trends over time."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - time_window

            # Filter errors to time window
            recent_errors = [
                error
                for error in self._errors
                if error.get("timestamp", 0) >= cutoff_time
            ]

            if not recent_errors:
                return {
                    "total_errors": 0,
                    "error_rate": 0.0,
                    "trend": "stable",
                    "severity_distribution": {},
                    "category_distribution": {},
                }

            # Calculate trend
            time_buckets = {}
            bucket_size = time_window / 10  # 10 buckets

            for error in recent_errors:
                bucket = int((error.get("timestamp", 0) - cutoff_time) / bucket_size)
                time_buckets[bucket] = time_buckets.get(bucket, 0) + 1

            # Determine trend
            if len(time_buckets) >= 2:
                first_half = sum(time_buckets.get(i, 0) for i in range(5))
                second_half = sum(time_buckets.get(i, 0) for i in range(5, 10))

                if second_half > first_half * 1.2:
                    trend = "increasing"
                elif second_half < first_half * 0.8:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Calculate distributions
            severity_dist = defaultdict(int)
            category_dist = defaultdict(int)

            for error in recent_errors:
                severity_dist[error.get("severity", "medium")] += 1
                category_dist[error.get("category", "system")] += 1

            return {
                "total_errors": len(recent_errors),
                "error_rate": len(recent_errors) / (time_window / 3600.0),
                "trend": trend,
                "severity_distribution": dict(severity_dist),
                "category_distribution": dict(category_dist),
                "time_buckets": dict(time_buckets),
            }

    def clear_old_errors(self, max_age: float = 86400.0) -> int:
        """Clear errors older than max_age seconds."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - max_age

            # Count old errors
            old_count = sum(
                1 for error in self._errors if error.get("timestamp", 0) < cutoff_time
            )

            # Remove old errors
            self._errors = deque(
                (
                    error
                    for error in self._errors
                    if error.get("timestamp", 0) >= cutoff_time
                ),
                maxlen=self.max_errors,
            )

            return old_count


class ErrorReporter(ABC):
    """Abstract error reporter."""

    @abstractmethod
    def report_error(self, error: DubChainError) -> None:
        """Report an error."""
        pass

    @abstractmethod
    def report_metrics(self, metrics: ErrorMetrics) -> None:
        """Report error metrics."""
        pass


class LogErrorReporter(ErrorReporter):
    """Log-based error reporter."""

    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)

    def report_error(self, error: DubChainError) -> None:
        """Report error to logs."""
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error}")
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR: {error}")
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR: {error}")
        else:
            self.logger.info(f"LOW SEVERITY ERROR: {error}")

    def report_metrics(self, metrics: ErrorMetrics) -> None:
        """Report metrics to logs."""
        self.logger.info(
            f"Error metrics: {metrics.total_errors} total errors, "
            f"rate: {metrics.error_rate:.2f}/hour"
        )


class FileErrorReporter(ErrorReporter):
    """File-based error reporter."""

    def __init__(self, file_path: str, max_file_size: int = 10 * 1024 * 1024):  # 10MB
        self.file_path = file_path
        self.max_file_size = max_file_size
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

    def report_error(self, error: DubChainError) -> None:
        """Report error to file."""
        with self._lock:
            try:
                # Check file size and rotate if needed
                self._rotate_if_needed()

                # Write error to file
                with open(self.file_path, "a") as f:
                    f.write(json.dumps(error.to_dict()) + "\n")

            except Exception as e:
                self._logger.error(f"Error writing to error file: {e}")

    def report_metrics(self, metrics: ErrorMetrics) -> None:
        """Report metrics to file."""
        with self._lock:
            try:
                # Check file size and rotate if needed
                self._rotate_if_needed()

                # Write metrics to file
                with open(self.file_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "type": "metrics",
                                "timestamp": time.time(),
                                "metrics": {
                                    "total_errors": metrics.total_errors,
                                    "error_rate": metrics.error_rate,
                                    "errors_by_severity": metrics.errors_by_severity,
                                    "errors_by_category": metrics.errors_by_category,
                                },
                            }
                        )
                        + "\n"
                    )

            except Exception as e:
                self._logger.error(f"Error writing metrics to file: {e}")

    def _rotate_if_needed(self) -> None:
        """Rotate file if it's too large."""
        try:
            import os

            if os.path.exists(self.file_path):
                file_size = os.path.getsize(self.file_path)
                if file_size > self.max_file_size:
                    # Rotate file
                    backup_path = f"{self.file_path}.{int(time.time())}"
                    os.rename(self.file_path, backup_path)
        except Exception as e:
            self._logger.error(f"Error rotating file: {e}")


class ErrorDashboard:
    """Error dashboard for monitoring and visualization."""

    def __init__(self, aggregator: ErrorAggregator):
        self.aggregator = aggregator
        self._logger = logging.getLogger(__name__)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        metrics = self.aggregator.get_metrics()
        trends = self.aggregator.get_error_trends()
        top_errors = self.aggregator.get_top_errors()

        return {
            "summary": {
                "total_errors": metrics.total_errors,
                "error_rate": metrics.error_rate,
                "system_impact_score": metrics.system_impact_score,
            },
            "distributions": {
                "by_severity": metrics.errors_by_severity,
                "by_category": metrics.errors_by_category,
                "by_type": metrics.errors_by_type,
            },
            "trends": trends,
            "top_errors": top_errors,
            "recommendations": self._generate_recommendations(metrics, trends),
            "timestamp": time.time(),
        }

    def _generate_recommendations(
        self, metrics: ErrorMetrics, trends: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on metrics and trends."""
        recommendations = []

        # High error rate
        if metrics.error_rate > 10.0:
            recommendations.append(
                "High error rate detected. Consider investigating root causes."
            )

        # Increasing trend
        if trends.get("trend") == "increasing":
            recommendations.append(
                "Error trend is increasing. Monitor closely and consider preventive measures."
            )

        # High critical errors
        critical_count = metrics.errors_by_severity.get("critical", 0)
        if critical_count > 0:
            recommendations.append(
                f"{critical_count} critical errors detected. Immediate attention required."
            )

        # High network errors
        network_count = metrics.errors_by_category.get("network", 0)
        if network_count > 0:
            recommendations.append(
                "Network errors detected. Check connectivity and network configuration."
            )

        # High storage errors
        storage_count = metrics.errors_by_category.get("storage", 0)
        if storage_count > 0:
            recommendations.append(
                "Storage errors detected. Check disk space and storage health."
            )

        return recommendations

    def generate_report(
        self, time_range: Optional[Dict[str, float]] = None
    ) -> ErrorReport:
        """Generate error report."""
        if time_range is None:
            time_range = {
                "start_time": time.time() - 3600.0,  # Last hour
                "end_time": time.time(),
            }

        metrics = self.aggregator.get_metrics()
        trends = self.aggregator.get_error_trends()
        top_errors = self.aggregator.get_top_errors()

        return ErrorReport(
            report_id=f"error_report_{int(time.time())}",
            timestamp=time.time(),
            time_range=time_range,
            summary={
                "total_errors": metrics.total_errors,
                "error_rate": metrics.error_rate,
                "system_impact_score": metrics.system_impact_score,
            },
            top_errors=top_errors,
            trends=trends,
            recommendations=self._generate_recommendations(metrics, trends),
            metadata={"generated_by": "ErrorDashboard", "version": "1.0"},
        )


class ErrorTelemetry:
    """Main error telemetry system."""

    def __init__(self, aggregator: ErrorAggregator = None):
        self.aggregator = aggregator or ErrorAggregator()
        self.reporters: List[ErrorReporter] = []
        self.dashboard = ErrorDashboard(self.aggregator)
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # Add default reporter
        self.add_reporter(LogErrorReporter())

    def add_reporter(self, reporter: ErrorReporter) -> None:
        """Add an error reporter."""
        with self._lock:
            self.reporters.append(reporter)
            self._logger.info(f"Added error reporter: {reporter.__class__.__name__}")

    def remove_reporter(self, reporter: ErrorReporter) -> None:
        """Remove an error reporter."""
        with self._lock:
            if reporter in self.reporters:
                self.reporters.remove(reporter)
                self._logger.info(
                    f"Removed error reporter: {reporter.__class__.__name__}"
                )

    def report_error(self, error: DubChainError) -> None:
        """Report an error through all reporters."""
        with self._lock:
            # Add to aggregator
            self.aggregator.add_error(error)

            # Report through all reporters
            for reporter in self.reporters:
                try:
                    reporter.report_error(error)
                except Exception as e:
                    self._logger.error(
                        f"Error in reporter {reporter.__class__.__name__}: {e}"
                    )

    def report_metrics(self) -> None:
        """Report metrics through all reporters."""
        with self._lock:
            metrics = self.aggregator.get_metrics()

            for reporter in self.reporters:
                try:
                    reporter.report_metrics(metrics)
                except Exception as e:
                    self._logger.error(
                        f"Error reporting metrics in {reporter.__class__.__name__}: {e}"
                    )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        return self.dashboard.get_dashboard_data()

    def generate_report(
        self, time_range: Optional[Dict[str, float]] = None
    ) -> ErrorReport:
        """Generate error report."""
        return self.dashboard.generate_report(time_range)

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on telemetry system."""
        with self._lock:
            return {
                "status": "healthy",
                "aggregator_errors": len(self.aggregator._errors),
                "reporters_count": len(self.reporters),
                "last_metrics_update": self.aggregator._last_metrics_update,
                "timestamp": time.time(),
            }
