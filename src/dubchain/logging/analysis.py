"""Log analysis and metrics for DubChain.

This module provides log analysis, metrics collection, dashboard,
and reporting capabilities for the DubChain logging system.
"""

import json
import logging
import re
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .core import LogEntry, LogLevel


@dataclass
class LogMetrics:
    """Log metrics data structure."""

    # Counts
    total_logs: int = 0
    logs_by_level: Dict[str, int] = field(default_factory=dict)
    logs_by_logger: Dict[str, int] = field(default_factory=dict)
    logs_by_component: Dict[str, int] = field(default_factory=dict)

    # Rates
    log_rate: float = 0.0
    log_rate_by_level: Dict[str, float] = field(default_factory=dict)
    log_rate_by_logger: Dict[str, float] = field(default_factory=dict)

    # Time series
    log_timeline: List[Dict[str, Any]] = field(default_factory=list)
    log_frequency: Dict[str, int] = field(default_factory=dict)

    # Performance metrics
    average_log_size: float = 0.0
    max_log_size: int = 0
    min_log_size: int = 0

    # Error analysis
    error_rate: float = 0.0
    critical_error_rate: float = 0.0
    warning_rate: float = 0.0

    def update_rates(self, time_window: float = 3600.0) -> None:
        """Update log rates based on time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window

        # Filter timeline to time window
        recent_logs = [
            log for log in self.log_timeline if log.get("timestamp", 0) >= cutoff_time
        ]

        # Calculate rates
        self.log_rate = len(recent_logs) / (time_window / 3600.0)  # logs per hour

        # Calculate rates by level
        for level in LogLevel:
            level_logs = [log for log in recent_logs if log.get("level") == level.value]
            self.log_rate_by_level[level.value] = len(level_logs) / (
                time_window / 3600.0
            )

        # Calculate rates by logger
        logger_counts = Counter(log.get("logger") for log in recent_logs)
        for logger, count in logger_counts.items():
            self.log_rate_by_logger[logger] = count / (time_window / 3600.0)

    def add_log(self, entry: LogEntry) -> None:
        """Add log entry to metrics."""
        self.total_logs += 1

        # Update by level
        level = entry.level.value
        self.logs_by_level[level] = self.logs_by_level.get(level, 0) + 1

        # Update by logger
        logger = entry.logger_name
        self.logs_by_logger[logger] = self.logs_by_logger.get(logger, 0) + 1

        # Update by component
        if entry.context and entry.context.component:
            component = entry.context.component
            self.logs_by_component[component] = (
                self.logs_by_component.get(component, 0) + 1
            )

        # Add to timeline
        self.log_timeline.append(
            {
                "timestamp": entry.timestamp,
                "level": level,
                "logger": logger,
                "component": entry.context.component if entry.context else None,
                "message": entry.message,
            }
        )

        # Update frequency
        message_key = f"{level}:{entry.message}"
        self.log_frequency[message_key] = self.log_frequency.get(message_key, 0) + 1

        # Update size metrics
        log_size = len(entry.message)
        if self.total_logs == 1:
            self.average_log_size = log_size
            self.max_log_size = log_size
            self.min_log_size = log_size
        else:
            self.average_log_size = (
                self.average_log_size * (self.total_logs - 1) + log_size
            ) / self.total_logs
            self.max_log_size = max(self.max_log_size, log_size)
            self.min_log_size = min(self.min_log_size, log_size)

        # Update error rates
        if level in ["error", "critical", "fatal"]:
            self.error_rate = (
                self.logs_by_level.get("error", 0)
                + self.logs_by_level.get("critical", 0)
                + self.logs_by_level.get("fatal", 0)
            ) / self.total_logs

        if level in ["critical", "fatal"]:
            self.critical_error_rate = (
                self.logs_by_level.get("critical", 0)
                + self.logs_by_level.get("fatal", 0)
            ) / self.total_logs

        if level == "warning":
            self.warning_rate = self.logs_by_level.get("warning", 0) / self.total_logs


class LogQuery:
    """Log query builder."""

    def __init__(self):
        self.filters = []
        self.sort_by = None
        self.sort_order = "asc"
        self.limit = None
        self.offset = 0

    def filter_by_level(self, level: LogLevel) -> "LogQuery":
        """Filter by log level."""
        self.filters.append(lambda entry: entry.level == level)
        return self

    def filter_by_logger(self, logger_name: str) -> "LogQuery":
        """Filter by logger name."""
        self.filters.append(lambda entry: entry.logger_name == logger_name)
        return self

    def filter_by_component(self, component: str) -> "LogQuery":
        """Filter by component."""
        self.filters.append(
            lambda entry: entry.context and entry.context.component == component
        )
        return self

    def filter_by_time_range(self, start_time: float, end_time: float) -> "LogQuery":
        """Filter by time range."""
        self.filters.append(lambda entry: start_time <= entry.timestamp <= end_time)
        return self

    def filter_by_message_pattern(self, pattern: str) -> "LogQuery":
        """Filter by message pattern."""
        regex = re.compile(pattern)
        self.filters.append(lambda entry: regex.search(entry.message) is not None)
        return self

    def filter_by_custom(self, filter_func: Callable[[LogEntry], bool]) -> "LogQuery":
        """Filter by custom function."""
        self.filters.append(filter_func)
        return self

    def sort_by_timestamp(self, order: str = "asc") -> "LogQuery":
        """Sort by timestamp."""
        self.sort_by = "timestamp"
        self.sort_order = order
        return self

    def sort_by_level(self, order: str = "asc") -> "LogQuery":
        """Sort by level."""
        self.sort_by = "level"
        self.sort_order = order
        return self

    def limit_results(self, limit: int) -> "LogQuery":
        """Limit results."""
        self.limit = limit
        return self

    def offset_results(self, offset: int) -> "LogQuery":
        """Offset results."""
        self.offset = offset
        return self

    def execute(self, entries: List[LogEntry]) -> List[LogEntry]:
        """Execute query on entries."""
        # Apply filters
        filtered_entries = entries
        for filter_func in self.filters:
            filtered_entries = [
                entry for entry in filtered_entries if filter_func(entry)
            ]

        # Sort
        if self.sort_by == "timestamp":
            filtered_entries.sort(
                key=lambda x: x.timestamp, reverse=(self.sort_order == "desc")
            )
        elif self.sort_by == "level":
            level_order = {level.value: i for i, level in enumerate(LogLevel)}
            filtered_entries.sort(
                key=lambda x: level_order.get(x.level.value, 999),
                reverse=(self.sort_order == "desc"),
            )

        # Apply offset and limit
        if self.offset > 0:
            filtered_entries = filtered_entries[self.offset :]

        if self.limit is not None:
            filtered_entries = filtered_entries[: self.limit]

        return filtered_entries


class LogSearch:
    """Log search functionality."""

    def __init__(self, entries: List[LogEntry]):
        self.entries = entries
        self._index = self._build_index()

    def _build_index(self) -> Dict[str, List[int]]:
        """Build search index."""
        index = defaultdict(list)

        for i, entry in enumerate(self.entries):
            # Index by message words
            words = re.findall(r"\w+", entry.message.lower())
            for word in words:
                index[word].append(i)

            # Index by logger name
            index[entry.logger_name.lower()].append(i)

            # Index by component
            if entry.context and entry.context.component:
                index[entry.context.component.lower()].append(i)

        return dict(index)

    def search(self, query: str) -> List[LogEntry]:
        """Search logs by query."""
        query_words = re.findall(r"\w+", query.lower())

        if not query_words:
            return []

        # Find entries that contain all query words
        result_indices = set()

        for word in query_words:
            if word in self._index:
                if not result_indices:
                    result_indices = set(self._index[word])
                else:
                    result_indices &= set(self._index[word])

        return [self.entries[i] for i in sorted(result_indices)]

    def search_by_regex(self, pattern: str) -> List[LogEntry]:
        """Search logs by regex pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [entry for entry in self.entries if regex.search(entry.message)]

    def search_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[LogEntry]:
        """Search logs by time range."""
        return [
            entry for entry in self.entries if start_time <= entry.timestamp <= end_time
        ]


class LogAnalyzer:
    """Log analyzer for processing and analyzing logs."""

    def __init__(self, entries: List[LogEntry] = None):
        self.entries = entries or []
        self.metrics = LogMetrics()
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # Initialize metrics
        for entry in self.entries:
            self.metrics.add_log(entry)

    def add_entry(self, entry: LogEntry) -> None:
        """Add log entry to analyzer."""
        with self._lock:
            self.entries.append(entry)
            self.metrics.add_log(entry)

    def add_entries(self, entries: List[LogEntry]) -> None:
        """Add multiple log entries to analyzer."""
        with self._lock:
            self.entries.extend(entries)
            for entry in entries:
                self.metrics.add_log(entry)

    def get_metrics(self) -> LogMetrics:
        """Get log metrics."""
        with self._lock:
            self.metrics.update_rates()
            return self.metrics

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze log patterns."""
        with self._lock:
            patterns = {
                "frequent_messages": self._get_frequent_messages(),
                "error_patterns": self._get_error_patterns(),
                "time_patterns": self._get_time_patterns(),
                "component_patterns": self._get_component_patterns(),
            }

            return patterns

    def _get_frequent_messages(self) -> List[Dict[str, Any]]:
        """Get frequent log messages."""
        message_counts = Counter(entry.message for entry in self.entries)
        return [
            {
                "message": message,
                "count": count,
                "percentage": (count / len(self.entries)) * 100,
            }
            for message, count in message_counts.most_common(10)
        ]

    def _get_error_patterns(self) -> Dict[str, Any]:
        """Get error patterns."""
        error_entries = [
            entry
            for entry in self.entries
            if entry.level.value in ["error", "critical", "fatal"]
        ]

        if not error_entries:
            return {"total_errors": 0, "error_rate": 0.0, "common_errors": []}

        error_counts = Counter(entry.message for entry in error_entries)

        return {
            "total_errors": len(error_entries),
            "error_rate": len(error_entries) / len(self.entries),
            "common_errors": [
                {"message": message, "count": count}
                for message, count in error_counts.most_common(5)
            ],
        }

    def _get_time_patterns(self) -> Dict[str, Any]:
        """Get time patterns."""
        if not self.entries:
            return {}

        # Group by hour
        hourly_counts = defaultdict(int)
        for entry in self.entries:
            hour = time.strftime("%H", time.gmtime(entry.timestamp))
            hourly_counts[hour] += 1

        # Group by day of week
        daily_counts = defaultdict(int)
        for entry in self.entries:
            day = time.strftime("%A", time.gmtime(entry.timestamp))
            daily_counts[day] += 1

        return {
            "hourly_distribution": dict(hourly_counts),
            "daily_distribution": dict(daily_counts),
        }

    def _get_component_patterns(self) -> Dict[str, Any]:
        """Get component patterns."""
        component_counts = defaultdict(int)
        component_errors = defaultdict(int)

        for entry in self.entries:
            if entry.context and entry.context.component:
                component = entry.context.component
                component_counts[component] += 1

                if entry.level.value in ["error", "critical", "fatal"]:
                    component_errors[component] += 1

        return {
            "component_activity": dict(component_counts),
            "component_errors": dict(component_errors),
        }

    def generate_report(
        self, time_range: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generate log analysis report."""
        with self._lock:
            if time_range:
                start_time = time_range.get("start_time", 0)
                end_time = time_range.get("end_time", time.time())
                filtered_entries = [
                    entry
                    for entry in self.entries
                    if start_time <= entry.timestamp <= end_time
                ]
            else:
                filtered_entries = self.entries

            if not filtered_entries:
                return {"error": "No logs found in specified time range"}

            # Calculate metrics for filtered entries
            temp_analyzer = LogAnalyzer(filtered_entries)
            metrics = temp_analyzer.get_metrics()
            patterns = temp_analyzer.analyze_patterns()

            return {
                "time_range": time_range,
                "total_logs": len(filtered_entries),
                "metrics": {
                    "log_rate": metrics.log_rate,
                    "error_rate": metrics.error_rate,
                    "critical_error_rate": metrics.critical_error_rate,
                    "warning_rate": metrics.warning_rate,
                    "logs_by_level": metrics.logs_by_level,
                    "logs_by_logger": metrics.logs_by_logger,
                    "logs_by_component": metrics.logs_by_component,
                },
                "patterns": patterns,
                "recommendations": self._generate_recommendations(metrics, patterns),
            }

    def _generate_recommendations(
        self, metrics: LogMetrics, patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # High error rate
        if metrics.error_rate > 0.1:
            recommendations.append(
                "High error rate detected. Consider investigating error patterns."
            )

        # High critical error rate
        if metrics.critical_error_rate > 0.01:
            recommendations.append(
                "Critical errors detected. Immediate attention required."
            )

        # High log rate
        if metrics.log_rate > 1000:
            recommendations.append(
                "High log rate detected. Consider log level optimization."
            )

        # Frequent error messages
        if patterns.get("error_patterns", {}).get("common_errors"):
            recommendations.append(
                "Common error patterns detected. Consider addressing root causes."
            )

        return recommendations


class LogDashboard:
    """Log dashboard for monitoring and visualization."""

    def __init__(self, analyzer: LogAnalyzer):
        self.analyzer = analyzer
        self._logger = logging.getLogger(__name__)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        metrics = self.analyzer.get_metrics()
        patterns = self.analyzer.analyze_patterns()

        return {
            "summary": {
                "total_logs": metrics.total_logs,
                "log_rate": metrics.log_rate,
                "error_rate": metrics.error_rate,
                "critical_error_rate": metrics.critical_error_rate,
            },
            "distributions": {
                "by_level": metrics.logs_by_level,
                "by_logger": metrics.logs_by_logger,
                "by_component": metrics.logs_by_component,
            },
            "patterns": patterns,
            "recommendations": self._generate_recommendations(metrics, patterns),
            "timestamp": time.time(),
        }

    def _generate_recommendations(
        self, metrics: LogMetrics, patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate dashboard recommendations."""
        recommendations = []

        # High error rate
        if metrics.error_rate > 0.1:
            recommendations.append(
                "High error rate detected. Monitor error patterns closely."
            )

        # High critical error rate
        if metrics.critical_error_rate > 0.01:
            recommendations.append(
                "Critical errors detected. Immediate investigation required."
            )

        # High log rate
        if metrics.log_rate > 1000:
            recommendations.append(
                "High log rate detected. Consider log level optimization."
            )

        # Frequent error messages
        if patterns.get("error_patterns", {}).get("common_errors"):
            recommendations.append(
                "Common error patterns detected. Address root causes."
            )

        return recommendations


class LogReporter:
    """Log reporter for generating reports."""

    def __init__(self, analyzer: LogAnalyzer):
        self.analyzer = analyzer
        self._logger = logging.getLogger(__name__)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report."""
        metrics = self.analyzer.get_metrics()
        patterns = self.analyzer.analyze_patterns()

        return {
            "report_type": "summary",
            "timestamp": time.time(),
            "total_logs": metrics.total_logs,
            "log_rate": metrics.log_rate,
            "error_rate": metrics.error_rate,
            "critical_error_rate": metrics.critical_error_rate,
            "top_loggers": list(metrics.logs_by_logger.keys())[:5],
            "top_components": list(metrics.logs_by_component.keys())[:5],
            "frequent_messages": patterns.get("frequent_messages", [])[:5],
            "error_patterns": patterns.get("error_patterns", {}),
        }

    def generate_detailed_report(
        self, time_range: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generate detailed report."""
        return self.analyzer.generate_report(time_range)

    def export_to_json(self, report: Dict[str, Any], file_path: str) -> bool:
        """Export report to JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            return True
        except Exception as e:
            self._logger.error(f"Failed to export report to JSON: {e}")
            return False

    def export_to_csv(self, entries: List[LogEntry], file_path: str) -> bool:
        """Export log entries to CSV file."""
        try:
            import csv

            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Write header
                writer.writerow(
                    ["timestamp", "level", "logger", "component", "message"]
                )

                # Write data
                for entry in entries:
                    writer.writerow(
                        [
                            time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.gmtime(entry.timestamp)
                            ),
                            entry.level.value,
                            entry.logger_name,
                            entry.context.component if entry.context else "",
                            entry.message,
                        ]
                    )

            return True
        except Exception as e:
            self._logger.error(f"Failed to export logs to CSV: {e}")
            return False
