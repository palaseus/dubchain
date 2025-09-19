"""Log filters for DubChain.

This module provides various log filters including level, context, regex,
time, and custom filters for the DubChain logging system.
"""

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from .core import LogEntry, LogFilter, LogLevel


class LevelFilter(LogFilter):
    """Filter logs by level."""

    def __init__(self, min_level: LogLevel, max_level: LogLevel = None):
        self.min_level = min_level
        self.max_level = max_level or LogLevel.FATAL

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by level."""
        level_values = {level.value: i for i, level in enumerate(LogLevel)}
        entry_level = level_values.get(entry.level.value, 0)
        min_level = level_values.get(self.min_level.value, 0)
        max_level = level_values.get(self.max_level.value, 999)

        return min_level <= entry_level <= max_level


class ContextFilter(LogFilter):
    """Filter logs by context."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.node_id = node_id
        self.component = component
        self.operation = operation
        self.user_id = user_id
        self.request_id = request_id
        self.session_id = session_id

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by context."""
        if not entry.context:
            return False

        if self.node_id and entry.context.node_id != self.node_id:
            return False

        if self.component and entry.context.component != self.component:
            return False

        if self.operation and entry.context.operation != self.operation:
            return False

        if self.user_id and entry.context.user_id != self.user_id:
            return False

        if self.request_id and entry.context.request_id != self.request_id:
            return False

        if self.session_id and entry.context.session_id != self.session_id:
            return False

        return True


class RegexFilter(LogFilter):
    """Filter logs by regex pattern."""

    def __init__(self, pattern: str, field: str = "message", flags: int = 0):
        self.pattern = re.compile(pattern, flags)
        self.field = field

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by regex pattern."""
        if self.field == "message":
            text = entry.message
        elif self.field == "logger":
            text = entry.logger_name
        elif self.field == "component" and entry.context:
            text = entry.context.component or ""
        elif self.field == "operation" and entry.context:
            text = entry.context.operation or ""
        else:
            return False

        return self.pattern.search(text) is not None


class TimeFilter(LogFilter):
    """Filter logs by time range."""

    def __init__(self, start_time: float, end_time: float):
        self.start_time = start_time
        self.end_time = end_time

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by time range."""
        return self.start_time <= entry.timestamp <= self.end_time


class CustomFilter(LogFilter):
    """Custom filter using a function."""

    def __init__(self, filter_func: Callable[[LogEntry], bool]):
        self.filter_func = filter_func

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry using custom function."""
        try:
            return self.filter_func(entry)
        except Exception as e:
            logging.error(f"Error in custom filter: {e}")
            return False


class CompositeFilter(LogFilter):
    """Composite filter combining multiple filters."""

    def __init__(self, filters: List[LogFilter], operator: str = "AND"):
        self.filters = filters
        self.operator = operator.upper()

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry using composite logic."""
        if not self.filters:
            return True

        if self.operator == "AND":
            return all(filter_obj.filter(entry) for filter_obj in self.filters)
        elif self.operator == "OR":
            return any(filter_obj.filter(entry) for filter_obj in self.filters)
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")


class LoggerFilter(LogFilter):
    """Filter logs by logger name."""

    def __init__(self, logger_names: List[str], exclude: bool = False):
        self.logger_names = logger_names
        self.exclude = exclude

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by logger name."""
        if self.exclude:
            return entry.logger_name not in self.logger_names
        else:
            return entry.logger_name in self.logger_names


class MessageFilter(LogFilter):
    """Filter logs by message content."""

    def __init__(
        self, keywords: List[str], exclude: bool = False, case_sensitive: bool = False
    ):
        self.keywords = keywords
        self.exclude = exclude
        self.case_sensitive = case_sensitive

        if not case_sensitive:
            self.keywords = [keyword.lower() for keyword in keywords]

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by message content."""
        message = entry.message
        if not self.case_sensitive:
            message = message.lower()

        if self.exclude:
            return not any(keyword in message for keyword in self.keywords)
        else:
            return any(keyword in message for keyword in self.keywords)


class ExceptionFilter(LogFilter):
    """Filter logs by exception type."""

    def __init__(self, exception_types: List[type], exclude: bool = False):
        self.exception_types = exception_types
        self.exclude = exclude

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by exception type."""
        if not entry.exception:
            return not self.exclude  # Include if no exception and not excluding

        exception_type = type(entry.exception)

        if self.exclude:
            return exception_type not in self.exception_types
        else:
            return exception_type in self.exception_types


class ThreadFilter(LogFilter):
    """Filter logs by thread ID."""

    def __init__(self, thread_ids: List[int], exclude: bool = False):
        self.thread_ids = thread_ids
        self.exclude = exclude

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by thread ID."""
        if self.exclude:
            return entry.thread_id not in self.thread_ids
        else:
            return entry.thread_id in self.thread_ids


class ProcessFilter(LogFilter):
    """Filter logs by process ID."""

    def __init__(self, process_ids: List[int], exclude: bool = False):
        self.process_ids = process_ids
        self.exclude = exclude

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by process ID."""
        if self.exclude:
            return entry.process_id not in self.process_ids
        else:
            return entry.process_id in self.process_ids


class MetadataFilter(LogFilter):
    """Filter logs by metadata."""

    def __init__(self, metadata_key: str, metadata_value: Any, exclude: bool = False):
        self.metadata_key = metadata_key
        self.metadata_value = metadata_value
        self.exclude = exclude

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by metadata."""
        if not entry.context or not entry.context.metadata:
            return not self.exclude

        actual_value = entry.context.metadata.get(self.metadata_key)

        if self.exclude:
            return actual_value != self.metadata_value
        else:
            return actual_value == self.metadata_value


class ExtraFilter(LogFilter):
    """Filter logs by extra data."""

    def __init__(self, extra_key: str, extra_value: Any, exclude: bool = False):
        self.extra_key = extra_key
        self.extra_value = extra_value
        self.exclude = exclude

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by extra data."""
        if not entry.extra:
            return not self.exclude

        actual_value = entry.extra.get(self.extra_key)

        if self.exclude:
            return actual_value != self.extra_value
        else:
            return actual_value == self.extra_value


class RateLimitFilter(LogFilter):
    """Rate limit filter to prevent log spam."""

    def __init__(self, max_per_second: float = 10.0, window_size: int = 100):
        self.max_per_second = max_per_second
        self.window_size = window_size
        self.timestamps = []
        self._lock = threading.RLock()

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by rate limit."""
        with self._lock:
            current_time = time.time()

            # Remove old timestamps
            self.timestamps = [ts for ts in self.timestamps if current_time - ts < 1.0]

            # Check if we're under the rate limit
            if len(self.timestamps) < self.max_per_second:
                self.timestamps.append(current_time)
                return True
            else:
                return False


class SamplingFilter(LogFilter):
    """Sampling filter to reduce log volume."""

    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate
        self._lock = threading.RLock()
        self._counter = 0

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by sampling rate."""
        with self._lock:
            self._counter += 1
            return (self._counter * self.sample_rate) % 1 < self.sample_rate


class DuplicateFilter(LogFilter):
    """Filter to remove duplicate log entries."""

    def __init__(self, window_size: int = 100, time_window: float = 60.0):
        self.window_size = window_size
        self.time_window = time_window
        self.recent_entries = []
        self._lock = threading.RLock()

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by duplicate detection."""
        with self._lock:
            current_time = time.time()

            # Remove old entries
            self.recent_entries = [
                (ts, msg)
                for ts, msg in self.recent_entries
                if current_time - ts < self.time_window
            ]

            # Check for duplicates
            entry_key = (entry.level.value, entry.message, entry.logger_name)

            for ts, (level, message, logger) in self.recent_entries:
                if (level, message, logger) == entry_key:
                    return False  # Duplicate found

            # Add to recent entries
            self.recent_entries.append((current_time, entry_key))

            # Limit window size
            if len(self.recent_entries) > self.window_size:
                self.recent_entries.pop(0)

            return True


class SensitiveDataFilter(LogFilter):
    """Filter to remove sensitive data from logs."""

    def __init__(self, sensitive_patterns: List[str]):
        self.sensitive_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in sensitive_patterns
        ]

    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry by sensitive data detection."""
        # Check message for sensitive patterns
        for pattern in self.sensitive_patterns:
            if pattern.search(entry.message):
                return False  # Block entry with sensitive data

        # Check context metadata for sensitive patterns
        if entry.context and entry.context.metadata:
            for value in entry.context.metadata.values():
                if isinstance(value, str):
                    for pattern in self.sensitive_patterns:
                        if pattern.search(value):
                            return False  # Block entry with sensitive data

        # Check extra data for sensitive patterns
        if entry.extra:
            for value in entry.extra.values():
                if isinstance(value, str):
                    for pattern in self.sensitive_patterns:
                        if pattern.search(value):
                            return False  # Block entry with sensitive data

        return True
