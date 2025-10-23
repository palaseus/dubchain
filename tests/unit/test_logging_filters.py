"""
Tests for logging filters module.
"""

import logging

logger = logging.getLogger(__name__)
import re
import threading
import time
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from src.dubchain.logging.core import LogContext, LogEntry, LogLevel
from src.dubchain.logging.filters import (
    CompositeFilter,
    ContextFilter,
    CustomFilter,
    DuplicateFilter,
    ExceptionFilter,
    ExtraFilter,
    LevelFilter,
    LoggerFilter,
    MessageFilter,
    MetadataFilter,
    ProcessFilter,
    RateLimitFilter,
    RegexFilter,
    SamplingFilter,
    SensitiveDataFilter,
    ThreadFilter,
    TimeFilter,
)


class TestLevelFilter:
    """Test LevelFilter class."""

    def test_level_filter_initialization(self):
        """Test level filter initialization."""
        filter_obj = LevelFilter(LogLevel.INFO)

        assert filter_obj.min_level == LogLevel.INFO
        assert filter_obj.max_level == LogLevel.FATAL

    def test_level_filter_custom_max_level(self):
        """Test level filter with custom max level."""
        filter_obj = LevelFilter(LogLevel.INFO, LogLevel.ERROR)

        assert filter_obj.min_level == LogLevel.INFO
        assert filter_obj.max_level == LogLevel.ERROR

    def test_level_filter_allow_level(self):
        """Test level filter allows appropriate levels."""
        filter_obj = LevelFilter(LogLevel.INFO)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_level_filter_block_level(self):
        """Test level filter blocks inappropriate levels."""
        filter_obj = LevelFilter(LogLevel.INFO)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.DEBUG,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False

    def test_level_filter_range(self):
        """Test level filter with range."""
        filter_obj = LevelFilter(LogLevel.INFO, LogLevel.ERROR)

        # Should allow INFO
        entry1 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )
        assert filter_obj.filter(entry1) is True

        # Should allow ERROR
        entry2 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.ERROR,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )
        assert filter_obj.filter(entry2) is True

        # Should block DEBUG
        entry3 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.DEBUG,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )
        assert filter_obj.filter(entry3) is False

        # Should block FATAL
        entry4 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.FATAL,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )
        assert filter_obj.filter(entry4) is False


class TestContextFilter:
    """Test ContextFilter class."""

    def test_context_filter_initialization(self):
        """Test context filter initialization."""
        filter_obj = ContextFilter(
            node_id="node_001", component="test_component", operation="test_operation"
        )

        assert filter_obj.node_id == "node_001"
        assert filter_obj.component == "test_component"
        assert filter_obj.operation == "test_operation"

    def test_context_filter_match(self):
        """Test context filter matches correctly."""
        filter_obj = ContextFilter(node_id="node_001", component="test_component")

        context = LogContext(
            node_id="node_001", component="test_component", operation="test_operation"
        )

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_context_filter_no_match(self):
        """Test context filter doesn't match incorrectly."""
        filter_obj = ContextFilter(node_id="node_001", component="test_component")

        context = LogContext(
            node_id="node_002", component="test_component"  # Different node
        )

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False

    def test_context_filter_no_context(self):
        """Test context filter with no context."""
        filter_obj = ContextFilter(node_id="node_001", component="test_component")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=None,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False


class TestRegexFilter:
    """Test RegexFilter class."""

    def test_regex_filter_initialization(self):
        """Test regex filter initialization."""
        filter_obj = RegexFilter(r"test.*message", "message")

        assert filter_obj.pattern.pattern == r"test.*message"
        assert filter_obj.field == "message"

    def test_regex_filter_message_match(self):
        """Test regex filter matches message."""
        filter_obj = RegexFilter(r"test.*message", "message")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is a test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_regex_filter_message_no_match(self):
        """Test regex filter doesn't match message."""
        filter_obj = RegexFilter(r"test.*message", "message")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is not a match",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False

    def test_regex_filter_logger_match(self):
        """Test regex filter matches logger name."""
        filter_obj = RegexFilter(r"test.*", "logger")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_regex_filter_component_match(self):
        """Test regex filter matches component."""
        filter_obj = RegexFilter(r"test.*", "component")

        context = LogContext(component="test_component")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_regex_filter_case_insensitive(self):
        """Test regex filter with case insensitive flag."""
        filter_obj = RegexFilter(r"TEST.*MESSAGE", "message", re.IGNORECASE)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is a test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True


class TestTimeFilter:
    """Test TimeFilter class."""

    def test_time_filter_initialization(self):
        """Test time filter initialization."""
        start_time = time.time() - 3600  # 1 hour ago
        end_time = time.time() + 3600  # 1 hour from now

        filter_obj = TimeFilter(start_time, end_time)

        assert filter_obj.start_time == start_time
        assert filter_obj.end_time == end_time

    def test_time_filter_within_range(self):
        """Test time filter allows entries within range."""
        current_time = time.time()
        start_time = current_time - 3600
        end_time = current_time + 3600

        filter_obj = TimeFilter(start_time, end_time)

        entry = LogEntry(
            timestamp=current_time,
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_time_filter_outside_range(self):
        """Test time filter blocks entries outside range."""
        current_time = time.time()
        start_time = current_time + 3600  # Future start time
        end_time = current_time + 7200  # Future end time

        filter_obj = TimeFilter(start_time, end_time)

        entry = LogEntry(
            timestamp=current_time,
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False


class TestCustomFilter:
    """Test CustomFilter class."""

    def test_custom_filter_initialization(self):
        """Test custom filter initialization."""

        def custom_func(entry):
            return entry.level == LogLevel.ERROR

        filter_obj = CustomFilter(custom_func)

        assert filter_obj.filter_func == custom_func

    def test_custom_filter_function(self):
        """Test custom filter with function."""

        def custom_func(entry):
            return "error" in entry.message.lower()

        filter_obj = CustomFilter(custom_func)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is an error message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_custom_filter_exception_handling(self):
        """Test custom filter handles exceptions."""

        def custom_func(entry):
            raise ValueError("Test error")

        filter_obj = CustomFilter(custom_func)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        # Should return False when exception occurs
        assert filter_obj.filter(entry) is False


class TestCompositeFilter:
    """Test CompositeFilter class."""

    def test_composite_filter_initialization(self):
        """Test composite filter initialization."""
        filters = [
            LevelFilter(LogLevel.INFO),
            ContextFilter(component="test_component"),
        ]

        filter_obj = CompositeFilter(filters, "AND")

        assert len(filter_obj.filters) == 2
        assert filter_obj.operator == "AND"

    def test_composite_filter_and_operator(self):
        """Test composite filter with AND operator."""
        filters = [
            LevelFilter(LogLevel.INFO),
            ContextFilter(component="test_component"),
        ]

        filter_obj = CompositeFilter(filters, "AND")

        context = LogContext(component="test_component")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_composite_filter_or_operator(self):
        """Test composite filter with OR operator."""
        filters = [
            LevelFilter(LogLevel.ERROR),
            ContextFilter(component="test_component"),
        ]

        filter_obj = CompositeFilter(filters, "OR")

        context = LogContext(component="test_component")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,  # Doesn't match level filter
            logger_name="test_logger",
            message="Test message",
            context=context,  # Matches context filter
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_composite_filter_no_filters(self):
        """Test composite filter with no filters."""
        filter_obj = CompositeFilter([], "AND")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_composite_filter_invalid_operator(self):
        """Test composite filter with invalid operator."""
        filters = [LevelFilter(LogLevel.INFO)]

        filter_obj = CompositeFilter(filters, "INVALID")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        with pytest.raises(ValueError):
            filter_obj.filter(entry)


class TestLoggerFilter:
    """Test LoggerFilter class."""

    def test_logger_filter_initialization(self):
        """Test logger filter initialization."""
        filter_obj = LoggerFilter(["logger1", "logger2"])

        assert filter_obj.logger_names == ["logger1", "logger2"]
        assert filter_obj.exclude is False

    def test_logger_filter_include(self):
        """Test logger filter includes specified loggers."""
        filter_obj = LoggerFilter(["logger1", "logger2"])

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="logger1",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_logger_filter_exclude(self):
        """Test logger filter excludes specified loggers."""
        filter_obj = LoggerFilter(["logger1", "logger2"], exclude=True)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="logger3",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True


class TestMessageFilter:
    """Test MessageFilter class."""

    def test_message_filter_initialization(self):
        """Test message filter initialization."""
        filter_obj = MessageFilter(["error", "warning"])

        assert filter_obj.keywords == ["error", "warning"]
        assert filter_obj.exclude is False
        assert filter_obj.case_sensitive is False

    def test_message_filter_include(self):
        """Test message filter includes messages with keywords."""
        filter_obj = MessageFilter(["error", "warning"])

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is an error message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_message_filter_exclude(self):
        """Test message filter excludes messages with keywords."""
        filter_obj = MessageFilter(["error", "warning"], exclude=True)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is an error message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False

    def test_message_filter_case_sensitive(self):
        """Test message filter with case sensitivity."""
        filter_obj = MessageFilter(["ERROR"], case_sensitive=True)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is an error message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False


class TestExceptionFilter:
    """Test ExceptionFilter class."""

    def test_exception_filter_initialization(self):
        """Test exception filter initialization."""
        filter_obj = ExceptionFilter([ValueError, TypeError])

        assert filter_obj.exception_types == [ValueError, TypeError]
        assert filter_obj.exclude is False

    def test_exception_filter_include(self):
        """Test exception filter includes specified exception types."""
        filter_obj = ExceptionFilter([ValueError, TypeError])

        try:
            raise ValueError("Test error")
        except ValueError as e:
            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.ERROR,
                logger_name="test_logger",
                message="Test error message",
                context=LogContext(),
                thread_id=12345,
                process_id=67890,
                exception=e,
            )

            assert filter_obj.filter(entry) is True

    def test_exception_filter_exclude(self):
        """Test exception filter excludes specified exception types."""
        filter_obj = ExceptionFilter([ValueError, TypeError], exclude=True)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.ERROR,
                logger_name="test_logger",
                message="Test error message",
                context=LogContext(),
                thread_id=12345,
                process_id=67890,
                exception=e,
            )

            assert filter_obj.filter(entry) is False

    def test_exception_filter_no_exception(self):
        """Test exception filter with no exception."""
        filter_obj = ExceptionFilter([ValueError, TypeError])

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True


class TestThreadFilter:
    """Test ThreadFilter class."""

    def test_thread_filter_initialization(self):
        """Test thread filter initialization."""
        filter_obj = ThreadFilter([12345, 67890])

        assert filter_obj.thread_ids == [12345, 67890]
        assert filter_obj.exclude is False

    def test_thread_filter_include(self):
        """Test thread filter includes specified thread IDs."""
        filter_obj = ThreadFilter([12345, 67890])

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_thread_filter_exclude(self):
        """Test thread filter excludes specified thread IDs."""
        filter_obj = ThreadFilter([12345, 67890], exclude=True)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=54321,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True


class TestProcessFilter:
    """Test ProcessFilter class."""

    def test_process_filter_initialization(self):
        """Test process filter initialization."""
        filter_obj = ProcessFilter([12345, 67890])

        assert filter_obj.process_ids == [12345, 67890]
        assert filter_obj.exclude is False

    def test_process_filter_include(self):
        """Test process filter includes specified process IDs."""
        filter_obj = ProcessFilter([12345, 67890])

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_process_filter_exclude(self):
        """Test process filter excludes specified process IDs."""
        filter_obj = ProcessFilter([12345, 67890], exclude=True)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=54321,
        )

        assert filter_obj.filter(entry) is True


class TestMetadataFilter:
    """Test MetadataFilter class."""

    def test_metadata_filter_initialization(self):
        """Test metadata filter initialization."""
        filter_obj = MetadataFilter("key1", "value1")

        assert filter_obj.metadata_key == "key1"
        assert filter_obj.metadata_value == "value1"
        assert filter_obj.exclude is False

    def test_metadata_filter_include(self):
        """Test metadata filter includes entries with matching metadata."""
        filter_obj = MetadataFilter("key1", "value1")

        context = LogContext(metadata={"key1": "value1", "key2": "value2"})

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_metadata_filter_exclude(self):
        """Test metadata filter excludes entries with matching metadata."""
        filter_obj = MetadataFilter("key1", "value1", exclude=True)

        context = LogContext(metadata={"key1": "value1", "key2": "value2"})

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False

    def test_metadata_filter_no_metadata(self):
        """Test metadata filter with no metadata."""
        filter_obj = MetadataFilter("key1", "value1")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True


class TestExtraFilter:
    """Test ExtraFilter class."""

    def test_extra_filter_initialization(self):
        """Test extra filter initialization."""
        filter_obj = ExtraFilter("key1", "value1")

        assert filter_obj.extra_key == "key1"
        assert filter_obj.extra_value == "value1"
        assert filter_obj.exclude is False

    def test_extra_filter_include(self):
        """Test extra filter includes entries with matching extra data."""
        filter_obj = ExtraFilter("key1", "value1")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
            extra={"key1": "value1", "key2": "value2"},
        )

        assert filter_obj.filter(entry) is True

    def test_extra_filter_exclude(self):
        """Test extra filter excludes entries with matching extra data."""
        filter_obj = ExtraFilter("key1", "value1", exclude=True)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
            extra={"key1": "value1", "key2": "value2"},
        )

        assert filter_obj.filter(entry) is False

    def test_extra_filter_no_extra(self):
        """Test extra filter with no extra data."""
        filter_obj = ExtraFilter("key1", "value1")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True


class TestRateLimitFilter:
    """Test RateLimitFilter class."""

    def test_rate_limit_filter_initialization(self):
        """Test rate limit filter initialization."""
        filter_obj = RateLimitFilter(max_per_second=5.0, window_size=50)

        assert filter_obj.max_per_second == 5.0
        assert filter_obj.window_size == 50
        assert len(filter_obj.timestamps) == 0

    def test_rate_limit_filter_under_limit(self):
        """Test rate limit filter allows entries under limit."""
        filter_obj = RateLimitFilter(max_per_second=10.0)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True
        assert len(filter_obj.timestamps) == 1

    def test_rate_limit_filter_over_limit(self):
        """Test rate limit filter blocks entries over limit."""
        filter_obj = RateLimitFilter(max_per_second=1.0)

        # Add multiple entries quickly
        for i in range(5):
            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test_logger",
                message=f"Test message {i}",
                context=LogContext(),
                thread_id=12345,
                process_id=67890,
            )

            result = filter_obj.filter(entry)
            if i < 1:  # First entry should pass
                assert result is True
            else:  # Subsequent entries should be blocked
                assert result is False


class TestSamplingFilter:
    """Test SamplingFilter class."""

    def test_sampling_filter_initialization(self):
        """Test sampling filter initialization."""
        filter_obj = SamplingFilter(sample_rate=0.1)

        assert filter_obj.sample_rate == 0.1
        assert filter_obj._counter == 0

    def test_sampling_filter_sampling(self):
        """Test sampling filter samples entries."""
        filter_obj = SamplingFilter(sample_rate=0.5)

        results = []
        for i in range(10):
            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test_logger",
                message=f"Test message {i}",
                context=LogContext(),
                thread_id=12345,
                process_id=67890,
            )

            results.append(filter_obj.filter(entry))

        # Should have some True and some False results
        assert any(results)
        assert not all(results)


class TestDuplicateFilter:
    """Test DuplicateFilter class."""

    def test_duplicate_filter_initialization(self):
        """Test duplicate filter initialization."""
        filter_obj = DuplicateFilter(window_size=50, time_window=60.0)

        assert filter_obj.window_size == 50
        assert filter_obj.time_window == 60.0
        assert len(filter_obj.recent_entries) == 0

    def test_duplicate_filter_unique_entries(self):
        """Test duplicate filter allows unique entries."""
        filter_obj = DuplicateFilter()

        entry1 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message 1",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        entry2 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message 2",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry1) is True
        assert filter_obj.filter(entry2) is True

    def test_duplicate_filter_duplicate_entries(self):
        """Test duplicate filter blocks duplicate entries."""
        filter_obj = DuplicateFilter()

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        # First entry should pass
        assert filter_obj.filter(entry) is True

        # Duplicate entry should be blocked
        assert filter_obj.filter(entry) is False


class TestSensitiveDataFilter:
    """Test SensitiveDataFilter class."""

    def test_sensitive_data_filter_initialization(self):
        """Test sensitive data filter initialization."""
        patterns = [r"password", r"secret", r"token"]
        filter_obj = SensitiveDataFilter(patterns)

        assert len(filter_obj.sensitive_patterns) == 3

    def test_sensitive_data_filter_clean_message(self):
        """Test sensitive data filter allows clean messages."""
        patterns = [r"password", r"secret", r"token"]
        filter_obj = SensitiveDataFilter(patterns)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="This is a normal message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is True

    def test_sensitive_data_filter_sensitive_message(self):
        """Test sensitive data filter blocks sensitive messages."""
        patterns = [r"password", r"secret", r"token"]
        filter_obj = SensitiveDataFilter(patterns)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="User password is 123456",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False

    def test_sensitive_data_filter_sensitive_metadata(self):
        """Test sensitive data filter blocks sensitive metadata."""
        patterns = [r"password", r"secret", r"token"]
        filter_obj = SensitiveDataFilter(patterns)

        context = LogContext(metadata={"user": "john", "key": "secret_value"})

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Normal message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        assert filter_obj.filter(entry) is False

    def test_sensitive_data_filter_sensitive_extra(self):
        """Test sensitive data filter blocks sensitive extra data."""
        patterns = [r"password", r"secret", r"token"]
        filter_obj = SensitiveDataFilter(patterns)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Normal message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
            extra={"key": "token_value"},
        )

        assert filter_obj.filter(entry) is False


class TestFilterIntegration:
    """Test filter integration and edge cases."""

    def test_all_filters_with_same_entry(self):
        """Test all filters with the same log entry."""
        context = LogContext(
            node_id="node_001",
            component="test_component",
            operation="test_operation",
            metadata={"key": "value"},
        )

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
            extra={"key": "value"},
        )

        filters = [
            LevelFilter(LogLevel.INFO),
            ContextFilter(component="test_component"),
            RegexFilter(r"test.*message", "message"),
            TimeFilter(time.time() - 3600, time.time() + 3600),
            LoggerFilter(["test_logger"]),
            MessageFilter(["test"]),
            ThreadFilter([12345]),
            ProcessFilter([67890]),
            MetadataFilter("key", "value"),
            ExtraFilter("key", "value"),
        ]

        for filter_obj in filters:
            result = filter_obj.filter(entry)
            assert isinstance(result, bool)

    def test_filters_with_edge_cases(self):
        """Test filters with edge cases."""
        # Empty context
        entry1 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="",
            message="",
            context=None,
            thread_id=0,
            process_id=0,
        )

        # Empty metadata
        context = LogContext(metadata={})
        entry2 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=context,
            thread_id=12345,
            process_id=67890,
        )

        # Empty extra
        entry3 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test_logger",
            message="Test message",
            context=LogContext(),
            thread_id=12345,
            process_id=67890,
            extra={},
        )

        filters = [
            LevelFilter(LogLevel.INFO),
            ContextFilter(),
            RegexFilter(r".*", "message"),
            TimeFilter(time.time() - 3600, time.time() + 3600),
            LoggerFilter([]),
            MessageFilter([]),
            ThreadFilter([]),
            ProcessFilter([]),
            MetadataFilter("nonexistent", "value"),
            ExtraFilter("nonexistent", "value"),
        ]

        for filter_obj in filters:
            for entry in [entry1, entry2, entry3]:
                result = filter_obj.filter(entry)
                assert isinstance(result, bool)
