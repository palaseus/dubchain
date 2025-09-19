"""Tests for logging core module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.logging.core import (
    DubChainLogger,
    LogConfig,
    LogContext,
    LogEntry,
    LogLevel,
    LogManager,
)


class TestLogLevel:
    """Test LogLevel enum."""

    def test_log_level_values(self):
        """Test log level values."""
        assert LogLevel.TRACE.value == "trace"
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"
        assert LogLevel.FATAL.value == "fatal"

    def test_log_level_comparison(self):
        """Test log level comparison."""
        # LogLevel doesn't support ordering comparison, test equality instead
        assert LogLevel.TRACE != LogLevel.DEBUG
        assert LogLevel.DEBUG != LogLevel.INFO
        assert LogLevel.INFO != LogLevel.WARNING
        assert LogLevel.WARNING != LogLevel.ERROR


class TestLogContext:
    """Test LogContext functionality."""

    def test_log_context_creation(self):
        """Test creating log context."""
        context = LogContext(
            node_id="node1",
            component="test_component",
            operation="test_operation",
            user_id="user123",
            request_id="req456",
        )

        assert context.node_id == "node1"
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.user_id == "user123"
        assert context.request_id == "req456"

    def test_log_context_defaults(self):
        """Test log context defaults."""
        context = LogContext()

        assert context.node_id is None
        assert context.component is None
        assert context.operation is None
        assert context.user_id is None
        assert context.request_id is None
        assert isinstance(context.metadata, dict)


class TestLogEntry:
    """Test LogEntry functionality."""

    def test_log_entry_creation(self):
        """Test creating log entry."""
        context = LogContext(component="test")
        entry = LogEntry(
            timestamp=1234567890.123,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )

        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.logger_name == "test.logger"
        assert entry.timestamp == 1234567890.123
        assert entry.context == context

    def test_log_entry_to_dict(self):
        """Test log entry to dict conversion."""
        context = LogContext(component="test")
        entry = LogEntry(
            timestamp=1234567890.123,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )

        entry_dict = entry.to_dict()
        assert entry_dict["level"] == "info"
        assert entry_dict["message"] == "Test message"
        assert entry_dict["logger_name"] == "test.logger"


class TestLogConfig:
    """Test LogConfig functionality."""

    def test_log_config_creation(self):
        """Test creating log config."""
        config = LogConfig(
            name="test_logger",
            level=LogLevel.DEBUG,
            format_type="json",
            handlers=["console", "file"],
            filters=["level_filter"],
        )

        assert config.name == "test_logger"
        assert config.level == LogLevel.DEBUG
        assert config.format_type == "json"
        assert config.handlers == ["console", "file"]
        assert config.filters == ["level_filter"]

    def test_log_config_defaults(self):
        """Test log config defaults."""
        config = LogConfig()

        assert config.name == "dubchain"
        assert config.level == LogLevel.INFO
        assert config.format_type == "json"
        assert config.handlers == ["console"]
        assert config.filters == []
        assert config.propagate is False


class TestDubChainLogger:
    """Test DubChainLogger functionality."""

    @pytest.fixture
    def log_config(self):
        """Fixture for log configuration."""
        return LogConfig(level=LogLevel.DEBUG)

    @pytest.fixture
    def log_manager(self, log_config):
        """Fixture for log manager."""
        return LogManager(log_config)

    @pytest.fixture
    def logger(self, log_manager):
        """Fixture for logger."""
        return DubChainLogger("test.logger", log_manager)

    def test_logger_creation(self, log_manager):
        """Test creating logger."""
        logger = DubChainLogger("test.logger", log_manager)

        assert logger.name == "test.logger"
        assert logger.manager == log_manager
        assert logger.level == LogLevel.INFO

    def test_logger_debug(self, logger):
        """Test logger debug method."""
        # Set logger level to DEBUG to enable debug logging
        logger.set_level(LogLevel.DEBUG)
        with patch.object(logger.manager, "log") as mock_log:
            logger.debug("Debug message")
            mock_log.assert_called_once()

    def test_logger_info(self, logger):
        """Test logger info method."""
        with patch.object(logger.manager, "log") as mock_log:
            logger.info("Info message")
            mock_log.assert_called_once()

    def test_logger_warning(self, logger):
        """Test logger warning method."""
        with patch.object(logger.manager, "log") as mock_log:
            logger.warning("Warning message")
            mock_log.assert_called_once()

    def test_logger_error(self, logger):
        """Test logger error method."""
        with patch.object(logger.manager, "log") as mock_log:
            logger.error("Error message")
            mock_log.assert_called_once()

    def test_logger_critical(self, logger):
        """Test logger critical method."""
        with patch.object(logger.manager, "log") as mock_log:
            logger.critical("Critical message")
            mock_log.assert_called_once()

    def test_logger_is_enabled_for(self, logger):
        """Test logger level checking."""
        logger.set_level(LogLevel.WARNING)

        assert logger.is_enabled_for(LogLevel.WARNING) is True
        assert logger.is_enabled_for(LogLevel.ERROR) is True
        assert logger.is_enabled_for(LogLevel.INFO) is False
        assert logger.is_enabled_for(LogLevel.DEBUG) is False


class TestLogManager:
    """Test LogManager functionality."""

    @pytest.fixture
    def log_config(self):
        """Fixture for log configuration."""
        return LogConfig(level=LogLevel.INFO)

    @pytest.fixture
    def log_manager(self, log_config):
        """Fixture for log manager."""
        return LogManager(log_config)

    def test_log_manager_creation(self, log_config):
        """Test creating log manager."""
        manager = LogManager(log_config)

        assert manager.config == log_config
        assert isinstance(manager.loggers, dict)
        assert isinstance(manager.handlers, dict)
        assert isinstance(manager.formatters, dict)

    def test_get_logger(self, log_manager):
        """Test getting logger."""
        logger1 = log_manager.get_logger("test.logger1")
        logger2 = log_manager.get_logger("test.logger2")
        logger1_again = log_manager.get_logger("test.logger1")

        assert logger1.name == "test.logger1"
        assert logger2.name == "test.logger2"
        assert logger1 is logger1_again  # Should return same instance
        assert logger1 is not logger2

    def test_add_handler(self, log_manager):
        """Test adding handler."""
        mock_handler = Mock()
        log_manager.add_handler("test_handler", mock_handler)

        assert "test_handler" in log_manager.handlers
        assert log_manager.handlers["test_handler"] == mock_handler

    def test_add_formatter(self, log_manager):
        """Test adding formatter."""
        mock_formatter = Mock()
        log_manager.add_formatter("test_formatter", mock_formatter)

        assert "test_formatter" in log_manager.formatters
        assert log_manager.formatters["test_formatter"] == mock_formatter

    def test_set_context(self, log_manager):
        """Test setting global context."""
        context = LogContext(component="test_component")
        log_manager.set_context(context)

        retrieved_context = log_manager.get_context()
        assert retrieved_context.component == "test_component"
