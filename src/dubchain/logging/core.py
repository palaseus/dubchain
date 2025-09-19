"""Core logging interfaces and data structures for DubChain.

This module defines the fundamental logging interfaces, data structures,
and configuration options for the DubChain logging system.
"""

import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class LogLevel(Enum):
    """Log levels."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class LogContext:
    """Log context information."""

    node_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "node_id": self.node_id,
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata,
        }


@dataclass
class LogEntry:
    """Log entry data structure."""

    timestamp: float
    level: LogLevel
    message: str
    logger_name: str
    context: LogContext
    exception: Optional[Exception] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    thread_id: Optional[int] = None
    process_id: Optional[int] = None

    def __post_init__(self):
        if self.thread_id is None:
            self.thread_id = threading.get_ident()
        if self.process_id is None:
            import os

            self.process_id = os.getpid()

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "message": self.message,
            "logger_name": self.logger_name,
            "context": self.context.to_dict(),
            "exception": str(self.exception) if self.exception else None,
            "extra": self.extra,
            "thread_id": self.thread_id,
            "process_id": self.process_id,
        }

    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class LogConfig:
    """Log configuration."""

    def __init__(
        self,
        name: str = "dubchain",
        level: LogLevel = LogLevel.INFO,
        format_type: str = "json",
        handlers: List[str] = None,
        filters: List[str] = None,
        propagate: bool = False,
        disable_existing_loggers: bool = False,
    ):
        self.name = name
        self.level = level
        self.format_type = format_type
        self.handlers = handlers or ["console"]
        self.filters = filters or []
        self.propagate = propagate
        self.disable_existing_loggers = disable_existing_loggers

        # Handler configurations
        self.handler_configs: Dict[str, Dict[str, Any]] = {}

        # Formatter configurations
        self.formatter_configs: Dict[str, Dict[str, Any]] = {}

        # Filter configurations
        self.filter_configs: Dict[str, Dict[str, Any]] = {}

    def add_handler_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add handler configuration."""
        self.handler_configs[name] = config

    def add_formatter_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add formatter configuration."""
        self.formatter_configs[name] = config

    def add_filter_config(self, name: str, config: Dict[str, Any]) -> None:
        """Add filter configuration."""
        self.filter_configs[name] = config


class LogFilter(ABC):
    """Abstract log filter."""

    @abstractmethod
    def filter(self, entry: LogEntry) -> bool:
        """Filter log entry. Return True to allow, False to block."""
        pass


class LogFormatter(ABC):
    """Abstract log formatter."""

    @abstractmethod
    def format(self, entry: LogEntry) -> str:
        """Format log entry."""
        pass


class LogHandler(ABC):
    """Abstract log handler."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.formatter: Optional[LogFormatter] = None
        self.filters: List[LogFilter] = []
        self.level: LogLevel = LogLevel.DEBUG
        self._lock = threading.RLock()

    def set_formatter(self, formatter: LogFormatter) -> None:
        """Set formatter."""
        with self._lock:
            self.formatter = formatter

    def add_filter(self, filter_obj: LogFilter) -> None:
        """Add filter."""
        with self._lock:
            self.filters.append(filter_obj)

    def remove_filter(self, filter_obj: LogFilter) -> None:
        """Remove filter."""
        with self._lock:
            if filter_obj in self.filters:
                self.filters.remove(filter_obj)

    def set_level(self, level: LogLevel) -> None:
        """Set log level."""
        with self._lock:
            self.level = level

    def should_handle(self, entry: LogEntry) -> bool:
        """Check if handler should handle the entry."""
        with self._lock:
            # Check level - define level hierarchy
            level_hierarchy = {
                LogLevel.TRACE: 0,
                LogLevel.DEBUG: 1,
                LogLevel.INFO: 2,
                LogLevel.WARNING: 3,
                LogLevel.ERROR: 4,
                LogLevel.CRITICAL: 5,
                LogLevel.FATAL: 6,
            }

            if level_hierarchy.get(entry.level, 0) < level_hierarchy.get(self.level, 0):
                return False

            # Check filters
            for filter_obj in self.filters:
                if not filter_obj.filter(entry):
                    return False

            return True

    @abstractmethod
    def emit(self, entry: LogEntry) -> None:
        """Emit log entry."""
        pass

    def handle(self, entry: LogEntry) -> None:
        """Handle log entry."""
        if self.should_handle(entry):
            self.emit(entry)


class LogProcessor(ABC):
    """Abstract log processor."""

    @abstractmethod
    def process(self, entry: LogEntry) -> LogEntry:
        """Process log entry."""
        pass


class LogManager:
    """Log manager for orchestrating logging operations."""

    def __init__(self, config: LogConfig = None):
        self.config = config or LogConfig()
        self.loggers: Dict[str, "DubChainLogger"] = {}
        self.handlers: Dict[str, LogHandler] = {}
        self.formatters: Dict[str, LogFormatter] = {}
        self.filters: Dict[str, LogFilter] = {}
        self.processors: List[LogProcessor] = []
        self._lock = threading.RLock()
        self._context = LogContext()

        # Setup default components
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Setup default logging components."""
        # Default formatter
        from .formatters import JSONFormatter

        self.add_formatter("json", JSONFormatter())

        # Default handler
        from .handlers import ConsoleHandler

        self.add_handler("console", ConsoleHandler())

    def add_logger(self, name: str, logger: "DubChainLogger") -> None:
        """Add logger."""
        with self._lock:
            self.loggers[name] = logger

    def get_logger(self, name: str) -> "DubChainLogger":
        """Get logger."""
        with self._lock:
            if name not in self.loggers:
                self.loggers[name] = DubChainLogger(name, self)
            return self.loggers[name]

    def add_handler(self, name: str, handler: LogHandler) -> None:
        """Add handler."""
        with self._lock:
            self.handlers[name] = handler

    def remove_handler(self, name: str) -> None:
        """Remove handler."""
        with self._lock:
            if name in self.handlers:
                del self.handlers[name]

    def add_formatter(self, name: str, formatter: LogFormatter) -> None:
        """Add formatter."""
        with self._lock:
            self.formatters[name] = formatter

    def remove_formatter(self, name: str) -> None:
        """Remove formatter."""
        with self._lock:
            if name in self.formatters:
                del self.formatters[name]

    def add_filter(self, name: str, filter_obj: LogFilter) -> None:
        """Add filter."""
        with self._lock:
            self.filters[name] = filter_obj

    def remove_filter(self, name: str) -> None:
        """Remove filter."""
        with self._lock:
            if name in self.filters:
                del self.filters[name]

    def add_processor(self, processor: LogProcessor) -> None:
        """Add processor."""
        with self._lock:
            self.processors.append(processor)

    def remove_processor(self, processor: LogProcessor) -> None:
        """Remove processor."""
        with self._lock:
            if processor in self.processors:
                self.processors.remove(processor)

    def set_context(self, context: LogContext) -> None:
        """Set global context."""
        with self._lock:
            self._context = context

    def get_context(self) -> LogContext:
        """Get global context."""
        with self._lock:
            return self._context

    def log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "root",
        context: LogContext = None,
        exception: Exception = None,
        extra: Dict[str, Any] = None,
    ) -> None:
        """Log a message."""
        with self._lock:
            # Merge contexts
            if context is None:
                context = self._context
            else:
                # Merge with global context
                merged_context = LogContext()
                merged_context.node_id = context.node_id or self._context.node_id
                merged_context.component = context.component or self._context.component
                merged_context.operation = context.operation or self._context.operation
                merged_context.user_id = context.user_id or self._context.user_id
                merged_context.request_id = (
                    context.request_id or self._context.request_id
                )
                merged_context.session_id = (
                    context.session_id or self._context.session_id
                )
                merged_context.correlation_id = (
                    context.correlation_id or self._context.correlation_id
                )
                merged_context.trace_id = context.trace_id or self._context.trace_id
                merged_context.span_id = context.span_id or self._context.span_id
                merged_context.metadata = {**self._context.metadata, **context.metadata}
                context = merged_context

            # Create log entry
            entry = LogEntry(
                timestamp=time.time(),
                level=level,
                message=message,
                logger_name=logger_name,
                context=context,
                exception=exception,
                extra=extra or {},
            )

            # Process entry
            for processor in self.processors:
                entry = processor.process(entry)

            # Handle entry
            for handler_name in self.config.handlers:
                if handler_name in self.handlers:
                    self.handlers[handler_name].handle(entry)

    def shutdown(self) -> None:
        """Shutdown log manager."""
        with self._lock:
            # Close all handlers
            for handler in self.handlers.values():
                if hasattr(handler, "close"):
                    handler.close()

            # Clear all components
            self.loggers.clear()
            self.handlers.clear()
            self.formatters.clear()
            self.filters.clear()
            self.processors.clear()


class DubChainLogger:
    """DubChain logger implementation."""

    def __init__(self, name: str, manager: LogManager):
        self.name = name
        self.manager = manager
        self.level = LogLevel.INFO
        self._lock = threading.RLock()

    def set_level(self, level: LogLevel) -> None:
        """Set log level."""
        with self._lock:
            self.level = level

    def is_enabled_for(self, level: LogLevel) -> bool:
        """Check if logger is enabled for level."""
        with self._lock:
            level_values = {l.value: i for i, l in enumerate(LogLevel)}
            entry_level = level_values.get(level.value, 0)
            logger_level = level_values.get(self.level.value, 0)
            return entry_level >= logger_level

    def log(
        self,
        level: LogLevel,
        message: str,
        context: LogContext = None,
        exception: Exception = None,
        extra: Dict[str, Any] = None,
    ) -> None:
        """Log a message."""
        if self.is_enabled_for(level):
            self.manager.log(
                level=level,
                message=message,
                logger_name=self.name,
                context=context,
                exception=exception,
                extra=extra,
            )

    def trace(self, message: str, **kwargs) -> None:
        """Log trace message."""
        self.log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def fatal(self, message: str, **kwargs) -> None:
        """Log fatal message."""
        self.log(LogLevel.FATAL, message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception message."""
        import sys

        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            kwargs["exception"] = exc_info[1]
        self.log(LogLevel.ERROR, message, **kwargs)


# Global log manager instance
_global_manager: Optional[LogManager] = None


def get_logger(name: str = "root") -> DubChainLogger:
    """Get logger instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = LogManager()
    return _global_manager.get_logger(name)


def setup_logging(config: LogConfig) -> LogManager:
    """Setup logging with configuration."""
    global _global_manager
    _global_manager = LogManager(config)
    return _global_manager


def shutdown_logging() -> None:
    """Shutdown logging."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.shutdown()
        _global_manager = None
