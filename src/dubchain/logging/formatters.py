"""Log formatters for DubChain.

This module provides various log formatters including JSON, structured,
text, and custom formatters for the DubChain logging system.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .core import LogEntry, LogFormatter, LogLevel


class JSONFormatter(LogFormatter):
    """JSON log formatter."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_context: bool = True,
        include_exception: bool = True,
        include_extra: bool = True,
        include_thread: bool = True,
        include_process: bool = True,
        timestamp_format: str = "iso",
        indent: Optional[int] = None,
        ensure_ascii: bool = False,
    ):
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_context = include_context
        self.include_exception = include_exception
        self.include_extra = include_extra
        self.include_thread = include_thread
        self.include_process = include_process
        self.timestamp_format = timestamp_format
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def format(self, entry: LogEntry) -> str:
        """Format log entry as JSON."""
        data = {}

        if self.include_timestamp:
            data["timestamp"] = self._format_timestamp(entry.timestamp)

        if self.include_level:
            data["level"] = entry.level.value

        if self.include_logger:
            data["logger"] = entry.logger_name

        if self.include_context:
            data["context"] = entry.context.to_dict()

        if self.include_exception and entry.exception:
            data["exception"] = {
                "type": type(entry.exception).__name__,
                "message": str(entry.exception),
                "traceback": self._get_traceback(entry.exception),
            }

        if self.include_extra and entry.extra:
            data["extra"] = entry.extra

        if self.include_thread:
            data["thread_id"] = entry.thread_id

        if self.include_process:
            data["process_id"] = entry.process_id

        data["message"] = entry.message

        return json.dumps(
            data, indent=self.indent, ensure_ascii=self.ensure_ascii, default=str
        )

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp."""
        if self.timestamp_format == "iso":
            return (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp))
                + f".{int((timestamp % 1) * 1000000):06d}Z"
            )
        elif self.timestamp_format == "unix":
            return str(timestamp)
        else:
            return time.strftime(self.timestamp_format, time.gmtime(timestamp))

    def _get_traceback(self, exception: Exception) -> str:
        """Get traceback for exception."""
        import traceback

        return traceback.format_exc()


class StructuredFormatter(LogFormatter):
    """Structured log formatter."""

    def __init__(
        self,
        format_string: str = None,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_context: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_context = include_context
        self.timestamp_format = timestamp_format
        self.format_string = format_string or self._get_default_format()

    def _get_default_format(self) -> str:
        """Get default format string."""
        parts = []

        if self.include_timestamp:
            parts.append("%(timestamp)s")

        if self.include_level:
            parts.append("[%(level)s]")

        if self.include_logger:
            parts.append("%(logger)s:")

        parts.append("%(message)s")

        if self.include_context:
            parts.append("| %(context)s")

        return " ".join(parts)

    def format(self, entry: LogEntry) -> str:
        """Format log entry."""
        format_data = {
            "timestamp": self._format_timestamp(entry.timestamp),
            "level": entry.level.value.upper(),
            "logger": entry.logger_name,
            "message": entry.message,
            "context": self._format_context(entry.context),
        }

        return self.format_string % format_data

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp."""
        return time.strftime(self.timestamp_format, time.gmtime(timestamp))

    def _format_context(self, context) -> str:
        """Format context."""
        if not context:
            return ""

        parts = []
        if context.node_id:
            parts.append(f"node={context.node_id}")
        if context.component:
            parts.append(f"component={context.component}")
        if context.operation:
            parts.append(f"operation={context.operation}")
        if context.request_id:
            parts.append(f"request={context.request_id}")

        return " ".join(parts)


class TextFormatter(LogFormatter):
    """Text log formatter."""

    def __init__(
        self,
        format_string: str = None,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.timestamp_format = timestamp_format
        self.format_string = format_string or self._get_default_format()

    def _get_default_format(self) -> str:
        """Get default format string."""
        parts = []

        if self.include_timestamp:
            parts.append("%(timestamp)s")

        if self.include_level:
            parts.append("[%(level)s]")

        if self.include_logger:
            parts.append("%(logger)s:")

        parts.append("%(message)s")

        return " ".join(parts)

    def format(self, entry: LogEntry) -> str:
        """Format log entry."""
        format_data = {
            "timestamp": self._format_timestamp(entry.timestamp),
            "level": entry.level.value.upper(),
            "logger": entry.logger_name,
            "message": entry.message,
        }

        return self.format_string % format_data

    def _format_timestamp(self, timestamp: float) -> str:
        """Format timestamp."""
        return time.strftime(self.timestamp_format, time.gmtime(timestamp))


class CustomFormatter(LogFormatter):
    """Custom log formatter."""

    def __init__(self, formatter_func: callable):
        self.formatter_func = formatter_func

    def format(self, entry: LogEntry) -> str:
        """Format log entry using custom function."""
        return self.formatter_func(entry)


class ColoredFormatter(TextFormatter):
    """Colored text formatter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = {
            LogLevel.TRACE: "\033[37m",  # White
            LogLevel.DEBUG: "\033[36m",  # Cyan
            LogLevel.INFO: "\033[32m",  # Green
            LogLevel.WARNING: "\033[33m",  # Yellow
            LogLevel.ERROR: "\033[31m",  # Red
            LogLevel.CRITICAL: "\033[35m",  # Magenta
            LogLevel.FATAL: "\033[41m",  # Red background
        }
        self.reset_color = "\033[0m"

    def format(self, entry: LogEntry) -> str:
        """Format log entry with colors."""
        formatted = super().format(entry)

        if entry.level in self.colors:
            color = self.colors[entry.level]
            # Apply color to the level part
            formatted = formatted.replace(
                f"[{entry.level.value.upper()}]",
                f"{color}[{entry.level.value.upper()}]{self.reset_color}",
            )

        return formatted


class CompactFormatter(LogFormatter):
    """Compact log formatter."""

    def __init__(self, separator: str = " | "):
        self.separator = separator

    def format(self, entry: LogEntry) -> str:
        """Format log entry in compact format."""
        parts = [
            time.strftime("%H:%M:%S", time.gmtime(entry.timestamp)),
            entry.level.value.upper()[:1],  # First letter of level
            entry.logger_name,
            entry.message,
        ]

        return self.separator.join(parts)


class MultiLineFormatter(LogFormatter):
    """Multi-line log formatter."""

    def __init__(self, indent: str = "  "):
        self.indent = indent

    def format(self, entry: LogEntry) -> str:
        """Format log entry in multi-line format."""
        lines = [
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(entry.timestamp))} [{entry.level.value.upper()}] {entry.logger_name}",
            f"{self.indent}Message: {entry.message}",
        ]

        if entry.context and any(entry.context.to_dict().values()):
            lines.append(f"{self.indent}Context:")
            for key, value in entry.context.to_dict().items():
                if value:
                    lines.append(f"{self.indent}  {key}: {value}")

        if entry.extra:
            lines.append(f"{self.indent}Extra:")
            for key, value in entry.extra.items():
                lines.append(f"{self.indent}  {key}: {value}")

        if entry.exception:
            lines.append(f"{self.indent}Exception: {entry.exception}")

        return "\n".join(lines)


class TemplateFormatter(LogFormatter):
    """Template-based log formatter."""

    def __init__(self, template: str):
        self.template = template

    def format(self, entry: LogEntry) -> str:
        """Format log entry using template."""
        # Create format data
        format_data = {
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(entry.timestamp)
            ),
            "level": entry.level.value.upper(),
            "logger": entry.logger_name,
            "message": entry.message,
            "thread_id": entry.thread_id,
            "process_id": entry.process_id,
        }

        # Add context fields
        if entry.context:
            context_dict = entry.context.to_dict()
            for key, value in context_dict.items():
                format_data[f"context_{key}"] = value or ""

        # Add extra fields
        if entry.extra:
            for key, value in entry.extra.items():
                format_data[f"extra_{key}"] = value

        # Add exception info
        if entry.exception:
            format_data["exception"] = str(entry.exception)
        else:
            format_data["exception"] = ""

        return self.template.format(**format_data)
