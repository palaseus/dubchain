"""Log handlers for DubChain.

This module provides various log handlers including file, network,
database, console, and memory handlers for the DubChain logging system.
"""

import gzip
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .core import LogEntry, LogHandler, LogLevel


class FileHandler(LogHandler):
    """File log handler."""

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: str = "utf-8",
        delay: bool = False,
    ):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.stream = None
        self._lock = threading.RLock()

        if not self.delay:
            self._open()

    def _open(self) -> None:
        """Open file stream."""
        if self.stream is None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            self.stream = open(self.filename, self.mode, encoding=self.encoding)

    def _close(self) -> None:
        """Close file stream."""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to file."""
        with self._lock:
            if self.stream is None:
                self._open()

            if self.formatter:
                formatted = self.formatter.format(entry)
            else:
                # Simple default formatting
                formatted = f"{entry.timestamp} [{entry.level.value.upper()}] {entry.logger_name}: {entry.message}"

            self.stream.write(formatted + "\n")
            self.stream.flush()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            self._close()


class RotatingFileHandler(LogHandler):
    """Rotating file log handler."""

    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = "utf-8",
    ):
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self.stream = None
        self._lock = threading.RLock()

        self._open()

    def _open(self) -> None:
        """Open file stream."""
        if self.stream is None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            self.stream = open(self.filename, "a", encoding=self.encoding)

    def _close(self) -> None:
        """Close file stream."""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def _should_rollover(self) -> bool:
        """Check if rollover is needed."""
        if self.stream is None:
            return False

        self.stream.seek(0, 2)  # Seek to end
        return self.stream.tell() >= self.max_bytes

    def _do_rollover(self) -> None:
        """Perform rollover."""
        if self.stream is not None:
            self._close()

        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            old_file = f"{self.filename}.{i}"
            new_file = f"{self.filename}.{i + 1}"
            if os.path.exists(old_file):
                if i == self.backup_count - 1:
                    os.remove(old_file)
                else:
                    shutil.move(old_file, new_file)

        # Move current file to .1
        if os.path.exists(self.filename):
            shutil.move(self.filename, f"{self.filename}.1")

        self._open()

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to file."""
        with self._lock:
            if self._should_rollover():
                self._do_rollover()

            if self.formatter:
                formatted = self.formatter.format(entry)
                self.stream.write(formatted + "\n")
                self.stream.flush()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            self._close()


class TimedRotatingFileHandler(LogHandler):
    """Timed rotating file log handler."""

    def __init__(
        self,
        filename: str,
        when: str = "midnight",
        interval: int = 1,
        backup_count: int = 5,
        encoding: str = "utf-8",
    ):
        super().__init__()
        self.filename = filename
        self.when = when
        self.interval = interval
        self.backup_count = backup_count
        self.encoding = encoding
        self.stream = None
        self._lock = threading.RLock()

        # Calculate next rollover time
        self._next_rollover = self._calculate_next_rollover()

        self._open()

    def _open(self) -> None:
        """Open file stream."""
        if self.stream is None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            self.stream = open(self.filename, "a", encoding=self.encoding)

    def _close(self) -> None:
        """Close file stream."""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def _calculate_next_rollover(self) -> float:
        """Calculate next rollover time."""
        now = time.time()

        if self.when == "midnight":
            # Next midnight
            import datetime

            today = datetime.date.today()
            tomorrow = today + datetime.timedelta(days=1)
            midnight = datetime.datetime.combine(tomorrow, datetime.time.min)
            return midnight.timestamp()
        elif self.when == "hour":
            # Next hour
            return now + 3600
        elif self.when == "day":
            # Next day
            return now + 86400
        else:
            # Default to hour
            return now + 3600

    def _should_rollover(self) -> bool:
        """Check if rollover is needed."""
        return time.time() >= self._next_rollover

    def _do_rollover(self) -> None:
        """Perform rollover."""
        if self.stream is not None:
            self._close()

        # Generate suffix based on when
        if self.when == "midnight":
            suffix = time.strftime("%Y-%m-%d", time.gmtime())
        elif self.when == "hour":
            suffix = time.strftime("%Y-%m-%d_%H", time.gmtime())
        elif self.when == "day":
            suffix = time.strftime("%Y-%m-%d", time.gmtime())
        else:
            suffix = time.strftime("%Y-%m-%d_%H", time.gmtime())

        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            old_file = f"{self.filename}.{suffix}.{i}"
            new_file = f"{self.filename}.{suffix}.{i + 1}"
            if os.path.exists(old_file):
                if i == self.backup_count - 1:
                    os.remove(old_file)
                else:
                    shutil.move(old_file, new_file)

        # Move current file
        if os.path.exists(self.filename):
            shutil.move(self.filename, f"{self.filename}.{suffix}.1")

        # Update next rollover time
        self._next_rollover = self._calculate_next_rollover()

        self._open()

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to file."""
        with self._lock:
            if self._should_rollover():
                self._do_rollover()

            if self.formatter:
                formatted = self.formatter.format(entry)
                self.stream.write(formatted + "\n")
                self.stream.flush()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            self._close()


class NetworkHandler(LogHandler):
    """Network log handler."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 514,
        protocol: str = "udp",
        timeout: float = 5.0,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.protocol = protocol
        self.timeout = timeout
        self.socket = None
        self._lock = threading.RLock()

    def _connect(self) -> None:
        """Connect to network endpoint."""
        try:
            import socket

            if self.protocol == "udp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif self.protocol == "tcp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.connect((self.host, self.port))
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")

        except Exception as e:
            # Log error but don't raise
            print(f"Failed to connect to {self.host}:{self.port}: {e}")

    def _disconnect(self) -> None:
        """Disconnect from network endpoint."""
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            finally:
                self.socket = None

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to network."""
        with self._lock:
            if self.socket is None:
                self._connect()

            if self.socket is not None:
                try:
                    if self.formatter:
                        formatted = self.formatter.format(entry)
                    else:
                        # Simple default formatting
                        formatted = f"{entry.timestamp} [{entry.level.value.upper()}] {entry.logger_name}: {entry.message}"

                    data = formatted.encode("utf-8")

                    if self.protocol == "udp":
                        self.socket.sendto(data, (self.host, self.port))
                    elif self.protocol == "tcp":
                        self.socket.send(data)

                except Exception as e:
                    # Reconnect on error
                    self._disconnect()
                    self._connect()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            self._disconnect()


class DatabaseHandler(LogHandler):
    """Database log handler."""

    def __init__(
        self,
        connection,
        table_name: str = "logs",
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        super().__init__()
        self.connection = connection
        self.table_name = table_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffer = []
        self._lock = threading.RLock()
        self._last_flush = time.time()

    def _connect(self) -> None:
        """Connect to database."""
        # Connection is already provided in constructor
        if self.connection is None:
            raise ValueError("Database connection is required")

    def _disconnect(self) -> None:
        """Disconnect from database."""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as e:
                print(f"Error closing database connection: {e}")
            finally:
                self.connection = None

    def _flush_buffer(self) -> None:
        """Flush buffer to database."""
        if not self.buffer:
            return

        try:
            # This is a simplified implementation
            # In practice, you'd use a proper database library
            for entry in self.buffer:
                # Insert log entry into database
                if self.connection:
                    self.connection.execute(
                        "INSERT INTO logs VALUES (?, ?, ?, ?)",
                        (
                            entry["timestamp"],
                            entry["level"],
                            entry["message"],
                            entry["formatted"],
                        ),
                    )

            self.buffer.clear()
            self._last_flush = time.time()

        except Exception as e:
            print(f"Failed to flush logs to database: {e}")

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to database."""
        with self._lock:
            if self.formatter:
                formatted = self.formatter.format(entry)
            else:
                # Simple default formatting
                formatted = f"{entry.timestamp} [{entry.level.value.upper()}] {entry.logger_name}: {entry.message}"

            self.buffer.append(
                {
                    "timestamp": entry.timestamp,
                    "level": entry.level.value,
                    "message": entry.message,
                    "logger_name": entry.logger_name,
                    "formatted": formatted,
                }
            )

            # Flush if buffer is full or enough time has passed
            if (
                len(self.buffer) >= self.batch_size
                or time.time() - self._last_flush >= self.flush_interval
            ):
                self._flush_buffer()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            self._flush_buffer()
            self._disconnect()


class ConsoleHandler(LogHandler):
    """Console log handler."""

    def __init__(self, stream: Any = None):
        super().__init__()
        self.stream = stream or sys.stdout
        self._lock = threading.RLock()

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to console."""
        with self._lock:
            if self.formatter:
                formatted = self.formatter.format(entry)
            else:
                # Simple default formatting
                formatted = f"{entry.timestamp} [{entry.level.value.upper()}] {entry.logger_name}: {entry.message}"

            self.stream.write(formatted + "\n")
            self.stream.flush()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            if self.stream is not None:
                self.stream.close()
                self.stream = None


class MemoryHandler(LogHandler):
    """Memory log handler."""

    def __init__(self, max_size: int = 1000):
        super().__init__()
        self.max_size = max_size
        self.buffer = []
        self._lock = threading.RLock()

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry to memory."""
        with self._lock:
            if self.formatter:
                formatted = self.formatter.format(entry)
                self.buffer.append(
                    {
                        "timestamp": entry.timestamp,
                        "level": entry.level.value,
                        "message": entry.message,
                        "logger_name": entry.logger_name,
                        "formatted": formatted,
                    }
                )

                # Remove old entries if buffer is full
                if len(self.buffer) > self.max_size:
                    self.buffer.pop(0)

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs from memory."""
        with self._lock:
            return self.buffer.copy()

    def clear_logs(self) -> None:
        """Clear all logs from memory."""
        with self._lock:
            self.buffer.clear()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            self.buffer.clear()


class AsyncHandler(LogHandler):
    """Asynchronous log handler."""

    def __init__(self, target_handler: LogHandler, queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.queue_size = queue_size
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = None
        self.running = False
        self._lock = threading.RLock()

        self._start_worker()

    def _start_worker(self) -> None:
        """Start worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        """Worker thread."""
        while self.running:
            try:
                entry = self.queue.get(timeout=1.0)
                if entry is None:  # Shutdown signal
                    break

                self.target_handler.emit(entry)
                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async handler worker: {e}")

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry asynchronously."""
        try:
            self.queue.put_nowait(entry)
        except queue.Full:
            # Drop entry if queue is full
            pass

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            self.running = False

            # Send shutdown signal
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass

            # Wait for worker to finish
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5.0)

            # Close target handler
            if self.target_handler:
                self.target_handler.close()


class CompressionHandler(LogHandler):
    """Compression log handler."""

    def __init__(
        self,
        target_handler: LogHandler,
        compression_level: int = 6,
        compress_after: int = 1000,
    ):
        super().__init__()
        self.target_handler = target_handler
        self.compression_level = compression_level
        self.compress_after = compress_after
        self.buffer = []
        self._lock = threading.RLock()

    def emit(self, entry: LogEntry) -> None:
        """Emit log entry with compression."""
        with self._lock:
            if self.formatter:
                formatted = self.formatter.format(entry)
                self.buffer.append(formatted)

                # Compress and forward if buffer is full
                if len(self.buffer) >= self.compress_after:
                    self._compress_and_forward()

    def _compress_and_forward(self) -> None:
        """Compress buffer and forward to target handler."""
        if not self.buffer:
            return

        try:
            # Compress buffer
            data = "\n".join(self.buffer).encode("utf-8")
            compressed = gzip.compress(data, compresslevel=self.compression_level)

            # Create a compressed log entry
            compressed_entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                message=f"Compressed {len(self.buffer)} log entries",
                logger_name="compression_handler",
                context=None,
            )
            compressed_entry.extra = {
                "compressed_size": len(compressed),
                "original_size": len(data),
                "compression_ratio": len(compressed) / len(data),
                "entry_count": len(self.buffer),
            }

            # Forward to target handler
            self.target_handler.emit(compressed_entry)

            # Clear buffer
            self.buffer.clear()

        except Exception as e:
            print(f"Error compressing logs: {e}")
            # Clear buffer on error
            self.buffer.clear()

    def close(self) -> None:
        """Close handler."""
        with self._lock:
            # Flush remaining buffer
            if self.buffer:
                self._compress_and_forward()

            # Close target handler
            if self.target_handler:
                self.target_handler.close()
