"""Tests for logging handlers module."""

import logging

logger = logging.getLogger(__name__)
import gzip
import os
import queue
import tempfile
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from dubchain.logging.core import LogContext, LogEntry, LogLevel
from dubchain.logging.handlers import (
    AsyncHandler,
    CompressionHandler,
    ConsoleHandler,
    DatabaseHandler,
    FileHandler,
    MemoryHandler,
    NetworkHandler,
    RotatingFileHandler,
    TimedRotatingFileHandler,
)


class TestConsoleHandler:
    """Test ConsoleHandler functionality."""

    @pytest.fixture
    def console_handler(self):
        """Fixture for console handler."""
        return ConsoleHandler()

    def test_console_handler_creation(self):
        """Test creating console handler."""
        handler = ConsoleHandler()

        assert handler.name == "ConsoleHandler"
        assert handler.level == LogLevel.DEBUG

    def test_console_handler_emit(self, console_handler):
        """Test console handler emit."""
        # Mock the stream directly
        mock_stream = Mock()
        console_handler.stream = mock_stream

        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )

        console_handler.emit(entry)

        # Should write to stream
        mock_stream.write.assert_called()
        mock_stream.flush.assert_called()

    def test_console_handler_level_filtering(self, console_handler):
        """Test console handler level filtering."""
        console_handler.set_level(LogLevel.WARNING)

        context = LogContext()
        debug_entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.DEBUG,
            message="Debug message",
            logger_name="test.logger",
            context=context,
        )

        warning_entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.WARNING,
            message="Warning message",
            logger_name="test.logger",
            context=context,
        )

        assert not console_handler.should_handle(debug_entry)
        assert console_handler.should_handle(warning_entry)


class TestFileHandler:
    """Test FileHandler functionality."""

    @pytest.fixture
    def temp_file(self):
        """Fixture for temporary file."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def file_handler(self, temp_file):
        """Fixture for file handler."""
        return FileHandler(temp_file)

    def test_file_handler_creation(self, temp_file):
        """Test creating file handler."""
        handler = FileHandler(temp_file)

        assert handler.filename == temp_file
        assert handler.level == LogLevel.DEBUG

    def test_file_handler_emit(self, file_handler, temp_file):
        """Test file handler emit."""
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )

        file_handler.emit(entry)

        # Check file was written
        with open(temp_file, "r") as f:
            content = f.read()
            assert "Test message" in content

    def test_file_handler_multiple_writes(self, file_handler, temp_file):
        """Test file handler multiple writes."""
        context = LogContext()

        for i in range(3):
            entry = LogEntry(
                timestamp=1234567890.0,
                level=LogLevel.INFO,
                message=f"Message {i}",
                logger_name="test.logger",
                context=context,
            )
            file_handler.emit(entry)

        # Check all messages were written
        with open(temp_file, "r") as f:
            content = f.read()
            assert "Message 0" in content
            assert "Message 1" in content
            assert "Message 2" in content


class TestRotatingFileHandler:
    """Test RotatingFileHandler functionality."""

    @pytest.fixture
    def temp_file(self):
        """Fixture for temporary file."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        yield path
        # Clean up rotated files
        for i in range(5):
            rotated_path = f"{path}.{i}"
            if os.path.exists(rotated_path):
                os.unlink(rotated_path)
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def rotating_handler(self, temp_file):
        """Fixture for rotating file handler."""
        return RotatingFileHandler(temp_file, max_bytes=1024, backup_count=3)

    def test_rotating_handler_creation(self, temp_file):
        """Test creating rotating file handler."""
        handler = RotatingFileHandler(temp_file, max_bytes=1024, backup_count=3)

        assert handler.filename == temp_file
        assert handler.max_bytes == 1024
        assert handler.backup_count == 3

    def test_rotating_handler_emit(self, rotating_handler, temp_file):
        """Test rotating file handler emit."""
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )

        rotating_handler.emit(entry)

        # Check file was written
        assert os.path.exists(temp_file)


class TestNetworkHandler:
    """Test NetworkHandler functionality."""

    @pytest.fixture
    def network_handler(self):
        """Fixture for network handler."""
        return NetworkHandler("localhost", 514)

    def test_network_handler_creation(self):
        """Test creating network handler."""
        handler = NetworkHandler("localhost", 514)

        assert handler.host == "localhost"
        assert handler.port == 514

    @patch("socket.socket")
    def test_network_handler_emit(self, mock_socket, network_handler):
        """Test network handler emit."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock

        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )

        network_handler.emit(entry)

        # Should attempt to send data
        mock_sock.sendto.assert_called()


class TestDatabaseHandler:
    """Test DatabaseHandler functionality."""

    @pytest.fixture
    def database_handler(self):
        """Fixture for database handler."""
        mock_connection = Mock()
        return DatabaseHandler(mock_connection, "logs")

    def test_database_handler_creation(self):
        """Test creating database handler."""
        mock_connection = Mock()
        handler = DatabaseHandler(mock_connection, "logs")

        assert handler.connection == mock_connection
        assert handler.table_name == "logs"

    def test_database_handler_emit(self, database_handler):
        """Test database handler emit."""
        # Set batch size to 1 to force immediate flush
        database_handler.batch_size = 1

        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )

        database_handler.emit(entry)

        # Should execute database query
        database_handler.connection.execute.assert_called()


class TestFileHandlerAdvanced:
    """Test advanced FileHandler functionality."""

    @pytest.fixture
    def temp_file(self):
        """Fixture for temporary file."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        # Remove the file so it doesn't exist initially
        os.unlink(path)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    def test_file_handler_delayed_open(self, temp_file):
        """Test file handler with delayed opening."""
        handler = FileHandler(temp_file, delay=True)
        
        # File should not exist yet
        assert not os.path.exists(temp_file)
        
        # Emit should create the file
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        # File should now exist
        assert os.path.exists(temp_file)
        
        handler.close()

    def test_file_handler_with_formatter(self, temp_file):
        """Test file handler with custom formatter."""
        handler = FileHandler(temp_file)
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        # Check that formatter was called
        mock_formatter.format.assert_called_once_with(entry)
        
        # Check file content
        with open(temp_file, "r") as f:
            content = f.read()
            assert "FORMATTED: Test message" in content
        
        handler.close()

    def test_file_handler_close(self, temp_file):
        """Test file handler close functionality."""
        handler = FileHandler(temp_file)
        
        # Emit a message to open the file
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        handler.emit(entry)
        
        # Close handler
        handler.close()
        
        # Stream should be None after close
        assert handler.stream is None


class TestRotatingFileHandlerAdvanced:
    """Test advanced RotatingFileHandler functionality."""

    @pytest.fixture
    def temp_file(self):
        """Fixture for temporary file."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        yield path
        # Clean up rotated files
        for i in range(5):
            rotated_path = f"{path}.{i}"
            if os.path.exists(rotated_path):
                os.unlink(rotated_path)
        if os.path.exists(path):
            os.unlink(path)

    def test_rotating_handler_rollover_trigger(self, temp_file):
        """Test rotating handler rollover trigger."""
        handler = RotatingFileHandler(temp_file, max_bytes=50, backup_count=3)
        
        # Mock formatter to write exactly 50 bytes (49 chars + newline)
        mock_formatter = Mock()
        mock_formatter.format.return_value = "x" * 49
        handler.formatter = mock_formatter
        
        # Write first entry (should be exactly 50 bytes)
        context = LogContext()
        entry1 = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message 1",
            logger_name="test.logger",
            context=context,
        )
        handler.emit(entry1)
        
        # Write second entry (should trigger rollover)
        entry2 = LogEntry(
            timestamp=1234567890.1,
            level=LogLevel.INFO,
            message="Test message 2",
            logger_name="test.logger",
            context=context,
        )
        handler.emit(entry2)
        
        # Check that rollover occurred
        assert os.path.exists(f"{temp_file}.1")
        
        handler.close()

    def test_rotating_handler_should_rollover(self, temp_file):
        """Test should rollover check."""
        handler = RotatingFileHandler(temp_file, max_bytes=50, backup_count=3)
        
        # Mock formatter to write exactly 50 bytes (49 chars + newline)
        mock_formatter = Mock()
        mock_formatter.format.return_value = "x" * 49
        handler.formatter = mock_formatter
        
        # Initially should not rollover
        assert not handler._should_rollover()
        
        # Write first entry (should be exactly 50 bytes)
        context = LogContext()
        entry1 = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message 1",
            logger_name="test.logger",
            context=context,
        )
        handler.emit(entry1)
        
        # Now should rollover (file size is exactly 50 bytes)
        assert handler._should_rollover()
        
        handler.close()

    def test_rotating_handler_do_rollover(self, temp_file):
        """Test rollover functionality."""
        handler = RotatingFileHandler(temp_file, max_bytes=100, backup_count=3)
        
        # Create some content
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        handler.emit(entry)
        
        # Manually trigger rollover
        handler._do_rollover()
        
        # Check that rollover files were created
        assert os.path.exists(f"{temp_file}.1")
        
        handler.close()

    def test_rotating_handler_with_formatter(self, temp_file):
        """Test rotating handler with formatter."""
        handler = RotatingFileHandler(temp_file, max_bytes=100, backup_count=3)
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        # Check that formatter was called
        mock_formatter.format.assert_called_once_with(entry)
        
        handler.close()


class TestTimedRotatingFileHandler:
    """Test TimedRotatingFileHandler functionality."""

    @pytest.fixture
    def temp_file(self):
        """Fixture for temporary file."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        yield path
        # Clean up rotated files
        for i in range(5):
            rotated_path = f"{path}.{i}"
            if os.path.exists(rotated_path):
                os.unlink(rotated_path)
        if os.path.exists(path):
            os.unlink(path)

    def test_timed_rotating_handler_creation(self, temp_file):
        """Test creating timed rotating file handler."""
        handler = TimedRotatingFileHandler(temp_file, when="hour", interval=1, backup_count=3)
        
        assert handler.filename == temp_file
        assert handler.when == "hour"
        assert handler.interval == 1
        assert handler.backup_count == 3

    def test_timed_rotating_handler_calculate_next_rollover_midnight(self, temp_file):
        """Test next rollover calculation for midnight."""
        handler = TimedRotatingFileHandler(temp_file, when="midnight")
        
        next_rollover = handler._calculate_next_rollover()
        
        # Should be in the future
        assert next_rollover > time.time()

    def test_timed_rotating_handler_calculate_next_rollover_hour(self, temp_file):
        """Test next rollover calculation for hour."""
        handler = TimedRotatingFileHandler(temp_file, when="hour")
        
        next_rollover = handler._calculate_next_rollover()
        
        # Should be approximately 1 hour in the future
        expected = time.time() + 3600
        assert abs(next_rollover - expected) < 10  # Allow 10 seconds tolerance

    def test_timed_rotating_handler_calculate_next_rollover_day(self, temp_file):
        """Test next rollover calculation for day."""
        handler = TimedRotatingFileHandler(temp_file, when="day")
        
        next_rollover = handler._calculate_next_rollover()
        
        # Should be approximately 1 day in the future
        expected = time.time() + 86400
        assert abs(next_rollover - expected) < 10  # Allow 10 seconds tolerance

    def test_timed_rotating_handler_calculate_next_rollover_default(self, temp_file):
        """Test next rollover calculation for default case."""
        handler = TimedRotatingFileHandler(temp_file, when="unknown")
        
        next_rollover = handler._calculate_next_rollover()
        
        # Should default to hour
        expected = time.time() + 3600
        assert abs(next_rollover - expected) < 10

    def test_timed_rotating_handler_should_rollover(self, temp_file):
        """Test should rollover check."""
        handler = TimedRotatingFileHandler(temp_file, when="hour")
        
        # Initially should not rollover
        assert not handler._should_rollover()
        
        # Set next rollover time in the past
        handler._next_rollover = time.time() - 1
        
        # Now should rollover
        assert handler._should_rollover()

    def test_timed_rotating_handler_do_rollover(self, temp_file):
        """Test rollover functionality."""
        handler = TimedRotatingFileHandler(temp_file, when="hour", backup_count=3)
        
        # Create some content
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        handler.emit(entry)
        
        # Manually trigger rollover
        handler._do_rollover()
        
        # Check that rollover files were created
        # The exact filename depends on the timestamp
        rotated_files = [f for f in os.listdir(os.path.dirname(temp_file)) 
                        if f.startswith(os.path.basename(temp_file))]
        assert len(rotated_files) > 1  # Should have at least the original and one rotated file
        
        handler.close()

    def test_timed_rotating_handler_emit_with_rollover(self, temp_file):
        """Test emit with automatic rollover."""
        handler = TimedRotatingFileHandler(temp_file, when="hour", backup_count=3)
        
        # Set next rollover time in the past to trigger rollover
        handler._next_rollover = time.time() - 1
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        # Should have triggered rollover
        assert handler._next_rollover > time.time()
        
        handler.close()


class TestNetworkHandlerAdvanced:
    """Test advanced NetworkHandler functionality."""

    def test_network_handler_creation_with_custom_params(self):
        """Test creating network handler with custom parameters."""
        handler = NetworkHandler("example.com", 8080, "tcp", 10.0)
        
        assert handler.host == "example.com"
        assert handler.port == 8080
        assert handler.protocol == "tcp"
        assert handler.timeout == 10.0

    @patch("socket.socket")
    def test_network_handler_connect_udp(self, mock_socket):
        """Test UDP connection."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        handler = NetworkHandler("localhost", 514, "udp")
        handler._connect()
        
        mock_socket.assert_called_once()
        assert handler.socket == mock_sock

    @patch("socket.socket")
    def test_network_handler_connect_tcp(self, mock_socket):
        """Test TCP connection."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        handler = NetworkHandler("localhost", 8080, "tcp", 5.0)
        handler._connect()
        
        mock_socket.assert_called_once()
        mock_sock.settimeout.assert_called_once_with(5.0)
        mock_sock.connect.assert_called_once_with(("localhost", 8080))
        assert handler.socket == mock_sock

    @patch("socket.socket")
    def test_network_handler_connect_unsupported_protocol(self, mock_socket):
        """Test connection with unsupported protocol."""
        handler = NetworkHandler("localhost", 8080, "invalid")
        
        # Should not raise exception, just print error
        handler._connect()
        
        assert handler.socket is None

    @patch("socket.socket")
    def test_network_handler_connect_exception(self, mock_socket):
        """Test connection with exception."""
        mock_socket.side_effect = Exception("Connection failed")
        
        handler = NetworkHandler("localhost", 8080, "tcp")
        
        # Should not raise exception, just print error
        handler._connect()
        
        assert handler.socket is None

    def test_network_handler_disconnect(self):
        """Test disconnection."""
        handler = NetworkHandler("localhost", 8080)
        mock_socket = Mock()
        handler.socket = mock_socket
        
        handler._disconnect()
        
        mock_socket.close.assert_called_once()
        assert handler.socket is None

    def test_network_handler_disconnect_exception(self):
        """Test disconnection with exception."""
        handler = NetworkHandler("localhost", 8080)
        mock_socket = Mock()
        mock_socket.close.side_effect = Exception("Close failed")
        handler.socket = mock_socket
        
        # Should not raise exception
        handler._disconnect()
        
        assert handler.socket is None

    @patch("socket.socket")
    def test_network_handler_emit_udp(self, mock_socket):
        """Test UDP emit."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        handler = NetworkHandler("localhost", 514, "udp")
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        mock_sock.sendto.assert_called_once()
        args, kwargs = mock_sock.sendto.call_args
        data, addr = args
        assert addr == ("localhost", 514)
        assert b"Test message" in data

    @patch("socket.socket")
    def test_network_handler_emit_tcp(self, mock_socket):
        """Test TCP emit."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        handler = NetworkHandler("localhost", 8080, "tcp")
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        mock_sock.send.assert_called_once()
        args, kwargs = mock_sock.send.call_args
        data = args[0]
        assert b"Test message" in data

    @patch("socket.socket")
    def test_network_handler_emit_with_formatter(self, mock_socket):
        """Test emit with formatter."""
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        handler = NetworkHandler("localhost", 514, "udp")
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        mock_formatter.format.assert_called_once_with(entry)
        mock_sock.sendto.assert_called_once()

    @patch("socket.socket")
    def test_network_handler_emit_exception_reconnect(self, mock_socket):
        """Test emit with exception and reconnection."""
        mock_sock = Mock()
        mock_sock.sendto.side_effect = Exception("Send failed")
        mock_socket.return_value = mock_sock
        
        handler = NetworkHandler("localhost", 514, "udp")
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        # Should not raise exception, should attempt reconnection
        handler.emit(entry)
        
        # Should have called connect twice (initial + reconnect)
        assert mock_socket.call_count == 2


class TestDatabaseHandlerAdvanced:
    """Test advanced DatabaseHandler functionality."""

    @pytest.fixture
    def database_handler(self):
        """Fixture for database handler."""
        mock_connection = Mock()
        return DatabaseHandler(mock_connection, "logs", batch_size=2, flush_interval=1.0)

    def test_database_handler_creation_with_custom_params(self):
        """Test creating database handler with custom parameters."""
        mock_connection = Mock()
        handler = DatabaseHandler(mock_connection, "custom_logs", 50, 10.0)
        
        assert handler.connection == mock_connection
        assert handler.table_name == "custom_logs"
        assert handler.batch_size == 50
        assert handler.flush_interval == 10.0

    def test_database_handler_flush_buffer_empty(self, database_handler):
        """Test flushing empty buffer."""
        database_handler._flush_buffer()
        
        # Should not execute any queries
        database_handler.connection.execute.assert_not_called()

    def test_database_handler_flush_buffer_with_data(self, database_handler):
        """Test flushing buffer with data."""
        # Add data to buffer
        database_handler.buffer = [
            {
                "timestamp": 1234567890.0,
                "level": "INFO",
                "message": "Test message",
                "formatted": "FORMATTED: Test message",
            }
        ]
        
        database_handler._flush_buffer()
        
        # Should execute query
        database_handler.connection.execute.assert_called_once()
        args, kwargs = database_handler.connection.execute.call_args
        query, params = args
        assert "INSERT INTO logs VALUES" in query
        assert params[0] == 1234567890.0
        assert params[1] == "INFO"
        assert params[2] == "Test message"
        assert params[3] == "FORMATTED: Test message"

    def test_database_handler_flush_buffer_exception(self, database_handler):
        """Test flushing buffer with exception."""
        # Add data to buffer
        database_handler.buffer = [
            {
                "timestamp": 1234567890.0,
                "level": "INFO",
                "message": "Test message",
                "formatted": "FORMATTED: Test message",
            }
        ]
        
        # Make connection.execute raise exception
        database_handler.connection.execute.side_effect = Exception("DB Error")
        
        # Should not raise exception
        database_handler._flush_buffer()
        
        # Buffer should NOT be cleared on error (actual implementation behavior)
        assert len(database_handler.buffer) == 1

    def test_database_handler_emit_batch_flush(self, database_handler):
        """Test emit with batch flush."""
        context = LogContext()
        
        # Emit enough entries to trigger batch flush
        for i in range(3):
            entry = LogEntry(
                timestamp=1234567890.0 + i,
                level=LogLevel.INFO,
                message=f"Test message {i}",
                logger_name="test.logger",
                context=context,
            )
            database_handler.emit(entry)
        
        # Should have flushed buffer
        assert len(database_handler.buffer) == 1  # Only the last entry should remain
        database_handler.connection.execute.assert_called()

    def test_database_handler_emit_time_flush(self, database_handler):
        """Test emit with time-based flush."""
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        # Set last flush time in the past
        database_handler._last_flush = time.time() - 2.0
        
        database_handler.emit(entry)
        
        # Should have flushed due to time
        database_handler.connection.execute.assert_called()

    def test_database_handler_emit_with_formatter(self, database_handler):
        """Test emit with formatter."""
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        database_handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        database_handler.emit(entry)
        
        mock_formatter.format.assert_called_once_with(entry)
        
        # Check buffer content
        assert len(database_handler.buffer) == 1
        assert database_handler.buffer[0]["formatted"] == "FORMATTED: Test message"

    def test_database_handler_close(self, database_handler):
        """Test close functionality."""
        # Add data to buffer
        database_handler.buffer = [
            {
                "timestamp": 1234567890.0,
                "level": "INFO",
                "message": "Test message",
                "formatted": "FORMATTED: Test message",
            }
        ]
        
        # Store reference to connection before close
        connection = database_handler.connection
        
        database_handler.close()
        
        # Should flush buffer and close connection
        connection.execute.assert_called()
        connection.close.assert_called()
        
        # Connection should be None after close
        assert database_handler.connection is None

    def test_database_handler_disconnect(self, database_handler):
        """Test disconnection."""
        # Store reference to connection before disconnect
        connection = database_handler.connection
        
        database_handler._disconnect()
        
        connection.close.assert_called_once()
        assert database_handler.connection is None

    def test_database_handler_disconnect_exception(self, database_handler):
        """Test disconnection with exception."""
        # Store reference to connection before disconnect
        connection = database_handler.connection
        connection.close.side_effect = Exception("Close failed")
        
        # Should not raise exception
        database_handler._disconnect()
        
        assert database_handler.connection is None


class TestMemoryHandler:
    """Test MemoryHandler functionality."""

    def test_memory_handler_creation(self):
        """Test creating memory handler."""
        handler = MemoryHandler(max_size=500)
        
        assert handler.max_size == 500
        assert len(handler.buffer) == 0

    def test_memory_handler_emit(self):
        """Test memory handler emit."""
        handler = MemoryHandler(max_size=10)
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        assert len(handler.buffer) == 1
        assert handler.buffer[0]["message"] == "Test message"
        assert handler.buffer[0]["level"] == "info"
        assert handler.buffer[0]["logger_name"] == "test.logger"

    def test_memory_handler_emit_with_formatter(self):
        """Test memory handler emit with formatter."""
        handler = MemoryHandler(max_size=10)
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        mock_formatter.format.assert_called_once_with(entry)
        assert handler.buffer[0]["formatted"] == "FORMATTED: Test message"

    def test_memory_handler_emit_buffer_overflow(self):
        """Test memory handler buffer overflow."""
        handler = MemoryHandler(max_size=3)
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        
        # Add more entries than max_size
        for i in range(5):
            entry = LogEntry(
                timestamp=1234567890.0 + i,
                level=LogLevel.INFO,
                message=f"Test message {i}",
                logger_name="test.logger",
                context=context,
            )
            handler.emit(entry)
        
        # Should only keep the last 3 entries
        assert len(handler.buffer) == 3
        assert handler.buffer[0]["message"] == "Test message 2"
        assert handler.buffer[1]["message"] == "Test message 3"
        assert handler.buffer[2]["message"] == "Test message 4"

    def test_memory_handler_get_logs(self):
        """Test getting logs from memory."""
        handler = MemoryHandler(max_size=10)
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        
        logs = handler.get_logs()
        
        assert len(logs) == 1
        assert logs[0]["message"] == "Test message"
        # Should return a copy
        assert logs is not handler.buffer

    def test_memory_handler_clear_logs(self):
        """Test clearing logs from memory."""
        handler = MemoryHandler(max_size=10)
        
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        handler.emit(entry)
        assert len(handler.buffer) == 1
        
        handler.clear_logs()
        assert len(handler.buffer) == 0

    def test_memory_handler_close(self):
        """Test close functionality."""
        handler = MemoryHandler(max_size=10)
        
        # Should not raise exception
        handler.close()


class TestAsyncHandler:
    """Test AsyncHandler functionality."""

    @pytest.fixture
    def target_handler(self):
        """Fixture for target handler."""
        return Mock()

    @pytest.fixture
    def async_handler(self, target_handler):
        """Fixture for async handler."""
        return AsyncHandler(target_handler, queue_size=10)

    def test_async_handler_creation(self, target_handler):
        """Test creating async handler."""
        handler = AsyncHandler(target_handler, queue_size=100)
        
        assert handler.target_handler == target_handler
        assert handler.queue_size == 100
        assert handler.running is True
        assert handler.thread is not None

    def test_async_handler_emit(self, async_handler, target_handler):
        """Test async handler emit."""
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        async_handler.emit(entry)
        
        # Give worker thread time to process
        time.sleep(0.1)
        
        # Should have called target handler
        target_handler.emit.assert_called_once_with(entry)

    def test_async_handler_emit_queue_full(self, target_handler):
        """Test async handler emit with full queue."""
        handler = AsyncHandler(target_handler, queue_size=1)
        
        context = LogContext()
        entry1 = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message 1",
            logger_name="test.logger",
            context=context,
        )
        entry2 = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message 2",
            logger_name="test.logger",
            context=context,
        )
        
        # Fill queue
        handler.emit(entry1)
        
        # This should be dropped due to full queue
        handler.emit(entry2)
        
        # Give worker thread time to process
        time.sleep(0.1)
        
        # Should only have processed the first entry
        target_handler.emit.assert_called_once_with(entry1)

    def test_async_handler_worker_exception(self, target_handler):
        """Test async handler worker with exception."""
        target_handler.emit.side_effect = Exception("Worker error")
        
        handler = AsyncHandler(target_handler, queue_size=10)
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        # Should not raise exception
        handler.emit(entry)
        
        # Give worker thread time to process
        time.sleep(0.1)
        
        handler.close()

    def test_async_handler_close(self, async_handler, target_handler):
        """Test async handler close."""
        async_handler.close()
        
        # Should have stopped running
        assert async_handler.running is False
        
        # Should have closed target handler
        target_handler.close.assert_called_once()

    def test_async_handler_close_queue_full(self, target_handler):
        """Test async handler close with full queue."""
        handler = AsyncHandler(target_handler, queue_size=1)
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        # Fill queue
        handler.emit(entry)
        
        # Close should not raise exception even with full queue
        handler.close()


class TestCompressionHandler:
    """Test CompressionHandler functionality."""

    @pytest.fixture
    def target_handler(self):
        """Fixture for target handler."""
        return Mock()

    @pytest.fixture
    def compression_handler(self, target_handler):
        """Fixture for compression handler."""
        return CompressionHandler(target_handler, compression_level=6, compress_after=3)

    def test_compression_handler_creation(self, target_handler):
        """Test creating compression handler."""
        handler = CompressionHandler(target_handler, compression_level=9, compress_after=100)
        
        assert handler.target_handler == target_handler
        assert handler.compression_level == 9
        assert handler.compress_after == 100
        assert len(handler.buffer) == 0

    def test_compression_handler_emit(self, compression_handler, target_handler):
        """Test compression handler emit."""
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        compression_handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        compression_handler.emit(entry)
        
        # Should add to buffer
        assert len(compression_handler.buffer) == 1

    def test_compression_handler_emit_with_formatter(self, compression_handler, target_handler):
        """Test compression handler emit with formatter."""
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        compression_handler.formatter = mock_formatter
        
        context = LogContext()
        entry = LogEntry(
            timestamp=1234567890.0,
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            context=context,
        )
        
        compression_handler.emit(entry)
        
        mock_formatter.format.assert_called_once_with(entry)
        assert compression_handler.buffer[0] == "FORMATTED: Test message"

    def test_compression_handler_compress_and_forward(self, compression_handler, target_handler):
        """Test compression and forward functionality."""
        # Add data to buffer
        compression_handler.buffer = ["Message 1", "Message 2", "Message 3"]
        
        compression_handler._compress_and_forward()
        
        # Should have called target handler
        target_handler.emit.assert_called_once()
        
        # Check the compressed entry
        args, kwargs = target_handler.emit.call_args
        compressed_entry = args[0]
        
        assert compressed_entry.logger_name == "compression_handler"
        assert "Compressed 3 log entries" in compressed_entry.message
        assert "compressed_size" in compressed_entry.extra
        assert "original_size" in compressed_entry.extra
        assert "compression_ratio" in compressed_entry.extra
        assert "entry_count" in compressed_entry.extra
        assert compressed_entry.extra["entry_count"] == 3
        
        # Buffer should be cleared
        assert len(compression_handler.buffer) == 0

    def test_compression_handler_compress_and_forward_empty_buffer(self, compression_handler, target_handler):
        """Test compression and forward with empty buffer."""
        compression_handler._compress_and_forward()
        
        # Should not call target handler
        target_handler.emit.assert_not_called()

    def test_compression_handler_compress_and_forward_exception(self, compression_handler, target_handler):
        """Test compression and forward with exception."""
        # Add data to buffer
        compression_handler.buffer = ["Message 1", "Message 2", "Message 3"]
        
        # Mock gzip.compress to raise exception
        with patch('gzip.compress', side_effect=Exception("Compression failed")):
            compression_handler._compress_and_forward()
        
        # Should not call target handler
        target_handler.emit.assert_not_called()
        
        # Buffer should be cleared
        assert len(compression_handler.buffer) == 0

    def test_compression_handler_emit_trigger_compression(self, compression_handler, target_handler):
        """Test emit that triggers compression."""
        # Mock formatter
        mock_formatter = Mock()
        mock_formatter.format.return_value = "FORMATTED: Test message"
        compression_handler.formatter = mock_formatter
        
        context = LogContext()
        
        # Emit exactly the number of entries to trigger compression (compress_after=3)
        for i in range(3):
            entry = LogEntry(
                timestamp=1234567890.0 + i,
                level=LogLevel.INFO,
                message=f"Test message {i}",
                logger_name="test.logger",
                context=context,
            )
            compression_handler.emit(entry)
        
        # Should have triggered compression
        target_handler.emit.assert_called_once()
        
        # Buffer should be cleared
        assert len(compression_handler.buffer) == 0

    def test_compression_handler_close(self, compression_handler, target_handler):
        """Test close functionality."""
        # Add data to buffer
        compression_handler.buffer = ["Message 1", "Message 2"]
        
        compression_handler.close()
        
        # Should flush remaining buffer
        target_handler.emit.assert_called_once()
        
        # Should close target handler
        target_handler.close.assert_called_once()

    def test_compression_handler_close_empty_buffer(self, compression_handler, target_handler):
        """Test close with empty buffer."""
        compression_handler.close()
        
        # Should not emit anything
        target_handler.emit.assert_not_called()
        
        # Should close target handler
        target_handler.close.assert_called_once()
