"""
Tests for logging aggregation module.
"""

import logging

logger = logging.getLogger(__name__)
import json
import os
import socket
import tempfile
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.dubchain.logging.aggregation import (
    DatabaseLogForwarder,
    DistributedAggregator,
    FileLogForwarder,
    LogAggregator,
    LogBuffer,
    LogCollector,
    LogForwarder,
    NetworkLogForwarder,
)
from src.dubchain.logging.core import LogContext, LogEntry, LogLevel


class TestLogBuffer:
    """Test LogBuffer class."""

    def test_log_buffer_initialization(self):
        """Test log buffer initialization."""
        buffer = LogBuffer(max_size=100, max_age=30.0)

        assert buffer.max_size == 100
        assert buffer.max_age == 30.0
        assert len(buffer.entries) == 0
        assert buffer.created_at > 0

    def test_log_buffer_add_entry(self):
        """Test adding entries to buffer."""
        buffer = LogBuffer(max_size=2)

        entry1 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message 1",
            context=LogContext(),
        )

        entry2 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message 2",
            context=LogContext(),
        )

        # Add first entry
        is_full = buffer.add_entry(entry1)
        assert not is_full
        assert len(buffer.entries) == 1

        # Add second entry (should make buffer full)
        is_full = buffer.add_entry(entry2)
        assert is_full
        assert len(buffer.entries) == 2

    def test_log_buffer_is_expired(self):
        """Test buffer expiration."""
        buffer = LogBuffer(max_age=0.1)  # 100ms

        # Should not be expired immediately
        assert not buffer.is_expired()

        # Wait for expiration
        time.sleep(0.2)
        assert buffer.is_expired()

    def test_log_buffer_clear(self):
        """Test buffer clearing."""
        buffer = LogBuffer()

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        buffer.add_entry(entry)
        assert len(buffer.entries) == 1

        buffer.clear()
        assert len(buffer.entries) == 0
        assert buffer.created_at > 0

    def test_log_buffer_get_entries(self):
        """Test getting entries from buffer."""
        buffer = LogBuffer()

        entry1 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message 1",
            context=LogContext(),
        )

        entry2 = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message 2",
            context=LogContext(),
        )

        buffer.add_entry(entry1)
        buffer.add_entry(entry2)

        entries = buffer.get_entries()
        assert len(entries) == 2
        assert entries[0].message == "Test message 1"
        assert entries[1].message == "Test message 2"

        # Should return a copy
        entries.clear()
        assert len(buffer.entries) == 2


class TestLogCollector:
    """Test LogCollector abstract class."""

    def test_log_collector_abstract(self):
        """Test that LogCollector is abstract."""
        # The LogCollector class is actually concrete, not abstract
        # This test verifies it can be instantiated
        collector = LogCollector()
        assert collector is not None
        assert len(collector.sources) == 0


class TestLogForwarder:
    """Test LogForwarder abstract class."""

    def test_log_forwarder_abstract(self):
        """Test that LogForwarder is abstract."""
        with pytest.raises(TypeError):
            LogForwarder()


class TestNetworkLogForwarder:
    """Test NetworkLogForwarder class."""

    def test_network_log_forwarder_initialization(self):
        """Test network log forwarder initialization."""
        forwarder = NetworkLogForwarder(
            host="localhost", port=8080, protocol="tcp", use_ssl=False, timeout=5.0
        )

        assert forwarder.host == "localhost"
        assert forwarder.port == 8080
        assert forwarder.protocol == "tcp"
        assert forwarder.use_ssl is False
        assert forwarder.timeout == 5.0
        assert forwarder.socket is None

    def test_network_log_forwarder_forward_empty_logs(self):
        """Test forwarding empty log list."""
        forwarder = NetworkLogForwarder("localhost", 8080)

        # Should return True for empty list
        result = forwarder.forward_logs([])
        assert result is True

    def test_network_log_forwarder_forward_logs_tcp(self):
        """Test forwarding logs via TCP."""
        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_socket.return_value = mock_sock

            forwarder = NetworkLogForwarder("localhost", 8080, protocol="tcp")

            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test",
                message="Test message",
                context=LogContext(),
            )

            result = forwarder.forward_logs([entry])

            # Should attempt to connect and send data
            mock_sock.connect.assert_called_once_with(("localhost", 8080))
            assert mock_sock.send.call_count == 2  # Length + data
            assert result is True

    def test_network_log_forwarder_forward_logs_udp(self):
        """Test forwarding logs via UDP."""
        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_socket.return_value = mock_sock

            forwarder = NetworkLogForwarder("localhost", 8080, protocol="udp")

            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test",
                message="Test message",
                context=LogContext(),
            )

            result = forwarder.forward_logs([entry])

            # Should send data via UDP
            mock_sock.sendto.assert_called_once()
            assert result is True

    def test_network_log_forwarder_connection_failure(self):
        """Test handling connection failure."""
        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_sock.connect.side_effect = ConnectionError("Connection failed")
            mock_socket.return_value = mock_sock

            forwarder = NetworkLogForwarder("localhost", 8080)

            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test",
                message="Test message",
                context=LogContext(),
            )

            result = forwarder.forward_logs([entry])
            assert result is False

    def test_network_log_forwarder_close(self):
        """Test closing forwarder."""
        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_socket.return_value = mock_sock

            forwarder = NetworkLogForwarder("localhost", 8080)
            forwarder.socket = mock_sock

            forwarder.close()

            mock_sock.close.assert_called_once()
            assert forwarder.socket is None

    def test_network_log_forwarder_unsupported_protocol(self):
        """Test unsupported protocol."""
        with patch("socket.socket") as mock_socket:
            mock_sock = Mock()
            mock_socket.return_value = mock_sock

            forwarder = NetworkLogForwarder("localhost", 8080, protocol="invalid")

            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test",
                message="Test message",
                context=LogContext(),
            )

            result = forwarder.forward_logs([entry])
            assert result is False


class TestFileLogForwarder:
    """Test FileLogForwarder class."""

    def test_file_log_forwarder_initialization(self):
        """Test file log forwarder initialization."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            forwarder = FileLogForwarder(temp_path, append=True)

            assert forwarder.file_path == temp_path
            assert forwarder.append is True

        finally:
            os.unlink(temp_path)

    def test_file_log_forwarder_forward_empty_logs(self):
        """Test forwarding empty log list."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            forwarder = FileLogForwarder(temp_path)

            # Should return True for empty list
            result = forwarder.forward_logs([])
            assert result is True

        finally:
            os.unlink(temp_path)

    def test_file_log_forwarder_forward_logs_append(self):
        """Test forwarding logs with append mode."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            forwarder = FileLogForwarder(temp_path, append=True)

            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test",
                message="Test message",
                context=LogContext(),
            )

            result = forwarder.forward_logs([entry])
            assert result is True

            # Check file content
            with open(temp_path, "r") as f:
                content = f.read()
                assert "Test message" in content

        finally:
            os.unlink(temp_path)

    def test_file_log_forwarder_forward_logs_overwrite(self):
        """Test forwarding logs with overwrite mode."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b"existing content")

        try:
            forwarder = FileLogForwarder(temp_path, append=False)

            entry = LogEntry(
                timestamp=time.time(),
                level=LogLevel.INFO,
                logger_name="test",
                message="Test message",
                context=LogContext(),
            )

            result = forwarder.forward_logs([entry])
            assert result is True

            # Check file content (should be overwritten)
            with open(temp_path, "r") as f:
                content = f.read()
                assert "existing content" not in content
                assert "Test message" in content

        finally:
            os.unlink(temp_path)

    def test_file_log_forwarder_forward_logs_failure(self):
        """Test handling file write failure."""
        # Use a path that will cause permission error
        forwarder = FileLogForwarder("/root/readonly_file.log")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        result = forwarder.forward_logs([entry])
        assert result is False


class TestDatabaseLogForwarder:
    """Test DatabaseLogForwarder class."""

    def test_database_log_forwarder_initialization(self):
        """Test database log forwarder initialization."""
        forwarder = DatabaseLogForwarder(
            connection_string="sqlite:///test.db", table_name="logs"
        )

        assert forwarder.connection_string == "sqlite:///test.db"
        assert forwarder.table_name == "logs"
        assert forwarder.connection is None

    def test_database_log_forwarder_forward_empty_logs(self):
        """Test forwarding empty log list."""
        forwarder = DatabaseLogForwarder("sqlite:///test.db")

        # Should return True for empty list
        result = forwarder.forward_logs([])
        assert result is True

    def test_database_log_forwarder_forward_logs(self):
        """Test forwarding logs to database."""
        forwarder = DatabaseLogForwarder("sqlite:///test.db")

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        # Should return True (simplified implementation)
        result = forwarder.forward_logs([entry])
        assert result is True


class TestLogAggregator:
    """Test LogAggregator class."""

    def test_log_aggregator_initialization(self):
        """Test log aggregator initialization."""
        aggregator = LogAggregator(buffer_size=100, flush_interval=30.0, max_retries=3)

        assert aggregator.buffer_size == 100
        assert aggregator.flush_interval == 30.0
        assert aggregator.max_retries == 3
        assert len(aggregator.forwarders) == 0
        assert len(aggregator.processors) == 0
        assert aggregator._running is True

        aggregator.shutdown()

    def test_log_aggregator_add_remove_forwarder(self):
        """Test adding and removing forwarders."""
        aggregator = LogAggregator()

        forwarder1 = Mock(spec=LogForwarder)
        forwarder2 = Mock(spec=LogForwarder)

        # Add forwarders
        aggregator.add_forwarder(forwarder1)
        aggregator.add_forwarder(forwarder2)
        assert len(aggregator.forwarders) == 2

        # Remove forwarder
        aggregator.remove_forwarder(forwarder1)
        assert len(aggregator.forwarders) == 1
        assert forwarder2 in aggregator.forwarders

        aggregator.shutdown()

    def test_log_aggregator_add_remove_processor(self):
        """Test adding and removing processors."""
        aggregator = LogAggregator()

        def processor1(entry):
            return entry

        def processor2(entry):
            return entry

        # Add processors
        aggregator.add_processor(processor1)
        aggregator.add_processor(processor2)
        assert len(aggregator.processors) == 2

        # Remove processor
        aggregator.remove_processor(processor1)
        assert len(aggregator.processors) == 1
        assert processor2 in aggregator.processors

        aggregator.shutdown()

    def test_log_aggregator_add_log(self):
        """Test adding log entry."""
        aggregator = LogAggregator(buffer_size=2)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        # Add log entry
        aggregator.add_log(entry)
        assert len(aggregator.buffer.entries) == 1

        # Add another entry to trigger flush
        aggregator.add_log(entry)
        assert len(aggregator.buffer.entries) == 0  # Should be flushed

        aggregator.shutdown()

    def test_log_aggregator_add_log_with_processor(self):
        """Test adding log entry with processor."""
        aggregator = LogAggregator()

        def processor(entry):
            entry.message = "Processed: " + entry.message
            return entry

        aggregator.add_processor(processor)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        aggregator.add_log(entry)

        # Check that message was processed
        assert len(aggregator.buffer.entries) == 1
        assert aggregator.buffer.entries[0].message == "Processed: Test message"

        aggregator.shutdown()

    def test_log_aggregator_force_flush(self):
        """Test force flushing buffer."""
        aggregator = LogAggregator()

        forwarder = Mock(spec=LogForwarder)
        forwarder.forward_logs.return_value = True
        aggregator.add_forwarder(forwarder)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        aggregator.add_log(entry)
        assert len(aggregator.buffer.entries) == 1

        # Force flush
        aggregator.force_flush()
        assert len(aggregator.buffer.entries) == 0
        forwarder.forward_logs.assert_called_once()

        aggregator.shutdown()

    def test_log_aggregator_get_buffer_status(self):
        """Test getting buffer status."""
        aggregator = LogAggregator(buffer_size=100, flush_interval=60.0)

        status = aggregator.get_buffer_status()

        assert status["buffer_size"] == 0
        assert status["max_size"] == 100
        assert status["age"] >= 0
        assert status["max_age"] == 60.0
        assert status["is_expired"] is False
        assert status["forwarders_count"] == 0

        aggregator.shutdown()

    def test_log_aggregator_forwarder_retry(self):
        """Test forwarder retry mechanism."""
        aggregator = LogAggregator(max_retries=2)

        forwarder = Mock(spec=LogForwarder)
        forwarder.forward_logs.side_effect = [False, True]  # Fail first, succeed second
        aggregator.add_forwarder(forwarder)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        aggregator.add_log(entry)
        aggregator.force_flush()

        # Should have been called twice (retry)
        assert forwarder.forward_logs.call_count == 2

        aggregator.shutdown()

    def test_log_aggregator_forwarder_max_retries_exceeded(self):
        """Test forwarder max retries exceeded."""
        aggregator = LogAggregator(max_retries=2)

        forwarder = Mock(spec=LogForwarder)
        forwarder.forward_logs.return_value = False  # Always fail
        aggregator.add_forwarder(forwarder)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        aggregator.add_log(entry)
        aggregator.force_flush()

        # Should have been called max_retries times
        assert forwarder.forward_logs.call_count == 2

        aggregator.shutdown()

    def test_log_aggregator_shutdown(self):
        """Test aggregator shutdown."""
        aggregator = LogAggregator()

        forwarder = Mock(spec=LogForwarder)
        forwarder.close = Mock()
        aggregator.add_forwarder(forwarder)

        assert aggregator._running is True

        aggregator.shutdown()

        assert aggregator._running is False
        forwarder.close.assert_called_once()


class TestDistributedAggregator:
    """Test DistributedAggregator class."""

    def test_distributed_aggregator_initialization(self):
        """Test distributed aggregator initialization."""
        aggregator = DistributedAggregator(node_id="node1", sync_interval=30.0)

        assert aggregator.node_id == "node1"
        assert aggregator.sync_interval == 30.0
        assert len(aggregator.aggregators) == 0
        assert aggregator._running is True

        aggregator.shutdown()

    def test_distributed_aggregator_add_remove_aggregator(self):
        """Test adding and removing aggregators."""
        distributed = DistributedAggregator("node1")

        aggregator1 = LogAggregator()
        aggregator2 = LogAggregator()

        # Add aggregators
        distributed.add_aggregator(aggregator1)
        distributed.add_aggregator(aggregator2)
        assert len(distributed.aggregators) == 2

        # Remove aggregator
        distributed.remove_aggregator(aggregator1)
        assert len(distributed.aggregators) == 1
        assert aggregator2 in distributed.aggregators

        distributed.shutdown()
        aggregator2.shutdown()

    def test_distributed_aggregator_add_log(self):
        """Test adding log entry to all aggregators."""
        distributed = DistributedAggregator("node1")

        aggregator1 = Mock(spec=LogAggregator)
        aggregator2 = Mock(spec=LogAggregator)

        distributed.add_aggregator(aggregator1)
        distributed.add_aggregator(aggregator2)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        distributed.add_log(entry)

        # Both aggregators should have received the log
        aggregator1.add_log.assert_called_once_with(entry)
        aggregator2.add_log.assert_called_once_with(entry)

        distributed.shutdown()

    def test_distributed_aggregator_get_status(self):
        """Test getting distributed aggregator status."""
        distributed = DistributedAggregator("node1")

        aggregator = LogAggregator()
        distributed.add_aggregator(aggregator)

        status = distributed.get_status()

        assert status["node_id"] == "node1"
        assert status["aggregators_count"] == 1
        assert len(status["aggregators"]) == 1
        assert status["aggregators"][0]["index"] == 0

        distributed.shutdown()
        aggregator.shutdown()

    def test_distributed_aggregator_shutdown(self):
        """Test distributed aggregator shutdown."""
        distributed = DistributedAggregator("node1")

        aggregator = Mock(spec=LogAggregator)
        aggregator.shutdown = Mock()
        distributed.add_aggregator(aggregator)

        assert distributed._running is True

        distributed.shutdown()

        assert distributed._running is False
        aggregator.shutdown.assert_called_once()


class TestAggregationIntegration:
    """Test aggregation integration and edge cases."""

    def test_aggregator_with_multiple_forwarders(self):
        """Test aggregator with multiple forwarders."""
        aggregator = LogAggregator(buffer_size=1)

        forwarder1 = Mock(spec=LogForwarder)
        forwarder1.forward_logs.return_value = True
        forwarder2 = Mock(spec=LogForwarder)
        forwarder2.forward_logs.return_value = True

        aggregator.add_forwarder(forwarder1)
        aggregator.add_forwarder(forwarder2)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        aggregator.add_log(entry)

        # Both forwarders should receive the log
        forwarder1.forward_logs.assert_called_once()
        forwarder2.forward_logs.assert_called_once()

        aggregator.shutdown()

    def test_aggregator_with_multiple_processors(self):
        """Test aggregator with multiple processors."""
        aggregator = LogAggregator()

        def processor1(entry):
            entry.message = "P1: " + entry.message
            return entry

        def processor2(entry):
            entry.message = "P2: " + entry.message
            return entry

        aggregator.add_processor(processor1)
        aggregator.add_processor(processor2)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        aggregator.add_log(entry)

        # Message should be processed by both processors
        assert aggregator.buffer.entries[0].message == "P2: P1: Test message"

        aggregator.shutdown()

    def test_aggregator_thread_safety(self):
        """Test aggregator thread safety."""
        aggregator = LogAggregator(buffer_size=100)

        forwarder = Mock(spec=LogForwarder)
        forwarder.forward_logs.return_value = True
        aggregator.add_forwarder(forwarder)

        def worker():
            for i in range(10):
                entry = LogEntry(
                    timestamp=time.time(),
                    level=LogLevel.INFO,
                    logger_name="test",
                    message=f"Test message {i}",
                    context=LogContext(),
                )
                aggregator.add_log(entry)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Force flush to ensure all logs are processed
        aggregator.force_flush()

        # Should have received logs from all threads
        assert forwarder.forward_logs.call_count > 0

        aggregator.shutdown()

    def test_distributed_aggregator_with_multiple_nodes(self):
        """Test distributed aggregator with multiple nodes."""
        distributed1 = DistributedAggregator("node1")
        distributed2 = DistributedAggregator("node2")

        aggregator1 = LogAggregator()
        aggregator2 = LogAggregator()

        distributed1.add_aggregator(aggregator1)
        distributed2.add_aggregator(aggregator2)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        # Add log to both distributed aggregators
        distributed1.add_log(entry)
        distributed2.add_log(entry)

        # Both should have the log
        assert len(aggregator1.buffer.entries) == 1
        assert len(aggregator2.buffer.entries) == 1

        distributed1.shutdown()
        distributed2.shutdown()
        aggregator1.shutdown()
        aggregator2.shutdown()

    def test_aggregator_error_handling(self):
        """Test aggregator error handling."""
        aggregator = LogAggregator()

        # Add a processor that raises an exception
        def bad_processor(entry):
            raise ValueError("Processor error")

        aggregator.add_processor(bad_processor)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        # Should not crash on processor error
        aggregator.add_log(entry)

        aggregator.shutdown()

    def test_forwarder_error_handling(self):
        """Test forwarder error handling."""
        aggregator = LogAggregator()

        forwarder = Mock(spec=LogForwarder)
        forwarder.forward_logs.side_effect = Exception("Forwarder error")
        aggregator.add_forwarder(forwarder)

        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            logger_name="test",
            message="Test message",
            context=LogContext(),
        )

        aggregator.add_log(entry)
        aggregator.force_flush()

        # Should have attempted to forward despite error
        forwarder.forward_logs.assert_called()

        aggregator.shutdown()
