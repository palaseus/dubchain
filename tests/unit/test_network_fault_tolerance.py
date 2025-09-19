"""Test cases for network/fault_tolerance.py module."""

import time
from unittest.mock import Mock, patch

import pytest

from dubchain.network.fault_tolerance import (
    AutoRecovery,
    ByzantineDetector,
    FaultEvent,
    FaultTolerance,
    FaultToleranceConfig,
    FaultType,
    NetworkPartitionDetector,
)


class TestFaultType:
    """Test FaultType enum."""

    def test_fault_type_values(self):
        """Test FaultType enum values."""
        assert FaultType.NODE_FAILURE.value == "node_failure"
        assert FaultType.NETWORK_PARTITION.value == "network_partition"
        assert FaultType.BYZANTINE_BEHAVIOR.value == "byzantine_behavior"
        assert FaultType.MESSAGE_LOSS.value == "message_loss"
        assert FaultType.CONNECTION_TIMEOUT.value == "connection_timeout"


class TestFaultEvent:
    """Test FaultEvent dataclass."""

    def test_fault_event_creation(self):
        """Test FaultEvent creation."""
        event = FaultEvent(
            fault_type=FaultType.NODE_FAILURE,
            peer_id="peer1",
            timestamp=time.time(),
            severity=0.8,
            description="Node failure detected",
        )

        assert event.fault_type == FaultType.NODE_FAILURE
        assert event.peer_id == "peer1"
        assert event.severity == 0.8
        assert event.description == "Node failure detected"
        assert event.metadata == {}

    def test_fault_event_with_metadata(self):
        """Test FaultEvent creation with metadata."""
        metadata = {"error_code": 500, "retry_count": 3}
        event = FaultEvent(
            fault_type=FaultType.CONNECTION_TIMEOUT,
            peer_id="peer2",
            timestamp=time.time(),
            severity=0.5,
            description="Connection timeout",
            metadata=metadata,
        )

        assert event.metadata == metadata


class TestFaultToleranceConfig:
    """Test FaultToleranceConfig dataclass."""

    def test_fault_tolerance_config_defaults(self):
        """Test FaultToleranceConfig with default values."""
        config = FaultToleranceConfig()

        assert config.max_faulty_peers == 3
        assert config.heartbeat_interval == 30.0
        assert config.timeout_threshold == 60.0
        assert config.enable_byzantine_detection is True
        assert config.enable_auto_recovery is True
        assert config.metadata == {}

    def test_fault_tolerance_config_custom_values(self):
        """Test FaultToleranceConfig with custom values."""
        metadata = {"custom_setting": "value"}
        config = FaultToleranceConfig(
            max_faulty_peers=5,
            heartbeat_interval=15.0,
            timeout_threshold=30.0,
            enable_byzantine_detection=False,
            enable_auto_recovery=False,
            metadata=metadata,
        )

        assert config.max_faulty_peers == 5
        assert config.heartbeat_interval == 15.0
        assert config.timeout_threshold == 30.0
        assert config.enable_byzantine_detection is False
        assert config.enable_auto_recovery is False
        assert config.metadata == metadata


class TestByzantineDetector:
    """Test ByzantineDetector class."""

    def test_byzantine_detector_creation(self):
        """Test ByzantineDetector creation."""
        config = FaultToleranceConfig()
        detector = ByzantineDetector(config)

        assert detector.config == config
        assert detector.suspicious_peers == {}
        assert detector.byzantine_peers == set()

    def test_detect_byzantine_behavior_inconsistent_messages(self):
        """Test detect_byzantine_behavior with inconsistent messages."""
        config = FaultToleranceConfig()
        detector = ByzantineDetector(config)

        behavior_data = {"inconsistent_messages": 6}
        result = detector.detect_byzantine_behavior("peer1", behavior_data)

        assert result is True

    def test_detect_byzantine_behavior_malicious_actions(self):
        """Test detect_byzantine_behavior with malicious actions."""
        config = FaultToleranceConfig()
        detector = ByzantineDetector(config)

        behavior_data = {"malicious_actions": 1}
        result = detector.detect_byzantine_behavior("peer1", behavior_data)

        assert result is True

    def test_detect_byzantine_behavior_normal_behavior(self):
        """Test detect_byzantine_behavior with normal behavior."""
        config = FaultToleranceConfig()
        detector = ByzantineDetector(config)

        behavior_data = {"inconsistent_messages": 3, "malicious_actions": 0}
        result = detector.detect_byzantine_behavior("peer1", behavior_data)

        assert result is False

    def test_is_byzantine_peer_true(self):
        """Test is_byzantine_peer returns True for Byzantine peer."""
        config = FaultToleranceConfig()
        detector = ByzantineDetector(config)
        detector.byzantine_peers.add("peer1")

        result = detector.is_byzantine_peer("peer1")

        assert result is True

    def test_is_byzantine_peer_false(self):
        """Test is_byzantine_peer returns False for normal peer."""
        config = FaultToleranceConfig()
        detector = ByzantineDetector(config)

        result = detector.is_byzantine_peer("peer1")

        assert result is False


class TestNetworkPartitionDetector:
    """Test NetworkPartitionDetector class."""

    def test_network_partition_detector_creation(self):
        """Test NetworkPartitionDetector creation."""
        config = FaultToleranceConfig()
        detector = NetworkPartitionDetector(config)

        assert detector.config == config
        assert detector.partitions == []
        assert detector.last_connectivity_check == {}

    def test_detect_partition_single_partition(self):
        """Test detect_partition with single connected component."""
        config = FaultToleranceConfig()
        detector = NetworkPartitionDetector(config)

        peer_connections = {
            "peer1": {"peer2", "peer3"},
            "peer2": {"peer1", "peer3"},
            "peer3": {"peer1", "peer2"},
        }

        partitions = detector.detect_partition(peer_connections)

        assert len(partitions) == 1
        assert partitions[0] == {"peer1", "peer2", "peer3"}

    def test_detect_partition_multiple_partitions(self):
        """Test detect_partition with multiple disconnected components."""
        config = FaultToleranceConfig()
        detector = NetworkPartitionDetector(config)

        peer_connections = {
            "peer1": {"peer2"},
            "peer2": {"peer1"},
            "peer3": {"peer4"},
            "peer4": {"peer3"},
            "peer5": set(),
        }

        partitions = detector.detect_partition(peer_connections)

        assert len(partitions) == 3
        # Check that all peers are in exactly one partition
        all_peers = set()
        for partition in partitions:
            all_peers.update(partition)
        assert all_peers == {"peer1", "peer2", "peer3", "peer4", "peer5"}

    def test_detect_partition_empty_connections(self):
        """Test detect_partition with empty connections."""
        config = FaultToleranceConfig()
        detector = NetworkPartitionDetector(config)

        peer_connections = {}

        partitions = detector.detect_partition(peer_connections)

        assert len(partitions) == 0

    def test_dfs_partition_visited_peer(self):
        """Test _dfs_partition with already visited peer."""
        config = FaultToleranceConfig()
        detector = NetworkPartitionDetector(config)

        connections = {"peer1": {"peer2"}}
        visited = {"peer1"}

        result = detector._dfs_partition("peer1", connections, visited)

        assert result == set()

    def test_dfs_partition_new_peer(self):
        """Test _dfs_partition with new peer."""
        config = FaultToleranceConfig()
        detector = NetworkPartitionDetector(config)

        connections = {"peer1": {"peer2"}, "peer2": {"peer1"}}
        visited = set()

        result = detector._dfs_partition("peer1", connections, visited)

        assert result == {"peer1", "peer2"}
        assert visited == {"peer1", "peer2"}


class TestAutoRecovery:
    """Test AutoRecovery class."""

    def test_auto_recovery_creation(self):
        """Test AutoRecovery creation."""
        config = FaultToleranceConfig()
        recovery = AutoRecovery(config)

        assert recovery.config == config
        assert recovery.recovery_attempts == {}
        assert recovery.max_recovery_attempts == 3

    def test_attempt_recovery_connection_timeout(self):
        """Test attempt_recovery with connection timeout."""
        config = FaultToleranceConfig()
        recovery = AutoRecovery(config)

        result = recovery.attempt_recovery("peer1", FaultType.CONNECTION_TIMEOUT)

        assert result is True
        assert recovery.recovery_attempts["peer1"] == 1

    def test_attempt_recovery_other_fault_type(self):
        """Test attempt_recovery with other fault type."""
        config = FaultToleranceConfig()
        recovery = AutoRecovery(config)

        result = recovery.attempt_recovery("peer1", FaultType.NODE_FAILURE)

        assert result is False
        assert recovery.recovery_attempts["peer1"] == 1

    def test_attempt_recovery_max_attempts_reached(self):
        """Test attempt_recovery when max attempts reached."""
        config = FaultToleranceConfig()
        recovery = AutoRecovery(config)
        recovery.recovery_attempts["peer1"] = 3

        result = recovery.attempt_recovery("peer1", FaultType.CONNECTION_TIMEOUT)

        assert result is False
        assert recovery.recovery_attempts["peer1"] == 3  # Should not increment

    def test_attempt_recovery_multiple_attempts(self):
        """Test attempt_recovery with multiple attempts."""
        config = FaultToleranceConfig()
        recovery = AutoRecovery(config)

        # First attempt
        result1 = recovery.attempt_recovery("peer1", FaultType.CONNECTION_TIMEOUT)
        assert result1 is True
        assert recovery.recovery_attempts["peer1"] == 1

        # Second attempt
        result2 = recovery.attempt_recovery("peer1", FaultType.CONNECTION_TIMEOUT)
        assert result2 is True
        assert recovery.recovery_attempts["peer1"] == 2

        # Third attempt
        result3 = recovery.attempt_recovery("peer1", FaultType.CONNECTION_TIMEOUT)
        assert result3 is True
        assert recovery.recovery_attempts["peer1"] == 3

        # Fourth attempt should fail
        result4 = recovery.attempt_recovery("peer1", FaultType.CONNECTION_TIMEOUT)
        assert result4 is False
        assert recovery.recovery_attempts["peer1"] == 3

    def test_reset_recovery_attempts(self):
        """Test reset_recovery_attempts."""
        config = FaultToleranceConfig()
        recovery = AutoRecovery(config)
        recovery.recovery_attempts["peer1"] = 2

        recovery.reset_recovery_attempts("peer1")

        assert "peer1" not in recovery.recovery_attempts

    def test_reset_recovery_attempts_nonexistent_peer(self):
        """Test reset_recovery_attempts with nonexistent peer."""
        config = FaultToleranceConfig()
        recovery = AutoRecovery(config)

        # Should not raise exception
        recovery.reset_recovery_attempts("nonexistent_peer")


class TestFaultTolerance:
    """Test FaultTolerance class."""

    def test_fault_tolerance_creation(self):
        """Test FaultTolerance creation."""
        config = FaultToleranceConfig()
        fault_tolerance = FaultTolerance(config)

        assert fault_tolerance.config == config
        assert fault_tolerance.byzantine_detector is not None
        assert fault_tolerance.partition_detector is not None
        assert fault_tolerance.auto_recovery is not None
        assert fault_tolerance.fault_history == []

    def test_handle_fault_byzantine_behavior(self):
        """Test handle_fault with Byzantine behavior."""
        config = FaultToleranceConfig()
        fault_tolerance = FaultTolerance(config)

        fault_event = FaultEvent(
            fault_type=FaultType.BYZANTINE_BEHAVIOR,
            peer_id="peer1",
            timestamp=time.time(),
            severity=1.0,
            description="Byzantine behavior detected",
        )

        result = fault_tolerance.handle_fault(fault_event)

        assert result is False
        assert len(fault_tolerance.fault_history) == 1
        assert fault_tolerance.fault_history[0] == fault_event
        assert "peer1" in fault_tolerance.byzantine_detector.byzantine_peers

    def test_handle_fault_with_auto_recovery_enabled(self):
        """Test handle_fault with auto recovery enabled."""
        config = FaultToleranceConfig(enable_auto_recovery=True)
        fault_tolerance = FaultTolerance(config)

        fault_event = FaultEvent(
            fault_type=FaultType.CONNECTION_TIMEOUT,
            peer_id="peer1",
            timestamp=time.time(),
            severity=0.5,
            description="Connection timeout",
        )

        result = fault_tolerance.handle_fault(fault_event)

        assert result is True  # Auto recovery should succeed for connection timeout
        assert len(fault_tolerance.fault_history) == 1
        assert fault_tolerance.fault_history[0] == fault_event

    def test_handle_fault_with_auto_recovery_disabled(self):
        """Test handle_fault with auto recovery disabled."""
        config = FaultToleranceConfig(enable_auto_recovery=False)
        fault_tolerance = FaultTolerance(config)

        fault_event = FaultEvent(
            fault_type=FaultType.CONNECTION_TIMEOUT,
            peer_id="peer1",
            timestamp=time.time(),
            severity=0.5,
            description="Connection timeout",
        )

        result = fault_tolerance.handle_fault(fault_event)

        assert result is False  # Auto recovery disabled
        assert len(fault_tolerance.fault_history) == 1
        assert fault_tolerance.fault_history[0] == fault_event

    def test_detect_network_partitions(self):
        """Test detect_network_partitions."""
        config = FaultToleranceConfig()
        fault_tolerance = FaultTolerance(config)

        peer_connections = {
            "peer1": {"peer2"},
            "peer2": {"peer1"},
            "peer3": {"peer4"},
            "peer4": {"peer3"},
        }

        partitions = fault_tolerance.detect_network_partitions(peer_connections)

        assert len(partitions) == 2
        # Check that all peers are in exactly one partition
        all_peers = set()
        for partition in partitions:
            all_peers.update(partition)
        assert all_peers == {"peer1", "peer2", "peer3", "peer4"}

    def test_is_peer_faulty_true(self):
        """Test is_peer_faulty returns True for Byzantine peer."""
        config = FaultToleranceConfig()
        fault_tolerance = FaultTolerance(config)
        fault_tolerance.byzantine_detector.byzantine_peers.add("peer1")

        result = fault_tolerance.is_peer_faulty("peer1")

        assert result is True

    def test_is_peer_faulty_false(self):
        """Test is_peer_faulty returns False for normal peer."""
        config = FaultToleranceConfig()
        fault_tolerance = FaultTolerance(config)

        result = fault_tolerance.is_peer_faulty("peer1")

        assert result is False
