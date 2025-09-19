"""
Unit tests for bridge security module.
"""

from unittest.mock import Mock, patch

import pytest

from dubchain.bridge.bridge_security import (
    BridgeSecurity,
    FraudDetection,
    SecurityValidator,
)


class TestBridgeSecurity:
    """Test BridgeSecurity class."""

    def test_bridge_security_creation(self):
        """Test BridgeSecurity creation."""
        security = BridgeSecurity()
        assert security is not None
        assert hasattr(security, "validator")
        assert hasattr(security, "fraud_detection")

    def test_validate_transaction(self):
        """Test transaction validation."""
        security = BridgeSecurity()
        transaction = {
            "message_id": "tx_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": 100,
            "timestamp": 1234567890,
        }

        result = security.validate_transaction(transaction)
        assert isinstance(result, bool)

    def test_validate_large_transaction(self):
        """Test validation of large transaction (should be flagged as fraud)."""
        security = BridgeSecurity()
        transaction = {
            "message_id": "tx_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": 2000000,  # Large amount
            "timestamp": 1234567890,
        }

        result = security.validate_transaction(transaction)
        assert result is False  # Should be blocked due to fraud detection

    def test_get_security_metrics(self):
        """Test getting security metrics."""
        security = BridgeSecurity()
        metrics = security.get_security_metrics()

        assert isinstance(metrics, dict)
        assert "security_metrics" in metrics
        assert "monitoring_metrics" in metrics


class TestSecurityValidator:
    """Test SecurityValidator class."""

    def test_security_validator_creation(self):
        """Test SecurityValidator creation."""
        validator = SecurityValidator()
        assert validator is not None
        assert hasattr(validator, "security_rules")

    def test_validate_transaction(self):
        """Test transaction validation."""
        validator = SecurityValidator()
        transaction = {
            "message_id": "msg_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": 100,
            "timestamp": 1234567890,
        }

        result = validator.validate_transaction(transaction)
        assert isinstance(result, bool)

    def test_validate_zero_amount_transaction(self):
        """Test validation of zero amount transaction."""
        validator = SecurityValidator()
        transaction = {
            "message_id": "msg_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": 0,
            "timestamp": 1234567890,
        }

        result = validator.validate_transaction(transaction)
        assert result is False  # Should fail for zero amount

    def test_validate_negative_amount_transaction(self):
        """Test validation of negative amount transaction."""
        validator = SecurityValidator()
        transaction = {
            "message_id": "msg_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": -100,
            "timestamp": 1234567890,
        }

        result = validator.validate_transaction(transaction)
        assert result is False  # Should fail for negative amount


class TestFraudDetection:
    """Test FraudDetection class."""

    def test_fraud_detection_creation(self):
        """Test FraudDetection creation."""
        fraud_detection = FraudDetection()
        assert fraud_detection is not None
        assert hasattr(fraud_detection, "suspicious_patterns")
        assert hasattr(fraud_detection, "detection_threshold")

    def test_detect_fraud(self):
        """Test fraud detection."""
        fraud_detection = FraudDetection()
        transaction = {
            "message_id": "tx_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": 100,
            "timestamp": 1234567890,
        }

        result = fraud_detection.detect_fraud(transaction)
        assert isinstance(result, bool)

    def test_detect_fraud_large_amount(self):
        """Test fraud detection with large amount."""
        fraud_detection = FraudDetection()
        transaction = {
            "message_id": "tx_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": 2000000,  # Large amount
            "timestamp": 1234567890,
        }

        result = fraud_detection.detect_fraud(transaction)
        assert result is True  # Should detect fraud

    def test_detect_fraud_small_amount(self):
        """Test fraud detection with small amount."""
        fraud_detection = FraudDetection()
        transaction = {
            "message_id": "tx_1",
            "source_chain": "chain_1",
            "target_chain": "chain_2",
            "amount": 100,  # Small amount
            "timestamp": 1234567890,
        }

        result = fraud_detection.detect_fraud(transaction)
        assert result is False  # Should not detect fraud
