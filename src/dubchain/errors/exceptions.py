"""Exception hierarchy for DubChain.

This module defines a comprehensive exception hierarchy for the DubChain
blockchain platform, providing structured error handling and categorization.
"""

import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""

    VALIDATION = "validation"
    CRYPTOGRAPHIC = "cryptographic"
    NETWORK = "network"
    STORAGE = "storage"
    CONSENSUS = "consensus"
    TRANSACTION = "transaction"
    BLOCK = "block"
    CHAIN = "chain"
    NODE = "node"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for an error."""

    timestamp: float = field(default_factory=time.time)
    node_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }


class DubChainError(Exception):
    """Base exception for all DubChain errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        retryable: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.cause = cause
        self.retryable = retryable
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
            "retryable": self.retryable,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "traceback": self.traceback,
        }

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"{self.__class__.__name__}: {self.message}"]

        if self.error_code:
            parts.append(f"Code: {self.error_code}")

        if self.severity != ErrorSeverity.MEDIUM:
            parts.append(f"Severity: {self.severity.value}")

        if self.category != ErrorCategory.SYSTEM:
            parts.append(f"Category: {self.category.value}")

        if self.retryable:
            parts.append("Retryable: Yes")

        return " | ".join(parts)


class ValidationError(DubChainError):
    """Validation error."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        expected: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)
        self.field = field
        self.value = value
        self.expected = expected

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation error to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "field": self.field,
                "value": str(self.value) if self.value is not None else None,
                "expected": str(self.expected) if self.expected is not None else None,
            }
        )
        return data


class CryptographicError(DubChainError):
    """Cryptographic error."""

    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        key_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            category=ErrorCategory.CRYPTOGRAPHIC,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        self.algorithm = algorithm
        self.key_type = key_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert cryptographic error to dictionary."""
        data = super().to_dict()
        data.update({"algorithm": self.algorithm, "key_type": self.key_type})
        return data


class NetworkError(DubChainError):
    """Network error."""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message, category=ErrorCategory.NETWORK, retryable=True, **kwargs
        )
        self.endpoint = endpoint
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        """Convert network error to dictionary."""
        data = super().to_dict()
        data.update({"endpoint": self.endpoint, "status_code": self.status_code})
        return data


class StorageError(DubChainError):
    """Storage error."""

    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.STORAGE, **kwargs)
        self.storage_type = storage_type
        self.operation = operation

    def to_dict(self) -> Dict[str, Any]:
        """Convert storage error to dictionary."""
        data = super().to_dict()
        data.update({"storage_type": self.storage_type, "operation": self.operation})
        return data


class ConsensusError(DubChainError):
    """Consensus error."""

    def __init__(
        self,
        message: str,
        consensus_type: Optional[str] = None,
        round_number: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            category=ErrorCategory.CONSENSUS,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        self.consensus_type = consensus_type
        self.round_number = round_number

    def to_dict(self) -> Dict[str, Any]:
        """Convert consensus error to dictionary."""
        data = super().to_dict()
        data.update(
            {"consensus_type": self.consensus_type, "round_number": self.round_number}
        )
        return data


class TransactionError(DubChainError):
    """Transaction error."""

    def __init__(
        self,
        message: str,
        transaction_id: Optional[str] = None,
        transaction_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.TRANSACTION, **kwargs)
        self.transaction_id = transaction_id
        self.transaction_type = transaction_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction error to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "transaction_id": self.transaction_id,
                "transaction_type": self.transaction_type,
            }
        )
        return data


class BlockError(DubChainError):
    """Block error."""

    def __init__(
        self,
        message: str,
        block_hash: Optional[str] = None,
        block_height: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.BLOCK, **kwargs)
        self.block_hash = block_hash
        self.block_height = block_height

    def to_dict(self) -> Dict[str, Any]:
        """Convert block error to dictionary."""
        data = super().to_dict()
        data.update({"block_hash": self.block_hash, "block_height": self.block_height})
        return data


class ChainError(DubChainError):
    """Chain error."""

    def __init__(
        self,
        message: str,
        chain_id: Optional[str] = None,
        fork_depth: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message, category=ErrorCategory.CHAIN, severity=ErrorSeverity.HIGH, **kwargs
        )
        self.chain_id = chain_id
        self.fork_depth = fork_depth

    def to_dict(self) -> Dict[str, Any]:
        """Convert chain error to dictionary."""
        data = super().to_dict()
        data.update({"chain_id": self.chain_id, "fork_depth": self.fork_depth})
        return data


class NodeError(DubChainError):
    """Node error."""

    def __init__(
        self,
        message: str,
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.NODE, **kwargs)
        self.node_id = node_id
        self.node_type = node_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert node error to dictionary."""
        data = super().to_dict()
        data.update({"node_id": self.node_id, "node_type": self.node_type})
        return data


class ConfigurationError(DubChainError):
    """Configuration error."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        self.config_key = config_key
        self.config_value = config_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration error to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "config_key": self.config_key,
                "config_value": str(self.config_value)
                if self.config_value is not None
                else None,
            }
        )
        return data


class ResourceError(DubChainError):
    """Resource error."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, category=ErrorCategory.RESOURCE, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert resource error to dictionary."""
        data = super().to_dict()
        data.update(
            {"resource_type": self.resource_type, "resource_id": self.resource_id}
        )
        return data


class TimeoutError(DubChainError):
    """Timeout error."""

    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message, category=ErrorCategory.TIMEOUT, retryable=True, **kwargs
        )
        self.timeout_duration = timeout_duration
        self.operation = operation

    def to_dict(self) -> Dict[str, Any]:
        """Convert timeout error to dictionary."""
        data = super().to_dict()
        data.update(
            {"timeout_duration": self.timeout_duration, "operation": self.operation}
        )
        return data


class RetryableError(DubChainError):
    """Retryable error."""

    def __init__(
        self,
        message: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, retryable=True, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def to_dict(self) -> Dict[str, Any]:
        """Convert retryable error to dictionary."""
        data = super().to_dict()
        data.update({"max_retries": self.max_retries, "retry_delay": self.retry_delay})
        return data


class FatalError(DubChainError):
    """Fatal error that cannot be recovered from."""

    def __init__(self, message: str, shutdown_required: bool = True, **kwargs):
        super().__init__(
            message, severity=ErrorSeverity.CRITICAL, retryable=False, **kwargs
        )
        self.shutdown_required = shutdown_required

    def to_dict(self) -> Dict[str, Any]:
        """Convert fatal error to dictionary."""
        data = super().to_dict()
        data.update({"shutdown_required": self.shutdown_required})
        return data


# Convenience functions for common error patterns
def create_validation_error(
    field: str, value: Any, expected: Any, message: Optional[str] = None
) -> ValidationError:
    """Create a validation error."""
    if message is None:
        message = f"Invalid value for field '{field}': expected {expected}, got {value}"

    return ValidationError(message=message, field=field, value=value, expected=expected)


def create_network_error(
    endpoint: str, status_code: Optional[int] = None, message: Optional[str] = None
) -> NetworkError:
    """Create a network error."""
    if message is None:
        if status_code:
            message = f"Network error for endpoint '{endpoint}': HTTP {status_code}"
        else:
            message = f"Network error for endpoint '{endpoint}'"

    return NetworkError(message=message, endpoint=endpoint, status_code=status_code)


def create_timeout_error(
    operation: str, timeout_duration: float, message: Optional[str] = None
) -> TimeoutError:
    """Create a timeout error."""
    if message is None:
        message = f"Operation '{operation}' timed out after {timeout_duration} seconds"

    return TimeoutError(
        message=message, operation=operation, timeout_duration=timeout_duration
    )


def create_fatal_error(message: str, shutdown_required: bool = True) -> FatalError:
    """Create a fatal error."""
    return FatalError(message=message, shutdown_required=shutdown_required)
