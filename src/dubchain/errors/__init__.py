"""DubChain Error Handling System.

This module provides a comprehensive error handling system for the DubChain
blockchain platform, including exception hierarchies, recovery mechanisms,
graceful degradation, and telemetry.
"""

from .degradation import (
    DegradationLevel,
    DegradationStrategy,
    GracefulDegradationManager,
    HealthChecker,
    ServiceLevel,
)
from .exceptions import (
    BlockError,
    BridgeError,
    ChainError,
    ClientError,
    ConfigurationError,
    ConsensusError,
    CryptographicError,
    DubChainError,
    FatalError,
    MonitoringError,
    NetworkError,
    NodeError,
    ResourceError,
    RetryableError,
    StorageError,
    TimeoutError,
    TransactionError,
    ValidationError,
)
from .recovery import (
    BackoffStrategy,
    CircuitBreaker,
    ErrorRecoveryManager,
    RecoveryAction,
    RecoveryStrategy,
    RetryPolicy,
)
from .telemetry import (
    ErrorAggregator,
    ErrorDashboard,
    ErrorMetrics,
    ErrorReporter,
    ErrorTelemetry,
)

__all__ = [
    # Exceptions
    "DubChainError",
    "ValidationError",
    "CryptographicError",
    "NetworkError",
    "StorageError",
    "ConsensusError",
    "TransactionError",
    "BlockError",
    "ChainError",
    "NodeError",
    "ConfigurationError",
    "ResourceError",
    "TimeoutError",
    "RetryableError",
    "FatalError",
    "BridgeError",
    "MonitoringError",
    "ClientError",
    # Recovery
    "ErrorRecoveryManager",
    "RecoveryStrategy",
    "RecoveryAction",
    "CircuitBreaker",
    "RetryPolicy",
    "BackoffStrategy",
    # Degradation
    "GracefulDegradationManager",
    "DegradationLevel",
    "DegradationStrategy",
    "ServiceLevel",
    "HealthChecker",
    # Telemetry
    "ErrorTelemetry",
    "ErrorMetrics",
    "ErrorReporter",
    "ErrorAggregator",
    "ErrorDashboard",
]
