"""Error recovery mechanisms for DubChain.

This module provides comprehensive error recovery mechanisms including
circuit breakers, retry policies, backoff strategies, and recovery managers.
"""

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .exceptions import DubChainError, RetryableError


class RecoveryStrategy(Enum):
    """Recovery strategies."""

    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    ESCALATE = "escalate"


class RecoveryAction(Enum):
    """Recovery actions."""

    RETRY = "retry"
    FALLBACK = "fallback"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    ABORT = "abort"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class RetryPolicy:
    """Retry policy configuration."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[type] = field(default_factory=lambda: [RetryableError])

    def get_delay(self, attempt: int) -> float:
        """Get delay for the given attempt."""
        if attempt <= 0:
            return 0.0

        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter if enabled
        if self.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay


@dataclass
class BackoffStrategy:
    """Backoff strategy configuration."""

    strategy_type: str = "exponential"  # exponential, linear, fixed
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """Get delay for the given attempt."""
        if attempt <= 0:
            return 0.0

        if self.strategy_type == "exponential":
            delay = self.base_delay * (self.multiplier ** (attempt - 1))
        elif self.strategy_type == "linear":
            delay = self.base_delay * attempt
        elif self.strategy_type == "fixed":
            delay = self.base_delay
        else:
            delay = self.base_delay

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter if enabled
        if self.jitter:
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"{__name__}.{name}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._logger.info(
                        f"Circuit breaker {self.name} transitioning to HALF_OPEN"
                    )
                else:
                    raise DubChainError(
                        f"Circuit breaker {self.name} is OPEN",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        retryable=True,
                    )

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result

            except self.expected_exception as e:
                self._on_failure()
                raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._logger.info(
                    f"Circuit breaker {self.name} transitioning to CLOSED"
                )

            self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._logger.warning(
                    f"Circuit breaker {self.name} transitioning to OPEN "
                    f"(failure count: {self.failure_count})"
                )

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }

    def reset(self) -> None:
        """Reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            self._logger.info(f"Circuit breaker {self.name} reset")


class ErrorRecoveryManager:
    """Error recovery manager."""

    def __init__(self):
        self._recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self._retry_policies: Dict[str, RetryPolicy] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._fallback_handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # Default configurations
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Setup default recovery configurations."""
        # Default retry policy
        self._retry_policies["default"] = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Default circuit breaker
        self._circuit_breakers["default"] = CircuitBreaker(
            failure_threshold=5, recovery_timeout=60.0, name="default"
        )

    def register_recovery_strategy(
        self,
        operation: str,
        strategy: RecoveryStrategy,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a recovery strategy for an operation."""
        with self._lock:
            self._recovery_strategies[operation] = strategy

            if config:
                if strategy == RecoveryStrategy.RETRY:
                    self._retry_policies[operation] = RetryPolicy(**config)
                elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    self._circuit_breakers[operation] = CircuitBreaker(**config)

    def register_fallback_handler(self, operation: str, handler: Callable) -> None:
        """Register a fallback handler for an operation."""
        with self._lock:
            self._fallback_handlers[operation] = handler

    def execute_with_recovery(
        self, operation: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with recovery mechanisms."""
        strategy = self._recovery_strategies.get(operation, RecoveryStrategy.RETRY)

        if strategy == RecoveryStrategy.RETRY:
            return self._execute_with_retry(operation, func, *args, **kwargs)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._execute_with_circuit_breaker(operation, func, *args, **kwargs)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_with_fallback(operation, func, *args, **kwargs)
        elif strategy == RecoveryStrategy.IGNORE:
            return self._execute_with_ignore(operation, func, *args, **kwargs)
        else:
            # Default to retry
            return self._execute_with_retry(operation, func, *args, **kwargs)

    def _execute_with_retry(
        self, operation: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        policy = self._retry_policies.get(operation, self._retry_policies["default"])
        last_exception = None

        for attempt in range(policy.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if exception is retryable
                if not self._is_retryable_exception(e, policy):
                    raise

                # Check if we've exhausted retries
                if attempt >= policy.max_retries:
                    break

                # Calculate delay and wait
                delay = policy.get_delay(attempt + 1)
                self._logger.warning(
                    f"Retry attempt {attempt + 1}/{policy.max_retries} for operation '{operation}' "
                    f"after {delay:.2f}s delay. Error: {e}"
                )
                time.sleep(delay)

        # All retries exhausted
        raise DubChainError(
            f"Operation '{operation}' failed after {policy.max_retries} retries",
            error_code="MAX_RETRIES_EXCEEDED",
            cause=last_exception,
            retryable=False,
        )

    def _execute_with_circuit_breaker(
        self, operation: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        circuit_breaker = self._circuit_breakers.get(
            operation, self._circuit_breakers["default"]
        )
        return circuit_breaker.call(func, *args, **kwargs)

    def _execute_with_fallback(
        self, operation: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with fallback handler."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            fallback_handler = self._fallback_handlers.get(operation)
            if fallback_handler:
                self._logger.warning(
                    f"Operation '{operation}' failed, using fallback handler. Error: {e}"
                )
                return fallback_handler(*args, **kwargs)
            else:
                raise

    def _execute_with_ignore(
        self, operation: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function and ignore errors."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self._logger.warning(f"Ignoring error in operation '{operation}': {e}")
            return None

    def _is_retryable_exception(
        self, exception: Exception, policy: RetryPolicy
    ) -> bool:
        """Check if exception is retryable."""
        return any(
            isinstance(exception, exc_type) for exc_type in policy.retryable_exceptions
        )

    def get_circuit_breaker_state(self, operation: str) -> Optional[CircuitState]:
        """Get circuit breaker state for operation."""
        circuit_breaker = self._circuit_breakers.get(operation)
        if circuit_breaker:
            return circuit_breaker.get_state()
        return None

    def reset_circuit_breaker(self, operation: str) -> None:
        """Reset circuit breaker for operation."""
        circuit_breaker = self._circuit_breakers.get(operation)
        if circuit_breaker:
            circuit_breaker.reset()

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics."""
        with self._lock:
            metrics = {
                "recovery_strategies": dict(self._recovery_strategies),
                "circuit_breakers": {},
                "retry_policies": {},
            }

            for name, cb in self._circuit_breakers.items():
                metrics["circuit_breakers"][name] = cb.get_metrics()

            for name, policy in self._retry_policies.items():
                metrics["retry_policies"][name] = {
                    "max_retries": policy.max_retries,
                    "base_delay": policy.base_delay,
                    "max_delay": policy.max_delay,
                    "exponential_base": policy.exponential_base,
                    "jitter": policy.jitter,
                }

            return metrics

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on recovery mechanisms."""
        with self._lock:
            health = {
                "status": "healthy",
                "circuit_breakers": {},
                "timestamp": time.time(),
            }

            for name, cb in self._circuit_breakers.items():
                state = cb.get_state()
                health["circuit_breakers"][name] = {
                    "state": state.value,
                    "healthy": state != CircuitState.OPEN,
                }

                if state == CircuitState.OPEN:
                    health["status"] = "degraded"

            return health


# Convenience functions
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Decorator for retry functionality."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            policy = RetryPolicy(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
            )

            last_exception = None
            for attempt in range(policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt >= policy.max_retries:
                        break

                    delay = policy.get_delay(attempt + 1)
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type = Exception,
):
    """Decorator for circuit breaker functionality."""

    def decorator(func: Callable) -> Callable:
        cb = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=func.__name__,
        )

        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)

        return wrapper

    return decorator
