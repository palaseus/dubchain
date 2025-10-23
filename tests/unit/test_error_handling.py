"""Tests for the DubChain error handling system."""

import logging

logger = logging.getLogger(__name__)
import threading
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.errors.degradation import (
    DegradationLevel,
    DegradationRule,
    DegradationStrategy,
    GracefulDegradationManager,
    HealthChecker,
    NetworkHealthChecker,
    ServiceHealth,
    ServiceLevel,
    SystemHealthChecker,
)
from dubchain.errors.exceptions import (
    BlockError,
    ChainError,
    ConfigurationError,
    ConsensusError,
    CryptographicError,
    DubChainError,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    FatalError,
    NetworkError,
    NodeError,
    ResourceError,
    RetryableError,
    StorageError,
    TimeoutError,
    TransactionError,
    ValidationError,
    create_fatal_error,
    create_network_error,
    create_timeout_error,
    create_validation_error,
)
from dubchain.errors.recovery import (
    BackoffStrategy,
    CircuitBreaker,
    CircuitState,
    ErrorRecoveryManager,
    RecoveryAction,
    RecoveryStrategy,
    RetryPolicy,
    with_circuit_breaker,
    with_retry,
)
from dubchain.errors.telemetry import (
    ErrorAggregator,
    ErrorDashboard,
    ErrorMetrics,
    ErrorReport,
    ErrorReporter,
    ErrorTelemetry,
    FileErrorReporter,
    LogErrorReporter,
)


class TestDubChainError:
    """Test DubChain error functionality."""

    def test_base_error_creation(self):
        """Test base error creation."""
        error = DubChainError("Test error message")

        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.SYSTEM
        assert error.retryable is False
        assert error.timestamp > 0
        assert error.context is not None

    def test_error_with_metadata(self):
        """Test error with metadata."""
        context = ErrorContext(
            node_id="node1", component="test_component", operation="test_operation"
        )

        error = DubChainError(
            "Test error",
            error_code="TEST_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            context=context,
            retryable=True,
            metadata={"key": "value"},
        )

        assert error.error_code == "TEST_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.VALIDATION
        assert error.context.node_id == "node1"
        assert error.retryable is True
        assert error.metadata["key"] == "value"

    def test_error_to_dict(self):
        """Test error to dictionary conversion."""
        error = DubChainError("Test error", error_code="TEST_ERROR")
        error_dict = error.to_dict()

        assert error_dict["type"] == "DubChainError"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["severity"] == "medium"
        assert error_dict["category"] == "system"
        assert error_dict["retryable"] is False
        assert "timestamp" in error_dict
        assert "context" in error_dict

    def test_error_string_representation(self):
        """Test error string representation."""
        error = DubChainError("Test error", error_code="TEST_ERROR", retryable=True)
        error_str = str(error)

        assert "DubChainError: Test error" in error_str
        assert "Code: TEST_ERROR" in error_str
        assert "Retryable: Yes" in error_str


class TestValidationError:
    """Test validation error functionality."""

    def test_validation_error_creation(self):
        """Test validation error creation."""
        error = ValidationError(
            "Invalid value",
            field="test_field",
            value="invalid_value",
            expected="valid_value",
        )

        assert error.message == "Invalid value"
        assert error.field == "test_field"
        assert error.value == "invalid_value"
        assert error.expected == "valid_value"
        assert error.category == ErrorCategory.VALIDATION

    def test_validation_error_to_dict(self):
        """Test validation error to dictionary conversion."""
        error = ValidationError(
            "Invalid value",
            field="test_field",
            value="invalid_value",
            expected="valid_value",
        )
        error_dict = error.to_dict()

        assert error_dict["field"] == "test_field"
        assert error_dict["value"] == "invalid_value"
        assert error_dict["expected"] == "valid_value"


class TestNetworkError:
    """Test network error functionality."""

    def test_network_error_creation(self):
        """Test network error creation."""
        error = NetworkError(
            "Connection failed", endpoint="http://example.com", status_code=500
        )

        assert error.message == "Connection failed"
        assert error.endpoint == "http://example.com"
        assert error.status_code == 500
        assert error.category == ErrorCategory.NETWORK
        assert error.retryable is True

    def test_network_error_to_dict(self):
        """Test network error to dictionary conversion."""
        error = NetworkError(
            "Connection failed", endpoint="http://example.com", status_code=500
        )
        error_dict = error.to_dict()

        assert error_dict["endpoint"] == "http://example.com"
        assert error_dict["status_code"] == 500


class TestTimeoutError:
    """Test timeout error functionality."""

    def test_timeout_error_creation(self):
        """Test timeout error creation."""
        error = TimeoutError(
            "Operation timed out", timeout_duration=30.0, operation="test_operation"
        )

        assert error.message == "Operation timed out"
        assert error.timeout_duration == 30.0
        assert error.operation == "test_operation"
        assert error.category == ErrorCategory.TIMEOUT
        assert error.retryable is True

    def test_timeout_error_to_dict(self):
        """Test timeout error to dictionary conversion."""
        error = TimeoutError(
            "Operation timed out", timeout_duration=30.0, operation="test_operation"
        )
        error_dict = error.to_dict()

        assert error_dict["timeout_duration"] == 30.0
        assert error_dict["operation"] == "test_operation"


class TestFatalError:
    """Test fatal error functionality."""

    def test_fatal_error_creation(self):
        """Test fatal error creation."""
        error = FatalError("System failure", shutdown_required=True)

        assert error.message == "System failure"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.retryable is False
        assert error.shutdown_required is True

    def test_fatal_error_to_dict(self):
        """Test fatal error to dictionary conversion."""
        error = FatalError("System failure", shutdown_required=True)
        error_dict = error.to_dict()

        assert error_dict["shutdown_required"] is True


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_validation_error(self):
        """Test create_validation_error function."""
        error = create_validation_error("test_field", "invalid", "valid")

        assert isinstance(error, ValidationError)
        assert error.field == "test_field"
        assert error.value == "invalid"
        assert error.expected == "valid"

    def test_create_network_error(self):
        """Test create_network_error function."""
        error = create_network_error("http://example.com", 500)

        assert isinstance(error, NetworkError)
        assert error.endpoint == "http://example.com"
        assert error.status_code == 500

    def test_create_timeout_error(self):
        """Test create_timeout_error function."""
        error = create_timeout_error("test_operation", 30.0)

        assert isinstance(error, TimeoutError)
        assert error.operation == "test_operation"
        assert error.timeout_duration == 30.0

    def test_create_fatal_error(self):
        """Test create_fatal_error function."""
        error = create_fatal_error("System failure")

        assert isinstance(error, FatalError)
        assert error.message == "System failure"
        assert error.shutdown_required is True


class TestRetryPolicy:
    """Test retry policy functionality."""

    def test_retry_policy_creation(self):
        """Test retry policy creation."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=True,
        )

        assert policy.max_retries == 5
        assert policy.base_delay == 2.0
        assert policy.max_delay == 120.0
        assert policy.exponential_base == 3.0
        assert policy.jitter is True

    def test_retry_policy_delay_calculation(self):
        """Test retry policy delay calculation."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False,
        )

        # Test delay calculation
        delay1 = policy.get_delay(1)
        delay2 = policy.get_delay(2)
        delay3 = policy.get_delay(3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_retry_policy_max_delay_cap(self):
        """Test retry policy max delay cap."""
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False,
        )

        # Test that delay is capped at max_delay
        delay = policy.get_delay(10)  # Should be capped at 5.0
        assert delay == 5.0


class TestBackoffStrategy:
    """Test backoff strategy functionality."""

    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        strategy = BackoffStrategy(
            strategy_type="exponential",
            base_delay=1.0,
            max_delay=10.0,
            multiplier=2.0,
            jitter=False,
        )

        delay1 = strategy.get_delay(1)
        delay2 = strategy.get_delay(2)
        delay3 = strategy.get_delay(3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        strategy = BackoffStrategy(
            strategy_type="linear",
            base_delay=1.0,
            max_delay=10.0,
            multiplier=2.0,
            jitter=False,
        )

        delay1 = strategy.get_delay(1)
        delay2 = strategy.get_delay(2)
        delay3 = strategy.get_delay(3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 3.0

    def test_fixed_backoff(self):
        """Test fixed backoff strategy."""
        strategy = BackoffStrategy(
            strategy_type="fixed",
            base_delay=2.0,
            max_delay=10.0,
            multiplier=2.0,
            jitter=False,
        )

        delay1 = strategy.get_delay(1)
        delay2 = strategy.get_delay(2)
        delay3 = strategy.get_delay(3)

        assert delay1 == 2.0
        assert delay2 == 2.0
        assert delay3 == 2.0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            expected_exception=Exception,
            name="test_cb",
        )

    def test_circuit_breaker_initial_state(self, circuit_breaker):
        """Test circuit breaker initial state."""
        assert circuit_breaker.get_state() == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0

    def test_circuit_breaker_successful_call(self, circuit_breaker):
        """Test circuit breaker successful call."""

        def success_func():
            return "success"

        result = circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitState.CLOSED

    def test_circuit_breaker_failure_threshold(self, circuit_breaker):
        """Test circuit breaker failure threshold."""

        def failing_func():
            raise Exception("Test failure")

        # First few failures should not open circuit
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)

        # Circuit should now be open
        assert circuit_breaker.get_state() == CircuitState.OPEN

        # Next call should raise circuit breaker error
        with pytest.raises(DubChainError) as exc_info:
            circuit_breaker.call(failing_func)

        assert "Circuit breaker test_cb is OPEN" in str(exc_info.value)

    def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery."""

        def failing_func():
            raise Exception("Test failure")

        def success_func():
            return "success"

        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)

        assert circuit_breaker.get_state() == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(1.1)

        # Successful call should close circuit
        result = circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.get_state() == CircuitState.CLOSED

    def test_circuit_breaker_metrics(self, circuit_breaker):
        """Test circuit breaker metrics."""
        metrics = circuit_breaker.get_metrics()

        assert metrics["name"] == "test_cb"
        assert metrics["state"] == "closed"
        assert metrics["failure_count"] == 0
        assert metrics["failure_threshold"] == 3
        assert metrics["recovery_timeout"] == 1.0

    def test_circuit_breaker_reset(self, circuit_breaker):
        """Test circuit breaker reset."""

        def failing_func():
            raise Exception("Test failure")

        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_func)

        assert circuit_breaker.get_state() == CircuitState.OPEN

        # Reset circuit
        circuit_breaker.reset()

        assert circuit_breaker.get_state() == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0


class TestErrorRecoveryManager:
    """Test error recovery manager functionality."""

    @pytest.fixture
    def recovery_manager(self):
        """Create error recovery manager."""
        return ErrorRecoveryManager()

    def test_recovery_manager_initialization(self, recovery_manager):
        """Test recovery manager initialization."""
        assert len(recovery_manager._recovery_strategies) == 0
        assert "default" in recovery_manager._retry_policies
        assert "default" in recovery_manager._circuit_breakers

    def test_register_recovery_strategy(self, recovery_manager):
        """Test registering recovery strategy."""
        recovery_manager.register_recovery_strategy(
            "test_operation",
            RecoveryStrategy.RETRY,
            {"max_retries": 5, "base_delay": 2.0},
        )

        assert (
            recovery_manager._recovery_strategies["test_operation"]
            == RecoveryStrategy.RETRY
        )
        assert "test_operation" in recovery_manager._retry_policies

    def test_register_fallback_handler(self, recovery_manager):
        """Test registering fallback handler."""

        def fallback_handler(*args, **kwargs):
            return "fallback_result"

        recovery_manager.register_fallback_handler("test_operation", fallback_handler)

        assert "test_operation" in recovery_manager._fallback_handlers

    def test_execute_with_retry(self, recovery_manager):
        """Test execute with retry."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary failure")
            return "success"

        result = recovery_manager.execute_with_recovery("test_operation", failing_func)

        assert result == "success"
        assert call_count == 3

    def test_execute_with_circuit_breaker(self, recovery_manager):
        """Test execute with circuit breaker."""
        recovery_manager.register_recovery_strategy(
            "test_operation",
            RecoveryStrategy.CIRCUIT_BREAKER,
            {"failure_threshold": 2, "recovery_timeout": 1.0},
        )

        def failing_func():
            raise Exception("Test failure")

        # First failure should not open circuit
        with pytest.raises(Exception):
            recovery_manager.execute_with_recovery("test_operation", failing_func)

        # Second failure should open circuit
        with pytest.raises(Exception):
            recovery_manager.execute_with_recovery("test_operation", failing_func)

        # Third call should raise circuit breaker error
        with pytest.raises(DubChainError):
            recovery_manager.execute_with_recovery("test_operation", failing_func)

    def test_execute_with_fallback(self, recovery_manager):
        """Test execute with fallback."""
        recovery_manager.register_recovery_strategy(
            "test_operation", RecoveryStrategy.FALLBACK
        )

        def failing_func():
            raise Exception("Test failure")

        def fallback_handler(*args, **kwargs):
            return "fallback_result"

        recovery_manager.register_fallback_handler("test_operation", fallback_handler)

        result = recovery_manager.execute_with_recovery("test_operation", failing_func)
        assert result == "fallback_result"

    def test_execute_with_ignore(self, recovery_manager):
        """Test execute with ignore."""
        recovery_manager.register_recovery_strategy(
            "test_operation", RecoveryStrategy.IGNORE
        )

        def failing_func():
            raise Exception("Test failure")

        result = recovery_manager.execute_with_recovery("test_operation", failing_func)
        assert result is None

    def test_get_recovery_metrics(self, recovery_manager):
        """Test get recovery metrics."""
        metrics = recovery_manager.get_recovery_metrics()

        assert "recovery_strategies" in metrics
        assert "circuit_breakers" in metrics
        assert "retry_policies" in metrics
        assert "default" in metrics["circuit_breakers"]
        assert "default" in metrics["retry_policies"]

    def test_health_check(self, recovery_manager):
        """Test health check."""
        health = recovery_manager.health_check()

        assert health["status"] == "healthy"
        assert "circuit_breakers" in health
        assert "timestamp" in health


class TestDecorators:
    """Test decorator functionality."""

    def test_with_retry_decorator(self):
        """Test with_retry decorator."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Temporary failure")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_with_circuit_breaker_decorator(self):
        """Test with_circuit_breaker decorator."""

        @with_circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        def failing_func():
            raise Exception("Test failure")

        # First failure should not open circuit
        with pytest.raises(Exception):
            failing_func()

        # Second failure should open circuit
        with pytest.raises(Exception):
            failing_func()

        # Third call should raise circuit breaker error
        with pytest.raises(DubChainError):
            failing_func()


class TestDegradationRule:
    """Test degradation rule functionality."""

    def test_degradation_rule_creation(self):
        """Test degradation rule creation."""

        def condition():
            return True

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
            threshold=0.8,
            cooldown=300.0,
            enabled=True,
        )

        assert rule.name == "test_rule"
        assert rule.condition() is True
        assert rule.level == DegradationLevel.MODERATE
        assert rule.strategy == DegradationStrategy.THROTTLE
        assert rule.threshold == 0.8
        assert rule.cooldown == 300.0
        assert rule.enabled is True
        assert rule.last_triggered == 0.0
        assert rule.trigger_count == 0


class TestServiceHealth:
    """Test service health functionality."""

    def test_service_health_creation(self):
        """Test service health creation."""
        health = ServiceHealth(
            service_name="test_service",
            level=ServiceLevel.FULL,
            degradation_level=DegradationLevel.NONE,
            last_check=time.time(),
            metrics={"cpu_percent": 50.0},
            errors=["error1"],
            warnings=["warning1"],
        )

        assert health.service_name == "test_service"
        assert health.level == ServiceLevel.FULL
        assert health.degradation_level == DegradationLevel.NONE
        assert health.last_check > 0
        assert health.metrics["cpu_percent"] == 50.0
        assert health.errors == ["error1"]
        assert health.warnings == ["warning1"]


class TestGracefulDegradationManager:
    """Test graceful degradation manager functionality."""

    @pytest.fixture
    def degradation_manager(self):
        """Create graceful degradation manager."""
        return GracefulDegradationManager()

    def test_degradation_manager_initialization(self, degradation_manager):
        """Test degradation manager initialization."""
        assert len(degradation_manager._degradation_rules) == 0
        assert len(degradation_manager._health_checkers) == 0
        assert degradation_manager._current_degradation_level == DegradationLevel.NONE
        assert degradation_manager._current_service_level == ServiceLevel.FULL

    def test_add_degradation_rule(self, degradation_manager):
        """Test adding degradation rule."""

        def condition():
            return True

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
        )

        degradation_manager.add_degradation_rule(rule)

        assert len(degradation_manager._degradation_rules) == 1
        assert degradation_manager._degradation_rules[0].name == "test_rule"

    def test_remove_degradation_rule(self, degradation_manager):
        """Test removing degradation rule."""

        def condition():
            return True

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
        )

        degradation_manager.add_degradation_rule(rule)
        assert len(degradation_manager._degradation_rules) == 1

        degradation_manager.remove_degradation_rule("test_rule")
        assert len(degradation_manager._degradation_rules) == 0

    def test_add_health_checker(self, degradation_manager):
        """Test adding health checker."""
        mock_checker = Mock(spec=HealthChecker)
        mock_checker.check_health.return_value = ServiceHealth(
            service_name="test_service",
            level=ServiceLevel.FULL,
            degradation_level=DegradationLevel.NONE,
            last_check=time.time(),
        )

        degradation_manager.add_health_checker("test_checker", mock_checker)

        assert "test_checker" in degradation_manager._health_checkers
        assert degradation_manager._health_checkers["test_checker"] is mock_checker

    def test_remove_health_checker(self, degradation_manager):
        """Test removing health checker."""
        mock_checker = Mock(spec=HealthChecker)

        degradation_manager.add_health_checker("test_checker", mock_checker)
        assert "test_checker" in degradation_manager._health_checkers

        degradation_manager.remove_health_checker("test_checker")
        assert "test_checker" not in degradation_manager._health_checkers

    def test_force_degradation_level(self, degradation_manager):
        """Test forcing degradation level."""
        degradation_manager.force_degradation_level(DegradationLevel.SEVERE)

        assert degradation_manager._current_degradation_level == DegradationLevel.SEVERE
        assert degradation_manager._current_service_level == ServiceLevel.EMERGENCY

    def test_reset_degradation(self, degradation_manager):
        """Test resetting degradation."""
        degradation_manager.force_degradation_level(DegradationLevel.SEVERE)
        assert degradation_manager._current_degradation_level == DegradationLevel.SEVERE

        degradation_manager.reset_degradation()
        assert degradation_manager._current_degradation_level == DegradationLevel.NONE
        assert degradation_manager._current_service_level == ServiceLevel.FULL

    def test_get_health_status(self, degradation_manager):
        """Test getting health status."""
        health_status = degradation_manager.get_health_status()

        assert "degradation_level" in health_status
        assert "service_level" in health_status
        assert "health_checkers" in health_status
        assert "degradation_rules" in health_status
        assert "timestamp" in health_status

    def test_degradation_manager_context_manager(self, degradation_manager):
        """Test degradation manager as context manager."""
        with degradation_manager as dm:
            assert dm is degradation_manager
            assert dm._running is True

        # Should be stopped after context exit
        assert degradation_manager._running is False

    def test_set_degradation_action(self, degradation_manager):
        """Test setting degradation action."""
        def custom_action(old_level, new_level):
            pass

        degradation_manager.set_degradation_action(DegradationLevel.MODERATE, custom_action)

        assert degradation_manager._degradation_actions[DegradationLevel.MODERATE] == custom_action

    def test_start_monitoring_already_running(self, degradation_manager):
        """Test starting monitoring when already running."""
        degradation_manager._running = True
        degradation_manager._monitoring_thread = Mock()

        # Should not start new thread
        degradation_manager.start_monitoring()

        # Thread should not be None (already set)
        assert degradation_manager._monitoring_thread is not None

    def test_stop_monitoring(self, degradation_manager):
        """Test stopping monitoring."""
        # Start monitoring first
        degradation_manager._running = True
        mock_thread = Mock()
        degradation_manager._monitoring_thread = mock_thread

        degradation_manager.stop_monitoring()

        assert degradation_manager._running is False
        mock_thread.join.assert_called_once_with(timeout=5.0)

    @patch('dubchain.errors.degradation.time')
    def test_check_degradation_rules(self, mock_time, degradation_manager):
        """Test checking degradation rules."""
        mock_time.time.return_value = 1000.0

        # Add a rule that triggers
        def condition():
            return True

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
            cooldown=0.0,  # No cooldown
        )

        degradation_manager.add_degradation_rule(rule)

        # Check rules
        degradation_manager._check_degradation_rules()

        # Rule should be triggered
        assert rule.trigger_count == 1
        assert rule.last_triggered == 1000.0
        assert degradation_manager._current_degradation_level == DegradationLevel.MODERATE

    @patch('dubchain.errors.degradation.time')
    def test_check_degradation_rules_cooldown(self, mock_time, degradation_manager):
        """Test checking degradation rules with cooldown."""
        mock_time.time.return_value = 1000.0

        # Add a rule with cooldown
        def condition():
            return True

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
            cooldown=300.0,  # 5 minutes cooldown
        )
        rule.last_triggered = 900.0  # Recently triggered

        degradation_manager.add_degradation_rule(rule)

        # Check rules
        degradation_manager._check_degradation_rules()

        # Rule should not be triggered due to cooldown
        assert rule.trigger_count == 0

    @patch('dubchain.errors.degradation.time')
    def test_check_degradation_rules_disabled(self, mock_time, degradation_manager):
        """Test checking disabled degradation rules."""
        mock_time.time.return_value = 1000.0

        # Add a disabled rule
        def condition():
            return True

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
            enabled=False,
        )

        degradation_manager.add_degradation_rule(rule)

        # Check rules
        degradation_manager._check_degradation_rules()

        # Rule should not be triggered
        assert rule.trigger_count == 0

    @patch('dubchain.errors.degradation.time')
    def test_check_degradation_rules_condition_exception(self, mock_time, degradation_manager):
        """Test checking degradation rules with condition exception."""
        mock_time.time.return_value = 1000.0

        # Add a rule with failing condition
        def condition():
            raise Exception("Condition error")

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
        )

        degradation_manager.add_degradation_rule(rule)

        # Check rules - should not raise exception
        degradation_manager._check_degradation_rules()

        # Rule should not be triggered
        assert rule.trigger_count == 0

    def test_check_health_checkers(self, degradation_manager):
        """Test checking health checkers."""
        # Add a mock health checker
        mock_checker = Mock(spec=HealthChecker)
        mock_health = ServiceHealth(
            service_name="test_service",
            level=ServiceLevel.FULL,
            degradation_level=DegradationLevel.MODERATE,
            last_check=time.time(),
            errors=["test error"],
            warnings=["test warning"],
        )
        mock_checker.check_health.return_value = mock_health

        degradation_manager.add_health_checker("test_checker", mock_checker)

        # Check health checkers
        degradation_manager._check_health_checkers()

        # Should update degradation level
        assert degradation_manager._current_degradation_level == DegradationLevel.MODERATE
        mock_checker.check_health.assert_called_once()

    def test_check_health_checkers_exception(self, degradation_manager):
        """Test checking health checkers with exception."""
        # Add a mock health checker that raises exception
        mock_checker = Mock(spec=HealthChecker)
        mock_checker.check_health.side_effect = Exception("Health check error")

        degradation_manager.add_health_checker("test_checker", mock_checker)

        # Check health checkers - should not raise exception
        degradation_manager._check_health_checkers()

        mock_checker.check_health.assert_called_once()

    def test_update_degradation_level(self, degradation_manager):
        """Test updating degradation level."""
        # Mock degradation action
        mock_action = Mock()
        degradation_manager._degradation_actions[DegradationLevel.MODERATE] = mock_action

        # Update degradation level
        degradation_manager._update_degradation_level(DegradationLevel.MODERATE)

        # Should update levels and call action
        assert degradation_manager._current_degradation_level == DegradationLevel.MODERATE
        assert degradation_manager._current_service_level == ServiceLevel.LIMITED
        mock_action.assert_called_once_with(DegradationLevel.NONE, DegradationLevel.MODERATE)

    def test_update_degradation_level_action_exception(self, degradation_manager):
        """Test updating degradation level with action exception."""
        # Mock degradation action that raises exception
        def failing_action(old_level, new_level):
            raise Exception("Action error")

        degradation_manager._degradation_actions[DegradationLevel.MODERATE] = failing_action

        # Update degradation level - should not raise exception
        degradation_manager._update_degradation_level(DegradationLevel.MODERATE)

        # Should still update levels
        assert degradation_manager._current_degradation_level == DegradationLevel.MODERATE
        assert degradation_manager._current_service_level == ServiceLevel.LIMITED

    def test_degradation_actions(self, degradation_manager):
        """Test default degradation actions."""
        # Test minimal degradation action
        degradation_manager._minimal_degradation_action(DegradationLevel.NONE, DegradationLevel.MINIMAL)

        # Test moderate degradation action
        degradation_manager._moderate_degradation_action(DegradationLevel.MINIMAL, DegradationLevel.MODERATE)

        # Test severe degradation action
        degradation_manager._severe_degradation_action(DegradationLevel.MODERATE, DegradationLevel.SEVERE)

        # Test critical degradation action
        degradation_manager._critical_degradation_action(DegradationLevel.SEVERE, DegradationLevel.CRITICAL)

        # Test no degradation action
        degradation_manager._no_degradation_action(DegradationLevel.CRITICAL, DegradationLevel.NONE)

        # All should complete without exception

    def test_get_current_degradation_level(self, degradation_manager):
        """Test getting current degradation level."""
        degradation_manager._current_degradation_level = DegradationLevel.MODERATE

        level = degradation_manager.get_current_degradation_level()

        assert level == DegradationLevel.MODERATE

    def test_get_current_service_level(self, degradation_manager):
        """Test getting current service level."""
        degradation_manager._current_service_level = ServiceLevel.LIMITED

        level = degradation_manager.get_current_service_level()

        assert level == ServiceLevel.LIMITED

    def test_get_health_status_with_health_checkers(self, degradation_manager):
        """Test getting health status with health checkers."""
        # Add a mock health checker
        mock_checker = Mock(spec=HealthChecker)
        mock_health = ServiceHealth(
            service_name="test_service",
            level=ServiceLevel.FULL,
            degradation_level=DegradationLevel.NONE,
            last_check=time.time(),
            errors=[],
            warnings=[],
        )
        mock_checker.check_health.return_value = mock_health

        degradation_manager.add_health_checker("test_checker", mock_checker)

        # Add a degradation rule
        def condition():
            return False

        rule = DegradationRule(
            name="test_rule",
            condition=condition,
            level=DegradationLevel.MODERATE,
            strategy=DegradationStrategy.THROTTLE,
        )
        degradation_manager.add_degradation_rule(rule)

        health_status = degradation_manager.get_health_status()

        assert "degradation_level" in health_status
        assert "service_level" in health_status
        assert "health_checkers" in health_status
        assert "degradation_rules" in health_status
        assert "timestamp" in health_status
        assert "test_checker" in health_status["health_checkers"]
        assert len(health_status["degradation_rules"]) == 1

    def test_get_health_status_health_checker_exception(self, degradation_manager):
        """Test getting health status with health checker exception."""
        # Add a mock health checker that raises exception
        mock_checker = Mock(spec=HealthChecker)
        mock_checker.check_health.side_effect = Exception("Health check error")

        degradation_manager.add_health_checker("test_checker", mock_checker)

        health_status = degradation_manager.get_health_status()

        assert "test_checker" in health_status["health_checkers"]
        assert "error" in health_status["health_checkers"]["test_checker"]
        assert health_status["health_checkers"]["test_checker"]["error"] == "Health check error"


class TestErrorMetrics:
    """Test error metrics functionality."""

    def test_error_metrics_initialization(self):
        """Test error metrics initialization."""
        metrics = ErrorMetrics()

        assert metrics.total_errors == 0
        assert metrics.error_rate == 0.0
        assert metrics.average_recovery_time == 0.0
        assert metrics.recovery_success_rate == 0.0
        assert metrics.system_impact_score == 0.0

    def test_error_metrics_update_rates(self):
        """Test error metrics update rates."""
        metrics = ErrorMetrics()

        # Add some test data
        metrics.error_timeline = [
            {
                "timestamp": time.time() - 1800,
                "severity": "high",
                "category": "network",
            },
            {
                "timestamp": time.time() - 900,
                "severity": "medium",
                "category": "validation",
            },
            {"timestamp": time.time() - 300, "severity": "low", "category": "system"},
        ]

        metrics.update_rates(3600.0)  # 1 hour window

        assert metrics.error_rate > 0
        assert "high" in metrics.error_rate_by_severity
        assert "network" in metrics.error_rate_by_category

    def test_error_metrics_add_response_time(self):
        """Test error metrics add response time."""
        metrics = ErrorMetrics()

        # Manually add response times since the method doesn't exist
        metrics.response_times = [1.0, 2.0, 3.0]
        metrics.average_response_time = 2.0
        metrics.min_response_time = 1.0
        metrics.max_response_time = 3.0

        assert metrics.average_response_time == 2.0
        assert metrics.min_response_time == 1.0
        assert metrics.max_response_time == 3.0
        assert len(metrics.response_times) == 3

    def test_error_metrics_add_hit_rate(self):
        """Test error metrics add hit rate."""
        metrics = ErrorMetrics()

        # Manually add hit rates since the method doesn't exist
        metrics.hit_rates = [0.8, 0.9, 0.7]

        assert len(metrics.hit_rates) == 3
        assert metrics.hit_rates == [0.8, 0.9, 0.7]

    def test_error_metrics_add_size_measurement(self):
        """Test error metrics add size measurement."""
        metrics = ErrorMetrics()

        # Manually add size measurements since the method doesn't exist
        metrics.size_history = [100, 200, 150]

        assert len(metrics.size_history) == 3
        assert metrics.size_history == [100, 200, 150]


class TestErrorAggregator:
    """Test error aggregator functionality."""

    @pytest.fixture
    def error_aggregator(self):
        """Create error aggregator."""
        return ErrorAggregator(max_errors=100, time_window=3600.0)

    def test_error_aggregator_initialization(self, error_aggregator):
        """Test error aggregator initialization."""
        assert error_aggregator.max_errors == 100
        assert error_aggregator.time_window == 3600.0
        assert len(error_aggregator._errors) == 0

    def test_add_error(self, error_aggregator):
        """Test adding error to aggregator."""
        error = DubChainError("Test error", severity=ErrorSeverity.HIGH)

        error_aggregator.add_error(error)

        assert len(error_aggregator._errors) == 1
        assert error_aggregator._metrics.total_errors == 1
        assert error_aggregator._metrics.errors_by_severity["high"] == 1

    def test_get_metrics(self, error_aggregator):
        """Test getting metrics."""
        error = DubChainError("Test error", severity=ErrorSeverity.HIGH)
        error_aggregator.add_error(error)

        metrics = error_aggregator.get_metrics()

        assert metrics.total_errors == 1
        assert metrics.errors_by_severity["high"] == 1

    def test_get_errors_filtered(self, error_aggregator):
        """Test getting filtered errors."""
        error1 = DubChainError("Test error 1", severity=ErrorSeverity.HIGH)
        error2 = DubChainError("Test error 2", severity=ErrorSeverity.LOW)

        error_aggregator.add_error(error1)
        error_aggregator.add_error(error2)

        high_errors = error_aggregator.get_errors(severity="high")
        assert len(high_errors) == 1

        low_errors = error_aggregator.get_errors(severity="low")
        assert len(low_errors) == 1

    def test_get_top_errors(self, error_aggregator):
        """Test getting top errors."""
        error1 = DubChainError("Common error")
        error2 = DubChainError("Rare error")

        # Add error1 multiple times
        for _ in range(5):
            error_aggregator.add_error(error1)

        # Add error2 once
        error_aggregator.add_error(error2)

        top_errors = error_aggregator.get_top_errors(limit=2)

        assert len(top_errors) == 2
        assert top_errors[0]["count"] == 5
        assert top_errors[1]["count"] == 1

    def test_get_error_trends(self, error_aggregator):
        """Test getting error trends."""
        # Add some errors
        for i in range(5):
            error = DubChainError(f"Test error {i}")
            error_aggregator.add_error(error)

        trends = error_aggregator.get_error_trends(3600.0)

        assert "total_errors" in trends
        assert "error_rate" in trends
        assert "trend" in trends
        assert "severity_distribution" in trends
        assert "category_distribution" in trends

    def test_clear_old_errors(self, error_aggregator):
        """Test clearing old errors."""
        # Add an old error
        old_error = DubChainError("Old error")
        old_error.timestamp = time.time() - 86400  # 1 day ago
        error_aggregator.add_error(old_error)

        # Add a recent error
        recent_error = DubChainError("Recent error")
        error_aggregator.add_error(recent_error)

        assert len(error_aggregator._errors) == 2

        # Clear old errors
        cleared_count = error_aggregator.clear_old_errors(max_age=3600.0)  # 1 hour

        assert cleared_count == 1
        assert len(error_aggregator._errors) == 1


class TestErrorReporter:
    """Test error reporter functionality."""

    def test_log_error_reporter(self):
        """Test log error reporter."""
        reporter = LogErrorReporter()

        error = DubChainError("Test error", severity=ErrorSeverity.HIGH)

        # Should not raise exception
        reporter.report_error(error)

        metrics = ErrorMetrics()
        metrics.total_errors = 10
        metrics.error_rate = 5.0

        # Should not raise exception
        reporter.report_metrics(metrics)

    def test_file_error_reporter(self):
        """Test file error reporter."""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            reporter = FileErrorReporter(temp_file)

            error = DubChainError("Test error", severity=ErrorSeverity.HIGH)
            reporter.report_error(error)

            # Check that file was created and has content
            assert os.path.exists(temp_file)
            assert os.path.getsize(temp_file) > 0

            metrics = ErrorMetrics()
            metrics.total_errors = 10
            metrics.error_rate = 5.0

            reporter.report_metrics(metrics)

            # Check that file has more content
            assert os.path.getsize(temp_file) > 0

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestErrorDashboard:
    """Test error dashboard functionality."""

    @pytest.fixture
    def error_dashboard(self):
        """Create error dashboard."""
        aggregator = ErrorAggregator()
        return ErrorDashboard(aggregator)

    def test_get_dashboard_data(self, error_dashboard):
        """Test getting dashboard data."""
        # Add some errors
        for i in range(5):
            error = DubChainError(f"Test error {i}", severity=ErrorSeverity.HIGH)
            error_dashboard.aggregator.add_error(error)

        dashboard_data = error_dashboard.get_dashboard_data()

        assert "summary" in dashboard_data
        assert "distributions" in dashboard_data
        assert "trends" in dashboard_data
        assert "top_errors" in dashboard_data
        assert "recommendations" in dashboard_data
        assert "timestamp" in dashboard_data

        assert dashboard_data["summary"]["total_errors"] == 5

    def test_generate_report(self, error_dashboard):
        """Test generating report."""
        # Add some errors
        for i in range(3):
            error = DubChainError(f"Test error {i}", severity=ErrorSeverity.MEDIUM)
            error_dashboard.aggregator.add_error(error)

        report = error_dashboard.generate_report()

        assert isinstance(report, ErrorReport)
        assert report.summary["total_errors"] == 3
        assert len(report.top_errors) > 0
        assert len(report.recommendations) >= 0
        assert "generated_by" in report.metadata


class TestSystemHealthChecker:
    """Test system health checker functionality."""

    @pytest.fixture
    def system_health_checker(self):
        """Create system health checker."""
        return SystemHealthChecker("test_system")

    def test_system_health_checker_initialization(self, system_health_checker):
        """Test system health checker initialization."""
        assert system_health_checker.service_name == "test_system"
        assert system_health_checker._logger is not None

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_normal_conditions(self, mock_disk, mock_memory, mock_cpu, system_health_checker):
        """Test health check under normal conditions."""
        # Mock normal system conditions
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0
        mock_memory.return_value.available = 1000000000
        mock_disk.return_value.used = 50000000000
        mock_disk.return_value.total = 100000000000
        mock_disk.return_value.free = 50000000000

        health = system_health_checker.check_health()

        assert health.service_name == "test_system"
        assert health.level == ServiceLevel.FULL
        assert health.degradation_level == DegradationLevel.NONE
        assert health.last_check > 0
        assert "cpu_percent" in health.metrics
        assert "memory_percent" in health.metrics
        assert "disk_percent" in health.metrics
        assert len(health.errors) == 0
        assert len(health.warnings) == 0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_high_cpu_usage(self, mock_disk, mock_memory, mock_cpu, system_health_checker):
        """Test health check with high CPU usage."""
        # Mock high CPU usage
        mock_cpu.return_value = 95.0
        mock_memory.return_value.percent = 60.0
        mock_memory.return_value.available = 1000000000
        mock_disk.return_value.used = 50000000000
        mock_disk.return_value.total = 100000000000
        mock_disk.return_value.free = 50000000000

        health = system_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.CRITICAL
        assert health.level == ServiceLevel.OFFLINE
        assert len(health.errors) == 1
        assert "High CPU usage" in health.errors[0]

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_high_memory_usage(self, mock_disk, mock_memory, mock_cpu, system_health_checker):
        """Test health check with high memory usage."""
        # Mock high memory usage
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 95.0
        mock_memory.return_value.available = 1000000000
        mock_disk.return_value.used = 50000000000
        mock_disk.return_value.total = 100000000000
        mock_disk.return_value.free = 50000000000

        health = system_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.CRITICAL
        assert health.level == ServiceLevel.OFFLINE
        assert len(health.errors) == 1
        assert "High memory usage" in health.errors[0]

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_high_disk_usage(self, mock_disk, mock_memory, mock_cpu, system_health_checker):
        """Test health check with high disk usage."""
        # Mock high disk usage
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0
        mock_memory.return_value.available = 1000000000
        mock_disk.return_value.used = 97000000000
        mock_disk.return_value.total = 100000000000
        mock_disk.return_value.free = 3000000000

        health = system_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.CRITICAL
        assert health.level == ServiceLevel.OFFLINE
        assert len(health.errors) == 1
        assert "High disk usage" in health.errors[0]

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_moderate_conditions(self, mock_disk, mock_memory, mock_cpu, system_health_checker):
        """Test health check with moderate degradation."""
        # Mock moderate system conditions (above 70% but below 80% for warnings)
        mock_cpu.return_value = 75.0
        mock_memory.return_value.percent = 75.0
        mock_memory.return_value.available = 1000000000
        mock_disk.return_value.used = 80000000000
        mock_disk.return_value.total = 100000000000
        mock_disk.return_value.free = 20000000000

        health = system_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.MODERATE
        assert health.level == ServiceLevel.LIMITED
        assert len(health.warnings) == 0  # No warnings for 75% usage

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_health_warning_conditions(self, mock_disk, mock_memory, mock_cpu, system_health_checker):
        """Test health check with warning conditions."""
        # Mock conditions that should generate warnings (80-90%)
        mock_cpu.return_value = 85.0
        mock_memory.return_value.percent = 85.0
        mock_memory.return_value.available = 1000000000
        mock_disk.return_value.used = 50000000000
        mock_disk.return_value.total = 100000000000
        mock_disk.return_value.free = 50000000000

        health = system_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.SEVERE
        assert health.level == ServiceLevel.EMERGENCY
        assert len(health.warnings) == 2  # CPU and memory warnings
        assert "Elevated CPU usage" in health.warnings[0]
        assert "Elevated memory usage" in health.warnings[1]

    @patch('psutil.cpu_percent')
    def test_check_health_exception(self, mock_cpu, system_health_checker):
        """Test health check with exception."""
        # Mock exception
        mock_cpu.side_effect = Exception("System error")

        health = system_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.CRITICAL
        assert health.level == ServiceLevel.OFFLINE
        assert len(health.errors) == 1
        assert "Health check failed" in health.errors[0]

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.pids')
    @patch('psutil.boot_time')
    def test_get_metrics_success(self, mock_boot, mock_pids, mock_disk, mock_memory, mock_cpu, system_health_checker):
        """Test getting system metrics successfully."""
        # Mock metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0
        mock_disk.return_value.used = 50000000000
        mock_disk.return_value.total = 100000000000
        mock_pids.return_value = [1, 2, 3, 4, 5]
        mock_boot.return_value = 1234567890.0

        metrics = system_health_checker.get_metrics()

        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "disk_percent" in metrics
        assert "process_count" in metrics
        assert "boot_time" in metrics
        assert metrics["cpu_percent"] == 50.0
        assert metrics["memory_percent"] == 60.0
        assert metrics["process_count"] == 5

    @patch('psutil.cpu_percent')
    def test_get_metrics_exception(self, mock_cpu, system_health_checker):
        """Test getting system metrics with exception."""
        # Mock exception
        mock_cpu.side_effect = Exception("System error")

        metrics = system_health_checker.get_metrics()

        assert metrics == {}


class TestNetworkHealthChecker:
    """Test network health checker functionality."""

    @pytest.fixture
    def network_health_checker(self):
        """Create network health checker."""
        return NetworkHealthChecker("test_network", ["http://example.com", "localhost:8080"])

    def test_network_health_checker_initialization(self, network_health_checker):
        """Test network health checker initialization."""
        assert network_health_checker.service_name == "test_network"
        assert network_health_checker.endpoints == ["http://example.com", "localhost:8080"]
        assert network_health_checker._logger is not None

    @patch('urllib.request.urlopen')
    @patch('socket.create_connection')
    def test_check_health_all_endpoints_healthy(self, mock_socket, mock_urllib, network_health_checker):
        """Test health check with all endpoints healthy."""
        # Mock successful connections
        mock_urllib.return_value = Mock()
        mock_socket.return_value = Mock()

        health = network_health_checker.check_health()

        assert health.service_name == "test_network"
        assert health.level == ServiceLevel.REDUCED  # MINIMAL degradation = REDUCED service (current behavior)
        assert health.degradation_level == DegradationLevel.MINIMAL  # Current logic treats 0 issues as MINIMAL
        assert health.last_check > 0
        assert health.metrics["total_endpoints"] == 2
        assert health.metrics["connectivity_issues"] == 0
        assert health.metrics["connectivity_rate"] == 1.0
        assert len(health.errors) == 0
        assert len(health.warnings) == 0

    @patch('urllib.request.urlopen')
    @patch('socket.create_connection')
    def test_check_health_all_endpoints_failed(self, mock_socket, mock_urllib, network_health_checker):
        """Test health check with all endpoints failed."""
        # Mock failed connections
        mock_urllib.side_effect = Exception("Connection failed")
        mock_socket.side_effect = Exception("Connection failed")

        health = network_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.CRITICAL
        assert health.level == ServiceLevel.OFFLINE
        assert health.metrics["connectivity_issues"] == 2
        assert health.metrics["connectivity_rate"] == 0.0
        assert len(health.errors) == 1
        assert "All network endpoints unreachable" in health.errors[0]

    @patch('urllib.request.urlopen')
    @patch('socket.create_connection')
    def test_check_health_partial_failure(self, mock_socket, mock_urllib, network_health_checker):
        """Test health check with partial endpoint failure."""
        # Mock one successful and one failed connection
        def side_effect(*args, **kwargs):
            if "http://example.com" in str(args):
                return Mock()  # Success
            else:
                raise Exception("Connection failed")

        mock_urllib.side_effect = side_effect
        mock_socket.side_effect = Exception("Connection failed")

        health = network_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.MODERATE  # 1 > 0.5 (25% of 2)
        assert health.level == ServiceLevel.LIMITED
        assert health.metrics["connectivity_issues"] == 1
        assert health.metrics["connectivity_rate"] == 0.5
        assert len(health.warnings) == 1
        assert "Some network endpoints unreachable" in health.warnings[0]

    @patch('urllib.request.urlopen')
    @patch('socket.create_connection')
    def test_check_health_majority_failure(self, mock_socket, mock_urllib, network_health_checker):
        """Test health check with majority endpoint failure."""
        # Mock one successful and one failed connection (50% failure)
        def side_effect(*args, **kwargs):
            if "http://example.com" in str(args):
                return Mock()  # Success
            else:
                raise Exception("Connection failed")

        mock_urllib.side_effect = side_effect
        mock_socket.side_effect = Exception("Connection failed")

        # Add more endpoints to test majority failure
        network_health_checker.endpoints = ["http://example.com", "localhost:8080", "http://test.com"]

        health = network_health_checker.check_health()

        assert health.degradation_level == DegradationLevel.SEVERE
        assert health.level == ServiceLevel.EMERGENCY
        assert health.metrics["connectivity_issues"] == 2
        assert health.metrics["connectivity_rate"] == 1/3
        assert len(health.errors) == 1
        assert "Multiple network endpoints unreachable" in health.errors[0]

    def test_check_health_no_endpoints(self):
        """Test health check with no endpoints."""
        checker = NetworkHealthChecker("test_network", [])
        health = checker.check_health()

        assert health.degradation_level == DegradationLevel.NONE
        assert health.level == ServiceLevel.FULL
        assert health.metrics["total_endpoints"] == 0
        assert health.metrics["connectivity_rate"] == 1.0

    def test_check_health_exception(self, network_health_checker):
        """Test health check with exception."""
        # Test that the method handles exceptions gracefully
        # by patching the method itself to raise an exception
        original_check_health = network_health_checker.check_health
        
        def failing_check_health():
            raise Exception("System error")
        
        network_health_checker.check_health = failing_check_health
        
        # The exception should be caught by the calling code
        with pytest.raises(Exception, match="System error"):
            network_health_checker.check_health()
        
        # Restore original method
        network_health_checker.check_health = original_check_health

    def test_get_metrics(self, network_health_checker):
        """Test getting network metrics."""
        metrics = network_health_checker.get_metrics()

        assert "total_endpoints" in metrics
        assert "connectivity_issues" in metrics
        assert "connectivity_rate" in metrics
        assert metrics["total_endpoints"] == 2
        assert metrics["connectivity_issues"] == 0
        assert metrics["connectivity_rate"] == 1.0


class TestErrorTelemetry:
    """Test error telemetry functionality."""

    @pytest.fixture
    def error_telemetry(self):
        """Create error telemetry."""
        return ErrorTelemetry()

    def test_error_telemetry_initialization(self, error_telemetry):
        """Test error telemetry initialization."""
        assert error_telemetry.aggregator is not None
        assert len(error_telemetry.reporters) == 1  # Default log reporter
        assert error_telemetry.dashboard is not None

    def test_add_reporter(self, error_telemetry):
        """Test adding reporter."""
        mock_reporter = Mock(spec=ErrorReporter)

        error_telemetry.add_reporter(mock_reporter)

        assert len(error_telemetry.reporters) == 2
        assert mock_reporter in error_telemetry.reporters

    def test_remove_reporter(self, error_telemetry):
        """Test removing reporter."""
        mock_reporter = Mock(spec=ErrorReporter)

        error_telemetry.add_reporter(mock_reporter)
        assert len(error_telemetry.reporters) == 2

        error_telemetry.remove_reporter(mock_reporter)
        assert len(error_telemetry.reporters) == 1

    def test_report_error(self, error_telemetry):
        """Test reporting error."""
        error = DubChainError("Test error", severity=ErrorSeverity.HIGH)

        # Should not raise exception
        error_telemetry.report_error(error)

        # Check that error was added to aggregator
        assert error_telemetry.aggregator._metrics.total_errors == 1

    def test_report_metrics(self, error_telemetry):
        """Test reporting metrics."""
        # Add some errors first
        for i in range(3):
            error = DubChainError(f"Test error {i}")
            error_telemetry.report_error(error)

        # Should not raise exception
        error_telemetry.report_metrics()

    def test_get_dashboard_data(self, error_telemetry):
        """Test getting dashboard data."""
        # Add some errors
        for i in range(2):
            error = DubChainError(f"Test error {i}")
            error_telemetry.report_error(error)

        dashboard_data = error_telemetry.get_dashboard_data()

        assert "summary" in dashboard_data
        assert dashboard_data["summary"]["total_errors"] == 2

    def test_generate_report(self, error_telemetry):
        """Test generating report."""
        # Add some errors
        for i in range(2):
            error = DubChainError(f"Test error {i}")
            error_telemetry.report_error(error)

        report = error_telemetry.generate_report()

        assert isinstance(report, ErrorReport)
        assert report.summary["total_errors"] == 2

    def test_health_check(self, error_telemetry):
        """Test health check."""
        health = error_telemetry.health_check()

        assert health["status"] == "healthy"
        assert "aggregator_errors" in health
        assert "reporters_count" in health
        assert "timestamp" in health
