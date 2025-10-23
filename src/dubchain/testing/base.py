"""Base testing infrastructure for DubChain.

This module provides the foundational testing classes and utilities
for the DubChain testing framework.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)
import os
import sys
import threading
import time
import traceback
import unittest
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from unittest.mock import MagicMock, Mock

from ..logging import get_logger


class ExecutionStatus(Enum):
    """Test execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class ExecutionType(Enum):
    """Test type classification."""

    UNIT = "unit"
    INTEGRATION = "integration"
    PROPERTY = "property"
    FUZZ = "fuzz"
    PERFORMANCE = "performance"
    BENCHMARK = "benchmark"
    LOAD = "load"
    STRESS = "stress"


@dataclass
class ExecutionResult:
    """Test execution result."""

    test_name: str
    test_type: ExecutionType
    status: ExecutionStatus
    start_time: float
    end_time: float
    duration: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    assertions_count: int = 0
    assertions_passed: int = 0
    assertions_failed: int = 0
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if test was successful."""
        return self.status == ExecutionStatus.PASSED

    @property
    def is_failure(self) -> bool:
        """Check if test failed."""
        return self.status in [ExecutionStatus.FAILED, ExecutionStatus.ERROR]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "test_type": self.test_type.value,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "assertions_count": self.assertions_count,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "custom_metrics": self.custom_metrics,
        }


@dataclass
class ExecutionConfig:
    """Test configuration."""

    # General settings
    test_type: ExecutionType = ExecutionType.UNIT
    timeout: float = 30.0
    retry_count: int = 0
    parallel: bool = False
    max_workers: int = 4

    # Output settings
    verbose: bool = False
    capture_output: bool = True
    show_traceback: bool = True

    # Coverage settings
    collect_coverage: bool = False
    coverage_threshold: float = 80.0

    # Performance settings
    collect_metrics: bool = False
    memory_profiling: bool = False
    cpu_profiling: bool = False

    # Fuzz testing settings
    fuzz_iterations: int = 1000
    fuzz_timeout: float = 60.0

    # Property testing settings
    property_examples: int = 100
    property_timeout: float = 30.0

    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionEnvironment:
    """Test execution environment."""

    name: str
    config: ExecutionConfig
    setup_hooks: List[Callable] = field(default_factory=list)
    teardown_hooks: List[Callable] = field(default_factory=list)
    fixtures: Dict[str, Any] = field(default_factory=dict)
    mocks: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def setup(self) -> None:
        """Setup test environment."""
        for hook in self.setup_hooks:
            try:
                hook()
            except Exception as e:
                logging.error(f"Error in setup hook: {e}")

    def teardown(self) -> None:
        """Teardown test environment."""
        for hook in self.teardown_hooks:
            try:
                hook()
            except Exception as e:
                logging.error(f"Error in teardown hook: {e}")


class ExecutionData:
    """Test data management."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def set(self, key: str, value: Any) -> None:
        """Set test data."""
        with self._lock:
            self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get test data."""
        with self._lock:
            return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if test data exists."""
        with self._lock:
            return key in self.data

    def clear(self) -> None:
        """Clear all test data."""
        with self._lock:
            self.data.clear()


class FixtureManager:
    """Test fixture management."""

    def __init__(
        self, name: str, setup_func: Callable = None, teardown_func: Callable = None
    ):
        self.name = name
        self.setup_func = setup_func
        self.teardown_func = teardown_func
        self.data: Dict[str, Any] = {}
        self._setup_called = False

    def setup(self) -> None:
        """Setup fixture."""
        if self.setup_func and not self._setup_called:
            self.data = self.setup_func() or {}
            self._setup_called = True

    def teardown(self) -> None:
        """Teardown fixture."""
        if self.teardown_func:
            self.teardown_func(self.data)
        self.data.clear()
        self._setup_called = False

    def get(self, key: str, default: Any = None) -> Any:
        """Get fixture data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set fixture data."""
        self.data[key] = value


class EnhancedMock:
    """Enhanced mock object for testing."""

    def __init__(
        self, spec: Any = None, return_value: Any = None, side_effect: Any = None
    ):
        self.mock = Mock(spec=spec, return_value=return_value, side_effect=side_effect)
        self.calls: List[Any] = []
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        """Get mock attribute."""
        return getattr(self.mock, name)

    def __call__(self, *args, **kwargs) -> Any:
        """Call mock."""
        with self._lock:
            self.calls.append((args, kwargs))
        return self.mock(*args, **kwargs)

    def assert_called_with(self, *args, **kwargs) -> None:
        """Assert mock was called with specific arguments."""
        self.mock.assert_called_with(*args, **kwargs)

    def assert_called_once(self) -> None:
        """Assert mock was called exactly once."""
        self.mock.assert_called_once()

    def assert_not_called(self) -> None:
        """Assert mock was not called."""
        self.mock.assert_not_called()

    def reset_mock(self) -> None:
        """Reset mock state."""
        self.mock.reset_mock()
        with self._lock:
            self.calls.clear()


class AssertionUtils:
    """Enhanced assertion utilities."""

    @staticmethod
    def assert_true(condition: bool, message: str = "Assertion failed") -> None:
        """Assert condition is true."""
        if not condition:
            raise AssertionError(message)

    @staticmethod
    def assert_false(condition: bool, message: str = "Assertion failed") -> None:
        """Assert condition is false."""
        if condition:
            raise AssertionError(message)

    @staticmethod
    def assert_equal(actual: Any, expected: Any, message: str = None) -> None:
        """Assert values are equal."""
        if actual != expected:
            msg = message or f"Expected {expected}, got {actual}"
            raise AssertionError(msg)

    @staticmethod
    def assert_not_equal(actual: Any, expected: Any, message: str = None) -> None:
        """Assert values are not equal."""
        if actual == expected:
            msg = message or f"Expected {actual} to not equal {expected}"
            raise AssertionError(msg)

    @staticmethod
    def assert_is(actual: Any, expected: Any, message: str = None) -> None:
        """Assert objects are the same."""
        if actual is not expected:
            msg = message or f"Expected {expected} to be {actual}"
            raise AssertionError(msg)

    @staticmethod
    def assert_is_not(actual: Any, expected: Any, message: str = None) -> None:
        """Assert objects are not the same."""
        if actual is expected:
            msg = message or f"Expected {actual} to not be {expected}"
            raise AssertionError(msg)

    @staticmethod
    def assert_is_none(value: Any, message: str = None) -> None:
        """Assert value is None."""
        if value is not None:
            msg = message or f"Expected None, got {value}"
            raise AssertionError(msg)

    @staticmethod
    def assert_is_not_none(value: Any, message: str = None) -> None:
        """Assert value is not None."""
        if value is None:
            msg = message or f"Expected not None, got {value}"
            raise AssertionError(msg)

    @staticmethod
    def assert_in(item: Any, container: Any, message: str = None) -> None:
        """Assert item is in container."""
        if item not in container:
            msg = message or f"Expected {item} to be in {container}"
            raise AssertionError(msg)

    @staticmethod
    def assert_not_in(item: Any, container: Any, message: str = None) -> None:
        """Assert item is not in container."""
        if item in container:
            msg = message or f"Expected {item} to not be in {container}"
            raise AssertionError(msg)

    @staticmethod
    def assert_raises(
        expected_exception: Type[Exception], callable_obj: Callable, *args, **kwargs
    ) -> Exception:
        """Assert callable raises expected exception."""
        try:
            callable_obj(*args, **kwargs)
        except expected_exception as e:
            return e
        except Exception as e:
            raise AssertionError(
                f"Expected {expected_exception.__name__}, got {type(e).__name__}: {e}"
            )
        else:
            raise AssertionError(f"Expected {expected_exception.__name__} to be raised")

    @staticmethod
    def assert_almost_equal(
        actual: float, expected: float, places: int = 7, message: str = None
    ) -> None:
        """Assert floats are almost equal."""
        if abs(actual - expected) > 10 ** (-places):
            msg = (
                message
                or f"Expected {expected} to be almost equal to {actual} (places={places})"
            )
            raise AssertionError(msg)


class BaseTestCase(ABC):
    """Base test case class."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = get_logger(f"test.{self.name}")
        self.environment: Optional[ExecutionEnvironment] = None
        self.fixtures: Dict[str, FixtureManager] = {}
        self.mocks: Dict[str, EnhancedMock] = {}
        self.data = ExecutionData()
        self.result: Optional[ExecutionResult] = None
        self._lock = threading.RLock()

    def setup(self) -> None:
        """Setup test case."""
        pass

    def teardown(self) -> None:
        """Teardown test case."""
        pass

    @abstractmethod
    def run_test(self) -> None:
        """Run the actual test."""
        pass

    def run(self, config: ExecutionConfig = None) -> ExecutionResult:
        """Run test case."""
        config = config or ExecutionConfig()

        start_time = time.time()
        self.result = ExecutionResult(
            test_name=self.name,
            test_type=config.test_type,
            status=ExecutionStatus.RUNNING,
            start_time=start_time,
            end_time=0,
            duration=0,
        )

        try:
            # Setup
            self.setup()
            if self.environment:
                self.environment.setup()

            # Run test
            self.run_test()

            # Success
            self.result.status = ExecutionStatus.PASSED

        except AssertionError as e:
            self.result.status = ExecutionStatus.FAILED
            self.result.error_message = str(e)
            self.result.error_traceback = traceback.format_exc()

        except Exception as e:
            self.result.status = ExecutionStatus.ERROR
            self.result.error_message = str(e)
            self.result.error_traceback = traceback.format_exc()

        finally:
            # Teardown
            try:
                self.teardown()
                if self.environment:
                    self.environment.teardown()
            except Exception as e:
                self.logger.error(f"Error in teardown: {e}")

            # Finalize result
            end_time = time.time()
            self.result.end_time = end_time
            self.result.duration = end_time - start_time

        return self.result

    def add_fixture(self, name: str, fixture: FixtureManager) -> None:
        """Add test fixture."""
        self.fixtures[name] = fixture

    def get_fixture(self, name: str) -> FixtureManager:
        """Get test fixture."""
        if name not in self.fixtures:
            raise KeyError(f"Fixture '{name}' not found")
        return self.fixtures[name]

    def add_mock(self, name: str, mock: EnhancedMock) -> None:
        """Add test mock."""
        self.mocks[name] = mock

    def get_mock(self, name: str) -> EnhancedMock:
        """Get test mock."""
        if name not in self.mocks:
            raise KeyError(f"Mock '{name}' not found")
        return self.mocks[name]

    def assert_true(self, condition: bool, message: str = "Assertion failed") -> None:
        """Assert condition is true."""
        self.result.assertions_count += 1
        try:
            AssertionUtils.assert_true(condition, message)
            self.result.assertions_passed += 1
        except AssertionError:
            self.result.assertions_failed += 1
            raise

    def assert_false(self, condition: bool, message: str = "Assertion failed") -> None:
        """Assert condition is false."""
        self.result.assertions_count += 1
        try:
            AssertionUtils.assert_false(condition, message)
            self.result.assertions_passed += 1
        except AssertionError:
            self.result.assertions_failed += 1
            raise

    def assert_equal(self, actual: Any, expected: Any, message: str = None) -> None:
        """Assert values are equal."""
        self.result.assertions_count += 1
        try:
            AssertionUtils.assert_equal(actual, expected, message)
            self.result.assertions_passed += 1
        except AssertionError:
            self.result.assertions_failed += 1
            raise

    def assert_not_equal(self, actual: Any, expected: Any, message: str = None) -> None:
        """Assert values are not equal."""
        self.result.assertions_count += 1
        try:
            AssertionUtils.assert_not_equal(actual, expected, message)
            self.result.assertions_passed += 1
        except AssertionError:
            self.result.assertions_failed += 1
            raise

    def assert_raises(
        self,
        expected_exception: Type[Exception],
        callable_obj: Callable,
        *args,
        **kwargs,
    ) -> Exception:
        """Assert callable raises expected exception."""
        self.result.assertions_count += 1
        try:
            exception = AssertionUtils.assert_raises(
                expected_exception, callable_obj, *args, **kwargs
            )
            self.result.assertions_passed += 1
            return exception
        except AssertionError:
            self.result.assertions_failed += 1
            raise


class AsyncTestCase(BaseTestCase):
    """Async test case class."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def setup(self) -> None:
        """Setup async test case."""
        super().setup()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def teardown(self) -> None:
        """Teardown async test case."""
        if self.loop:
            self.loop.close()
        super().teardown()

    @abstractmethod
    async def run_test_async(self) -> None:
        """Run the actual async test."""
        pass

    def run_test(self) -> None:
        """Run async test."""
        if self.loop:
            self.loop.run_until_complete(self.run_test_async())


class SuiteManager:
    """Test suite for organizing and running multiple tests."""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.tests: List[BaseTestCase] = []
        self.results: List[ExecutionResult] = []
        self.config: ExecutionConfig = ExecutionConfig()
        self._lock = threading.RLock()

    def add_test(self, test: BaseTestCase) -> None:
        """Add test to suite."""
        with self._lock:
            self.tests.append(test)

    def remove_test(self, test: BaseTestCase) -> None:
        """Remove test from suite."""
        with self._lock:
            if test in self.tests:
                self.tests.remove(test)

    def run(self, config: ExecutionConfig = None) -> List[ExecutionResult]:
        """Run all tests in suite."""
        config = config or self.config

        with self._lock:
            self.results.clear()

            for test in self.tests:
                try:
                    result = test.run(config)
                    self.results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = ExecutionResult(
                        test_name=test.name,
                        test_type=config.test_type,
                        status=ExecutionStatus.ERROR,
                        start_time=time.time(),
                        end_time=time.time(),
                        duration=0,
                        error_message=str(e),
                        error_traceback=traceback.format_exc(),
                    )
                    self.results.append(error_result)

        return self.results.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary."""
        with self._lock:
            total = len(self.results)
            passed = sum(1 for r in self.results if r.status == ExecutionStatus.PASSED)
            failed = sum(1 for r in self.results if r.status == ExecutionStatus.FAILED)
            error = sum(1 for r in self.results if r.status == ExecutionStatus.ERROR)
            skipped = sum(
                1 for r in self.results if r.status == ExecutionStatus.SKIPPED
            )

            total_duration = sum(r.duration for r in self.results)

            return {
                "suite_name": self.name,
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "error": error,
                "skipped": skipped,
                "success_rate": (passed / total * 100) if total > 0 else 0,
                "total_duration": total_duration,
                "average_duration": total_duration / total if total > 0 else 0,
            }


class RunnerManager:
    """Test runner for executing test suites."""

    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.suites: List[SuiteManager] = []
        self.results: List[ExecutionResult] = []
        self.logger = get_logger("test_runner")

    def add_suite(self, suite: SuiteManager) -> None:
        """Add test suite to runner."""
        self.suites.append(suite)

    def run_all(self) -> List[ExecutionResult]:
        """Run all test suites."""
        self.results.clear()

        for suite in self.suites:
            self.logger.info(f"Running test suite: {suite.name}")
            suite_results = suite.run(self.config)
            self.results.extend(suite_results)

        return self.results.copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get overall test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == ExecutionStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == ExecutionStatus.FAILED)
        error = sum(1 for r in self.results if r.status == ExecutionStatus.ERROR)
        skipped = sum(1 for r in self.results if r.status == ExecutionStatus.SKIPPED)

        total_duration = sum(r.duration for r in self.results)

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "error": error,
            "skipped": skipped,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / total if total > 0 else 0,
            "suites_count": len(self.suites),
        }


class EnhancedMock(Mock):
    """Enhanced mock object with additional tracking capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []  # Track calls in a list for iteration

    def __call__(self, *args, **kwargs):
        """Override call to track in our calls list."""
        result = super().__call__(*args, **kwargs)
        self.calls.append((args, kwargs))
        return result

    def reset_mock(self, visited=None, return_value=False, side_effect=False):
        """Reset the mock including our custom calls tracking."""
        # Call parent reset_mock with correct arguments
        try:
            if visited is not None:
                super().reset_mock(visited, return_value, side_effect)
            else:
                super().reset_mock(return_value, side_effect)
        except TypeError:
            # Fallback for different mock types
            super().reset_mock()
        self.calls = []
