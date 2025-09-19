"""Tests for unit testing infrastructure."""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from src.dubchain.testing.unit import (
    UnitTestCase,
    UnitTestSuite,
    UnitTestRunner,
    MockFactory,
    StubFactory,
    SpyFactory,
    UnitTestHelpers,
)
from src.dubchain.testing.base import ExecutionConfig, ExecutionType, ExecutionStatus, EnhancedMock


class TestUnitTestCase:
    """Test UnitTestCase functionality."""

    def test_init(self):
        """Test UnitTestCase initialization."""
        test_case = UnitTestCase("test_name")
        assert test_case.name == "test_name"
        assert test_case.config.test_type == ExecutionType.UNIT
        assert test_case.isolated_components == {}
        assert test_case.mocked_dependencies == {}

    def test_setup_teardown(self):
        """Test setup and teardown methods."""
        test_case = UnitTestCase()
        test_case.setup()
        test_case.teardown()
        # Should not raise any exceptions

    def test_mock_dependency(self):
        """Test mocking dependencies."""
        test_case = UnitTestCase()
        mock = test_case.mock_dependency("test_dep", return_value="test_value")
        
        assert "test_dep" in test_case.mocked_dependencies
        assert mock.return_value == "test_value"

    def test_get_mocked_dependency(self):
        """Test getting mocked dependencies."""
        test_case = UnitTestCase()
        test_case.mock_dependency("test_dep")
        
        mock = test_case.get_mocked_dependency("test_dep")
        assert mock is not None

    def test_get_mocked_dependency_not_found(self):
        """Test getting non-existent mocked dependency."""
        test_case = UnitTestCase()
        
        with pytest.raises(KeyError):
            test_case.get_mocked_dependency("nonexistent")

    def test_assert_mock_called_with(self):
        """Test mock assertion methods."""
        test_case = UnitTestCase()
        mock = test_case.mock_dependency("test_dep")
        mock("arg1", "arg2", key="value")
        
        test_case.assert_mock_called_with("test_dep", "arg1", "arg2", key="value")

    def test_assert_mock_called_once(self):
        """Test mock called once assertion."""
        test_case = UnitTestCase()
        mock = test_case.mock_dependency("test_dep")
        mock()
        
        test_case.assert_mock_called_once("test_dep")

    def test_assert_mock_not_called(self):
        """Test mock not called assertion."""
        test_case = UnitTestCase()
        test_case.mock_dependency("test_dep")
        
        test_case.assert_mock_not_called("test_dep")

    def test_assert_mock_call_count(self):
        """Test mock call count assertion."""
        test_case = UnitTestCase()
        test_case.setup()  # Initialize self.result
        mock = test_case.mock_dependency("test_dep")
        mock()
        mock()
        
        test_case.assert_mock_call_count("test_dep", 2)


class TestUnitTestSuite:
    """Test UnitTestSuite functionality."""

    def test_init(self):
        """Test UnitTestSuite initialization."""
        suite = UnitTestSuite("test_suite")
        assert suite.name == "test_suite"
        assert suite.config.test_type == ExecutionType.UNIT
        assert suite.test_categories == {}

    def test_add_test(self):
        """Test adding tests to suite."""
        suite = UnitTestSuite()
        test_case = UnitTestCase("test1")
        
        suite.add_test(test_case, "category1")
        
        assert len(suite.tests) == 1
        assert "category1" in suite.test_categories
        assert test_case in suite.test_categories["category1"]

    def test_run_by_category(self):
        """Test running tests by category."""
        suite = UnitTestSuite()
        test_case = UnitTestCase("test1")
        suite.add_test(test_case, "category1")
        
        results = suite.run_by_category("category1")
        assert len(results) == 1

    def test_run_by_category_empty(self):
        """Test running tests by non-existent category."""
        suite = UnitTestSuite()
        results = suite.run_by_category("nonexistent")
        assert results == []

    def test_get_category_summary(self):
        """Test getting category summary."""
        suite = UnitTestSuite()
        test_case = UnitTestCase("test1")
        suite.add_test(test_case, "category1")
        
        summary = suite.get_category_summary("category1")
        assert summary["category"] == "category1"
        assert summary["total_tests"] == 1

    def test_get_category_summary_empty(self):
        """Test getting summary for non-existent category."""
        suite = UnitTestSuite()
        summary = suite.get_category_summary("nonexistent")
        assert summary["total_tests"] == 0


class TestUnitTestRunner:
    """Test UnitTestRunner functionality."""

    def test_init(self):
        """Test UnitTestRunner initialization."""
        runner = UnitTestRunner()
        assert runner.config.test_type == ExecutionType.UNIT
        assert runner.coverage_collector is None
        assert runner.performance_monitor is None

    def test_run_with_coverage(self):
        """Test running tests with coverage."""
        runner = UnitTestRunner()
        results = runner.run_with_coverage()
        assert isinstance(results, list)

    def test_run_with_performance_monitoring(self):
        """Test running tests with performance monitoring."""
        runner = UnitTestRunner()
        results = runner.run_with_performance_monitoring()
        assert isinstance(results, list)


class TestMockFactory:
    """Test MockFactory functionality."""

    def test_create_mock(self):
        """Test creating basic mock."""
        mock = MockFactory.create_mock(spec=str, return_value="test")
        assert mock.return_value == "test"

    def test_create_magic_mock(self):
        """Test creating magic mock."""
        mock = MockFactory.create_magic_mock(return_value="test")
        assert isinstance(mock, MagicMock)
        assert mock.return_value == "test"

    def test_create_async_mock(self):
        """Test creating async mock."""
        mock = MockFactory.create_async_mock(return_value="test")
        assert mock.return_value == "test"

    def test_create_context_manager_mock(self):
        """Test creating context manager mock."""
        mock = MockFactory.create_context_manager_mock(return_value="test")
        assert hasattr(mock, "__enter__")
        assert hasattr(mock, "__exit__")

    def test_create_property_mock(self):
        """Test creating property mock."""
        mock = MockFactory.create_property_mock(get_value="get", set_value="set")
        assert hasattr(mock, "get")
        assert hasattr(mock, "set")


class TestStubFactory:
    """Test StubFactory functionality."""

    def test_create_stub(self):
        """Test creating stub with methods."""
        def test_method():
            return "test"
        
        stub = StubFactory.create_stub({"test_method": test_method})
        assert hasattr(stub, "test_method")
        assert stub.test_method() == "test"

    def test_create_data_stub(self):
        """Test creating data stub."""
        data = {"key1": "value1", "key2": "value2"}
        stub = StubFactory.create_data_stub(data)
        assert stub.key1 == "value1"
        assert stub.key2 == "value2"

    def test_create_service_stub(self):
        """Test creating service stub."""
        methods = {"method1": "return1", "method2": lambda: "return2"}
        stub = StubFactory.create_service_stub("test_service", methods)
        assert stub.service_name == "test_service"
        assert stub.method1 == "return1"
        assert callable(stub.method2)


class TestSpyFactory:
    """Test SpyFactory functionality."""

    def test_create_spy(self):
        """Test creating spy."""
        spy = SpyFactory.create_spy()
        assert hasattr(spy, "calls")
        assert hasattr(spy, "call_count")
        assert spy.calls == []
        assert spy.call_count == 0

    def test_create_spy_with_target(self):
        """Test creating spy with target function."""
        def target_func(x):
            return x * 2
        
        spy = SpyFactory.create_spy(target_func)
        result = spy.mock(5)
        assert result == 10
        assert spy.call_count == 1

    def test_create_method_spy(self):
        """Test creating method spy."""
        class TestClass:
            def test_method(self, x):
                return x * 2
        
        obj = TestClass()
        spy = SpyFactory.create_method_spy(obj, "test_method")
        
        result = obj.test_method(5)
        assert result == 10
        assert spy.call_count == 1
        
        # Restore original method
        spy.restore()
        assert obj.test_method(5) == 10

    def test_create_property_spy(self):
        """Test creating property spy."""
        class TestClass:
            def __init__(self):
                self._value = 0
            
            @property
            def value(self):
                return self._value
            
            @value.setter
            def value(self, val):
                self._value = val
        
        obj = TestClass()
        spy = SpyFactory.create_property_spy(obj, "value")
        
        obj.value = 42
        assert obj.value == 42
        assert len(spy.set_calls) == 1
        assert len(spy.get_calls) == 1
        
        # Restore original property
        spy.restore()


class TestUnitTestHelpers:
    """Test UnitTestHelpers functionality."""

    def test_patch_method(self):
        """Test patching method."""
        class TestClass:
            def test_method(self):
                return "original"
        
        obj = TestClass()
        original = UnitTestHelpers.patch_method(obj, "test_method", lambda self: "patched")
        
        assert obj.test_method() == "patched"
        
        # Restore
        UnitTestHelpers.restore_method(obj, "test_method", original)
        assert obj.test_method() == "original"

    def test_patch_property(self):
        """Test patching property."""
        class TestClass:
            def __init__(self):
                self.value = "original"
        
        obj = TestClass()
        original = UnitTestHelpers.patch_property(obj, "value", "patched")
        
        assert obj.value == "patched"
        
        # Restore
        UnitTestHelpers.restore_property(obj, "value", original)
        assert obj.value == "original"

    def test_create_test_data(self):
        """Test creating test data."""
        user_data = UnitTestHelpers.create_test_data("user", id="test_id", name="Test User")
        assert user_data["id"] == "test_id"
        assert user_data["name"] == "Test User"
        assert user_data["email"] == "test@example.com"

        tx_data = UnitTestHelpers.create_test_data("transaction", amount=100)
        assert tx_data["amount"] == 100
        assert tx_data["from_address"] == "test_from"

        block_data = UnitTestHelpers.create_test_data("block", nonce=123)
        assert block_data["nonce"] == 123
        assert block_data["previous_hash"] == "test_prev_hash"

        custom_data = UnitTestHelpers.create_test_data("custom", key="value")
        assert custom_data["key"] == "value"

    def test_assert_called_in_order(self):
        """Test asserting calls in order."""
        mock1 = EnhancedMock()
        mock2 = EnhancedMock()
        mock1("call1")
        mock2("call2")
        
        UnitTestHelpers.assert_called_in_order([mock1, mock2], ("call1",), ("call2",))

    def test_assert_called_with_any_of(self):
        """Test asserting calls with any of expected values."""
        mock = EnhancedMock()
        mock("call1")
        
        UnitTestHelpers.assert_called_with_any_of(mock, ("call1",), ("call2",))

    def test_assert_called_with_pattern(self):
        """Test asserting calls with pattern."""
        mock = EnhancedMock()
        mock("test_call_123")
        
        UnitTestHelpers.assert_called_with_pattern(mock, r"test_call_\d+")

    def test_create_mock_chain(self):
        """Test creating mock chain."""
        chain = UnitTestHelpers.create_mock_chain("method1", "method2", "method3")
        assert hasattr(chain, "method1")
        assert hasattr(chain.method1, "method2")
        assert hasattr(chain.method1.method2, "method3")

    def test_assert_mock_chain_called(self):
        """Test asserting mock chain calls."""
        chain = UnitTestHelpers.create_mock_chain("method1", "method2")
        chain.method1.method2()
        
        result = UnitTestHelpers.assert_mock_chain_called(chain, "method1", "method2")
        assert result is not None
