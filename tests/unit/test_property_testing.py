"""Tests for property-based testing infrastructure."""

import logging

logger = logging.getLogger(__name__)
import pytest
import random
from unittest.mock import Mock, patch

from src.dubchain.testing.property import (
    PropertyExecutionResult,
    PropertyGenerator,
    IntGenerator,
    StringGenerator,
    ListGenerator,
    DictGenerator,
    CompositeGenerator,
    PropertyValidator,
    PropertyReporter,
    PropertyTestCase,
    PropertyTestSuite,
    PropertyTestRunner,
    int_gen,
    string_gen,
    list_gen,
    dict_gen,
    composite_gen,
    always_true,
    always_false,
    is_positive,
    is_non_negative,
    is_even,
    is_odd,
    is_palindrome,
    list_is_sorted,
    dict_has_key,
)
from src.dubchain.testing.base import ExecutionStatus, ExecutionType


class TestPropertyExecutionResult:
    """Test PropertyExecutionResult functionality."""

    def test_init(self):
        """Test PropertyExecutionResult initialization."""
        result = PropertyExecutionResult(
            test_name="test",
            test_type=ExecutionType.PROPERTY,
            status=ExecutionStatus.PASSED,
            start_time=1234567890.0,
            end_time=1234567891.0,
            duration=1.0,
            property_name="prop",
            examples_tested=100,
            examples_passed=95,
            examples_failed=5
        )
        assert result.test_name == "test"
        assert result.property_name == "prop"
        assert result.examples_tested == 100
        assert result.examples_passed == 95
        assert result.examples_failed == 5


class TestIntGenerator:
    """Test IntGenerator functionality."""

    def test_init(self):
        """Test IntGenerator initialization."""
        gen = IntGenerator(0, 100)
        assert gen.min_value == 0
        assert gen.max_value == 100

    def test_generate(self):
        """Test integer generation."""
        gen = IntGenerator(0, 10)
        with patch('random.randint', return_value=5):
            result = gen.generate()
            assert result == 5

    def test_shrink(self):
        """Test integer shrinking."""
        gen = IntGenerator(0, 100)
        
        # Test shrinking positive value
        shrunk_values = list(gen.shrink(5))
        assert 4 in shrunk_values
        assert 0 in shrunk_values
        
        # Test shrinking negative value
        shrunk_values = list(gen.shrink(-3))
        assert 3 in shrunk_values


class TestStringGenerator:
    """Test StringGenerator functionality."""

    def test_init(self):
        """Test StringGenerator initialization."""
        gen = StringGenerator(0, 100, "abc")
        assert gen.min_length == 0
        assert gen.max_length == 100
        assert gen.charset == "abc"

    def test_init_default_charset(self):
        """Test StringGenerator with default charset."""
        gen = StringGenerator()
        assert gen.charset is not None

    def test_generate(self):
        """Test string generation."""
        gen = StringGenerator(0, 5, "ab")
        with patch('random.randint', return_value=3):
            with patch('random.choices', return_value=['a', 'b', 'c']):
                result = gen.generate()
                assert result == "abc"

    def test_shrink(self):
        """Test string shrinking."""
        gen = StringGenerator(0, 100)
        
        # Test shrinking longer string
        shrunk_values = list(gen.shrink("hello"))
        assert "hell" in shrunk_values  # Remove from end
        assert "ello" in shrunk_values  # Remove from beginning
        assert "helo" in shrunk_values  # Remove from middle


class TestListGenerator:
    """Test ListGenerator functionality."""

    def test_init(self):
        """Test ListGenerator initialization."""
        int_gen = IntGenerator()
        list_gen = ListGenerator(int_gen, 0, 10)
        assert list_gen.element_generator == int_gen
        assert list_gen.min_length == 0
        assert list_gen.max_length == 10

    def test_generate(self):
        """Test list generation."""
        int_gen = IntGenerator()
        list_gen = ListGenerator(int_gen, 0, 3)
        
        with patch('random.randint', return_value=2):
            with patch.object(int_gen, 'generate', side_effect=[1, 2]):
                result = list_gen.generate()
                assert result == [1, 2]

    def test_shrink(self):
        """Test list shrinking."""
        int_gen = IntGenerator()
        list_gen = ListGenerator(int_gen, 0, 10)
        
        # Test shrinking longer list
        shrunk_values = list(list_gen.shrink([1, 2, 3]))
        assert [1, 2] in shrunk_values  # Remove from end
        assert [2, 3] in shrunk_values  # Remove from beginning
        assert [1, 3] in shrunk_values  # Remove from middle


class TestDictGenerator:
    """Test DictGenerator functionality."""

    def test_init(self):
        """Test DictGenerator initialization."""
        key_gen = StringGenerator()
        value_gen = IntGenerator()
        dict_gen = DictGenerator(key_gen, value_gen, 0, 10)
        assert dict_gen.key_generator == key_gen
        assert dict_gen.value_generator == value_gen
        assert dict_gen.min_size == 0
        assert dict_gen.max_size == 10

    def test_generate(self):
        """Test dictionary generation."""
        key_gen = StringGenerator()
        value_gen = IntGenerator()
        dict_gen = DictGenerator(key_gen, value_gen, 0, 2)
        
        with patch('random.randint', return_value=1):
            with patch.object(key_gen, 'generate', return_value="key"):
                with patch.object(value_gen, 'generate', return_value=42):
                    result = dict_gen.generate()
                    assert result == {"key": 42}

    def test_shrink(self):
        """Test dictionary shrinking."""
        key_gen = StringGenerator()
        value_gen = IntGenerator()
        dict_gen = DictGenerator(key_gen, value_gen, 0, 10)
        
        # Test shrinking dictionary
        test_dict = {"key1": 1, "key2": 2}
        shrunk_values = list(dict_gen.shrink(test_dict))
        # Should have shrunk versions with fewer keys
        assert any(len(d) < len(test_dict) for d in shrunk_values)


class TestCompositeGenerator:
    """Test CompositeGenerator functionality."""

    def test_init(self):
        """Test CompositeGenerator initialization."""
        generators = {"int": IntGenerator(), "str": StringGenerator()}
        comp_gen = CompositeGenerator(generators)
        assert comp_gen.generators == generators

    def test_generate(self):
        """Test composite generation."""
        int_gen = IntGenerator()
        str_gen = StringGenerator()
        comp_gen = CompositeGenerator({"int": int_gen, "str": str_gen})
        
        with patch.object(int_gen, 'generate', return_value=42):
            with patch.object(str_gen, 'generate', return_value="test"):
                result = comp_gen.generate()
                assert result == {"int": 42, "str": "test"}

    def test_shrink(self):
        """Test composite shrinking."""
        int_gen = IntGenerator()
        str_gen = StringGenerator()
        comp_gen = CompositeGenerator({"int": int_gen, "str": str_gen})
        
        test_value = {"int": 5, "str": "hello"}
        shrunk_values = list(comp_gen.shrink(test_value))
        # Should have shrunk versions
        assert len(shrunk_values) > 0


class TestPropertyValidator:
    """Test PropertyValidator functionality."""

    def test_init(self):
        """Test PropertyValidator initialization."""
        def test_prop(x):
            return x > 0
        
        validator = PropertyValidator(test_prop, "positive")
        assert validator.property_func == test_prop
        assert validator.name == "positive"

    def test_init_default_name(self):
        """Test PropertyValidator with default name."""
        def test_prop(x):
            return x > 0
        
        validator = PropertyValidator(test_prop)
        assert validator.name == "test_prop"

    def test_validate(self):
        """Test property validation."""
        def positive_prop(x):
            return x > 0
        
        validator = PropertyValidator(positive_prop)
        assert validator.validate(5) is True
        assert validator.validate(-1) is False

    def test_validate_with_exception(self):
        """Test property validation with exception handling."""
        def failing_prop(x):
            if x < 0:
                raise ValueError("Negative value")
            return x > 0
        
        validator = PropertyValidator(failing_prop)
        
        is_valid, exception = validator.validate_with_exception(5)
        assert is_valid is True
        assert exception is None
        
        is_valid, exception = validator.validate_with_exception(-1)
        assert is_valid is False
        assert isinstance(exception, ValueError)


class TestPropertyReporter:
    """Test PropertyReporter functionality."""

    def test_init(self):
        """Test PropertyReporter initialization."""
        reporter = PropertyReporter()
        assert reporter.results == []

    def test_add_result(self):
        """Test adding result."""
        reporter = PropertyReporter()
        result = PropertyExecutionResult(
            test_name="test",
            test_type=ExecutionType.PROPERTY,
            status=ExecutionStatus.PASSED,
            start_time=1234567890.0,
            end_time=1234567891.0,
            duration=1.0
        )
        reporter.add_result(result)
        assert len(reporter.results) == 1

    def test_generate_report_empty(self):
        """Test generating report with no results."""
        reporter = PropertyReporter()
        report = reporter.generate_report()
        assert report["total_tests"] == 0

    def test_generate_report(self):
        """Test generating report with results."""
        reporter = PropertyReporter()
        
        # Add passed result
        passed_result = PropertyExecutionResult(
            test_name="test1",
            test_type=ExecutionType.PROPERTY,
            status=ExecutionStatus.PASSED,
            start_time=1234567890.0,
            end_time=1234567891.0,
            duration=1.0,
            examples_tested=100,
            examples_passed=100,
            examples_failed=0
        )
        reporter.add_result(passed_result)
        
        # Add failed result
        failed_result = PropertyExecutionResult(
            test_name="test2",
            test_type=ExecutionType.PROPERTY,
            status=ExecutionStatus.FAILED,
            start_time=1234567890.0,
            end_time=1234567891.0,
            duration=1.0,
            examples_tested=100,
            examples_passed=95,
            examples_failed=5
        )
        reporter.add_result(failed_result)
        
        report = reporter.generate_report()
        assert report["total_tests"] == 2
        assert report["passed"] == 1
        assert report["failed"] == 1
        assert report["total_examples_tested"] == 200
        assert report["examples_passed"] == 195
        assert report["examples_failed"] == 5


class TestPropertyTestCase:
    """Test PropertyTestCase functionality."""

    def test_init(self):
        """Test PropertyTestCase initialization."""
        test_case = PropertyTestCase("test")
        assert test_case.name == "test"
        assert test_case.generators == {}
        assert test_case.validators == []
        assert test_case.examples_count == 100
        assert test_case.shrinking_enabled is True

    def test_setup_teardown(self):
        """Test setup and teardown."""
        test_case = PropertyTestCase()
        test_case.setup()
        test_case.teardown()
        # Should not raise exceptions

    def test_add_generator(self):
        """Test adding generator."""
        test_case = PropertyTestCase()
        gen = IntGenerator()
        test_case.add_generator("int_gen", gen)
        assert test_case.generators["int_gen"] == gen

    def test_get_generator(self):
        """Test getting generator."""
        test_case = PropertyTestCase()
        gen = IntGenerator()
        test_case.add_generator("int_gen", gen)
        retrieved_gen = test_case.get_generator("int_gen")
        assert retrieved_gen == gen

    def test_get_generator_not_found(self):
        """Test getting non-existent generator."""
        test_case = PropertyTestCase()
        with pytest.raises(KeyError):
            test_case.get_generator("nonexistent")

    def test_add_validator(self):
        """Test adding validator."""
        test_case = PropertyTestCase()
        validator = PropertyValidator(lambda x: x > 0)
        test_case.add_validator(validator)
        assert validator in test_case.validators

    def test_for_all(self):
        """Test for_all property testing."""
        test_case = PropertyTestCase()
        test_case.examples_count = 10  # Reduce for testing
        
        def positive_prop(x):
            return x > 0
        
        int_gen = IntGenerator(1, 10)  # Only positive numbers
        
        # This should not raise an exception since all generated values are positive
        test_case.for_all(int_gen, property_func=positive_prop)

    def test_for_all_failing(self):
        """Test for_all with failing property."""
        test_case = PropertyTestCase()
        test_case.examples_count = 10
        
        def always_false_prop(x):
            return False
        
        int_gen = IntGenerator(1, 10)
        
        with pytest.raises(AssertionError):
            test_case.for_all(int_gen, property_func=always_false_prop)


class TestPropertyTestSuite:
    """Test PropertyTestSuite functionality."""

    def test_init(self):
        """Test PropertyTestSuite initialization."""
        suite = PropertyTestSuite("test_suite")
        assert suite.name == "test_suite"
        assert suite.reporter is not None

    def test_add_test(self):
        """Test adding test to suite."""
        suite = PropertyTestSuite()
        test_case = PropertyTestCase("test")
        suite.add_test(test_case)
        assert test_case in suite.tests

    def test_get_property_report(self):
        """Test getting property report."""
        suite = PropertyTestSuite()
        report = suite.get_property_report()
        assert isinstance(report, dict)


class TestPropertyTestRunner:
    """Test PropertyTestRunner functionality."""

    def test_init(self):
        """Test PropertyTestRunner initialization."""
        runner = PropertyTestRunner()
        assert runner.global_generators == {}
        assert runner.global_validators == []

    def test_add_global_generator(self):
        """Test adding global generator."""
        runner = PropertyTestRunner()
        gen = IntGenerator()
        runner.add_global_generator("int_gen", gen)
        assert runner.global_generators["int_gen"] == gen

    def test_add_global_validator(self):
        """Test adding global validator."""
        runner = PropertyTestRunner()
        validator = PropertyValidator(lambda x: x > 0)
        runner.add_global_validator(validator)
        assert validator in runner.global_validators

    def test_run_with_custom_examples(self):
        """Test running with custom examples count."""
        runner = PropertyTestRunner()
        results = runner.run_with_custom_examples(50)
        assert isinstance(results, list)

    def test_run_with_shrinking(self):
        """Test running with shrinking enabled."""
        runner = PropertyTestRunner()
        results = runner.run_with_shrinking(50)
        assert isinstance(results, list)


class TestPredefinedGenerators:
    """Test predefined generator functions."""

    def test_int_gen(self):
        """Test int_gen function."""
        gen = int_gen(0, 100)
        assert isinstance(gen, IntGenerator)
        assert gen.min_value == 0
        assert gen.max_value == 100

    def test_string_gen(self):
        """Test string_gen function."""
        gen = string_gen(0, 50, "abc")
        assert isinstance(gen, StringGenerator)
        assert gen.min_length == 0
        assert gen.max_length == 50
        assert gen.charset == "abc"

    def test_list_gen(self):
        """Test list_gen function."""
        int_gen = IntGenerator()
        gen = list_gen(int_gen, 0, 5)
        assert isinstance(gen, ListGenerator)
        assert gen.element_generator == int_gen
        assert gen.min_length == 0
        assert gen.max_length == 5

    def test_dict_gen(self):
        """Test dict_gen function."""
        key_gen = StringGenerator()
        value_gen = IntGenerator()
        gen = dict_gen(key_gen, value_gen, 0, 3)
        assert isinstance(gen, DictGenerator)
        assert gen.key_generator == key_gen
        assert gen.value_generator == value_gen
        assert gen.min_size == 0
        assert gen.max_size == 3

    def test_composite_gen(self):
        """Test composite_gen function."""
        int_gen = IntGenerator()
        str_gen = StringGenerator()
        gen = composite_gen(int=int_gen, str=str_gen)
        assert isinstance(gen, CompositeGenerator)
        assert "int" in gen.generators
        assert "str" in gen.generators


class TestPropertyValidators:
    """Test predefined property validator functions."""

    def test_always_true(self):
        """Test always_true property."""
        assert always_true() is True
        assert always_true(1, 2, 3) is True

    def test_always_false(self):
        """Test always_false property."""
        assert always_false() is False
        assert always_false(1, 2, 3) is False

    def test_is_positive(self):
        """Test is_positive property."""
        assert is_positive(5) is True
        assert is_positive(0) is False
        assert is_positive(-1) is False

    def test_is_non_negative(self):
        """Test is_non_negative property."""
        assert is_non_negative(5) is True
        assert is_non_negative(0) is True
        assert is_non_negative(-1) is False

    def test_is_even(self):
        """Test is_even property."""
        assert is_even(4) is True
        assert is_even(5) is False
        assert is_even(0) is True

    def test_is_odd(self):
        """Test is_odd property."""
        assert is_odd(5) is True
        assert is_odd(4) is False
        assert is_odd(0) is False

    def test_is_palindrome(self):
        """Test is_palindrome property."""
        assert is_palindrome("racecar") is True
        assert is_palindrome("hello") is False
        assert is_palindrome("") is True
        assert is_palindrome("a") is True

    def test_list_is_sorted(self):
        """Test list_is_sorted property."""
        assert list_is_sorted([1, 2, 3]) is True
        assert list_is_sorted([3, 2, 1]) is False
        assert list_is_sorted([]) is True
        assert list_is_sorted([1]) is True

    def test_dict_has_key(self):
        """Test dict_has_key property."""
        d = {"key1": "value1", "key2": "value2"}
        assert dict_has_key(d, "key1") is True
        assert dict_has_key(d, "key3") is False
