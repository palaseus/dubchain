"""Property-based testing infrastructure for DubChain.

This module provides property-based testing capabilities using hypothesis
and custom generators for testing invariants and properties.
"""

import logging

logger = logging.getLogger(__name__)
import random
import string
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union

from ..logging import get_logger
from .base import (
    AssertionUtils,
    BaseTestCase,
    ExecutionConfig,
    ExecutionResult,
    ExecutionStatus,
    ExecutionType,
    RunnerManager,
    SuiteManager,
)


@dataclass
class PropertyExecutionResult(ExecutionResult):
    """Property test result with additional property-specific data."""

    property_name: str = ""
    examples_tested: int = 0
    examples_passed: int = 0
    examples_failed: int = 0
    shrinking_attempts: int = 0
    minimal_failing_example: Optional[Any] = None
    property_violations: List[Any] = field(default_factory=list)


class PropertyGenerator(ABC):
    """Abstract property generator."""

    @abstractmethod
    def generate(self, size: int = 10) -> Any:
        """Generate test data."""
        pass

    @abstractmethod
    def shrink(self, value: Any) -> Generator[Any, None, None]:
        """Shrink failing example."""
        pass


class IntGenerator(PropertyGenerator):
    """Integer generator."""

    def __init__(self, min_value: int = 0, max_value: int = 100):
        self.min_value = min_value
        self.max_value = max_value

    def generate(self, size: int = 10) -> int:
        """Generate random integer."""
        return random.randint(self.min_value, self.max_value)

    def shrink(self, value: int) -> Generator[int, None, None]:
        """Shrink integer value."""
        if value > self.min_value:
            yield value - 1
        if value > 0:
            yield 0
        if value < 0:
            yield -value


class StringGenerator(PropertyGenerator):
    """String generator."""

    def __init__(self, min_length: int = 0, max_length: int = 100, charset: str = None):
        self.min_length = min_length
        self.max_length = max_length
        self.charset = charset or string.ascii_letters + string.digits

    def generate(self, size: int = 10) -> str:
        """Generate random string."""
        length = random.randint(self.min_length, min(self.max_length, size))
        return "".join(random.choices(self.charset, k=length))

    def shrink(self, value: str) -> Generator[str, None, None]:
        """Shrink string value."""
        if len(value) > self.min_length:
            # Remove characters from the end
            yield value[:-1]
            # Remove characters from the beginning
            yield value[1:]
            # Remove characters from the middle
            if len(value) > 2:
                mid = len(value) // 2
                yield value[:mid] + value[mid + 1 :]


class ListGenerator(PropertyGenerator):
    """List generator."""

    def __init__(
        self,
        element_generator: PropertyGenerator,
        min_length: int = 0,
        max_length: int = 10,
    ):
        self.element_generator = element_generator
        self.min_length = min_length
        self.max_length = max_length

    def generate(self, size: int = 10) -> List[Any]:
        """Generate random list."""
        length = random.randint(self.min_length, min(self.max_length, size))
        return [self.element_generator.generate(size) for _ in range(length)]

    def shrink(self, value: List[Any]) -> Generator[List[Any], None, None]:
        """Shrink list value."""
        if len(value) > self.min_length:
            # Remove elements from the end
            yield value[:-1]
            # Remove elements from the beginning
            yield value[1:]
            # Remove elements from the middle
            if len(value) > 2:
                mid = len(value) // 2
                yield value[:mid] + value[mid + 1 :]

        # Shrink individual elements
        for i, element in enumerate(value):
            for shrunk_element in self.element_generator.shrink(element):
                new_list = value.copy()
                new_list[i] = shrunk_element
                yield new_list


class DictGenerator(PropertyGenerator):
    """Dictionary generator."""

    def __init__(
        self,
        key_generator: PropertyGenerator,
        value_generator: PropertyGenerator,
        min_size: int = 0,
        max_size: int = 10,
    ):
        self.key_generator = key_generator
        self.value_generator = value_generator
        self.min_size = min_size
        self.max_size = max_size

    def generate(self, size: int = 10) -> Dict[Any, Any]:
        """Generate random dictionary."""
        dict_size = random.randint(self.min_size, min(self.max_size, size))
        result = {}
        for _ in range(dict_size):
            key = self.key_generator.generate(size)
            value = self.value_generator.generate(size)
            result[key] = value
        return result

    def shrink(self, value: Dict[Any, Any]) -> Generator[Dict[Any, Any], None, None]:
        """Shrink dictionary value."""
        if len(value) > self.min_size:
            # Remove random key
            keys = list(value.keys())
            if keys:
                key_to_remove = random.choice(keys)
                new_dict = value.copy()
                del new_dict[key_to_remove]
                yield new_dict

        # Shrink values
        for key, val in value.items():
            for shrunk_value in self.value_generator.shrink(val):
                new_dict = value.copy()
                new_dict[key] = shrunk_value
                yield new_dict


class CompositeGenerator(PropertyGenerator):
    """Composite generator for complex data structures."""

    def __init__(self, generators: Dict[str, PropertyGenerator]):
        self.generators = generators

    def generate(self, size: int = 10) -> Dict[str, Any]:
        """Generate composite data structure."""
        return {name: gen.generate(size) for name, gen in self.generators.items()}

    def shrink(self, value: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Shrink composite data structure."""
        for field_name, field_value in value.items():
            if field_name in self.generators:
                for shrunk_value in self.generators[field_name].shrink(field_value):
                    new_value = value.copy()
                    new_value[field_name] = shrunk_value
                    yield new_value


class PropertyValidator:
    """Property validator for checking invariants."""

    def __init__(self, property_func: Callable, name: str = ""):
        self.property_func = property_func
        self.name = name or property_func.__name__

    def validate(self, *args, **kwargs) -> bool:
        """Validate property with given arguments."""
        try:
            return bool(self.property_func(*args, **kwargs))
        except Exception:
            return False

    def validate_with_exception(
        self, *args, **kwargs
    ) -> tuple[bool, Optional[Exception]]:
        """Validate property and return exception if any."""
        try:
            result = self.property_func(*args, **kwargs)
            return bool(result), None
        except Exception as e:
            return False, e


class PropertyReporter:
    """Property test reporter."""

    def __init__(self):
        self.results: List[PropertyExecutionResult] = []
        self.logger = get_logger("property_reporter")

    def add_result(self, result: PropertyExecutionResult) -> None:
        """Add test result."""
        self.results.append(result)

    def generate_report(self) -> Dict[str, Any]:
        """Generate property test report."""
        if not self.results:
            return {"total_tests": 0, "passed": 0, "failed": 0, "error": 0}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == ExecutionStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == ExecutionStatus.FAILED)
        error = sum(1 for r in self.results if r.status == ExecutionStatus.ERROR)

        total_examples = sum(r.examples_tested for r in self.results)
        passed_examples = sum(r.examples_passed for r in self.results)
        failed_examples = sum(r.examples_failed for r in self.results)

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "error": error,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_examples_tested": total_examples,
            "examples_passed": passed_examples,
            "examples_failed": failed_examples,
            "example_success_rate": (passed_examples / total_examples * 100)
            if total_examples > 0
            else 0,
            "failing_properties": [
                {
                    "property_name": r.property_name,
                    "examples_tested": r.examples_tested,
                    "examples_failed": r.examples_failed,
                    "minimal_failing_example": r.minimal_failing_example,
                    "shrinking_attempts": r.shrinking_attempts,
                }
                for r in self.results
                if r.status == ExecutionStatus.FAILED
            ],
        }


class PropertyTestCase(BaseTestCase):
    """Property-based test case."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.config = ExecutionConfig()
        self.config.test_type = ExecutionType.PROPERTY
        self.generators: Dict[str, PropertyGenerator] = {}
        self.validators: List[PropertyValidator] = []
        self.examples_count: int = 100
        self.shrinking_enabled: bool = True
        self.max_shrinking_attempts: int = 100

    def setup(self) -> None:
        """Setup property test case."""
        super().setup()
        self._setup_generators()
        self._setup_validators()

    def teardown(self) -> None:
        """Teardown property test case."""
        self._cleanup_validators()
        self._cleanup_generators()
        super().teardown()

    def run_test(self) -> None:
        """Run the actual test."""
        # This is a base class - subclasses should override this method
        pass

    def _setup_generators(self) -> None:
        """Setup property generators."""
        pass

    def _cleanup_generators(self) -> None:
        """Cleanup property generators."""
        self.generators.clear()

    def _setup_validators(self) -> None:
        """Setup property validators."""
        pass

    def _cleanup_validators(self) -> None:
        """Cleanup property validators."""
        self.validators.clear()

    def add_generator(self, name: str, generator: PropertyGenerator) -> None:
        """Add property generator."""
        self.generators[name] = generator

    def get_generator(self, name: str) -> PropertyGenerator:
        """Get property generator."""
        if name not in self.generators:
            raise KeyError(f"Generator '{name}' not found")
        return self.generators[name]

    def add_validator(self, validator: PropertyValidator) -> None:
        """Add property validator."""
        self.validators.append(validator)

    def for_all(self, *generators, property_func: Callable) -> None:
        """Test property for all generated values."""
        validator = PropertyValidator(property_func)
        self.add_validator(validator)

        # Generate test data
        test_data = []
        for _ in range(self.examples_count):
            args = [gen.generate() for gen in generators]
            test_data.append(args)

        # Test property
        passed = 0
        failed = 0
        failing_examples = []

        for args in test_data:
            is_valid, exception = validator.validate_with_exception(*args)
            if is_valid:
                passed += 1
            else:
                failed += 1
                failing_examples.append((args, exception))

        # Update result
        if self.result:
            self.result.examples_tested = len(test_data)
            self.result.examples_passed = passed
            self.result.examples_failed = failed

        # Handle failures
        if failing_examples:
            if self.shrinking_enabled:
                self._shrink_failing_examples(failing_examples, generators, validator)

            # Use first failing example for assertion
            first_failing = failing_examples[0]
            args, exception = first_failing

            if exception:
                raise AssertionError(f"Property failed with exception: {exception}")
            else:
                raise AssertionError(f"Property failed for arguments: {args}")

    def _shrink_failing_examples(
        self,
        failing_examples: List[tuple],
        generators: List[PropertyGenerator],
        validator: PropertyValidator,
    ) -> None:
        """Shrink failing examples to find minimal counterexamples."""
        if not self.result:
            return

        shrinking_attempts = 0
        minimal_example = None

        for failing_args, _ in failing_examples:
            current_example = failing_args
            shrunk = True

            while shrunk and shrinking_attempts < self.max_shrinking_attempts:
                shrunk = False
                shrinking_attempts += 1

                # Try to shrink each argument
                for i, (arg, gen) in enumerate(zip(current_example, generators)):
                    for shrunk_arg in gen.shrink(arg):
                        new_args = list(current_example)
                        new_args[i] = shrunk_arg

                        is_valid, _ = validator.validate_with_exception(*new_args)
                        if not is_valid:
                            current_example = tuple(new_args)
                            shrunk = True
                            break

                    if shrunk:
                        break

            if minimal_example is None or len(str(current_example)) < len(
                str(minimal_example)
            ):
                minimal_example = current_example

        self.result.shrinking_attempts = shrinking_attempts
        self.result.minimal_failing_example = minimal_example

    def run_test(self) -> None:
        """Run property test."""
        # This will be implemented by subclasses
        pass


class PropertyTestSuite(SuiteManager):
    """Property test suite."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.config.test_type = ExecutionType.PROPERTY
        self.reporter = PropertyReporter()

    def add_test(self, test: PropertyTestCase) -> None:
        """Add property test to suite."""
        super().add_test(test)

    def run(self, config: ExecutionConfig = None) -> List[ExecutionResult]:
        """Run all property tests in suite."""
        config = config or self.config
        results = super().run(config)

        # Add results to reporter
        for result in results:
            if isinstance(result, PropertyExecutionResult):
                self.reporter.add_result(result)

        return results

    def get_property_report(self) -> Dict[str, Any]:
        """Get property test report."""
        return self.reporter.generate_report()


class PropertyTestRunner(RunnerManager):
    """Property test runner."""

    def __init__(self, config: ExecutionConfig = None):
        super().__init__(config)
        self.config.test_type = ExecutionType.PROPERTY
        self.global_generators: Dict[str, PropertyGenerator] = {}
        self.global_validators: List[PropertyValidator] = []

    def add_global_generator(self, name: str, generator: PropertyGenerator) -> None:
        """Add global generator."""
        self.global_generators[name] = generator

    def add_global_validator(self, validator: PropertyValidator) -> None:
        """Add global validator."""
        self.global_validators.append(validator)

    def run_with_custom_examples(
        self, examples_count: int = 1000
    ) -> List[ExecutionResult]:
        """Run tests with custom number of examples."""
        original_config = self.config
        self.config.examples_count = examples_count

        try:
            return self.run_all()
        finally:
            self.config = original_config

    def run_with_shrinking(self, max_attempts: int = 100) -> List[ExecutionResult]:
        """Run tests with shrinking enabled."""
        original_config = self.config
        self.config.shrinking_enabled = True
        self.config.max_shrinking_attempts = max_attempts

        try:
            return self.run_all()
        finally:
            self.config = original_config


# Predefined generators for common types
def int_gen(min_val: int = 0, max_val: int = 100) -> IntGenerator:
    """Create integer generator."""
    return IntGenerator(min_val, max_val)


def string_gen(
    min_len: int = 0, max_len: int = 100, charset: str = None
) -> StringGenerator:
    """Create string generator."""
    return StringGenerator(min_len, max_len, charset)


def list_gen(
    element_gen: PropertyGenerator, min_len: int = 0, max_len: int = 10
) -> ListGenerator:
    """Create list generator."""
    return ListGenerator(element_gen, min_len, max_len)


def dict_gen(
    key_gen: PropertyGenerator,
    value_gen: PropertyGenerator,
    min_size: int = 0,
    max_size: int = 10,
) -> DictGenerator:
    """Create dictionary generator."""
    return DictGenerator(key_gen, value_gen, min_size, max_size)


def composite_gen(**generators) -> CompositeGenerator:
    """Create composite generator."""
    return CompositeGenerator(generators)


# Common property validators
def always_true(*args, **kwargs) -> bool:
    """Always true property."""
    return True


def always_false(*args, **kwargs) -> bool:
    """Always false property."""
    return False


def is_positive(x: int) -> bool:
    """Check if integer is positive."""
    return x > 0


def is_non_negative(x: int) -> bool:
    """Check if integer is non-negative."""
    return x >= 0


def is_even(x: int) -> bool:
    """Check if integer is even."""
    return x % 2 == 0


def is_odd(x: int) -> bool:
    """Check if integer is odd."""
    return x % 2 == 1


def is_palindrome(s: str) -> bool:
    """Check if string is palindrome."""
    return s == s[::-1]


def list_is_sorted(lst: List[Any]) -> bool:
    """Check if list is sorted."""
    return lst == sorted(lst)


def dict_has_key(d: Dict[Any, Any], key: Any) -> bool:
    """Check if dictionary has key."""
    return key in d
