"""Fuzz testing infrastructure for DubChain.

This module provides fuzz testing capabilities for testing system
robustness with random and malformed inputs.
"""

import hashlib
import logging
import random
import string
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Type, Union

from ..logging import get_logger
from .base import (
    BaseTestCase,
    ExecutionConfig,
    ExecutionResult,
    ExecutionStatus,
    ExecutionType,
    RunnerManager,
    SuiteManager,
)


@dataclass
class FuzzExecutionResult(ExecutionResult):
    """Fuzz test result with additional fuzz-specific data."""

    fuzz_iterations: int = 0
    crashes_found: int = 0
    hangs_found: int = 0
    memory_leaks_found: int = 0
    unique_crashes: List[Any] = field(default_factory=list)
    unique_hangs: List[Any] = field(default_factory=list)
    coverage_achieved: float = 0.0
    fuzz_corpus_size: int = 0


class FuzzGenerator(ABC):
    """Abstract fuzz generator."""

    @abstractmethod
    def generate(self, seed: Any = None) -> bytes:
        """Generate fuzz data."""
        pass

    @abstractmethod
    def mutate(self, data: bytes, mutation_count: int = 1) -> bytes:
        """Mutate existing data."""
        pass


class RandomFuzzGenerator(FuzzGenerator):
    """Random fuzz generator."""

    def __init__(self, min_size: int = 1, max_size: int = 1024):
        self.min_size = min_size
        self.max_size = max_size

    def generate(self, seed: Any = None) -> bytes:
        """Generate random fuzz data."""
        if seed is not None:
            random.seed(seed)

        size = random.randint(self.min_size, self.max_size)
        return bytes([random.randint(0, 255) for _ in range(size)])

    def mutate(self, data: bytes, mutation_count: int = 1) -> bytes:
        """Mutate existing data."""
        result = bytearray(data)

        for _ in range(mutation_count):
            if not result:
                break

            mutation_type = random.choice(["flip", "insert", "delete", "replace"])

            if mutation_type == "flip":
                # Flip random bit
                pos = random.randint(0, len(result) - 1)
                result[pos] ^= 1 << random.randint(0, 7)

            elif mutation_type == "insert":
                # Insert random byte
                pos = random.randint(0, len(result))
                result.insert(pos, random.randint(0, 255))

            elif mutation_type == "delete":
                # Delete random byte
                if len(result) > 1:
                    pos = random.randint(0, len(result) - 1)
                    del result[pos]

            elif mutation_type == "replace":
                # Replace random byte
                pos = random.randint(0, len(result) - 1)
                result[pos] = random.randint(0, 255)

        return bytes(result)


class StringFuzzGenerator(FuzzGenerator):
    """String fuzz generator."""

    def __init__(
        self, min_length: int = 1, max_length: int = 1000, charset: str = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.charset = charset or string.printable

    def generate(self, seed: Any = None) -> bytes:
        """Generate random string fuzz data."""
        if seed is not None:
            random.seed(seed)

        length = random.randint(self.min_length, self.max_length)
        return "".join(random.choices(self.charset, k=length)).encode("utf-8")

    def mutate(self, data: bytes, mutation_count: int = 1) -> bytes:
        """Mutate string data."""
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # If not valid UTF-8, treat as binary
            return RandomFuzzGenerator().mutate(data, mutation_count)

        result = list(text)

        for _ in range(mutation_count):
            if not result:
                break

            mutation_type = random.choice(["insert", "delete", "replace", "duplicate"])

            if mutation_type == "insert":
                # Insert random character
                pos = random.randint(0, len(result))
                result.insert(pos, random.choice(self.charset))

            elif mutation_type == "delete":
                # Delete random character
                if len(result) > 1:
                    pos = random.randint(0, len(result) - 1)
                    del result[pos]

            elif mutation_type == "replace":
                # Replace random character
                pos = random.randint(0, len(result) - 1)
                result[pos] = random.choice(self.charset)

            elif mutation_type == "duplicate":
                # Duplicate random character
                pos = random.randint(0, len(result) - 1)
                result.insert(pos, result[pos])

        return "".join(result).encode("utf-8")


class StructuredFuzzGenerator(FuzzGenerator):
    """Structured fuzz generator for specific data formats."""

    def __init__(self, format_type: str = "json"):
        self.format_type = format_type

    def generate(self, seed: Any = None) -> bytes:
        """Generate structured fuzz data."""
        if seed is not None:
            random.seed(seed)

        if self.format_type == "json":
            return self._generate_json()
        elif self.format_type == "xml":
            return self._generate_xml()
        elif self.format_type == "csv":
            return self._generate_csv()
        else:
            return RandomFuzzGenerator().generate(seed)

    def _generate_json(self) -> bytes:
        """Generate JSON fuzz data."""
        import json

        # Generate random JSON structure
        data = {
            "string": "".join(
                random.choices(string.ascii_letters, k=random.randint(1, 50))
            ),
            "number": random.randint(-1000, 1000),
            "boolean": random.choice([True, False]),
            "array": [random.randint(0, 100) for _ in range(random.randint(0, 10))],
            "object": {
                "nested": random.choice([True, False, None]),
                "value": random.uniform(0, 1),
            },
        }

        return json.dumps(data).encode("utf-8")

    def _generate_xml(self) -> bytes:
        """Generate XML fuzz data."""
        root_tag = "".join(
            random.choices(string.ascii_letters, k=random.randint(1, 10))
        )
        content = "".join(
            random.choices(
                string.ascii_letters + string.digits, k=random.randint(0, 100)
            )
        )

        xml = f"<{root_tag}>{content}</{root_tag}>"
        return xml.encode("utf-8")

    def _generate_csv(self) -> bytes:
        """Generate CSV fuzz data."""
        rows = random.randint(1, 10)
        cols = random.randint(1, 5)

        csv_lines = []
        for _ in range(rows):
            row = [
                "".join(random.choices(string.ascii_letters, k=random.randint(1, 10)))
                for _ in range(cols)
            ]
            csv_lines.append(",".join(row))

        return "\n".join(csv_lines).encode("utf-8")

    def mutate(self, data: bytes, mutation_count: int = 1) -> bytes:
        """Mutate structured data."""
        # For structured data, we can apply format-specific mutations
        if self.format_type == "json":
            return self._mutate_json(data, mutation_count)
        else:
            # Fall back to random mutations
            return RandomFuzzGenerator().mutate(data, mutation_count)

    def _mutate_json(self, data: bytes, mutation_count: int) -> bytes:
        """Mutate JSON data."""
        try:
            import json

            text = data.decode("utf-8")
            parsed = json.loads(text)

            # Apply mutations to parsed data
            for _ in range(mutation_count):
                self._mutate_json_value(parsed)

            return json.dumps(parsed).encode("utf-8")
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If not valid JSON, use random mutations
            return RandomFuzzGenerator().mutate(data, mutation_count)

    def _mutate_json_value(self, value: Any) -> None:
        """Mutate JSON value in place."""
        if isinstance(value, dict):
            if value:
                key = random.choice(list(value.keys()))
                self._mutate_json_value(value[key])
        elif isinstance(value, list):
            if value:
                idx = random.randint(0, len(value) - 1)
                self._mutate_json_value(value[idx])
        elif isinstance(value, str):
            # Mutate string
            if value:
                pos = random.randint(0, len(value) - 1)
                value = (
                    value[:pos] + random.choice(string.ascii_letters) + value[pos + 1 :]
                )
        elif isinstance(value, (int, float)):
            # Mutate number
            if isinstance(value, int):
                value = random.randint(-1000, 1000)
            else:
                value = random.uniform(-100, 100)


class FuzzMutator:
    """Fuzz mutator for applying various mutation strategies."""

    def __init__(self):
        self.mutation_strategies = [
            self._bit_flip,
            self._byte_flip,
            self._arithmetic_mutation,
            self._interesting_values,
            self._havoc_mutation,
            self._splice_mutation,
        ]

    def mutate(self, data: bytes, strategy: str = None) -> bytes:
        """Apply mutation strategy to data."""
        if strategy:
            strategy_func = getattr(self, f"_{strategy}", None)
            if strategy_func:
                return strategy_func(data)

        # Use random strategy
        strategy_func = random.choice(self.mutation_strategies)
        return strategy_func(data)

    def _bit_flip(self, data: bytes) -> bytes:
        """Flip random bits."""
        result = bytearray(data)
        if not result:
            return bytes(result)

        num_flips = random.randint(1, min(8, len(result)))
        for _ in range(num_flips):
            pos = random.randint(0, len(result) - 1)
            bit = random.randint(0, 7)
            result[pos] ^= 1 << bit

        return bytes(result)

    def _byte_flip(self, data: bytes) -> bytes:
        """Flip random bytes."""
        result = bytearray(data)
        if not result:
            return bytes(result)

        num_flips = random.randint(1, min(4, len(result)))
        for _ in range(num_flips):
            pos = random.randint(0, len(result) - 1)
            result[pos] ^= 0xFF

        return bytes(result)

    def _arithmetic_mutation(self, data: bytes) -> bytes:
        """Apply arithmetic mutations."""
        result = bytearray(data)
        if len(result) < 4:
            return bytes(result)

        # Treat as 32-bit integer and apply arithmetic
        pos = random.randint(0, len(result) - 4)
        value = struct.unpack("<I", result[pos : pos + 4])[0]

        operations = ["add", "sub", "mul", "div"]
        operation = random.choice(operations)
        operand = random.randint(1, 100)

        if operation == "add":
            value = (value + operand) & 0xFFFFFFFF
        elif operation == "sub":
            value = (value - operand) & 0xFFFFFFFF
        elif operation == "mul":
            value = (value * operand) & 0xFFFFFFFF
        elif operation == "div":
            if operand != 0:
                value = (value // operand) & 0xFFFFFFFF

        result[pos : pos + 4] = struct.pack("<I", value)
        return bytes(result)

    def _interesting_values(self, data: bytes) -> bytes:
        """Replace with interesting values."""
        result = bytearray(data)
        if not result:
            return bytes(result)

        interesting_values = [
            0x00,
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,
            0x06,
            0x07,
            0x08,
            0x09,
            0x0A,
            0x0B,
            0x0C,
            0x0D,
            0x0E,
            0x0F,
            0x10,
            0x20,
            0x40,
            0x7F,
            0x80,
            0x81,
            0xFE,
            0xFF,
            0x100,
            0x101,
            0x200,
            0x201,
            0x400,
            0x401,
            0x800,
            0x801,
            0x1000,
            0x1001,
            0x2000,
            0x2001,
            0x4000,
            0x4001,
            0x8000,
            0x8001,
            0x10000,
            0x10001,
            0x20000,
            0x20001,
            0x40000,
            0x40001,
            0x80000,
            0x80001,
            0x100000,
            0x100001,
            0x200000,
            0x200001,
            0x400000,
            0x400001,
            0x800000,
            0x800001,
            0x1000000,
            0x1000001,
            0x2000000,
            0x2000001,
            0x4000000,
            0x4000001,
            0x8000000,
            0x8000001,
            0x10000000,
            0x10000001,
            0x20000000,
            0x20000001,
            0x40000000,
            0x40000001,
            0x80000000,
            0x80000001,
            0xFFFFFFFF,
        ]

        pos = random.randint(0, len(result) - 1)
        value = random.choice(interesting_values)

        if value <= 0xFF:
            result[pos] = value
        elif value <= 0xFFFF and pos < len(result) - 1:
            result[pos] = value & 0xFF
            result[pos + 1] = (value >> 8) & 0xFF
        elif value <= 0xFFFFFFFF and pos < len(result) - 3:
            result[pos] = value & 0xFF
            result[pos + 1] = (value >> 8) & 0xFF
            result[pos + 2] = (value >> 16) & 0xFF
            result[pos + 3] = (value >> 24) & 0xFF

        return bytes(result)

    def _havoc_mutation(self, data: bytes) -> bytes:
        """Apply havoc mutation (multiple random mutations)."""
        result = bytearray(data)
        if not result:
            return bytes(result)

        num_mutations = random.randint(1, 10)
        for _ in range(num_mutations):
            mutation_type = random.choice(["insert", "delete", "replace", "duplicate"])

            if mutation_type == "insert" and len(result) < 1000:
                pos = random.randint(0, len(result))
                result.insert(pos, random.randint(0, 255))

            elif mutation_type == "delete" and len(result) > 1:
                pos = random.randint(0, len(result) - 1)
                del result[pos]

            elif mutation_type == "replace":
                pos = random.randint(0, len(result) - 1)
                result[pos] = random.randint(0, 255)

            elif mutation_type == "duplicate" and len(result) < 1000:
                pos = random.randint(0, len(result) - 1)
                result.insert(pos, result[pos])

        return bytes(result)

    def _splice_mutation(self, data: bytes) -> bytes:
        """Apply splice mutation (combine with another input)."""
        # For simplicity, we'll just duplicate the data
        result = bytearray(data)
        if not result:
            return bytes(result)

        # Duplicate a random portion
        start = random.randint(0, len(result) - 1)
        end = random.randint(start, len(result))
        portion = result[start:end]

        # Insert at random position
        pos = random.randint(0, len(result))
        result[pos:pos] = portion

        return bytes(result)


class FuzzAnalyzer:
    """Fuzz analyzer for analyzing fuzz test results."""

    def __init__(self):
        self.crashes: List[Dict[str, Any]] = []
        self.hangs: List[Dict[str, Any]] = []
        self.coverage_data: Dict[str, Any] = {}
        self.unique_crashes: Dict[str, Any] = {}
        self.unique_hangs: Dict[str, Any] = {}

    def analyze_crash(self, input_data: bytes, crash_info: Dict[str, Any]) -> None:
        """Analyze crash information."""
        crash_hash = self._hash_input(input_data)

        crash_entry = {
            "hash": crash_hash,
            "input": input_data,
            "crash_info": crash_info,
            "timestamp": time.time(),
        }

        self.crashes.append(crash_entry)

        # Check if this is a unique crash
        if crash_hash not in self.unique_crashes:
            self.unique_crashes[crash_hash] = crash_entry

    def analyze_hang(self, input_data: bytes, hang_info: Dict[str, Any]) -> None:
        """Analyze hang information."""
        hang_hash = self._hash_input(input_data)

        hang_entry = {
            "hash": hang_hash,
            "input": input_data,
            "hang_info": hang_info,
            "timestamp": time.time(),
        }

        self.hangs.append(hang_entry)

        # Check if this is a unique hang
        if hang_hash not in self.unique_hangs:
            self.unique_hangs[hang_hash] = hang_entry

    def update_coverage(self, coverage_data: Dict[str, Any]) -> None:
        """Update coverage information."""
        self.coverage_data.update(coverage_data)

    def _hash_input(self, input_data: bytes) -> str:
        """Hash input data for uniqueness checking."""
        return hashlib.sha256(input_data).hexdigest()

    def get_analysis_report(self) -> Dict[str, Any]:
        """Get fuzz analysis report."""
        return {
            "total_crashes": len(self.crashes),
            "unique_crashes": len(self.unique_crashes),
            "total_hangs": len(self.hangs),
            "unique_hangs": len(self.unique_hangs),
            "coverage_data": self.coverage_data,
            "crash_summary": [
                {
                    "hash": crash["hash"],
                    "input_length": len(crash["input"]),
                    "crash_type": crash["crash_info"].get("type", "unknown"),
                }
                for crash in self.unique_crashes.values()
            ],
            "hang_summary": [
                {
                    "hash": hang["hash"],
                    "input_length": len(hang["input"]),
                    "hang_duration": hang["hang_info"].get("duration", 0),
                }
                for hang in self.unique_hangs.values()
            ],
        }


class FuzzTestCase(BaseTestCase):
    """Fuzz test case."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.config = ExecutionConfig()
        self.config.test_type = ExecutionType.FUZZ
        self.generator: Optional[FuzzGenerator] = None
        self.mutator: Optional[FuzzMutator] = None
        self.analyzer: Optional[FuzzAnalyzer] = None
        self.target_function: Optional[Callable] = None
        self.fuzz_iterations: int = 1000
        self.timeout: float = 1.0
        self.corpus: List[bytes] = []
        self.crashes: List[bytes] = []
        self.hangs: List[bytes] = []

    def setup(self) -> None:
        """Setup fuzz test case."""
        super().setup()
        self._setup_generator()
        self._setup_mutator()
        self._setup_analyzer()
        self._setup_corpus()

    def teardown(self) -> None:
        """Teardown fuzz test case."""
        self._cleanup_analyzer()
        self._cleanup_mutator()
        self._cleanup_generator()
        super().teardown()

    def _setup_generator(self) -> None:
        """Setup fuzz generator."""
        self.generator = RandomFuzzGenerator()

    def _cleanup_generator(self) -> None:
        """Cleanup fuzz generator."""
        self.generator = None

    def _setup_mutator(self) -> None:
        """Setup fuzz mutator."""
        self.mutator = FuzzMutator()

    def _cleanup_mutator(self) -> None:
        """Cleanup fuzz mutator."""
        self.mutator = None

    def _setup_analyzer(self) -> None:
        """Setup fuzz analyzer."""
        self.analyzer = FuzzAnalyzer()

    def _cleanup_analyzer(self) -> None:
        """Cleanup fuzz analyzer."""
        self.analyzer = None

    def _setup_corpus(self) -> None:
        """Setup initial fuzz corpus."""
        # Add some basic test cases
        self.corpus = [
            b"",  # Empty input
            b"a",  # Single character
            b"hello",  # Simple string
            b"\x00\x01\x02\x03",  # Binary data
            b"A" * 100,  # Long string
        ]

    def set_target_function(self, target_func: Callable) -> None:
        """Set target function to fuzz."""
        self.target_function = target_func

    def set_generator(self, generator: FuzzGenerator) -> None:
        """Set fuzz generator."""
        self.generator = generator

    def add_to_corpus(self, data: bytes) -> None:
        """Add data to fuzz corpus."""
        if data not in self.corpus:
            self.corpus.append(data)

    def run_test(self) -> None:
        """Run fuzz test."""
        if not self.target_function:
            raise ValueError("Target function not set")

        if not self.generator:
            raise ValueError("Fuzz generator not set")

        iterations = 0
        crashes = 0
        hangs = 0

        while iterations < self.fuzz_iterations:
            # Generate or mutate input
            if self.corpus and random.random() < 0.5:
                # Mutate from corpus
                base_input = random.choice(self.corpus)
                input_data = self.mutator.mutate(base_input)
            else:
                # Generate new input
                input_data = self.generator.generate()

            # Test the input
            try:
                result = self._test_input(input_data)
                if result["status"] == "crash":
                    crashes += 1
                    self.crashes.append(input_data)
                    if self.analyzer:
                        self.analyzer.analyze_crash(input_data, result)
                elif result["status"] == "hang":
                    hangs += 1
                    self.hangs.append(input_data)
                    if self.analyzer:
                        self.analyzer.analyze_hang(input_data, result)
                elif result["status"] == "success":
                    # Add to corpus if it's interesting
                    self.add_to_corpus(input_data)

            except Exception as e:
                crashes += 1
                self.crashes.append(input_data)
                if self.analyzer:
                    self.analyzer.analyze_crash(input_data, {"exception": str(e)})

            iterations += 1

        # Update result
        if self.result:
            self.result.fuzz_iterations = iterations
            self.result.crashes_found = crashes
            self.result.hangs_found = hangs
            self.result.fuzz_corpus_size = len(self.corpus)

            if self.analyzer:
                report = self.analyzer.get_analysis_report()
                self.result.unique_crashes = report["crash_summary"]
                self.result.unique_hangs = report["hang_summary"]

    def _test_input(self, input_data: bytes) -> Dict[str, Any]:
        """Test input data with target function."""
        import signal
        import threading

        result = {"status": "unknown", "output": None, "error": None}

        def timeout_handler(signum, frame):
            raise TimeoutError("Function timed out")

        def run_function():
            try:
                output = self.target_function(input_data)
                result["status"] = "success"
                result["output"] = output
            except Exception as e:
                result["status"] = "crash"
                result["error"] = str(e)

        # Set up timeout
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout))

        try:
            # Run in thread to detect hangs
            thread = threading.Thread(target=run_function)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.timeout)

            if thread.is_alive():
                result["status"] = "hang"
                result["error"] = "Function hung"

        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

        return result


class FuzzTestSuite(SuiteManager):
    """Fuzz test suite."""

    def __init__(self, name: str = None):
        super().__init__(name)
        self.config = ExecutionConfig()
        self.config.test_type = ExecutionType.FUZZ

    def add_test(self, test: FuzzTestCase) -> None:
        """Add fuzz test to suite."""
        super().add_test(test)


class FuzzTestRunner(RunnerManager):
    """Fuzz test runner."""

    def __init__(self, config: ExecutionConfig = None):
        super().__init__(config)
        if self.config:
            self.config.test_type = ExecutionType.FUZZ

    def run_with_iterations(self, iterations: int = 10000) -> List[ExecutionResult]:
        """Run fuzz tests with specified iterations."""
        original_config = self.config
        self.config.fuzz_iterations = iterations

        try:
            return self.run_all()
        finally:
            self.config = original_config

    def run_with_timeout(self, timeout: float = 5.0) -> List[ExecutionResult]:
        """Run fuzz tests with specified timeout."""
        original_config = self.config
        self.config.fuzz_timeout = timeout

        try:
            return self.run_all()
        finally:
            self.config = original_config
