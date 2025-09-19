"""Test utilities for DubChain.

This module provides utility functions and helpers for testing.
"""

import hashlib
import json
import logging
import random
import string
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..logging import get_logger


class TestUtils:
    """Test utility functions."""

    @staticmethod
    def generate_random_string(length: int = 10, charset: str = None) -> str:
        """Generate random string."""
        if charset is None:
            charset = string.ascii_letters + string.digits
        return "".join(random.choices(charset, k=length))

    @staticmethod
    def generate_random_int(min_val: int = 0, max_val: int = 100) -> int:
        """Generate random integer."""
        return random.randint(min_val, max_val)

    @staticmethod
    def generate_random_float(min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Generate random float."""
        return random.uniform(min_val, max_val)

    @staticmethod
    def generate_random_bytes(length: int = 10) -> bytes:
        """Generate random bytes."""
        return bytes([random.randint(0, 255) for _ in range(length)])

    @staticmethod
    def generate_random_dict(size: int = 5) -> Dict[str, Any]:
        """Generate random dictionary."""
        result = {}
        for _ in range(size):
            key = TestUtils.generate_random_string(5)
            value = random.choice(
                [
                    TestUtils.generate_random_string(),
                    TestUtils.generate_random_int(),
                    TestUtils.generate_random_float(),
                    TestUtils.generate_random_bytes(5),
                ]
            )
            result[key] = value
        return result

    @staticmethod
    def generate_random_list(size: int = 5) -> List[Any]:
        """Generate random list."""
        result = []
        for _ in range(size):
            value = random.choice(
                [
                    TestUtils.generate_random_string(),
                    TestUtils.generate_random_int(),
                    TestUtils.generate_random_float(),
                    TestUtils.generate_random_bytes(5),
                ]
            )
            result.append(value)
        return result

    @staticmethod
    def hash_data(data: Any) -> str:
        """Hash data using SHA-256."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        elif not isinstance(data, bytes):
            data = json.dumps(data, sort_keys=True).encode("utf-8")

        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def wait_for_condition(
        condition: Callable, timeout: float = 10.0, interval: float = 0.1
    ) -> bool:
        """Wait for condition to be true."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition():
                return True
            time.sleep(interval)
        return False

    @staticmethod
    def retry_on_failure(
        func: Callable, max_retries: int = 3, delay: float = 1.0
    ) -> Any:
        """Retry function on failure."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)

    @staticmethod
    def measure_execution_time(func: Callable, *args, **kwargs) -> tuple[Any, float]:
        """Measure function execution time."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    @staticmethod
    def create_temp_file(content: str = "", suffix: str = ".tmp") -> str:
        """Create temporary file with content."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name

    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """Cleanup temporary file."""
        import os

        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass


class TestHelpers:
    """Test helper functions."""

    @staticmethod
    def assert_dict_contains(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
        """Assert that dict1 contains all keys and values from dict2."""
        for key, value in dict2.items():
            if key not in dict1:
                raise AssertionError(f"Key '{key}' not found in dict1")
            if dict1[key] != value:
                raise AssertionError(
                    f"Value for key '{key}' differs: expected {value}, got {dict1[key]}"
                )

    @staticmethod
    def assert_list_contains(list1: List[Any], list2: List[Any]) -> None:
        """Assert that list1 contains all elements from list2."""
        for item in list2:
            if item not in list1:
                raise AssertionError(f"Item {item} not found in list1")

    @staticmethod
    def assert_approximately_equal(
        actual: float, expected: float, tolerance: float = 1e-6
    ) -> None:
        """Assert that two floats are approximately equal."""
        if abs(actual - expected) > tolerance:
            raise AssertionError(
                f"Values not approximately equal: expected {expected}, got {actual}"
            )

    @staticmethod
    def assert_string_contains(text: str, substring: str) -> None:
        """Assert that text contains substring."""
        if substring not in text:
            raise AssertionError(f"Substring '{substring}' not found in text")

    @staticmethod
    def assert_regex_match(text: str, pattern: str) -> None:
        """Assert that text matches regex pattern."""
        import re

        if not re.match(pattern, text):
            raise AssertionError(f"Text '{text}' does not match pattern '{pattern}'")

    @staticmethod
    def assert_file_exists(file_path: str) -> None:
        """Assert that file exists."""
        import os

        if not os.path.exists(file_path):
            raise AssertionError(f"File '{file_path}' does not exist")

    @staticmethod
    def assert_file_not_exists(file_path: str) -> None:
        """Assert that file does not exist."""
        import os

        if os.path.exists(file_path):
            raise AssertionError(f"File '{file_path}' exists")

    @staticmethod
    def assert_directory_exists(dir_path: str) -> None:
        """Assert that directory exists."""
        import os

        if not os.path.exists(dir_path):
            raise AssertionError(f"Directory '{dir_path}' does not exist")
        if not os.path.isdir(dir_path):
            raise AssertionError(f"Path '{dir_path}' is not a directory")


class TestDataGenerators:
    """Test data generators."""

    @staticmethod
    def generate_user_data(user_id: str = None) -> Dict[str, Any]:
        """Generate user test data."""
        return {
            "id": user_id or TestUtils.generate_random_string(8),
            "name": TestUtils.generate_random_string(10),
            "email": f"{TestUtils.generate_random_string(5)}@example.com",
            "age": TestUtils.generate_random_int(18, 65),
            "created_at": time.time(),
            "is_active": random.choice([True, False]),
        }

    @staticmethod
    def generate_transaction_data(transaction_id: str = None) -> Dict[str, Any]:
        """Generate transaction test data."""
        return {
            "id": transaction_id or TestUtils.generate_random_string(8),
            "from_address": TestUtils.generate_random_string(40),
            "to_address": TestUtils.generate_random_string(40),
            "amount": TestUtils.generate_random_float(0.1, 1000.0),
            "fee": TestUtils.generate_random_float(0.01, 10.0),
            "timestamp": time.time(),
            "status": random.choice(["pending", "confirmed", "failed"]),
        }

    @staticmethod
    def generate_block_data(block_id: str = None) -> Dict[str, Any]:
        """Generate block test data."""
        return {
            "id": block_id or TestUtils.generate_random_string(8),
            "previous_hash": TestUtils.generate_random_string(64),
            "merkle_root": TestUtils.generate_random_string(64),
            "timestamp": time.time(),
            "nonce": TestUtils.generate_random_int(0, 1000000),
            "difficulty": TestUtils.generate_random_int(1, 10),
            "transactions": [
                TestUtils.generate_random_string(8)
                for _ in range(random.randint(1, 10))
            ],
        }

    @staticmethod
    def generate_wallet_data(wallet_address: str = None) -> Dict[str, Any]:
        """Generate wallet test data."""
        return {
            "address": wallet_address or TestUtils.generate_random_string(40),
            "private_key": TestUtils.generate_random_string(64),
            "public_key": TestUtils.generate_random_string(64),
            "balance": TestUtils.generate_random_float(0.0, 10000.0),
            "created_at": time.time(),
            "last_activity": time.time(),
        }

    @staticmethod
    def generate_smart_contract_data(contract_id: str = None) -> Dict[str, Any]:
        """Generate smart contract test data."""
        return {
            "id": contract_id or TestUtils.generate_random_string(8),
            "name": TestUtils.generate_random_string(10),
            "bytecode": TestUtils.generate_random_string(100),
            "abi": [
                {
                    "name": TestUtils.generate_random_string(8),
                    "type": "function",
                    "inputs": [
                        {
                            "name": TestUtils.generate_random_string(5),
                            "type": random.choice(
                                ["address", "uint256", "string", "bool"]
                            ),
                        }
                    ],
                    "outputs": [
                        {
                            "name": "",
                            "type": random.choice(
                                ["address", "uint256", "string", "bool"]
                            ),
                        }
                    ],
                }
            ],
            "deployed_address": TestUtils.generate_random_string(40),
            "creator": TestUtils.generate_random_string(40),
            "created_at": time.time(),
        }

    @staticmethod
    def generate_network_message_data(message_id: str = None) -> Dict[str, Any]:
        """Generate network message test data."""
        return {
            "id": message_id or TestUtils.generate_random_string(8),
            "type": random.choice(
                ["ping", "pong", "block", "transaction", "request", "response"]
            ),
            "sender": TestUtils.generate_random_string(40),
            "receiver": TestUtils.generate_random_string(40),
            "payload": TestUtils.generate_random_dict(5),
            "timestamp": time.time(),
            "ttl": TestUtils.generate_random_int(1, 3600),
        }


class TestValidators:
    """Test validation functions."""

    @staticmethod
    def validate_user_data(user_data: Dict[str, Any]) -> bool:
        """Validate user data."""
        required_fields = ["id", "name", "email", "age", "created_at", "is_active"]

        for field in required_fields:
            if field not in user_data:
                return False

        if not isinstance(user_data["id"], str) or len(user_data["id"]) == 0:
            return False

        if not isinstance(user_data["name"], str) or len(user_data["name"]) == 0:
            return False

        if not isinstance(user_data["email"], str) or "@" not in user_data["email"]:
            return False

        if not isinstance(user_data["age"], int) or user_data["age"] < 0:
            return False

        if (
            not isinstance(user_data["created_at"], (int, float))
            or user_data["created_at"] <= 0
        ):
            return False

        if not isinstance(user_data["is_active"], bool):
            return False

        return True

    @staticmethod
    def validate_transaction_data(transaction_data: Dict[str, Any]) -> bool:
        """Validate transaction data."""
        required_fields = [
            "id",
            "from_address",
            "to_address",
            "amount",
            "fee",
            "timestamp",
            "status",
        ]

        for field in required_fields:
            if field not in transaction_data:
                return False

        if (
            not isinstance(transaction_data["id"], str)
            or len(transaction_data["id"]) == 0
        ):
            return False

        if (
            not isinstance(transaction_data["from_address"], str)
            or len(transaction_data["from_address"]) == 0
        ):
            return False

        if (
            not isinstance(transaction_data["to_address"], str)
            or len(transaction_data["to_address"]) == 0
        ):
            return False

        if (
            not isinstance(transaction_data["amount"], (int, float))
            or transaction_data["amount"] <= 0
        ):
            return False

        if (
            not isinstance(transaction_data["fee"], (int, float))
            or transaction_data["fee"] < 0
        ):
            return False

        if (
            not isinstance(transaction_data["timestamp"], (int, float))
            or transaction_data["timestamp"] <= 0
        ):
            return False

        if transaction_data["status"] not in ["pending", "confirmed", "failed"]:
            return False

        return True

    @staticmethod
    def validate_block_data(block_data: Dict[str, Any]) -> bool:
        """Validate block data."""
        required_fields = [
            "id",
            "previous_hash",
            "merkle_root",
            "timestamp",
            "nonce",
            "difficulty",
            "transactions",
        ]

        for field in required_fields:
            if field not in block_data:
                return False

        if not isinstance(block_data["id"], str) or len(block_data["id"]) == 0:
            return False

        if (
            not isinstance(block_data["previous_hash"], str)
            or len(block_data["previous_hash"]) == 0
        ):
            return False

        if (
            not isinstance(block_data["merkle_root"], str)
            or len(block_data["merkle_root"]) == 0
        ):
            return False

        if (
            not isinstance(block_data["timestamp"], (int, float))
            or block_data["timestamp"] <= 0
        ):
            return False

        if not isinstance(block_data["nonce"], int) or block_data["nonce"] < 0:
            return False

        if (
            not isinstance(block_data["difficulty"], int)
            or block_data["difficulty"] <= 0
        ):
            return False

        if not isinstance(block_data["transactions"], list):
            return False

        return True

    @staticmethod
    def validate_wallet_data(wallet_data: Dict[str, Any]) -> bool:
        """Validate wallet data."""
        required_fields = [
            "address",
            "private_key",
            "public_key",
            "balance",
            "created_at",
            "last_activity",
        ]

        for field in required_fields:
            if field not in wallet_data:
                return False

        if (
            not isinstance(wallet_data["address"], str)
            or len(wallet_data["address"]) == 0
        ):
            return False

        if (
            not isinstance(wallet_data["private_key"], str)
            or len(wallet_data["private_key"]) == 0
        ):
            return False

        if (
            not isinstance(wallet_data["public_key"], str)
            or len(wallet_data["public_key"]) == 0
        ):
            return False

        if (
            not isinstance(wallet_data["balance"], (int, float))
            or wallet_data["balance"] < 0
        ):
            return False

        if (
            not isinstance(wallet_data["created_at"], (int, float))
            or wallet_data["created_at"] <= 0
        ):
            return False

        if (
            not isinstance(wallet_data["last_activity"], (int, float))
            or wallet_data["last_activity"] <= 0
        ):
            return False

        return True


class TestComparators:
    """Test comparison functions."""

    @staticmethod
    def compare_objects(obj1: Any, obj2: Any) -> Dict[str, Any]:
        """Compare two objects and return differences."""
        differences = {
            "equal": obj1 == obj2,
            "type_equal": type(obj1) == type(obj2),
            "differences": [],
        }

        if obj1 != obj2:
            if isinstance(obj1, dict) and isinstance(obj2, dict):
                differences["differences"] = TestComparators._compare_dicts(obj1, obj2)
            elif isinstance(obj1, list) and isinstance(obj2, list):
                differences["differences"] = TestComparators._compare_lists(obj1, obj2)
            else:
                differences["differences"] = [f"Values differ: {obj1} vs {obj2}"]

        return differences

    @staticmethod
    def _compare_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> List[str]:
        """Compare two dictionaries."""
        differences = []

        # Check for missing keys
        for key in dict1:
            if key not in dict2:
                differences.append(f"Key '{key}' missing in dict2")

        for key in dict2:
            if key not in dict1:
                differences.append(f"Key '{key}' missing in dict1")

        # Check for different values
        for key in dict1:
            if key in dict2 and dict1[key] != dict2[key]:
                differences.append(f"Key '{key}' differs: {dict1[key]} vs {dict2[key]}")

        return differences

    @staticmethod
    def _compare_lists(list1: List[Any], list2: List[Any]) -> List[str]:
        """Compare two lists."""
        differences = []

        if len(list1) != len(list2):
            differences.append(f"Length differs: {len(list1)} vs {len(list2)}")

        min_length = min(len(list1), len(list2))
        for i in range(min_length):
            if list1[i] != list2[i]:
                differences.append(f"Index {i} differs: {list1[i]} vs {list2[i]}")

        return differences

    @staticmethod
    def compare_with_tolerance(
        actual: float, expected: float, tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """Compare floats with tolerance."""
        difference = abs(actual - expected)
        return {
            "equal": difference <= tolerance,
            "actual": actual,
            "expected": expected,
            "difference": difference,
            "tolerance": tolerance,
        }

    @staticmethod
    def compare_strings_ignore_case(str1: str, str2: str) -> Dict[str, Any]:
        """Compare strings ignoring case."""
        return {
            "equal": str1.lower() == str2.lower(),
            "str1": str1,
            "str2": str2,
            "str1_lower": str1.lower(),
            "str2_lower": str2.lower(),
        }
