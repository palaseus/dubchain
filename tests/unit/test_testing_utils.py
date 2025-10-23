"""
Unit tests for testing utilities.
"""

import logging

logger = logging.getLogger(__name__)
import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.testing.utils import (
    TestComparators,
    TestDataGenerators,
    TestHelpers,
    TestUtils,
    TestValidators,
)


class TestTestUtils:
    """Test TestUtils class."""

    def test_generate_random_string_default(self):
        """Test random string generation with default parameters."""
        result = TestUtils.generate_random_string()
        assert isinstance(result, str)
        assert len(result) == 10

    def test_generate_random_string_custom_length(self):
        """Test random string generation with custom length."""
        result = TestUtils.generate_random_string(length=20)
        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_random_int_default(self):
        """Test random integer generation with default parameters."""
        result = TestUtils.generate_random_int()
        assert isinstance(result, int)
        assert 0 <= result <= 100

    def test_generate_random_float_default(self):
        """Test random float generation with default parameters."""
        result = TestUtils.generate_random_float()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_generate_random_bytes_default(self):
        """Test random bytes generation with default parameters."""
        result = TestUtils.generate_random_bytes()
        assert isinstance(result, bytes)
        assert len(result) == 10

    def test_generate_random_dict_default(self):
        """Test random dictionary generation with default parameters."""
        result = TestUtils.generate_random_dict()
        assert isinstance(result, dict)
        assert len(result) == 5

    def test_generate_random_list_default(self):
        """Test random list generation with default parameters."""
        result = TestUtils.generate_random_list()
        assert isinstance(result, list)
        assert len(result) == 5

    def test_generate_random_string_custom_charset(self):
        """Test random string generation with custom charset."""
        charset = "0123456789"
        result = TestUtils.generate_random_string(length=5, charset=charset)
        assert isinstance(result, str)
        assert len(result) == 5
        assert all(c in charset for c in result)

    def test_generate_random_int_custom_range(self):
        """Test random integer generation with custom range."""
        result = TestUtils.generate_random_int(min_val=10, max_val=20)
        assert isinstance(result, int)
        assert 10 <= result <= 20

    def test_generate_random_float_custom_range(self):
        """Test random float generation with custom range."""
        result = TestUtils.generate_random_float(min_val=5.0, max_val=10.0)
        assert isinstance(result, float)
        assert 5.0 <= result <= 10.0

    def test_generate_random_bytes_custom_length(self):
        """Test random bytes generation with custom length."""
        result = TestUtils.generate_random_bytes(length=20)
        assert isinstance(result, bytes)
        assert len(result) == 20

    def test_generate_random_dict_custom_size(self):
        """Test random dictionary generation with custom size."""
        result = TestUtils.generate_random_dict(size=3)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_generate_random_list_custom_size(self):
        """Test random list generation with custom size."""
        result = TestUtils.generate_random_list(size=3)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_hash_data_string(self):
        """Test hashing string data."""
        data = "test string"
        result = TestUtils.hash_data(data)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest length

    def test_hash_data_bytes(self):
        """Test hashing bytes data."""
        data = b"test bytes"
        result = TestUtils.hash_data(data)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_dict(self):
        """Test hashing dictionary data."""
        data = {"key": "value", "number": 123}
        result = TestUtils.hash_data(data)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_list(self):
        """Test hashing list data."""
        data = [1, 2, 3, "test"]
        result = TestUtils.hash_data(data)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_wait_for_condition_success(self):
        """Test waiting for condition that becomes true."""
        condition_met = [False]
        
        def condition():
            return condition_met[0]
        
        def set_condition():
            time.sleep(0.1)
            condition_met[0] = True
        
        # Start a thread to set the condition after a short delay
        import threading
        thread = threading.Thread(target=set_condition)
        thread.start()
        
        result = TestUtils.wait_for_condition(condition, timeout=1.0, interval=0.05)
        assert result is True
        
        thread.join()

    def test_wait_for_condition_timeout(self):
        """Test waiting for condition that never becomes true."""
        def condition():
            return False
        
        result = TestUtils.wait_for_condition(condition, timeout=0.1, interval=0.01)
        assert result is False

    def test_retry_on_failure_success(self):
        """Test retry function that eventually succeeds."""
        attempts = [0]
        
        def failing_function():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not ready yet")
            return "success"
        
        result = TestUtils.retry_on_failure(failing_function, max_retries=3, delay=0.01)
        assert result == "success"
        assert attempts[0] == 3

    def test_retry_on_failure_max_retries(self):
        """Test retry function that fails after max retries."""
        def failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            TestUtils.retry_on_failure(failing_function, max_retries=2, delay=0.01)

    def test_measure_execution_time(self):
        """Test measuring function execution time."""
        def test_function():
            time.sleep(0.1)
            return "result"
        
        result, execution_time = TestUtils.measure_execution_time(test_function)
        assert result == "result"
        assert execution_time >= 0.1
        assert execution_time < 0.2  # Allow some tolerance

    def test_create_temp_file(self):
        """Test creating temporary file."""
        content = "test content"
        file_path = TestUtils.create_temp_file(content, ".test")
        
        try:
            assert os.path.exists(file_path)
            with open(file_path, 'r') as f:
                assert f.read() == content
        finally:
            TestUtils.cleanup_temp_file(file_path)

    def test_create_temp_file_default_content(self):
        """Test creating temporary file with default content."""
        file_path = TestUtils.create_temp_file()
        
        try:
            assert os.path.exists(file_path)
            with open(file_path, 'r') as f:
                assert f.read() == ""
        finally:
            TestUtils.cleanup_temp_file(file_path)

    def test_cleanup_temp_file(self):
        """Test cleaning up temporary file."""
        file_path = TestUtils.create_temp_file("test")
        assert os.path.exists(file_path)
        
        TestUtils.cleanup_temp_file(file_path)
        assert not os.path.exists(file_path)

    def test_cleanup_temp_file_nonexistent(self):
        """Test cleaning up nonexistent file."""
        # Should not raise exception
        TestUtils.cleanup_temp_file("nonexistent_file.txt")


class TestTestHelpers:
    """Test TestHelpers class."""

    def test_helpers_creation(self):
        """Test TestHelpers creation."""
        helpers = TestHelpers()
        assert helpers is not None

    def test_assert_dict_contains_success(self):
        """Test successful dict contains assertion."""
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"a": 1, "b": 2}
        
        # Should not raise exception
        TestHelpers.assert_dict_contains(dict1, dict2)

    def test_assert_dict_contains_missing_key(self):
        """Test dict contains assertion with missing key."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "c": 3}
        
        with pytest.raises(AssertionError, match="Key 'c' not found in dict1"):
            TestHelpers.assert_dict_contains(dict1, dict2)

    def test_assert_dict_contains_different_value(self):
        """Test dict contains assertion with different value."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 3}
        
        with pytest.raises(AssertionError, match="Value for key 'b' differs"):
            TestHelpers.assert_dict_contains(dict1, dict2)

    def test_assert_list_contains_success(self):
        """Test successful list contains assertion."""
        list1 = [1, 2, 3, 4, 5]
        list2 = [2, 4]
        
        # Should not raise exception
        TestHelpers.assert_list_contains(list1, list2)

    def test_assert_list_contains_missing_item(self):
        """Test list contains assertion with missing item."""
        list1 = [1, 2, 3]
        list2 = [2, 4]
        
        with pytest.raises(AssertionError, match="Item 4 not found in list1"):
            TestHelpers.assert_list_contains(list1, list2)

    def test_assert_approximately_equal_success(self):
        """Test successful approximately equal assertion."""
        # Should not raise exception
        TestHelpers.assert_approximately_equal(1.0, 1.0000001, tolerance=1e-6)

    def test_assert_approximately_equal_failure(self):
        """Test approximately equal assertion failure."""
        with pytest.raises(AssertionError, match="Values not approximately equal"):
            TestHelpers.assert_approximately_equal(1.0, 2.0, tolerance=1e-6)

    def test_assert_string_contains_success(self):
        """Test successful string contains assertion."""
        text = "Hello, world!"
        substring = "world"
        
        # Should not raise exception
        TestHelpers.assert_string_contains(text, substring)

    def test_assert_string_contains_failure(self):
        """Test string contains assertion failure."""
        text = "Hello, world!"
        substring = "universe"
        
        with pytest.raises(AssertionError, match="Substring 'universe' not found in text"):
            TestHelpers.assert_string_contains(text, substring)

    def test_assert_regex_match_success(self):
        """Test successful regex match assertion."""
        text = "Hello123"
        pattern = r"^[A-Za-z]+\d+$"
        
        # Should not raise exception
        TestHelpers.assert_regex_match(text, pattern)

    def test_assert_regex_match_failure(self):
        """Test regex match assertion failure."""
        text = "Hello123"
        pattern = r"^[A-Za-z]+$"
        
        with pytest.raises(AssertionError, match="Text 'Hello123' does not match pattern"):
            TestHelpers.assert_regex_match(text, pattern)

    def test_assert_file_exists_success(self):
        """Test successful file exists assertion."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            # Should not raise exception
            TestHelpers.assert_file_exists(temp_file)
        finally:
            os.unlink(temp_file)

    def test_assert_file_exists_failure(self):
        """Test file exists assertion failure."""
        with pytest.raises(AssertionError, match="File 'nonexistent.txt' does not exist"):
            TestHelpers.assert_file_exists("nonexistent.txt")

    def test_assert_file_not_exists_success(self):
        """Test successful file not exists assertion."""
        # Should not raise exception
        TestHelpers.assert_file_not_exists("nonexistent.txt")

    def test_assert_file_not_exists_failure(self):
        """Test file not exists assertion failure."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(AssertionError, match="File .* exists"):
                TestHelpers.assert_file_not_exists(temp_file)
        finally:
            os.unlink(temp_file)

    def test_assert_directory_exists_success(self):
        """Test successful directory exists assertion."""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Should not raise exception
            TestHelpers.assert_directory_exists(temp_dir)
        finally:
            os.rmdir(temp_dir)

    def test_assert_directory_exists_failure_nonexistent(self):
        """Test directory exists assertion failure for nonexistent directory."""
        with pytest.raises(AssertionError, match="Directory 'nonexistent' does not exist"):
            TestHelpers.assert_directory_exists("nonexistent")

    def test_assert_directory_exists_failure_not_directory(self):
        """Test directory exists assertion failure for file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(AssertionError, match="Path .* is not a directory"):
                TestHelpers.assert_directory_exists(temp_file)
        finally:
            os.unlink(temp_file)


class TestTestDataGenerators:
    """Test TestDataGenerators class."""

    def test_data_generators_creation(self):
        """Test TestDataGenerators creation."""
        generators = TestDataGenerators()
        assert generators is not None

    def test_generate_user_data_default(self):
        """Test generating user data with default parameters."""
        user_data = TestDataGenerators.generate_user_data()
        
        assert isinstance(user_data, dict)
        assert "id" in user_data
        assert "name" in user_data
        assert "email" in user_data
        assert "age" in user_data
        assert "created_at" in user_data
        assert "is_active" in user_data
        
        assert isinstance(user_data["id"], str)
        assert len(user_data["id"]) == 8
        assert isinstance(user_data["name"], str)
        assert len(user_data["name"]) == 10
        assert "@example.com" in user_data["email"]
        assert 18 <= user_data["age"] <= 65
        assert isinstance(user_data["created_at"], float)
        assert isinstance(user_data["is_active"], bool)

    def test_generate_user_data_custom_id(self):
        """Test generating user data with custom ID."""
        custom_id = "custom123"
        user_data = TestDataGenerators.generate_user_data(user_id=custom_id)
        
        assert user_data["id"] == custom_id

    def test_generate_transaction_data_default(self):
        """Test generating transaction data with default parameters."""
        transaction_data = TestDataGenerators.generate_transaction_data()
        
        assert isinstance(transaction_data, dict)
        assert "id" in transaction_data
        assert "from_address" in transaction_data
        assert "to_address" in transaction_data
        assert "amount" in transaction_data
        assert "fee" in transaction_data
        assert "timestamp" in transaction_data
        assert "status" in transaction_data
        
        assert isinstance(transaction_data["id"], str)
        assert len(transaction_data["id"]) == 8
        assert isinstance(transaction_data["from_address"], str)
        assert len(transaction_data["from_address"]) == 40
        assert isinstance(transaction_data["to_address"], str)
        assert len(transaction_data["to_address"]) == 40
        assert 0.1 <= transaction_data["amount"] <= 1000.0
        assert 0.01 <= transaction_data["fee"] <= 10.0
        assert isinstance(transaction_data["timestamp"], float)
        assert transaction_data["status"] in ["pending", "confirmed", "failed"]

    def test_generate_transaction_data_custom_id(self):
        """Test generating transaction data with custom ID."""
        custom_id = "tx123"
        transaction_data = TestDataGenerators.generate_transaction_data(transaction_id=custom_id)
        
        assert transaction_data["id"] == custom_id

    def test_generate_block_data_default(self):
        """Test generating block data with default parameters."""
        block_data = TestDataGenerators.generate_block_data()
        
        assert isinstance(block_data, dict)
        assert "id" in block_data
        assert "previous_hash" in block_data
        assert "merkle_root" in block_data
        assert "timestamp" in block_data
        assert "nonce" in block_data
        assert "difficulty" in block_data
        assert "transactions" in block_data
        
        assert isinstance(block_data["id"], str)
        assert len(block_data["id"]) == 8
        assert isinstance(block_data["previous_hash"], str)
        assert len(block_data["previous_hash"]) == 64
        assert isinstance(block_data["merkle_root"], str)
        assert len(block_data["merkle_root"]) == 64
        assert isinstance(block_data["timestamp"], float)
        assert 0 <= block_data["nonce"] <= 1000000
        assert 1 <= block_data["difficulty"] <= 10
        assert isinstance(block_data["transactions"], list)
        assert 1 <= len(block_data["transactions"]) <= 10

    def test_generate_block_data_custom_id(self):
        """Test generating block data with custom ID."""
        custom_id = "block123"
        block_data = TestDataGenerators.generate_block_data(block_id=custom_id)
        
        assert block_data["id"] == custom_id

    def test_generate_wallet_data_default(self):
        """Test generating wallet data with default parameters."""
        wallet_data = TestDataGenerators.generate_wallet_data()
        
        assert isinstance(wallet_data, dict)
        assert "address" in wallet_data
        assert "private_key" in wallet_data
        assert "public_key" in wallet_data
        assert "balance" in wallet_data
        assert "created_at" in wallet_data
        assert "last_activity" in wallet_data
        
        assert isinstance(wallet_data["address"], str)
        assert len(wallet_data["address"]) == 40
        assert isinstance(wallet_data["private_key"], str)
        assert len(wallet_data["private_key"]) == 64
        assert isinstance(wallet_data["public_key"], str)
        assert len(wallet_data["public_key"]) == 64
        assert 0.0 <= wallet_data["balance"] <= 10000.0
        assert isinstance(wallet_data["created_at"], float)
        assert isinstance(wallet_data["last_activity"], float)

    def test_generate_wallet_data_custom_address(self):
        """Test generating wallet data with custom address."""
        custom_address = "custom_address_123"
        wallet_data = TestDataGenerators.generate_wallet_data(wallet_address=custom_address)
        
        assert wallet_data["address"] == custom_address

    def test_generate_smart_contract_data_default(self):
        """Test generating smart contract data with default parameters."""
        contract_data = TestDataGenerators.generate_smart_contract_data()
        
        assert isinstance(contract_data, dict)
        assert "id" in contract_data
        assert "name" in contract_data
        assert "bytecode" in contract_data
        assert "abi" in contract_data
        assert "deployed_address" in contract_data
        assert "creator" in contract_data
        assert "created_at" in contract_data
        
        assert isinstance(contract_data["id"], str)
        assert len(contract_data["id"]) == 8
        assert isinstance(contract_data["name"], str)
        assert len(contract_data["name"]) == 10
        assert isinstance(contract_data["bytecode"], str)
        assert len(contract_data["bytecode"]) == 100
        assert isinstance(contract_data["abi"], list)
        assert len(contract_data["abi"]) == 1
        assert isinstance(contract_data["deployed_address"], str)
        assert len(contract_data["deployed_address"]) == 40
        assert isinstance(contract_data["creator"], str)
        assert len(contract_data["creator"]) == 40
        assert isinstance(contract_data["created_at"], float)

    def test_generate_smart_contract_data_custom_id(self):
        """Test generating smart contract data with custom ID."""
        custom_id = "contract123"
        contract_data = TestDataGenerators.generate_smart_contract_data(contract_id=custom_id)
        
        assert contract_data["id"] == custom_id

    def test_generate_network_message_data_default(self):
        """Test generating network message data with default parameters."""
        message_data = TestDataGenerators.generate_network_message_data()
        
        assert isinstance(message_data, dict)
        assert "id" in message_data
        assert "type" in message_data
        assert "sender" in message_data
        assert "receiver" in message_data
        assert "payload" in message_data
        assert "timestamp" in message_data
        assert "ttl" in message_data
        
        assert isinstance(message_data["id"], str)
        assert len(message_data["id"]) == 8
        assert message_data["type"] in ["ping", "pong", "block", "transaction", "request", "response"]
        assert isinstance(message_data["sender"], str)
        assert len(message_data["sender"]) == 40
        assert isinstance(message_data["receiver"], str)
        assert len(message_data["receiver"]) == 40
        assert isinstance(message_data["payload"], dict)
        assert len(message_data["payload"]) == 5
        assert isinstance(message_data["timestamp"], float)
        assert 1 <= message_data["ttl"] <= 3600

    def test_generate_network_message_data_custom_id(self):
        """Test generating network message data with custom ID."""
        custom_id = "msg123"
        message_data = TestDataGenerators.generate_network_message_data(message_id=custom_id)
        
        assert message_data["id"] == custom_id


class TestTestValidators:
    """Test TestValidators class."""

    def test_validators_creation(self):
        """Test TestValidators creation."""
        validators = TestValidators()
        assert validators is not None

    def test_validate_user_data_valid(self):
        """Test validating valid user data."""
        user_data = {
            "id": "user123",
            "name": "John Doe",
            "email": "john@example.com",
            "age": 25,
            "created_at": 1234567890.0,
            "is_active": True
        }
        
        assert TestValidators.validate_user_data(user_data) is True

    def test_validate_user_data_missing_field(self):
        """Test validating user data with missing field."""
        user_data = {
            "id": "user123",
            "name": "John Doe",
            "email": "john@example.com",
            "age": 25,
            "created_at": 1234567890.0
            # Missing "is_active"
        }
        
        assert TestValidators.validate_user_data(user_data) is False

    def test_validate_user_data_invalid_email(self):
        """Test validating user data with invalid email."""
        user_data = {
            "id": "user123",
            "name": "John Doe",
            "email": "invalid-email",  # No @ symbol
            "age": 25,
            "created_at": 1234567890.0,
            "is_active": True
        }
        
        assert TestValidators.validate_user_data(user_data) is False

    def test_validate_user_data_negative_age(self):
        """Test validating user data with negative age."""
        user_data = {
            "id": "user123",
            "name": "John Doe",
            "email": "john@example.com",
            "age": -5,  # Negative age
            "created_at": 1234567890.0,
            "is_active": True
        }
        
        assert TestValidators.validate_user_data(user_data) is False

    def test_validate_transaction_data_valid(self):
        """Test validating valid transaction data."""
        transaction_data = {
            "id": "tx123",
            "from_address": "0x1234567890abcdef1234567890abcdef12345678",
            "to_address": "0xabcdef1234567890abcdef1234567890abcdef12",
            "amount": 100.0,
            "fee": 0.1,
            "timestamp": 1234567890.0,
            "status": "confirmed"
        }
        
        assert TestValidators.validate_transaction_data(transaction_data) is True

    def test_validate_transaction_data_invalid_status(self):
        """Test validating transaction data with invalid status."""
        transaction_data = {
            "id": "tx123",
            "from_address": "0x1234567890abcdef1234567890abcdef12345678",
            "to_address": "0xabcdef1234567890abcdef1234567890abcdef12",
            "amount": 100.0,
            "fee": 0.1,
            "timestamp": 1234567890.0,
            "status": "invalid_status"  # Invalid status
        }
        
        assert TestValidators.validate_transaction_data(transaction_data) is False

    def test_validate_transaction_data_negative_amount(self):
        """Test validating transaction data with negative amount."""
        transaction_data = {
            "id": "tx123",
            "from_address": "0x1234567890abcdef1234567890abcdef12345678",
            "to_address": "0xabcdef1234567890abcdef1234567890abcdef12",
            "amount": -100.0,  # Negative amount
            "fee": 0.1,
            "timestamp": 1234567890.0,
            "status": "confirmed"
        }
        
        assert TestValidators.validate_transaction_data(transaction_data) is False

    def test_validate_block_data_valid(self):
        """Test validating valid block data."""
        block_data = {
            "id": "block123",
            "previous_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "merkle_root": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "timestamp": 1234567890.0,
            "nonce": 12345,
            "difficulty": 5,
            "transactions": ["tx1", "tx2", "tx3"]
        }
        
        assert TestValidators.validate_block_data(block_data) is True

    def test_validate_block_data_invalid_difficulty(self):
        """Test validating block data with invalid difficulty."""
        block_data = {
            "id": "block123",
            "previous_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "merkle_root": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "timestamp": 1234567890.0,
            "nonce": 12345,
            "difficulty": 0,  # Invalid difficulty (must be > 0)
            "transactions": ["tx1", "tx2", "tx3"]
        }
        
        assert TestValidators.validate_block_data(block_data) is False

    def test_validate_wallet_data_valid(self):
        """Test validating valid wallet data."""
        wallet_data = {
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "private_key": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "public_key": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "balance": 1000.0,
            "created_at": 1234567890.0,
            "last_activity": 1234567890.0
        }
        
        assert TestValidators.validate_wallet_data(wallet_data) is True

    def test_validate_wallet_data_negative_balance(self):
        """Test validating wallet data with negative balance."""
        wallet_data = {
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "private_key": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "public_key": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "balance": -1000.0,  # Negative balance
            "created_at": 1234567890.0,
            "last_activity": 1234567890.0
        }
        
        assert TestValidators.validate_wallet_data(wallet_data) is False


class TestTestComparators:
    """Test TestComparators class."""

    def test_comparators_creation(self):
        """Test TestComparators creation."""
        comparators = TestComparators()
        assert comparators is not None

    def test_compare_objects_equal(self):
        """Test comparing equal objects."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"a": 1, "b": 2}
        
        result = TestComparators.compare_objects(obj1, obj2)
        assert result["equal"] is True
        assert result["type_equal"] is True
        assert result["differences"] == []

    def test_compare_objects_different(self):
        """Test comparing different objects."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"a": 1, "b": 3}
        
        result = TestComparators.compare_objects(obj1, obj2)
        assert result["equal"] is False
        assert result["type_equal"] is True
        assert len(result["differences"]) > 0

    def test_compare_objects_different_types(self):
        """Test comparing objects of different types."""
        obj1 = {"a": 1}
        obj2 = [1, 2, 3]
        
        result = TestComparators.compare_objects(obj1, obj2)
        assert result["equal"] is False
        assert result["type_equal"] is False

    def test_compare_dicts_equal(self):
        """Test comparing equal dictionaries."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 2}
        
        differences = TestComparators._compare_dicts(dict1, dict2)
        assert differences == []

    def test_compare_dicts_missing_keys(self):
        """Test comparing dictionaries with missing keys."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "c": 3}
        
        differences = TestComparators._compare_dicts(dict1, dict2)
        assert "Key 'b' missing in dict2" in differences
        assert "Key 'c' missing in dict1" in differences

    def test_compare_dicts_different_values(self):
        """Test comparing dictionaries with different values."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 3}
        
        differences = TestComparators._compare_dicts(dict1, dict2)
        assert "Key 'b' differs: 2 vs 3" in differences

    def test_compare_lists_equal(self):
        """Test comparing equal lists."""
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]
        
        differences = TestComparators._compare_lists(list1, list2)
        assert differences == []

    def test_compare_lists_different_lengths(self):
        """Test comparing lists with different lengths."""
        list1 = [1, 2, 3]
        list2 = [1, 2]
        
        differences = TestComparators._compare_lists(list1, list2)
        assert "Length differs: 3 vs 2" in differences

    def test_compare_lists_different_values(self):
        """Test comparing lists with different values."""
        list1 = [1, 2, 3]
        list2 = [1, 4, 3]
        
        differences = TestComparators._compare_lists(list1, list2)
        assert "Index 1 differs: 2 vs 4" in differences

    def test_compare_with_tolerance_equal(self):
        """Test comparing floats with tolerance - equal case."""
        result = TestComparators.compare_with_tolerance(1.0, 1.0000001, tolerance=1e-6)
        assert result["equal"] is True
        assert result["actual"] == 1.0
        assert result["expected"] == 1.0000001
        assert result["difference"] <= 1e-6
        assert result["tolerance"] == 1e-6

    def test_compare_with_tolerance_different(self):
        """Test comparing floats with tolerance - different case."""
        result = TestComparators.compare_with_tolerance(1.0, 2.0, tolerance=1e-6)
        assert result["equal"] is False
        assert result["actual"] == 1.0
        assert result["expected"] == 2.0
        assert result["difference"] == 1.0
        assert result["tolerance"] == 1e-6

    def test_compare_strings_ignore_case_equal(self):
        """Test comparing strings ignoring case - equal case."""
        result = TestComparators.compare_strings_ignore_case("Hello", "HELLO")
        assert result["equal"] is True
        assert result["str1"] == "Hello"
        assert result["str2"] == "HELLO"
        assert result["str1_lower"] == "hello"
        assert result["str2_lower"] == "hello"

    def test_compare_strings_ignore_case_different(self):
        """Test comparing strings ignoring case - different case."""
        result = TestComparators.compare_strings_ignore_case("Hello", "World")
        assert result["equal"] is False
        assert result["str1"] == "Hello"
        assert result["str2"] == "World"
        assert result["str1_lower"] == "hello"
        assert result["str2_lower"] == "world"
