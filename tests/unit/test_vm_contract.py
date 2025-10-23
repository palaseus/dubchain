"""
Unit tests for VM contract functionality.
"""

import logging

logger = logging.getLogger(__name__)
import time

import pytest

from dubchain.crypto.hashing import Hash, SHA256Hasher
from dubchain.vm.contract import (
    ContractEvent,
    ContractMemory,
    ContractState,
    ContractStorage,
    ContractType,
    SmartContract,
)


class TestContractStorage:
    """Test ContractStorage class."""

    def test_storage_creation(self):
        """Test storage creation."""
        storage = ContractStorage()

        assert len(storage.storage) == 0
        assert len(storage.storage_keys) == 0

    def test_storage_get_default(self):
        """Test getting default storage value."""
        storage = ContractStorage()
        key = Hash(SHA256Hasher.hash(b"test_key").value)

        value = storage.get(key)
        assert value == b"\x00" * 32

    def test_storage_set_get(self):
        """Test setting and getting storage values."""
        storage = ContractStorage()
        key = Hash(SHA256Hasher.hash(b"test_key").value)
        value = b"test_value" + b"\x00" * 22  # 32 bytes total

        storage.set(key, value)
        assert storage.get(key) == value
        assert key in storage.storage_keys

    def test_storage_set_invalid_size(self):
        """Test setting storage value with invalid size."""
        storage = ContractStorage()
        key = Hash(SHA256Hasher.hash(b"test_key").value)
        value = b"short"  # Not 32 bytes

        with pytest.raises(ValueError, match="Storage values must be 32 bytes"):
            storage.set(key, value)

    def test_storage_delete(self):
        """Test deleting storage values."""
        storage = ContractStorage()
        key = Hash(SHA256Hasher.hash(b"test_key").value)
        value = b"test_value" + b"\x00" * 22  # 32 bytes total

        storage.set(key, value)
        assert key in storage.storage_keys

        storage.delete(key)
        assert key not in storage.storage_keys
        assert storage.get(key) == b"\x00" * 32

    def test_storage_clear(self):
        """Test clearing storage."""
        storage = ContractStorage()
        key1 = Hash(SHA256Hasher.hash(b"key1").value)
        key2 = Hash(SHA256Hasher.hash(b"key2").value)
        value = b"test_value" + b"\x00" * 22  # 32 bytes total

        storage.set(key1, value)
        storage.set(key2, value)

        storage.clear()
        assert len(storage.storage) == 0
        assert len(storage.storage_keys) == 0

    def test_storage_get_all_keys(self):
        """Test getting all storage keys."""
        storage = ContractStorage()
        key1 = Hash(SHA256Hasher.hash(b"key1").value)
        key2 = Hash(SHA256Hasher.hash(b"key2").value)
        value = b"test_value" + b"\x00" * 22  # 32 bytes total

        storage.set(key1, value)
        storage.set(key2, value)

        keys = storage.get_all_keys()
        assert len(keys) == 2
        assert key1 in keys
        assert key2 in keys

    def test_storage_size(self):
        """Test getting storage size."""
        storage = ContractStorage()
        key1 = Hash(SHA256Hasher.hash(b"key1").value)
        key2 = Hash(SHA256Hasher.hash(b"key2").value)
        value = b"test_value" + b"\x00" * 22  # 32 bytes total

        assert storage.get_storage_size() == 0

        storage.set(key1, value)
        assert storage.get_storage_size() == 1

        storage.set(key2, value)
        assert storage.get_storage_size() == 2

        storage.delete(key1)
        assert storage.get_storage_size() == 1


class TestContractMemory:
    """Test ContractMemory class."""

    def test_memory_creation(self):
        """Test memory creation."""
        memory = ContractMemory()

        assert len(memory.memory) == 0
        assert memory.max_size == 2**32

    def test_memory_creation_custom_size(self):
        """Test memory creation with custom size."""
        memory = ContractMemory(max_size=1024)

        assert memory.max_size == 1024

    def test_memory_creation_invalid_size(self):
        """Test memory creation with invalid size."""
        with pytest.raises(ValueError, match="Max memory size must be positive"):
            ContractMemory(max_size=0)

    def test_memory_get_default(self):
        """Test getting default memory value."""
        memory = ContractMemory()

        value = memory.get(0, 32)
        assert value == b"\x00" * 32

    def test_memory_set_get(self):
        """Test setting and getting memory values."""
        memory = ContractMemory()
        data = b"test_data"

        memory.set(0, data)
        assert memory.get(0, len(data)) == data

    def test_memory_set_get_offset(self):
        """Test setting and getting memory values with offset."""
        memory = ContractMemory()
        data = b"test_data"

        memory.set(100, data)
        assert memory.get(100, len(data)) == data

    def test_memory_set_invalid_offset(self):
        """Test setting memory with invalid offset."""
        memory = ContractMemory()
        data = b"test_data"

        with pytest.raises(ValueError, match="Offset must be non-negative"):
            memory.set(-1, data)

    def test_memory_set_exceeds_max_size(self):
        """Test setting memory that exceeds max size."""
        memory = ContractMemory(max_size=100)
        data = b"x" * 101

        with pytest.raises(ValueError, match="Memory access exceeds maximum size"):
            memory.set(0, data)

    def test_memory_get_invalid_offset(self):
        """Test getting memory with invalid offset."""
        memory = ContractMemory()

        with pytest.raises(ValueError, match="Offset and size must be non-negative"):
            memory.get(-1, 10)

    def test_memory_get_invalid_size(self):
        """Test getting memory with invalid size."""
        memory = ContractMemory()

        with pytest.raises(ValueError, match="Offset and size must be non-negative"):
            memory.get(0, -1)

    def test_memory_get_size(self):
        """Test getting memory size."""
        memory = ContractMemory()

        assert memory.get_size() == 0

        memory.set(0, b"test")
        assert memory.get_size() == 4

        memory.set(100, b"data")
        assert memory.get_size() == 104

    def test_memory_clear(self):
        """Test clearing memory."""
        memory = ContractMemory()

        memory.set(0, b"test")
        memory.set(100, b"data")

        memory.clear()
        assert memory.get_size() == 0


class TestContractEvent:
    """Test ContractEvent class."""

    def test_event_creation(self):
        """Test event creation."""
        topics = [
            Hash(SHA256Hasher.hash(b"topic1").value),
            Hash(SHA256Hasher.hash(b"topic2").value),
        ]
        data = b"event_data"
        block_number = 123
        transaction_hash = Hash(SHA256Hasher.hash(b"tx_hash").value)

        event = ContractEvent(
            address="0x123",
            topics=topics,
            data=data,
            block_number=block_number,
            transaction_hash=transaction_hash,
            log_index=0,
        )

        assert event.address == "0x123"
        assert event.topics == topics
        assert event.data == data
        assert event.block_number == block_number
        assert event.transaction_hash == transaction_hash
        assert event.log_index == 0
        assert not event.removed

    def test_event_creation_too_many_topics(self):
        """Test event creation with too many topics."""
        topics = [Hash(SHA256Hasher.hash(f"topic{i}".encode()).value) for i in range(5)]

        with pytest.raises(ValueError, match="Maximum 4 topics allowed per event"):
            ContractEvent(
                address="0x123",
                topics=topics,
                data=b"data",
                block_number=123,
                transaction_hash=Hash(SHA256Hasher.hash(b"tx_hash").value),
                log_index=0,
            )

    def test_event_to_dict(self):
        """Test event to dictionary conversion."""
        topics = [
            Hash(SHA256Hasher.hash(b"topic1").value),
            Hash(SHA256Hasher.hash(b"topic2").value),
        ]
        data = b"event_data"
        block_number = 123
        transaction_hash = Hash(SHA256Hasher.hash(b"tx_hash").value)

        event = ContractEvent(
            address="0x123",
            topics=topics,
            data=data,
            block_number=block_number,
            transaction_hash=transaction_hash,
            log_index=0,
        )

        event_dict = event.to_dict()

        assert event_dict["address"] == "0x123"
        assert len(event_dict["topics"]) == 2
        assert event_dict["data"] == data.hex()
        assert event_dict["block_number"] == 123
        assert event_dict["transaction_hash"] == transaction_hash.to_hex()
        assert event_dict["log_index"] == 0
        assert not event_dict["removed"]

    def test_event_matches_filter_address(self):
        """Test event matching with address filter."""
        event = ContractEvent(
            address="0x123",
            topics=[Hash(SHA256Hasher.hash(b"topic1").value)],
            data=b"data",
            block_number=123,
            transaction_hash=Hash(SHA256Hasher.hash(b"tx_hash").value),
            log_index=0,
        )

        assert event.matches_filter(address_filter="0x123")
        assert not event.matches_filter(address_filter="0x456")

    def test_event_matches_filter_topics(self):
        """Test event matching with topics filter."""
        topic1 = Hash(SHA256Hasher.hash(b"topic1").value)
        topic2 = Hash(SHA256Hasher.hash(b"topic2").value)

        event = ContractEvent(
            address="0x123",
            topics=[topic1, topic2],
            data=b"data",
            block_number=123,
            transaction_hash=Hash(SHA256Hasher.hash(b"tx_hash").value),
            log_index=0,
        )

        # Match all topics
        assert event.matches_filter(topics_filter=[topic1, topic2])

        # Match first topic only
        assert event.matches_filter(topics_filter=[topic1, None])

        # No match
        assert not event.matches_filter(
            topics_filter=[Hash(SHA256Hasher.hash(b"different").value)]
        )

        # Too many topics in filter
        assert not event.matches_filter(
            topics_filter=[topic1, topic2, Hash(SHA256Hasher.hash(b"extra").value)]
        )

    def test_event_matches_filter_combined(self):
        """Test event matching with combined filters."""
        topic1 = Hash(SHA256Hasher.hash(b"topic1").value)

        event = ContractEvent(
            address="0x123",
            topics=[topic1],
            data=b"data",
            block_number=123,
            transaction_hash=Hash(SHA256Hasher.hash(b"tx_hash").value),
            log_index=0,
        )

        # Both filters match
        assert event.matches_filter(address_filter="0x123", topics_filter=[topic1])

        # Address doesn't match
        assert not event.matches_filter(address_filter="0x456", topics_filter=[topic1])

        # Topic doesn't match
        assert not event.matches_filter(
            address_filter="0x123",
            topics_filter=[Hash(SHA256Hasher.hash(b"different").value)],
        )


class TestSmartContract:
    """Test SmartContract class."""

    def test_contract_creation(self):
        """Test contract creation."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        assert contract.address == "0x123"
        assert contract.bytecode == b"contract_bytecode"
        assert contract.contract_type == ContractType.STANDARD
        assert contract.state == ContractState.ACTIVE
        assert contract.balance == 0

    def test_contract_creation_empty_address(self):
        """Test contract creation with empty address."""
        with pytest.raises(ValueError, match="Contract address cannot be empty"):
            SmartContract(address="", bytecode=b"bytecode")

    def test_contract_creation_empty_bytecode(self):
        """Test contract creation with empty bytecode."""
        with pytest.raises(ValueError, match="Contract bytecode cannot be empty"):
            SmartContract(address="0x123", bytecode=b"")

    def test_contract_deploy(self):
        """Test contract deployment."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        creator = "0xcreator"
        creation_block = 100
        creation_transaction = Hash(SHA256Hasher.hash(b"tx_hash").value)
        initial_balance = 1000

        contract.deploy(creator, creation_block, creation_transaction, initial_balance)

        assert contract.creator == creator
        assert contract.creation_block == creation_block
        assert contract.creation_transaction == creation_transaction
        assert contract.balance == initial_balance
        assert contract.state == ContractState.ACTIVE
        assert contract.creation_time > 0

    def test_contract_execute_success(self):
        """Test successful contract execution."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        result = contract.execute(
            caller="0xcaller",
            value=100,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        assert result["success"] is True
        assert result["gas_used"] == 0
        assert result["return_data"] == b""
        assert result["events"] == []
        assert result["storage_changes"] == {}

    def test_contract_execute_not_active(self):
        """Test contract execution when not active."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")
        contract.state = ContractState.PAUSED

        with pytest.raises(RuntimeError, match="Contract is not active"):
            contract.execute(
                caller="0xcaller",
                value=100,
                data=b"call_data",
                gas_limit=100000,
                block_context={},
            )

    def test_contract_emit_event(self):
        """Test emitting contract event."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        topics = [
            Hash(SHA256Hasher.hash(b"topic1").value),
            Hash(SHA256Hasher.hash(b"topic2").value),
        ]
        data = b"event_data"
        block_number = 123
        transaction_hash = Hash(SHA256Hasher.hash(b"tx_hash").value)

        contract.emit_event(topics, data, block_number, transaction_hash)

        assert len(contract.events) == 1
        event = contract.events[0]
        assert event.address == "0x123"
        assert event.topics == topics
        assert event.data == data
        assert event.block_number == block_number
        assert event.transaction_hash == transaction_hash
        assert event.log_index == 0

    def test_contract_emit_event_too_many_topics(self):
        """Test emitting event with too many topics."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        topics = [Hash(SHA256Hasher.hash(f"topic{i}".encode()).value) for i in range(5)]

        with pytest.raises(ValueError, match="Maximum 4 topics allowed per event"):
            contract.emit_event(
                topics, b"data", 123, Hash(SHA256Hasher.hash(b"tx_hash").value)
            )

    def test_contract_get_events(self):
        """Test getting contract events."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        # Add some events
        for i in range(5):
            contract.emit_event(
                [Hash(SHA256Hasher.hash(f"topic{i}".encode()).value)],
                f"data{i}".encode(),
                i + 100,
                Hash(SHA256Hasher.hash(f"tx{i}".encode()).value),
            )

        # Get all events
        all_events = contract.get_events()
        assert len(all_events) == 5

        # Get events with block range
        filtered_events = contract.get_events(from_block=102, to_block=104)
        assert len(filtered_events) == 3

        # Get events with address filter
        filtered_events = contract.get_events(address_filter="0x123")
        assert len(filtered_events) == 5

        filtered_events = contract.get_events(address_filter="0x456")
        assert len(filtered_events) == 0

    def test_contract_storage_operations(self):
        """Test contract storage operations."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        key = Hash(SHA256Hasher.hash(b"storage_key").value)
        value = b"storage_value" + b"\x00" * 19  # 32 bytes total

        # Set storage value
        contract.set_storage_value(key, value)
        assert contract.get_storage_value(key) == value

        # Delete storage value
        contract.delete_storage_value(key)
        assert contract.get_storage_value(key) == b"\x00" * 32

    def test_contract_balance_operations(self):
        """Test contract balance operations."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        assert contract.get_balance() == 0

        # Add balance
        contract.add_balance(1000)
        assert contract.get_balance() == 1000

        # Subtract balance
        assert contract.subtract_balance(300)
        assert contract.get_balance() == 700

        # Try to subtract more than available
        assert not contract.subtract_balance(800)
        assert contract.get_balance() == 700

    def test_contract_balance_operations_invalid(self):
        """Test contract balance operations with invalid amounts."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        with pytest.raises(ValueError, match="Amount must be non-negative"):
            contract.add_balance(-100)

        with pytest.raises(ValueError, match="Amount must be non-negative"):
            contract.subtract_balance(-100)

    def test_contract_pause_unpause(self):
        """Test contract pause and unpause."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        assert contract.state == ContractState.ACTIVE

        # Pause contract
        contract.pause()
        assert contract.state == ContractState.PAUSED

        # Unpause contract
        contract.unpause()
        assert contract.state == ContractState.ACTIVE

    def test_contract_pause_invalid_state(self):
        """Test pausing contract in invalid state."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")
        contract.state = ContractState.DESTROYED

        with pytest.raises(
            RuntimeError, match="Cannot pause contract in state: destroyed"
        ):
            contract.pause()

    def test_contract_unpause_invalid_state(self):
        """Test unpausing contract in invalid state."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")
        contract.state = ContractState.ACTIVE

        with pytest.raises(
            RuntimeError, match="Cannot unpause contract in state: active"
        ):
            contract.unpause()

    def test_contract_destroy(self):
        """Test contract destruction."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        # Set some state
        contract.add_balance(1000)
        contract.set_storage_value(
            Hash(SHA256Hasher.hash(b"key").value), b"value" + b"\x00" * 27
        )

        # Destroy contract
        contract.destroy()

        assert contract.state == ContractState.DESTROYED
        assert contract.balance == 0
        assert contract.storage.get_storage_size() == 0
        assert contract.memory.get_size() == 0

    def test_contract_upgrade(self):
        """Test contract upgrade."""
        contract = SmartContract(address="0x123", bytecode=b"old_bytecode")

        new_bytecode = b"new_bytecode"
        contract.upgrade(new_bytecode)

        assert contract.bytecode == new_bytecode
        assert contract.state == ContractState.ACTIVE

    def test_contract_upgrade_invalid_state(self):
        """Test upgrading contract in invalid state."""
        contract = SmartContract(address="0x123", bytecode=b"old_bytecode")
        contract.state = ContractState.DESTROYED

        with pytest.raises(
            RuntimeError, match="Cannot upgrade contract in state: destroyed"
        ):
            contract.upgrade(b"new_bytecode")

    def test_contract_metadata(self):
        """Test getting contract metadata."""
        contract = SmartContract(
            address="0x123",
            bytecode=b"contract_bytecode",
            name="TestContract",
            version="1.0.0",
            description="A test contract",
        )

        metadata = contract.get_metadata()

        assert metadata["address"] == "0x123"
        assert metadata["contract_type"] == "standard"
        assert metadata["state"] == "active"
        assert metadata["name"] == "TestContract"
        assert metadata["version"] == "1.0.0"
        assert metadata["description"] == "A test contract"
        assert metadata["balance"] == 0
        assert metadata["storage_size"] == 0
        assert metadata["memory_size"] == 0
        assert metadata["event_count"] == 0
        assert metadata["call_stack_depth"] == 0

    def test_contract_storage_dump(self):
        """Test getting storage dump."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        # Add some storage
        key1 = Hash(SHA256Hasher.hash(b"key1").value)
        key2 = Hash(SHA256Hasher.hash(b"key2").value)
        value = b"value" + b"\x00" * 27  # 32 bytes

        contract.set_storage_value(key1, value)
        contract.set_storage_value(key2, value)

        storage_dump = contract.get_storage_dump()

        assert len(storage_dump) == 2
        assert key1.to_hex() in storage_dump
        assert key2.to_hex() in storage_dump
        assert storage_dump[key1.to_hex()] == value.hex()

    def test_contract_event_log(self):
        """Test getting event log."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        # Add some events
        for i in range(3):
            contract.emit_event(
                [Hash(SHA256Hasher.hash(f"topic{i}".encode()).value)],
                f"data{i}".encode(),
                i + 100,
                Hash(SHA256Hasher.hash(f"tx{i}".encode()).value),
            )

        # Get all events
        event_log = contract.get_event_log()
        assert len(event_log) == 3

        # Get limited events
        event_log = contract.get_event_log(limit=2)
        assert len(event_log) == 2

    def test_contract_abi_validation(self):
        """Test ABI validation."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        # Valid ABI
        valid_abi = {
            "functions": [{"name": "test", "signature": "test()"}],
            "events": [{"name": "TestEvent", "signature": "TestEvent(uint256)"}],
            "constructor": {"inputs": []},
        }

        assert contract.validate_abi(valid_abi)

        # Invalid ABI
        invalid_abi = {
            "functions": [{"name": "test", "signature": "test()"}],
            "events": [{"name": "TestEvent", "signature": "TestEvent(uint256)"}]
            # Missing constructor
        }

        assert not contract.validate_abi(invalid_abi)

        # Invalid ABI type
        assert not contract.validate_abi("not_a_dict")

    def test_contract_set_abi(self):
        """Test setting ABI."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        valid_abi = {
            "functions": [{"name": "test", "signature": "test()"}],
            "events": [{"name": "TestEvent", "signature": "TestEvent(uint256)"}],
            "constructor": {"inputs": []},
        }

        contract.set_abi(valid_abi)
        assert contract.abi == valid_abi

    def test_contract_set_invalid_abi(self):
        """Test setting invalid ABI."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        invalid_abi = {
            "functions": [{"name": "test", "signature": "test()"}],
            "events": [{"name": "TestEvent", "signature": "TestEvent(uint256)"}]
            # Missing constructor
        }

        with pytest.raises(ValueError, match="Invalid ABI format"):
            contract.set_abi(invalid_abi)

    def test_contract_get_function_signature(self):
        """Test getting function signature."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        abi = {
            "functions": [
                {"name": "test", "signature": "test()"},
                {"name": "test2", "signature": "test2(uint256)"},
            ],
            "events": [],
            "constructor": {"inputs": []},
        }

        contract.set_abi(abi)

        assert contract.get_function_signature("test") == "test()"
        assert contract.get_function_signature("test2") == "test2(uint256)"
        assert contract.get_function_signature("nonexistent") is None

    def test_contract_get_event_signature(self):
        """Test getting event signature."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        abi = {
            "functions": [],
            "events": [
                {"name": "TestEvent", "signature": "TestEvent(uint256)"},
                {"name": "TestEvent2", "signature": "TestEvent2(uint256,string)"},
            ],
            "constructor": {"inputs": []},
        }

        contract.set_abi(abi)

        assert contract.get_event_signature("TestEvent") == "TestEvent(uint256)"
        assert (
            contract.get_event_signature("TestEvent2") == "TestEvent2(uint256,string)"
        )
        assert contract.get_event_signature("nonexistent") is None

    def test_contract_string_representation(self):
        """Test contract string representation."""
        contract = SmartContract(address="0x123", bytecode=b"contract_bytecode")

        assert (
            str(contract) == "SmartContract(address=0x123, type=standard, state=active)"
        )
        assert (
            "SmartContract(address=0x123, type=standard, state=active, balance=0"
            in repr(contract)
        )
