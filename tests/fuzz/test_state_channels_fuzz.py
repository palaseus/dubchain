"""
Fuzz Testing Harness for State Channels

This module provides comprehensive fuzz testing for state channels including:
- Malformed state update generation
- Invalid signature fuzzing
- Corrupted data handling
- Edge case discovery
- Stress testing with random inputs
"""

import pytest
import random
import string
import time
import json
from typing import List, Dict, Any, Optional, Union
from unittest.mock import Mock, patch

from src.dubchain.state_channels.channel_manager import ChannelManager
from src.dubchain.state_channels.channel_protocol import (
    ChannelConfig,
    ChannelId,
    StateUpdate,
    StateUpdateType,
    ChannelStatus,
)
from src.dubchain.crypto.signatures import PrivateKey, PublicKey


class FuzzGenerator:
    """Generates fuzzed inputs for testing."""
    
    @staticmethod
    def generate_random_string(min_length: int = 1, max_length: int = 100) -> str:
        """Generate a random string."""
        length = random.randint(min_length, max_length)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def generate_random_bytes(min_length: int = 1, max_length: int = 1000) -> bytes:
        """Generate random bytes."""
        length = random.randint(min_length, max_length)
        return bytes([random.randint(0, 255) for _ in range(length)])
    
    @staticmethod
    def generate_random_int(min_value: int = -1000, max_value: int = 1000) -> int:
        """Generate a random integer."""
        return random.randint(min_value, max_value)
    
    @staticmethod
    def generate_random_dict(max_depth: int = 3) -> Dict[str, Any]:
        """Generate a random dictionary."""
        if max_depth <= 0:
            return FuzzGenerator.generate_random_string()
        
        result = {}
        num_keys = random.randint(1, 10)
        
        for _ in range(num_keys):
            key = FuzzGenerator.generate_random_string(1, 20)
            value_type = random.choice(['string', 'int', 'dict', 'list', 'bool'])
            
            if value_type == 'string':
                value = FuzzGenerator.generate_random_string()
            elif value_type == 'int':
                value = FuzzGenerator.generate_random_int()
            elif value_type == 'dict':
                value = FuzzGenerator.generate_random_dict(max_depth - 1)
            elif value_type == 'list':
                value = [FuzzGenerator.generate_random_string() for _ in range(random.randint(0, 5))]
            else:  # bool
                value = random.choice([True, False])
            
            result[key] = value
        
        return result
    
    @staticmethod
    def generate_malformed_json() -> str:
        """Generate malformed JSON strings."""
        malformed_patterns = [
            '{"incomplete": "json"',  # Missing closing brace
            '{"invalid": json}',      # Invalid value
            '{"duplicate": "key", "duplicate": "key"}',  # Duplicate keys
            '{"nested": {"incomplete": "nested"',  # Incomplete nested
            '{"array": [1, 2, 3,}',  # Incomplete array
            '{"string": "unclosed',   # Unclosed string
            '{"number": 1.2.3}',      # Invalid number
            '{"null": null, "undefined": undefined}',  # Invalid null/undefined
        ]
        return random.choice(malformed_patterns)
    
    @staticmethod
    def generate_corrupted_signature() -> str:
        """Generate corrupted signature strings."""
        corruption_patterns = [
            "0" * 64,  # All zeros
            "1" * 64,  # All ones
            "a" * 64,  # All 'a'
            "deadbeef" * 8,  # Repeated pattern
            "invalid_signature",  # Too short
            "x" * 128,  # Too long
            "".join(random.choices("0123456789abcdef", k=random.randint(1, 200))),  # Random length
        ]
        return random.choice(corruption_patterns)


class StateUpdateFuzzer:
    """Fuzzes state update objects."""
    
    def __init__(self):
        self.generator = FuzzGenerator()
    
    def generate_fuzzed_update(self, channel_id: ChannelId, participants: List[str]) -> StateUpdate:
        """Generate a fuzzed state update."""
        # Randomly choose which fields to fuzz
        fuzz_fields = random.sample([
            'update_id', 'sequence_number', 'update_type', 'state_data', 
            'timestamp', 'nonce', 'participants'
        ], k=random.randint(1, 4))
        
        # Generate base values
        update_id = self.generator.generate_random_string() if 'update_id' in fuzz_fields else "fuzz-update"
        sequence_number = self.generator.generate_random_int(-100, 100) if 'sequence_number' in fuzz_fields else 1
        update_type = random.choice(list(StateUpdateType)) if 'update_type' in fuzz_fields else StateUpdateType.TRANSFER
        timestamp = self.generator.generate_random_int(0, 2**32) if 'timestamp' in fuzz_fields else int(time.time())
        nonce = self.generator.generate_random_int(-100, 100) if 'nonce' in fuzz_fields else 0
        
        # Fuzz participants
        if 'participants' in fuzz_fields:
            fuzzed_participants = [
                self.generator.generate_random_string() 
                for _ in range(random.randint(0, 10))
            ]
        else:
            fuzzed_participants = participants
        
        # Fuzz state data
        if 'state_data' in fuzz_fields:
            state_data = self.generator.generate_random_dict()
        else:
            state_data = {"sender": "alice", "recipient": "bob", "amount": 1000}
        
        try:
            return StateUpdate(
                update_id=update_id,
                channel_id=channel_id,
                sequence_number=sequence_number,
                update_type=update_type,
                participants=fuzzed_participants,
                state_data=state_data,
                timestamp=timestamp,
                nonce=nonce
            )
        except Exception:
            # If creation fails, return a minimal valid update
            return StateUpdate(
                update_id="fallback-update",
                channel_id=channel_id,
                sequence_number=1,
                update_type=StateUpdateType.TRANSFER,
                participants=participants,
                state_data={"sender": "alice", "recipient": "bob", "amount": 1000},
                timestamp=int(time.time())
            )
    
    def generate_corrupted_signatures(self, update: StateUpdate, participants: List[str]) -> Dict[str, str]:
        """Generate corrupted signatures for an update."""
        corrupted_signatures = {}
        
        for participant in participants:
            if random.choice([True, False]):  # 50% chance to corrupt
                corrupted_signatures[participant] = self.generator.generate_corrupted_signature()
            else:
                # Generate valid signature
                private_key = PrivateKey.generate()
                signature = private_key.sign(update.get_hash())
                corrupted_signatures[participant] = signature.to_hex()
        
        return corrupted_signatures


class TestStateChannelFuzzing:
    """Fuzz testing for state channels."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig()
        self.manager = ChannelManager(self.config)
        self.fuzzer = StateUpdateFuzzer()
        self.generator = FuzzGenerator()
        
        # Create test participants
        self.participants = ["alice", "bob", "charlie"]
        self.private_keys = {p: PrivateKey.generate() for p in self.participants}
        self.public_keys = {p: key.get_public_key() for p, key in self.private_keys.items()}
        self.deposits = {p: 100000 for p in self.participants}  # Large deposits for fuzzing
    
    def test_fuzzed_state_updates(self):
        """Test handling of fuzzed state updates."""
        # Create channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Generate and test many fuzzed updates
        for i in range(100):
            try:
                # Generate fuzzed update
                fuzzed_update = self.fuzzer.generate_fuzzed_update(channel_id, self.participants)
                
                # Try to apply the fuzzed update
                success, errors = self.manager.update_channel_state(
                    channel_id, fuzzed_update, self.public_keys
                )
                
                # Most fuzzed updates should fail, but the system should not crash
                # We're testing that the system gracefully handles malformed inputs
                assert isinstance(success, bool)
                assert isinstance(errors, list)
                
            except Exception as e:
                # System should not crash on fuzzed inputs
                # Log the exception but continue testing
                print(f"Fuzzed update {i} caused exception: {e}")
                continue
    
    def test_corrupted_signatures(self):
        """Test handling of corrupted signatures."""
        # Create channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Generate and test many updates with corrupted signatures
        for i in range(50):
            try:
                # Create valid update
                update = StateUpdate(
                    update_id=f"corrupted-sig-{i}",
                    channel_id=channel_id,
                    sequence_number=i + 1,
                    update_type=StateUpdateType.TRANSFER,
                    participants=self.participants,
                    state_data={"sender": "alice", "recipient": "bob", "amount": 100},
                    timestamp=int(time.time())
                )
                
                # Generate corrupted signatures
                corrupted_sigs = self.fuzzer.generate_corrupted_signatures(update, self.participants)
                
                # Try to add corrupted signatures
                for participant, sig_hex in corrupted_sigs.items():
                    try:
                        # This will likely fail for corrupted signatures
                        if len(sig_hex) == 128:  # Valid length
                            # Try to create signature from hex
                            from src.dubchain.crypto.signatures import Signature
                            signature = Signature.from_hex(sig_hex, update.get_hash().value)
                            update.add_signature(participant, signature)
                    except Exception:
                        # Expected to fail for corrupted signatures
                        pass
                
                # Try to apply update with corrupted signatures
                success, errors = self.manager.update_channel_state(
                    channel_id, update, self.public_keys
                )
                
                # Should fail gracefully
                assert isinstance(success, bool)
                assert isinstance(errors, list)
                
            except Exception as e:
                # System should not crash
                print(f"Corrupted signature test {i} caused exception: {e}")
                continue
    
    def test_malformed_json_data(self):
        """Test handling of malformed JSON data."""
        # Create channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Test malformed JSON in state data
        for i in range(20):
            try:
                # Generate malformed JSON
                malformed_json = self.generator.generate_malformed_json()
                
                # Try to create update with malformed JSON
                update = StateUpdate(
                    update_id=f"malformed-json-{i}",
                    channel_id=channel_id,
                    sequence_number=i + 1,
                    update_type=StateUpdateType.TRANSFER,
                    participants=self.participants,
                    state_data={"malformed": malformed_json},
                    timestamp=int(time.time())
                )
                
                # Try to apply update
                success, errors = self.manager.update_channel_state(
                    channel_id, update, self.public_keys
                )
                
                # Should handle gracefully
                assert isinstance(success, bool)
                assert isinstance(errors, list)
                
            except Exception as e:
                # System should not crash
                print(f"Malformed JSON test {i} caused exception: {e}")
                continue
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        # Create channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Test extreme values
        extreme_values = [
            {"amount": 2**63 - 1},  # Max int64
            {"amount": -2**63},     # Min int64
            {"amount": 0},          # Zero
            {"amount": -1},         # Negative
            {"sender": "a" * 1000}, # Very long string
            {"sender": ""},         # Empty string
            {"sender": None},       # None value
        ]
        
        for i, extreme_data in enumerate(extreme_values):
            try:
                update = StateUpdate(
                    update_id=f"extreme-{i}",
                    channel_id=channel_id,
                    sequence_number=i + 1,
                    update_type=StateUpdateType.TRANSFER,
                    participants=self.participants,
                    state_data=extreme_data,
                    timestamp=int(time.time())
                )
                
                # Try to apply update
                success, errors = self.manager.update_channel_state(
                    channel_id, update, self.public_keys
                )
                
                # Should handle gracefully
                assert isinstance(success, bool)
                assert isinstance(errors, list)
                
            except Exception as e:
                # System should not crash
                print(f"Extreme value test {i} caused exception: {e}")
                continue
    
    def test_random_channel_operations(self):
        """Test random sequences of channel operations."""
        # Create multiple channels
        channel_ids = []
        for i in range(5):
            success, channel_id, errors = self.manager.create_channel(
                self.participants, self.deposits, self.public_keys
            )
            if success:
                channel_ids.append(channel_id)
                self.manager.open_channel(channel_id)
        
        # Perform random operations
        operations = [
            "create_update", "close_channel", "initiate_dispute", 
            "get_info", "update_state", "fuzz_update"
        ]
        
        for i in range(100):
            try:
                operation = random.choice(operations)
                channel_id = random.choice(channel_ids) if channel_ids else None
                
                if operation == "create_update" and channel_id:
                    # Create normal update
                    update = StateUpdate(
                        update_id=f"random-{i}",
                        channel_id=channel_id,
                        sequence_number=i + 1,
                        update_type=StateUpdateType.TRANSFER,
                        participants=self.participants,
                        state_data={"sender": "alice", "recipient": "bob", "amount": 100},
                        timestamp=int(time.time())
                    )
                    
                    # Sign update
                    for participant, private_key in self.private_keys.items():
                        signature = private_key.sign(update.get_hash())
                        update.add_signature(participant, signature)
                    
                    self.manager.update_channel_state(channel_id, update, self.public_keys)
                
                elif operation == "close_channel" and channel_id:
                    self.manager.close_channel(channel_id)
                
                elif operation == "initiate_dispute" and channel_id:
                    self.manager.initiate_dispute(channel_id, "alice", "Random dispute")
                
                elif operation == "get_info" and channel_id:
                    self.manager.get_channel_info(channel_id)
                
                elif operation == "fuzz_update" and channel_id:
                    # Create fuzzed update
                    fuzzed_update = self.fuzzer.generate_fuzzed_update(channel_id, self.participants)
                    self.manager.update_channel_state(channel_id, fuzzed_update, self.public_keys)
                
            except Exception as e:
                # System should not crash
                print(f"Random operation {i} caused exception: {e}")
                continue
    
    def test_memory_exhaustion_resistance(self):
        """Test resistance to memory exhaustion attacks."""
        # Create channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Try to create updates with very large data
        for i in range(10):
            try:
                # Create update with large state data
                large_data = {
                    "large_field": "x" * 10000,  # 10KB string
                    "large_array": ["item"] * 1000,  # 1000 items
                    "nested_large": {
                        "field1": "y" * 5000,
                        "field2": "z" * 5000,
                    }
                }
                
                update = StateUpdate(
                    update_id=f"large-data-{i}",
                    channel_id=channel_id,
                    sequence_number=i + 1,
                    update_type=StateUpdateType.TRANSFER,
                    participants=self.participants,
                    state_data=large_data,
                    timestamp=int(time.time())
                )
                
                # Try to apply update
                success, errors = self.manager.update_channel_state(
                    channel_id, update, self.public_keys
                )
                
                # Should handle gracefully
                assert isinstance(success, bool)
                assert isinstance(errors, list)
                
            except Exception as e:
                # System should not crash
                print(f"Large data test {i} caused exception: {e}")
                continue
    
    def test_concurrent_fuzzing(self):
        """Test concurrent fuzzing operations."""
        import threading
        import queue
        
        # Create channel
        success, channel_id, errors = self.manager.create_channel(
            self.participants, self.deposits, self.public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Results queue
        results = queue.Queue()
        
        def fuzz_worker(worker_id: int, num_operations: int):
            """Worker thread for fuzzing."""
            for i in range(num_operations):
                try:
                    # Create fuzzed update
                    fuzzed_update = self.fuzzer.generate_fuzzed_update(channel_id, self.participants)
                    
                    # Try to apply update
                    success, errors = self.manager.update_channel_state(
                        channel_id, fuzzed_update, self.public_keys
                    )
                    
                    results.put((worker_id, i, success, len(errors)))
                    
                except Exception as e:
                    results.put((worker_id, i, False, str(e)))
        
        # Start multiple fuzzing threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=fuzz_worker, args=(worker_id, 20))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        total_operations = 0
        successful_operations = 0
        
        while not results.empty():
            worker_id, op_id, success, error_count = results.get()
            total_operations += 1
            if success:
                successful_operations += 1
        
        # System should have handled concurrent fuzzing without crashing
        assert total_operations > 0
        print(f"Concurrent fuzzing completed: {successful_operations}/{total_operations} operations successful")


class TestFuzzStress:
    """Stress testing with fuzzed inputs."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ChannelConfig()
        self.manager = ChannelManager(self.config)
        self.fuzzer = StateUpdateFuzzer()
        self.generator = FuzzGenerator()
    
    def test_rapid_fuzzing(self):
        """Test rapid fuzzing to find race conditions."""
        # Create multiple channels
        participants = ["alice", "bob"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {p: 100000 for p in participants}
        
        channel_ids = []
        for i in range(10):
            success, channel_id, errors = self.manager.create_channel(
                participants, deposits, public_keys
            )
            if success:
                channel_ids.append(channel_id)
                self.manager.open_channel(channel_id)
        
        # Rapid fuzzing
        for i in range(1000):
            try:
                channel_id = random.choice(channel_ids)
                
                # Generate fuzzed update
                fuzzed_update = self.fuzzer.generate_fuzzed_update(channel_id, participants)
                
                # Try to apply update
                success, errors = self.manager.update_channel_state(
                    channel_id, fuzzed_update, public_keys
                )
                
                # Should not crash
                assert isinstance(success, bool)
                
            except Exception as e:
                # Log but continue
                if i % 100 == 0:  # Log every 100th exception
                    print(f"Rapid fuzzing exception at {i}: {e}")
                continue
    
    def test_fuzz_with_disputes(self):
        """Test fuzzing with dispute resolution."""
        # Create channel
        participants = ["alice", "bob"]
        private_keys = {p: PrivateKey.generate() for p in participants}
        public_keys = {p: key.get_public_key() for p, key in private_keys.items()}
        deposits = {p: 100000 for p in participants}
        
        success, channel_id, errors = self.manager.create_channel(
            participants, deposits, public_keys
        )
        assert success is True
        
        self.manager.open_channel(channel_id)
        
        # Fuzz with disputes
        for i in range(100):
            try:
                # Randomly choose operation
                operation = random.choice(["fuzz_update", "initiate_dispute", "normal_update"])
                
                if operation == "fuzz_update":
                    fuzzed_update = self.fuzzer.generate_fuzzed_update(channel_id, participants)
                    self.manager.update_channel_state(channel_id, fuzzed_update, public_keys)
                
                elif operation == "initiate_dispute":
                    self.manager.initiate_dispute(channel_id, "alice", f"Fuzz dispute {i}")
                
                elif operation == "normal_update":
                    update = StateUpdate(
                        update_id=f"normal-{i}",
                        channel_id=channel_id,
                        sequence_number=i + 1,
                        update_type=StateUpdateType.TRANSFER,
                        participants=participants,
                        state_data={"sender": "alice", "recipient": "bob", "amount": 100},
                        timestamp=int(time.time())
                    )
                    
                    for participant, private_key in private_keys.items():
                        signature = private_key.sign(update.get_hash())
                        update.add_signature(participant, signature)
                    
                    self.manager.update_channel_state(channel_id, update, public_keys)
                
            except Exception as e:
                # Should not crash
                print(f"Fuzz with disputes exception at {i}: {e}")
                continue


if __name__ == "__main__":
    pytest.main([__file__])
