"""
Off-chain State Management for State Channels

This module implements off-chain state management including:
- State persistence and synchronization
- State versioning and rollback
- State compression and optimization
- State validation and integrity checks
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..errors import ClientError
from dubchain.logging import get_logger
from .channel_protocol import ChannelState, ChannelParticipant, Payment, StateUpdateType
from .channel_protocol import ChannelStatus

logger = get_logger(__name__)

class StateVersion(Enum):
    """State version types."""
    CURRENT = "current"
    SNAPSHOT = "snapshot"
    CHECKPOINT = "checkpoint"
    BACKUP = "backup"

class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"

@dataclass
class StateSnapshot:
    """Snapshot of channel state."""
    snapshot_id: str
    channel_id: str
    state: ChannelState
    created_at: int
    version: StateVersion
    compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE
    size_bytes: int = 0
    checksum: str = ""

@dataclass
class StateDiff:
    """Difference between two states."""
    diff_id: str
    channel_id: str
    from_snapshot_id: str
    to_snapshot_id: str
    changes: List[Dict[str, Any]]
    created_at: int
    size_bytes: int = 0

@dataclass
class StateConfig:
    """Configuration for state management."""
    max_snapshots_per_channel: int = 100
    snapshot_interval: int = 3600  # 1 hour
    compression_enabled: bool = True
    compression_type: CompressionType = CompressionType.GZIP
    enable_versioning: bool = True
    enable_rollback: bool = True
    max_state_size: int = 10 * 1024 * 1024  # 10MB
    enable_checksums: bool = True

class StateCompressor:
    """Compresses and decompresses state data."""
    
    def __init__(self, config: StateConfig):
        """Initialize state compressor."""
        self.config = config
        logger.info("Initialized state compressor")
    
    def compress_state(self, state: ChannelState) -> Tuple[bytes, CompressionType]:
        """Compress state data."""
        try:
            if not getattr(self.config, 'compression_enabled', True):
                return self._serialize_state(state), CompressionType.NONE
            
            # Serialize state
            state_data = self._serialize_state(state)
            
            # Choose compression type
            compression_type = getattr(self.config, 'compression_type', CompressionType.GZIP)
            
            # Compress based on type
            if compression_type == CompressionType.GZIP:
                compressed_data = self._compress_gzip(state_data)
            elif compression_type == CompressionType.LZ4:
                compressed_data = self._compress_lz4(state_data)
            elif compression_type == CompressionType.ZSTD:
                compressed_data = self._compress_zstd(state_data)
            else:
                compressed_data = state_data
                compression_type = CompressionType.NONE
            
            logger.debug(f"Compressed state: {len(state_data)} -> {len(compressed_data)} bytes")
            return compressed_data, compression_type
            
        except Exception as e:
            logger.error(f"Error compressing state: {e}")
            return self._serialize_state(state), CompressionType.NONE
    
    def decompress_state(self, compressed_data: bytes, compression_type: CompressionType) -> ChannelState:
        """Decompress state data."""
        try:
            if compression_type == CompressionType.NONE:
                return self._deserialize_state(compressed_data)
            
            # Decompress based on type
            if compression_type == CompressionType.GZIP:
                decompressed_data = self._decompress_gzip(compressed_data)
            elif compression_type == CompressionType.LZ4:
                decompressed_data = self._decompress_lz4(compressed_data)
            elif compression_type == CompressionType.ZSTD:
                decompressed_data = self._decompress_zstd(compressed_data)
            else:
                decompressed_data = compressed_data
            
            return self._deserialize_state(decompressed_data)
            
        except Exception as e:
            logger.error(f"Error decompressing state: {e}")
            raise ClientError(f"Failed to decompress state: {e}")
    
    def _serialize_state(self, state: ChannelState) -> bytes:
        """Serialize state to bytes."""
        try:
            # Convert state to dictionary
            state_dict = {
                'channel_id': state.channel_id,
                'participants': {
                    addr: {
                        'address': p.address,
                        'public_key': p.public_key,
                        'balance': p.balance,
                        'nonce': p.nonce,
                        'is_active': p.is_active,
                        'last_activity': p.last_activity
                    }
                    for addr, p in state.participants.items()
                },
                'total_balance': state.total_balance,
                'nonce': state.nonce,
                'created_at': state.created_at,
                'last_updated': state.last_updated,
                'state_hash': state.state_hash,
                'payments': [
                    {
                        'payment_id': p.payment_id,
                        'from_address': p.from_address,
                        'to_address': p.to_address,
                        'amount': p.amount,
                        'payment_type': p.payment_type.value,
                        'timestamp': p.timestamp,
                        'signature': p.signature,
                        'metadata': p.metadata
                    }
                    for p in state.payments
                ],
                'metadata': state.metadata
            }
            
            return json.dumps(state_dict).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error serializing state: {e}")
            raise ClientError(f"Failed to serialize state: {e}")
    
    def _deserialize_state(self, data: bytes) -> ChannelState:
        """Deserialize state from bytes."""
        try:
            state_dict = json.loads(data.decode('utf-8'))
            
            # Reconstruct participants
            participants = {}
            for addr, p_data in state_dict['participants'].items():
                participants[addr] = ChannelParticipant(
                    address=p_data['address'],
                    public_key=p_data['public_key'],
                    balance=p_data['balance'],
                    nonce=p_data['nonce'],
                    is_active=p_data['is_active'],
                    last_activity=p_data['last_activity']
                )
            
            # Reconstruct payments
            payments = []
            for p_data in state_dict['payments']:
                payments.append(Payment(
                    payment_id=p_data['payment_id'],
                    from_address=p_data['from_address'],
                    to_address=p_data['to_address'],
                    amount=p_data['amount'],
                    payment_type=PaymentType(p_data['payment_type']),
                    timestamp=p_data['timestamp'],
                    signature=p_data.get('signature'),
                    metadata=p_data.get('metadata', {})
                ))
            
            # Reconstruct state
            state = ChannelState(
                channel_id=state_dict['channel_id'],
                participants=participants,
                total_balance=state_dict['total_balance'],
                nonce=state_dict['nonce'],
                created_at=state_dict['created_at'],
                last_updated=state_dict['last_updated'],
                state_hash=state_dict['state_hash'],
                payments=payments,
                metadata=state_dict.get('metadata', {})
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Error deserializing state: {e}")
            raise ClientError(f"Failed to deserialize state: {e}")
    
    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        try:
            import gzip
            return gzip.compress(data)
        except ImportError:
            logger.warning("gzip not available, using no compression")
            return data
    
    def _decompress_gzip(self, data: bytes) -> bytes:
        """Decompress data using gzip."""
        try:
            import gzip
            return gzip.decompress(data)
        except ImportError:
            logger.warning("gzip not available, returning original data")
            return data
    
    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress data using LZ4."""
        try:
            import lz4.frame
            return lz4.frame.compress(data)
        except ImportError:
            logger.warning("LZ4 not available, using no compression")
            return data
    
    def _decompress_lz4(self, data: bytes) -> bytes:
        """Decompress data using LZ4."""
        try:
            import lz4.frame
            return lz4.frame.decompress(data)
        except ImportError:
            logger.warning("LZ4 not available, returning original data")
            return data
    
    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress data using Zstandard."""
        try:
            import zstandard as zstd
            compressor = zstd.ZstdCompressor()
            return compressor.compress(data)
        except ImportError:
            logger.warning("Zstandard not available, using no compression")
            return data
    
    def _decompress_zstd(self, data: bytes) -> bytes:
        """Decompress data using Zstandard."""
        try:
            import zstandard as zstd
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        except ImportError:
            logger.warning("Zstandard not available, returning original data")
            return data

class StateVersioner:
    """Manages state versioning and rollback."""
    
    def __init__(self, config: StateConfig):
        """Initialize state versioner."""
        self.config = config
        self.snapshots: Dict[str, List[StateSnapshot]] = {}
        self.diffs: Dict[str, List[StateDiff]] = {}
        logger.info("Initialized state versioner")
    
    def create_snapshot(self, channel_id: str, state: ChannelState, version: StateVersion = StateVersion.SNAPSHOT) -> str:
        """Create a state snapshot."""
        try:
            snapshot_id = f"snapshot_{channel_id}_{int(time.time())}"
            current_time = int(time.time())
            
            # Create snapshot
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                channel_id=channel_id,
                state=state,
                created_at=current_time,
                version=version
            )
            
            # Store snapshot
            if channel_id not in self.snapshots:
                self.snapshots[channel_id] = []
            
            self.snapshots[channel_id].append(snapshot)
            
            # Limit snapshots per channel
            if len(self.snapshots[channel_id]) > getattr(self.config, 'max_snapshots_per_channel', 100):
                # Remove oldest snapshot
                self.snapshots[channel_id].pop(0)
            
            logger.info(f"Created snapshot {snapshot_id} for channel {channel_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            raise ClientError(f"Failed to create snapshot: {e}")
    
    def get_snapshot(self, channel_id: str, snapshot_id: str) -> Optional[StateSnapshot]:
        """Get a specific snapshot."""
        try:
            if channel_id not in self.snapshots:
                return None
            
            for snapshot in self.snapshots[channel_id]:
                if snapshot.snapshot_id == snapshot_id:
                    return snapshot
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting snapshot: {e}")
            return None
    
    def list_snapshots(self, channel_id: str) -> List[StateSnapshot]:
        """List all snapshots for a channel."""
        return self.snapshots.get(channel_id, [])
    
    def rollback_to_snapshot(self, channel_id: str, snapshot_id: str) -> Optional[ChannelState]:
        """Rollback to a specific snapshot."""
        try:
            if not getattr(self.config, 'enable_rollback', True):
                logger.error("Rollback not enabled")
                return None
            
            snapshot = self.get_snapshot(channel_id, snapshot_id)
            if not snapshot:
                logger.error(f"Snapshot {snapshot_id} not found")
                return None
            
            logger.info(f"Rolled back channel {channel_id} to snapshot {snapshot_id}")
            return snapshot.state
            
        except Exception as e:
            logger.error(f"Error rolling back to snapshot: {e}")
            return None
    
    def create_diff(self, channel_id: str, from_snapshot_id: str, to_snapshot_id: str) -> Optional[str]:
        """Create a diff between two snapshots."""
        try:
            from_snapshot = self.get_snapshot(channel_id, from_snapshot_id)
            to_snapshot = self.get_snapshot(channel_id, to_snapshot_id)
            
            if not from_snapshot or not to_snapshot:
                logger.error("One or both snapshots not found")
                return None
            
            # Calculate changes
            changes = self._calculate_changes(from_snapshot.state, to_snapshot.state)
            
            # Create diff
            diff_id = f"diff_{channel_id}_{from_snapshot_id}_{to_snapshot_id}"
            diff = StateDiff(
                diff_id=diff_id,
                channel_id=channel_id,
                from_snapshot_id=from_snapshot_id,
                to_snapshot_id=to_snapshot_id,
                changes=changes,
                created_at=int(time.time())
            )
            
            # Store diff
            if channel_id not in self.diffs:
                self.diffs[channel_id] = []
            
            self.diffs[channel_id].append(diff)
            
            logger.info(f"Created diff {diff_id} for channel {channel_id}")
            return diff_id
            
        except Exception as e:
            logger.error(f"Error creating diff: {e}")
            return None
    
    def _calculate_changes(self, from_state: ChannelState, to_state: ChannelState) -> List[Dict[str, Any]]:
        """Calculate changes between two states."""
        try:
            changes = []
            
            # Check participant changes
            for addr in set(from_state.participants.keys()) | set(to_state.participants.keys()):
                from_participant = from_state.participants.get(addr)
                to_participant = to_state.participants.get(addr)
                
                if from_participant and to_participant:
                    # Check balance changes
                    if from_participant.balance != to_participant.balance:
                        changes.append({
                            'type': 'balance_change',
                            'address': addr,
                            'from_balance': from_participant.balance,
                            'to_balance': to_participant.balance
                        })
                    
                    # Check nonce changes
                    if from_participant.nonce != to_participant.nonce:
                        changes.append({
                            'type': 'nonce_change',
                            'address': addr,
                            'from_nonce': from_participant.nonce,
                            'to_nonce': to_participant.nonce
                        })
                elif from_participant and not to_participant:
                    changes.append({
                        'type': 'participant_removed',
                        'address': addr
                    })
                elif not from_participant and to_participant:
                    changes.append({
                        'type': 'participant_added',
                        'address': addr,
                        'balance': to_participant.balance,
                        'public_key': to_participant.public_key
                    })
            
            # Check payment changes
            from_payment_ids = {p.payment_id for p in from_state.payments}
            to_payment_ids = {p.payment_id for p in to_state.payments}
            
            new_payments = to_payment_ids - from_payment_ids
            for payment_id in new_payments:
                payment = next(p for p in to_state.payments if p.payment_id == payment_id)
                changes.append({
                    'type': 'payment_added',
                    'payment_id': payment_id,
                    'from_address': payment.from_address,
                    'to_address': payment.to_address,
                    'amount': payment.amount
                })
            
            # Check total balance changes
            if from_state.total_balance != to_state.total_balance:
                changes.append({
                    'type': 'total_balance_change',
                    'from_balance': from_state.total_balance,
                    'to_balance': to_state.total_balance
                })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error calculating changes: {e}")
            return []

class StateValidator:
    """Validates state integrity and consistency."""
    
    def __init__(self, config: StateConfig):
        """Initialize state validator."""
        self.config = config
        self.custom_rules = {}  # Added for test compatibility
        logger.info("Initialized state validator")
    
    def validate_state(self, state: ChannelState) -> bool:
        """Validate state integrity."""
        try:
            # Check basic structure
            if not state.channel_id:
                logger.error("Channel ID required")
                return False
            
            if not state.participants:
                logger.error("Participants required")
                return False
            
            if state.total_balance < 0:
                logger.error("Total balance cannot be negative")
                return False
            
            # Check participant consistency
            if isinstance(state.participants, list):
                # participants is a list of strings
                for participant in state.participants:
                    if not isinstance(participant, str) or not participant:
                        logger.error(f"Invalid participant: {participant}")
                        return False
            elif isinstance(state.participants, dict):
                # participants is a dictionary of ChannelParticipant objects
                for addr, participant in state.participants.items():
                    if not self._validate_participant(participant):
                        logger.error(f"Invalid participant {addr}")
                        return False
            
            # Check balance consistency
            if not self._validate_balance_consistency(state):
                logger.error("Balance consistency violated")
                return False
            
            # Check state hash
            if not self._validate_state_hash(state):
                logger.error("Invalid state hash")
                return False
            
            # Check size limits
            if not self._validate_state_size(state):
                logger.error("State size exceeds limits")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state: {e}")
            return False
    
    def _validate_participant(self, participant: ChannelParticipant) -> bool:
        """Validate participant data."""
        try:
            if not participant.address:
                logger.error("Participant address required")
                return False
            
            if not participant.public_key:
                logger.error("Participant public key required")
                return False
            
            if participant.balance < 0:
                logger.error("Participant balance cannot be negative")
                return False
            
            if participant.nonce < 0:
                logger.error("Participant nonce cannot be negative")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating participant: {e}")
            return False
    
    def _validate_balance_consistency(self, state: ChannelState) -> bool:
        """Validate balance consistency."""
        try:
            # Calculate total balance from balances dictionary
            if state.balances:
                calculated_total = sum(state.balances.values())
            else:
                calculated_total = 0
            
            if calculated_total != state.total_balance:
                logger.error(f"Balance mismatch: calculated {calculated_total}, stored {state.total_balance}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating balance consistency: {e}")
            return False
    
    def _validate_state_hash(self, state: ChannelState) -> bool:
        """Validate state hash."""
        try:
            # Calculate expected hash
            expected_hash = self._calculate_state_hash(state)
            
            if state.state_hash != expected_hash:
                logger.error(f"State hash mismatch: expected {expected_hash}, got {state.state_hash}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state hash: {e}")
            return False
    
    def _validate_state_size(self, state: ChannelState) -> bool:
        """Validate state size."""
        try:
            # Estimate state size
            size = len(str(state.channel_id))
            if isinstance(state.participants, list):
                size += sum(len(p) for p in state.participants)
            elif isinstance(state.participants, dict):
                size += sum(len(addr) + len(p.public_key) for addr, p in state.participants.items())
            size += len(state.payments) * 100  # Estimate per payment
            size += len(state.metadata) * 50  # Estimate per metadata entry
            
            max_size = getattr(self.config, 'max_state_size', 10 * 1024 * 1024)  # Default 10MB
            if size > max_size:
                logger.error(f"State size {size} exceeds limit {max_size}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state size: {e}")
            return False
    
    def _calculate_state_hash(self, state: ChannelState) -> str:
        """Calculate state hash."""
        try:
            # Create hashable representation
            hash_data = {
                'channel_id': str(state.channel_id),
                'participants': state.participants,
                'balances': state.balances,
                'total_balance': state.total_balance,
                'sequence_number': state.sequence_number,
                'nonce': state.nonce,
                'created_at': state.created_at,
                'last_updated': state.last_updated
            }
            
            # Add payment hashes
            payment_hashes = []
            for payment in state.payments:
                payment_hash = hashlib.sha256(
                    f"{payment.payment_id}{payment.from_address}{payment.to_address}{payment.amount}{payment.timestamp}".encode()
                ).hexdigest()
                payment_hashes.append(payment_hash)
            
            hash_data['payment_hashes'] = payment_hashes
            
            # Calculate hash
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating state hash: {e}")
            return ""
    
    def validate_state_update(self, update: "StateUpdate", channel_state: "ChannelState", public_keys: Dict[str, "PublicKey"]) -> Tuple[bool, List[str]]:
        """Validate a state update."""
        errors = []
        
        try:
            # Check sequence number
            if update.sequence_number != (channel_state.sequence_number or 0) + 1:
                errors.append("Invalid sequence number")
            
            # Check participants match
            if update.participants != channel_state.participants:
                errors.append("Participant set mismatch")
            
            # Check signatures
            if not update.verify_signatures(public_keys):
                errors.append("Invalid signatures")
            
            # Check signature requirements
            if hasattr(self.config, 'require_all_signatures') and self.config.require_all_signatures:
                if not update.signatures or len(update.signatures) < len(channel_state.participants):
                    errors.append("Insufficient signatures")
            
            # Check update-specific validation
            if update.update_type == StateUpdateType.TRANSFER:
                sender = update.data.get("sender")
                recipient = update.data.get("recipient")
                amount = update.data.get("amount", 0)
                
                if sender not in channel_state.balances:
                    errors.append(f"Sender {sender} not found")
                elif channel_state.balances[sender] < amount:
                    errors.append("Insufficient balance")
                
                if recipient not in channel_state.balances:
                    errors.append(f"Recipient {recipient} not found")
            
            elif update.update_type == StateUpdateType.MULTI_PARTY:
                transfers = update.data.get("transfers", [])
                for transfer in transfers:
                    sender = transfer.get("sender")
                    recipient = transfer.get("recipient")
                    amount = transfer.get("amount", 0)
                    
                    if sender not in channel_state.balances:
                        errors.append(f"Sender {sender} not found")
                    elif channel_state.balances[sender] < amount:
                        errors.append(f"Insufficient balance for {sender}")
                    
                    if recipient not in channel_state.balances:
                        errors.append(f"Recipient {recipient} not found")
            
            elif update.update_type == StateUpdateType.CONDITIONAL:
                # Basic conditional validation
                condition = update.data.get("condition", {})
                if not condition:
                    errors.append("Condition required for conditional update")
            
            # Run custom validation rules
            for rule_name, rule_func in self.custom_rules.items():
                is_valid, rule_errors = rule_func(update, channel_state, public_keys)
                if not is_valid:
                    errors.extend(rule_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating state update: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def add_validation_rule(self, name: str, rule_func: Callable) -> None:
        """Add a custom validation rule."""
        self.custom_rules[name] = rule_func

class OffChainStateManager:
    """Main off-chain state management system."""
    
    def __init__(self, config: StateConfig):
        """Initialize off-chain state manager."""
        self.config = config
        self.compressor = StateCompressor(config)
        self.versioner = StateVersioner(config)
        self.validator = StateValidator(config)
        self.states: Dict[str, ChannelState] = {}
        self.pending_updates: Dict[str, List["StateUpdate"]] = {}  # Added for test compatibility
        logger.info("Initialized off-chain state manager")
    
    def store_state(self, channel_id: str, state: ChannelState) -> bool:
        """Store channel state."""
        try:
            # Validate state
            if not self.validator.validate_state(state):
                logger.error(f"Invalid state for channel {channel_id}")
                return False
            
            # Store state
            self.states[channel_id] = state
            
            # Create snapshot if needed
            if getattr(self.config, 'enable_versioning', True):
                self.versioner.create_snapshot(channel_id, state)
            
            logger.info(f"Stored state for channel {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing state: {e}")
            return False
    
    def get_state(self, channel_id: str) -> Optional[ChannelState]:
        """Get channel state."""
        return self.states.get(channel_id)
    
    def compress_state(self, channel_id: str) -> Optional[Tuple[bytes, CompressionType]]:
        """Compress channel state."""
        try:
            state = self.get_state(channel_id)
            if not state:
                logger.error(f"Channel {channel_id} not found")
                return None
            
            compressed_data, compression_type = self.compressor.compress_state(state)
            return compressed_data, compression_type
            
        except Exception as e:
            logger.error(f"Error compressing state: {e}")
            return None
    
    def decompress_state(self, compressed_data: bytes, compression_type: CompressionType) -> Optional[ChannelState]:
        """Decompress state data."""
        try:
            return self.compressor.decompress_state(compressed_data, compression_type)
        except Exception as e:
            logger.error(f"Error decompressing state: {e}")
            return None
    
    def create_snapshot(self, channel_id: str, version: StateVersion = StateVersion.SNAPSHOT) -> Optional[str]:
        """Create state snapshot."""
        try:
            state = self.get_state(channel_id)
            if not state:
                logger.error(f"Channel {channel_id} not found")
                return None
            
            return self.versioner.create_snapshot(channel_id, state, version)
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            return None
    
    def rollback_to_snapshot(self, channel_id: str, snapshot_id: str) -> bool:
        """Rollback to snapshot."""
        try:
            new_state = self.versioner.rollback_to_snapshot(channel_id, snapshot_id)
            if not new_state:
                logger.error(f"Failed to rollback to snapshot {snapshot_id}")
                return False
            
            # Store rolled back state
            self.states[channel_id] = new_state
            
            logger.info(f"Rolled back channel {channel_id} to snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back to snapshot: {e}")
            return False
    
    def list_snapshots(self, channel_id: str) -> List[StateSnapshot]:
        """List channel snapshots."""
        return self.versioner.list_snapshots(channel_id)
    
    def cleanup_old_snapshots(self, channel_id: str, keep_count: int = 10) -> int:
        """Clean up old snapshots."""
        try:
            snapshots = self.versioner.list_snapshots(channel_id)
            
            if len(snapshots) <= keep_count:
                return 0
            
            # Remove oldest snapshots
            snapshots_to_remove = snapshots[:-keep_count]
            for snapshot in snapshots_to_remove:
                self.versioner.snapshots[channel_id].remove(snapshot)
            
            logger.info(f"Cleaned up {len(snapshots_to_remove)} old snapshots for channel {channel_id}")
            return len(snapshots_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up snapshots: {e}")
            return 0
    
    def create_channel_state(self, channel_id: str, participants: List[str], deposits: Dict[str, int]) -> ChannelState:
        """Create a new channel state."""
        try:
            # Create initial balances from deposits
            balances = deposits.copy()
            total_balance = sum(deposits.values())
            
            # Create channel state
            state = ChannelState(
                channel_id=channel_id,
                participants=participants,
                deposits=deposits,
                balances=balances,
                total_balance=total_balance,
                sequence_number=0,
                last_update_timestamp=int(time.time()),
                status=ChannelStatus.PENDING,
                config=self.config
            )
            
            # Set state hash
            state.state_hash = self.validator._calculate_state_hash(state)
            
            # Store the state
            self.store_state(channel_id, state)
            
            logger.info(f"Created channel state for {channel_id}")
            return state
            
        except Exception as e:
            logger.error(f"Error creating channel state: {e}")
            raise
    
    def get_channel_state(self, channel_id: str) -> Optional[ChannelState]:
        """Get channel state by ID."""
        return self.get_state(channel_id)
    
    def update_channel_state(self, channel_id: str, update: "StateUpdate", public_keys: Dict[str, "PublicKey"]) -> Tuple[bool, List[str]]:
        """Update channel state with a state update."""
        try:
            # Get current state
            current_state = self.get_state(channel_id)
            if not current_state:
                logger.error(f"Channel {channel_id} not found")
                return False, [f"Channel {channel_id} not found"]
            
            # Validate the update
            is_valid, errors = self.validator.validate_state_update(update, current_state, public_keys)
            if not is_valid:
                logger.error(f"Invalid state update: {errors}")
                return False, errors
            
            # Apply the update
            if not current_state.apply_state_update(update):
                logger.error("Failed to apply state update")
                return False, ["Failed to apply state update"]
            
            # Update state hash
            current_state.state_hash = self.validator._calculate_state_hash(current_state)
            
            # Store updated state
            self.store_state(channel_id, current_state)
            
            logger.info(f"Updated channel state for {channel_id}")
            return True, []
            
        except Exception as e:
            logger.error(f"Error updating channel state: {e}")
            return False, [f"Error updating channel state: {str(e)}"]
    
    def sign_state_update(self, update: "StateUpdate", participant: str, private_key: "PrivateKey") -> "StateSignature":
        """Sign a state update."""
        try:
            # Create signature
            message_hash = update.get_hash().encode()
            signature = private_key.sign(message_hash)
            
            # Create state signature
            state_sig = StateSignature(
                channel_id=str(update.channel_id),
                state_hash=update.get_hash(),
                signer=participant,
                signature=signature,
                timestamp=int(time.time()),
                participant=participant
            )
            
            logger.info(f"Signed state update for participant {participant}")
            return state_sig
            
        except Exception as e:
            logger.error(f"Error signing state update: {e}")
            raise
    
    def collect_signatures(self, update: "StateUpdate", participants: List[str], private_keys: Dict[str, "PrivateKey"]) -> Dict[str, "StateSignature"]:
        """Collect signatures from multiple participants."""
        try:
            signatures = {}
            
            for participant in participants:
                if participant in private_keys:
                    signature = self.sign_state_update(update, participant, private_keys[participant])
                    signatures[participant] = signature
                    
                    # Add signature to update
                    update.add_signature(participant, signature.signature)
            
            logger.info(f"Collected {len(signatures)} signatures")
            return signatures
            
        except Exception as e:
            logger.error(f"Error collecting signatures: {e}")
            return {}
    
    def verify_state_consistency(self, channel_id: str) -> Tuple[bool, List[str]]:
        """Verify state consistency for a channel."""
        try:
            state = self.get_state(channel_id)
            if not state:
                logger.error(f"Channel {channel_id} not found")
                return False, [f"Channel {channel_id} not found"]
            
            errors = []
            
            # Validate state integrity
            if not self.validator.validate_state(state):
                logger.error(f"State validation failed for channel {channel_id}")
                errors.append("State validation failed")
            
            # Check balance consistency
            if state.balances:
                total_balances = sum(state.balances.values())
                if total_balances != state.total_balance:
                    logger.error(f"Balance conservation violated: {total_balances} != {state.total_balance}")
                    errors.append(f"Balance conservation violated: {total_balances} != {state.total_balance}")
            
            if errors:
                return False, errors
            
            logger.info(f"State consistency verified for channel {channel_id}")
            return True, []
            
        except Exception as e:
            logger.error(f"Error verifying state consistency: {e}")
            return False, [f"Error verifying state consistency: {str(e)}"]
    
    def resolve_state_conflicts(self, channel_id: str, conflicting_states: List[ChannelState]) -> Optional[ChannelState]:
        """Resolve state conflicts using latest wins strategy."""
        try:
            if not conflicting_states:
                return None
            
            # Sort by sequence number (latest first)
            sorted_states = sorted(conflicting_states, key=lambda s: s.sequence_number, reverse=True)
            
            # Take the latest state
            resolved_state = sorted_states[0]
            
            # Store the resolved state
            self.store_state(channel_id, resolved_state)
            
            logger.info(f"Resolved state conflicts for channel {channel_id} using latest wins")
            return resolved_state
            
        except Exception as e:
            logger.error(f"Error resolving state conflicts: {e}")
            return None
    
    def synchronize_states(self, channel_id: str, remote_states: List[ChannelState]) -> bool:
        """Synchronize local state with remote states."""
        try:
            if not remote_states:
                logger.error("No remote states provided")
                return False
            
            # Use the first remote state for synchronization
            remote_state = remote_states[0]
            local_state = self.get_state(channel_id)
            
            if not local_state:
                # No local state, use remote state
                self.store_state(channel_id, remote_state)
                logger.info(f"Stored remote state for channel {channel_id}")
                return True
            
            # Compare sequence numbers
            if remote_state.sequence_number > local_state.sequence_number:
                # Remote state is newer, update local
                self.store_state(channel_id, remote_state)
                logger.info(f"Updated local state with remote for channel {channel_id}")
                return True
            elif remote_state.sequence_number < local_state.sequence_number:
                # Local state is newer, keep local
                logger.info(f"Local state is newer for channel {channel_id}")
                return True
            else:
                # Same sequence number, check timestamps
                if remote_state.last_update_timestamp > local_state.last_update_timestamp:
                    self.store_state(channel_id, remote_state)
                    logger.info(f"Updated local state with newer remote for channel {channel_id}")
                    return True
                else:
                    logger.info(f"Local state is up to date for channel {channel_id}")
                    return True
            
        except Exception as e:
            logger.error(f"Error synchronizing states: {e}")
            return False
    
    def export_state(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Export channel state for import."""
        try:
            state = self.get_state(channel_id)
            if not state:
                logger.error(f"Channel {channel_id} not found")
                return None
            
            # Convert state to dictionary
            state_dict = {
                "channel_id": str(state.channel_id),
                "participants": state.participants,
                "deposits": state.deposits,
                "balances": state.balances,
                "total_balance": state.total_balance,
                "sequence_number": state.sequence_number,
                "last_update_timestamp": state.last_update_timestamp,
                "status": state.status.value if hasattr(state.status, 'value') else str(state.status),
                "nonce": state.nonce,
                "created_at": state.created_at,
                "last_updated": state.last_updated,
                "state_hash": state.state_hash,
                "metadata": state.metadata
            }
            
            logger.info(f"Exported state for channel {channel_id}")
            return state_dict
            
        except Exception as e:
            logger.error(f"Error exporting state: {e}")
            return None
    
    def import_state(self, state_data: Dict[str, Any]) -> bool:
        """Import channel state from exported data."""
        try:
            # Create ChannelState from imported data
            state = ChannelState(
                channel_id=state_data["channel_id"],
                participants=state_data["participants"],
                deposits=state_data["deposits"],
                balances=state_data["balances"],
                total_balance=state_data["total_balance"],
                sequence_number=state_data["sequence_number"],
                last_update_timestamp=state_data["last_update_timestamp"],
                status=ChannelStatus(state_data["status"]) if isinstance(state_data["status"], str) else state_data["status"],
                nonce=state_data["nonce"],
                created_at=state_data["created_at"],
                last_updated=state_data["last_updated"],
                state_hash=state_data["state_hash"],
                metadata=state_data["metadata"]
            )
            
            # Store the imported state
            self.store_state(state_data["channel_id"], state)
            
            logger.info(f"Imported state for channel {state_data['channel_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing state: {e}")
            return False
    
    def cleanup_channel(self, channel_id: str) -> bool:
        """Clean up channel data."""
        try:
            # Clean up snapshots but preserve the channel state
            self.cleanup_old_snapshots(channel_id, 0)
            
            # Clear pending updates
            if channel_id in self.pending_updates:
                del self.pending_updates[channel_id]
            
            logger.info(f"Cleaned up channel {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up channel: {e}")
            return False

@dataclass
class StateTransition:
    """State transition for off-chain state."""
    transition_id: str
    channel_id: str
    from_state: ChannelState
    to_state: ChannelState
    transition_type: str
    timestamp: int
    signature: Optional[str] = None
    validation_rules: Optional[List[str]] = None  # Added for test compatibility
    preconditions: Optional[Dict[str, Any]] = None  # Added for test compatibility
    postconditions: Optional[Dict[str, Any]] = None  # Added for test compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, transition_id: str = None, channel_id: str = None, 
                 from_state: ChannelState = None, to_state: ChannelState = None,
                 transition_type: str = None, timestamp: int = None,
                 validation_rules: List[str] = None, preconditions: Dict[str, Any] = None,
                 postconditions: Dict[str, Any] = None, **kwargs):
        """Initialize StateTransition with flexible parameters."""
        self.transition_id = transition_id or f"transition_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.channel_id = channel_id or "default_channel"
        self.from_state = from_state
        self.to_state = to_state
        self.transition_type = transition_type or "default"
        self.timestamp = timestamp or int(time.time())
        self.signature = kwargs.get('signature')
        self.validation_rules = validation_rules or []
        self.preconditions = preconditions or {}
        self.postconditions = postconditions or {}
        self.metadata = kwargs.get('metadata', {})
    
    def validate_preconditions(self, channel_state=None) -> bool:
        """Validate preconditions for the transition."""
        try:
            if not self.preconditions:
                return True
            
            # Use provided channel_state or self.from_state
            state_to_check = channel_state or self.from_state
            
            # Example precondition validation
            if "min_balance" in self.preconditions:
                min_balance_info = self.preconditions["min_balance"]
                participant = min_balance_info.get("participant")
                amount = min_balance_info.get("amount", 0)
                
                if participant and state_to_check:
                    if hasattr(state_to_check, 'balances') and state_to_check.balances:
                        if participant in state_to_check.balances:
                            return state_to_check.balances[participant] >= amount
                    elif isinstance(state_to_check, dict) and 'balances' in state_to_check:
                        if participant in state_to_check['balances']:
                            return state_to_check['balances'][participant] >= amount
            
            return True
        except Exception:
            return False
    
    def validate_postconditions(self, new_state=None) -> bool:
        """Validate postconditions for the transition."""
        try:
            if not self.postconditions:
                return True
            
            # Use provided new_state or self.to_state
            state_to_check = new_state or self.to_state
            
            # Example postcondition validation
            if "balance_conservation" in self.postconditions:
                if self.from_state and state_to_check:
                    # Calculate original total
                    if hasattr(self.from_state, 'balances') and self.from_state.balances:
                        original_total = sum(self.from_state.balances.values())
                    elif isinstance(self.from_state, dict) and 'balances' in self.from_state:
                        original_total = sum(self.from_state['balances'].values())
                    else:
                        original_total = 0
                    
                    # Calculate new total
                    if hasattr(state_to_check, 'balances') and state_to_check.balances:
                        new_total = sum(state_to_check.balances.values())
                    elif isinstance(state_to_check, dict) and 'balances' in state_to_check:
                        new_total = sum(state_to_check['balances'].values())
                    else:
                        new_total = 0
                    
                    # Balance conservation means total should remain the same
                    return original_total == new_total
            
            return True
        except Exception:
            return False
    
    def get_transition_hash(self) -> str:
        """Get hash of the transition."""
        import hashlib
        # Use a deterministic hash that doesn't include timestamps for consistency
        transition_data = f"{self.channel_id}:{self.transition_type}"
        if self.from_state:
            transition_data += f":{str(self.from_state)}"
        if self.to_state:
            transition_data += f":{str(self.to_state)}"
        return hashlib.sha256(transition_data.encode()).hexdigest()

@dataclass
class StateSignature:
    """Signature for state validation."""
    signature_id: str
    channel_id: str
    state_hash: str
    signer: str
    signature: str
    timestamp: int
    participant: Optional[str] = None  # For backward compatibility
    nonce: Optional[int] = None  # Added for test compatibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, signature_id: str = None, channel_id: str = None, state_hash: str = None, 
                 signer: str = None, signature: str = None, timestamp: int = None, 
                 participant: str = None, nonce: int = None, **kwargs):
        """Initialize StateSignature with flexible parameters."""
        self.signature_id = signature_id or f"sig_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.channel_id = channel_id or "default_channel"
        self.state_hash = state_hash or "default_hash"
        self.signer = signer or participant or "default_signer"
        self.signature = signature or ""
        self.timestamp = timestamp or int(time.time())
        self.participant = participant
        self.nonce = nonce
        self.metadata = kwargs.get('metadata', {})
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.participant is not None and self.signer != self.participant:
            # If participant is provided and different from signer, use participant as signer
            self.signer = self.participant
    
    def verify(self, public_key: "PublicKey", message_hash: bytes) -> bool:
        """Verify the signature."""
        try:
            # Handle both Signature objects and raw bytes
            if hasattr(self.signature, 'verify'):
                # It's a Signature object
                return public_key.verify(self.signature, message_hash)
            else:
                # It's raw bytes or string, try to verify directly
                return public_key.verify(self.signature, message_hash)
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signature_id": self.signature_id,
            "channel_id": self.channel_id,
            "state_hash": self.state_hash,
            "signer": self.signer,
            "signature": str(self.signature) if self.signature else "",
            "timestamp": self.timestamp,
            "participant": self.participant,
            "nonce": self.nonce,
            "metadata": self.metadata,
        }

__all__ = [
    "OffChainStateManager",
    "StateCompressor",
    "StateVersioner",
    "StateValidator",
    "StateSnapshot",
    "StateDiff",
    "StateConfig",
    "StateVersion",
    "CompressionType",
    "StateTransition",
    "StateSignature",
]