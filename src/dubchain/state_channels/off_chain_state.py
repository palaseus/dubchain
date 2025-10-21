"""
Off-chain State Management for State Channels

This module implements off-chain state management including:
- State persistence and synchronization
- State versioning and rollback
- State compression and optimization
- State validation and integrity checks
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..errors import ClientError
from dubchain.logging import get_logger
from .channel_protocol import ChannelState, ChannelParticipant, Payment

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
            if not self.config.compression_enabled:
                return self._serialize_state(state), CompressionType.NONE
            
            # Serialize state
            state_data = self._serialize_state(state)
            
            # Choose compression type
            compression_type = self.config.compression_type
            
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
            if len(self.snapshots[channel_id]) > self.config.max_snapshots_per_channel:
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
            if not self.config.enable_rollback:
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
            # Calculate total balance from participants
            calculated_total = sum(p.balance for p in state.participants.values())
            
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
            size = len(state.channel_id)
            size += sum(len(addr) + len(p.public_key) for addr, p in state.participants.items())
            size += len(state.payments) * 100  # Estimate per payment
            size += len(state.metadata) * 50  # Estimate per metadata entry
            
            if size > self.config.max_state_size:
                logger.error(f"State size {size} exceeds limit {self.config.max_state_size}")
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
                'channel_id': state.channel_id,
                'participants': {
                    addr: {
                        'balance': p.balance,
                        'nonce': p.nonce,
                        'public_key': p.public_key
                    }
                    for addr, p in state.participants.items()
                },
                'total_balance': state.total_balance,
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

class OffChainStateManager:
    """Main off-chain state management system."""
    
    def __init__(self, config: StateConfig):
        """Initialize off-chain state manager."""
        self.config = config
        self.compressor = StateCompressor(config)
        self.versioner = StateVersioner(config)
        self.validator = StateValidator(config)
        self.states: Dict[str, ChannelState] = {}
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
            if self.config.enable_versioning:
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
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, signature_id: str = None, channel_id: str = None, state_hash: str = None, 
                 signer: str = None, signature: str = None, timestamp: int = None, 
                 participant: str = None, **kwargs):
        """Initialize StateSignature with flexible parameters."""
        self.signature_id = signature_id or f"sig_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.channel_id = channel_id or "default_channel"
        self.state_hash = state_hash or "default_hash"
        self.signer = signer or participant or "default_signer"
        self.signature = signature or ""
        self.timestamp = timestamp or int(time.time())
        self.participant = participant
        self.metadata = kwargs.get('metadata', {})
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.participant is not None and self.signer != self.participant:
            # If participant is provided and different from signer, use participant as signer
            self.signer = self.participant

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