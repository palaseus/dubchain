"""
Channel Manager - Orchestrates State Channel Operations

This module provides the main interface for managing state channels including:
- Channel lifecycle management
- Multi-party coordination
- Integration with all security and dispute resolution components
- Performance monitoring and optimization
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature
from .channel_protocol import (
    ChannelConfig,
    ChannelId,
    ChannelState,
    StateChannel,
    StateUpdate,
    StateUpdateType,
    ChannelStatus,
    ChannelEvent,
    ChannelCloseReason,
)
from .dispute_resolution import DisputeManager, OnChainContract, DisputeEvidence
from .off_chain_state import OffChainStateManager, StateValidator
from .security import SecurityManager, SecurityEvent, FraudProof


@dataclass
class ChannelMetrics:
    """Metrics for channel performance monitoring."""
    channel_id: ChannelId
    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    average_update_time: float = 0.0
    total_volume: int = 0
    security_events: int = 0
    disputes_initiated: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    last_update_at: Optional[int] = None
    
    def update_success(self, update_time: float, volume: int = 0) -> None:
        """Record a successful update."""
        self.total_updates += 1
        self.successful_updates += 1
        self.total_volume += volume
        self.last_update_at = int(time.time())
        
        # Update average update time
        if self.average_update_time == 0:
            self.average_update_time = update_time
        else:
            self.average_update_time = (self.average_update_time + update_time) / 2
    
    def update_failure(self) -> None:
        """Record a failed update."""
        self.total_updates += 1
        self.failed_updates += 1
    
    def record_security_event(self) -> None:
        """Record a security event."""
        self.security_events += 1
    
    def record_dispute(self) -> None:
        """Record a dispute initiation."""
        self.disputes_initiated += 1
    
    def get_success_rate(self) -> float:
        """Get the success rate of updates."""
        if self.total_updates == 0:
            return 0.0
        return self.successful_updates / self.total_updates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_id": self.channel_id.value,
            "total_updates": self.total_updates,
            "successful_updates": self.successful_updates,
            "failed_updates": self.failed_updates,
            "success_rate": self.get_success_rate(),
            "average_update_time": self.average_update_time,
            "total_volume": self.total_volume,
            "security_events": self.security_events,
            "disputes_initiated": self.disputes_initiated,
            "created_at": self.created_at,
            "last_update_at": self.last_update_at,
        }


class ChannelManager:
    """Main manager for state channel operations."""
    
    def __init__(self, config: Optional[ChannelConfig] = None):
        self.config = config or ChannelConfig()
        
        # Core components
        self.off_chain_manager = OffChainStateManager(self.config)
        self.security_manager = SecurityManager(self.config)
        self.on_chain_contract = OnChainContract("0x" + "0" * 40)  # Placeholder address
        self.dispute_manager = DisputeManager(self.on_chain_contract)
        
        # Channel management
        self.active_channels: Dict[ChannelId, StateChannel] = {}
        self.channel_metrics: Dict[ChannelId, ChannelMetrics] = {}
        self.participant_channels: Dict[str, Set[ChannelId]] = {}  # participant -> set of channels
        
        # Event handling
        self.event_handlers: Dict[ChannelEvent, List[callable]] = {}
        self._initialize_event_handlers()
        
        # Performance monitoring
        self.performance_monitoring = True
        self.monitoring_interval = 60  # seconds
        
        # Async support
        self.executor = None  # Will be initialized when needed
    
    def _initialize_event_handlers(self) -> None:
        """Initialize event handlers for all channel events."""
        for event in ChannelEvent:
            self.event_handlers[event] = []
    
    def create_channel(
        self,
        participants: List[str],
        deposits: Dict[str, int],
        participant_keys: Dict[str, PublicKey],
        custom_config: Optional[ChannelConfig] = None
    ) -> Tuple[bool, Optional[ChannelId], List[str]]:
        """Create a new state channel."""
        try:
            # Validate inputs
            errors = []
            
            if len(participants) < 2:
                errors.append("At least 2 participants required")
            
            if len(participants) > self.config.max_participants:
                errors.append(f"Too many participants: {len(participants)} > {self.config.max_participants}")
            
            # Validate deposits
            for participant in participants:
                if participant not in deposits:
                    errors.append(f"Missing deposit for {participant}")
                elif deposits[participant] < self.config.min_deposit:
                    errors.append(f"Deposit too small for {participant}: {deposits[participant]} < {self.config.min_deposit}")
            
            # Validate participant keys
            for participant in participants:
                if participant not in participant_keys:
                    errors.append(f"Missing public key for {participant}")
            
            if errors:
                return False, None, errors
            
            # Generate channel ID
            channel_id = ChannelId.generate()
            
            # Use custom config if provided
            config = custom_config or self.config
            
            # Create channel
            channel = StateChannel(channel_id, config)
            
            # Create channel state
            success = channel.create_channel(participants, deposits, participant_keys)
            if not success:
                return False, None, ["Failed to create channel"]
            
            # Register with off-chain manager
            channel_state = channel.get_latest_state()
            if channel_state:
                self.off_chain_manager.state_cache[channel_id] = channel_state
                # Initialize pending updates list
                self.off_chain_manager.pending_updates[channel_id] = []
            
            # Register with security manager
            self.security_manager.get_channel_security(channel_id)
            
            # Register with on-chain contract
            self.on_chain_contract.register_channel(channel_id, participants, deposits, config)
            
            # Track channel
            self.active_channels[channel_id] = channel
            self.channel_metrics[channel_id] = ChannelMetrics(channel_id)
            
            # Track participant channels
            for participant in participants:
                if participant not in self.participant_channels:
                    self.participant_channels[participant] = set()
                self.participant_channels[participant].add(channel_id)
            
            # Emit event
            self._emit_event(ChannelEvent.CREATED, channel_id)
            
            return True, channel_id, []
        
        except Exception as e:
            return False, None, [f"Error creating channel: {str(e)}"]
    
    def open_channel(self, channel_id: ChannelId) -> Tuple[bool, List[str]]:
        """Open a channel for state updates."""
        try:
            if channel_id not in self.active_channels:
                return False, ["Channel not found"]
            
            channel = self.active_channels[channel_id]
            success = channel.open_channel()
            
            if success:
                self._emit_event(ChannelEvent.OPENED, channel_id)
                return True, []
            else:
                return False, ["Failed to open channel"]
        
        except Exception as e:
            return False, [f"Error opening channel: {str(e)}"]
    
    def update_channel_state(
        self,
        channel_id: ChannelId,
        update: StateUpdate,
        participant_keys: Dict[str, PublicKey]
    ) -> Tuple[bool, List[str]]:
        """Update a channel state with comprehensive validation."""
        start_time = time.time()
        
        try:
            if channel_id not in self.active_channels:
                return False, ["Channel not found"]
            
            channel = self.active_channels[channel_id]
            channel_state = channel.get_latest_state()
            
            if not channel_state:
                return False, ["Channel state not found"]
            
            # Security validation
            security_valid, security_events = self.security_manager.validate_channel_operation(
                channel_id, "state_update", {
                    "update": update,
                    "channel_state": channel_state,
                    "public_keys": participant_keys
                }
            )
            
            if not security_valid:
                # Record security events
                metrics = self.channel_metrics.get(channel_id)
                if metrics:
                    for event in security_events:
                        metrics.record_security_event()
                
                return False, [f"Security violation: {event.threat_type.value}" for event in security_events]
            
            # Off-chain validation and update
            success, errors = self.off_chain_manager.update_channel_state(
                channel_id, update, participant_keys
            )
            
            if not success:
                # Record failed update
                metrics = self.channel_metrics.get(channel_id)
                if metrics:
                    metrics.update_failure()
                
                return False, errors
            
            # Update channel state to match off-chain state
            updated_state = self.off_chain_manager.get_channel_state(channel_id)
            if updated_state:
                channel.state = updated_state
            
            # Record successful update
            update_time = time.time() - start_time
            metrics = self.channel_metrics.get(channel_id)
            if metrics:
                volume = self._calculate_update_volume(update)
                metrics.update_success(update_time, volume)
            
            # Emit event
            self._emit_event(ChannelEvent.STATE_UPDATED, channel_id)
            
            return True, []
        
        except Exception as e:
            # Record failed update
            metrics = self.channel_metrics.get(channel_id)
            if metrics:
                metrics.update_failure()
            
            return False, [f"Error updating channel: {str(e)}"]
    
    def initiate_dispute(
        self,
        channel_id: ChannelId,
        initiator: str,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], List[str]]:
        """Initiate a dispute resolution process."""
        try:
            if channel_id not in self.active_channels:
                return False, None, ["Channel not found"]
            
            # Create evidence if provided
            dispute_evidence = None
            if evidence:
                dispute_evidence = DisputeEvidence(
                    evidence_id=str(uuid.uuid4()),
                    channel_id=channel_id,
                    submitter=initiator,
                    evidence_type=evidence.get("type", "general"),
                    evidence_data=evidence,
                    timestamp=int(time.time())
                )
            
            # Initiate dispute
            dispute_id = self.dispute_manager.initiate_dispute(
                channel_id, initiator, reason, dispute_evidence
            )
            
            if dispute_id:
                # Record dispute
                metrics = self.channel_metrics.get(channel_id)
                if metrics:
                    metrics.record_dispute()
                
                # Update channel status
                channel = self.active_channels[channel_id]
                channel.initiate_dispute(evidence or {})
                
                # Emit event
                self._emit_event(ChannelEvent.DISPUTE_INITIATED, channel_id)
                
                return True, dispute_id, []
            else:
                return False, None, ["Failed to initiate dispute"]
        
        except Exception as e:
            return False, None, [f"Error initiating dispute: {str(e)}"]
    
    def close_channel(
        self,
        channel_id: ChannelId,
        reason: str = "cooperative",
        final_state: Optional[ChannelState] = None
    ) -> Tuple[bool, List[str]]:
        """Close a channel."""
        try:
            if channel_id not in self.active_channels:
                return False, ["Channel not found"]
            
            channel = self.active_channels[channel_id]
            
            # Close the channel
            close_reason = ChannelCloseReason.COOPERATIVE if reason == "cooperative" else ChannelCloseReason.DISPUTE
            success = channel.close_channel(close_reason)
            
            if success:
                # Clean up resources
                self._cleanup_channel(channel_id)
                
                # Emit events
                self._emit_event(ChannelEvent.CLOSING, channel_id)
                self._emit_event(ChannelEvent.CLOSED, channel_id)
                
                return True, []
            else:
                return False, ["Failed to close channel"]
        
        except Exception as e:
            return False, [f"Error closing channel: {str(e)}"]
    
    def get_channel_info(self, channel_id: ChannelId) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a channel."""
        if channel_id not in self.active_channels:
            # Try to get info from off-chain manager for closed channels
            state = self.off_chain_manager.get_channel_state(channel_id)
            if state:
                return {
                    "channel_id": channel_id.value,
                    "status": state.status.value,
                    "sequence_number": state.sequence_number,
                    "participants": state.participants,
                    "balances": state.balances,
                    "deposits": state.deposits,
                    "created_at": state.created_at,
                    "opened_at": state.opened_at,
                    "closed_at": state.closed_at,
                    "close_reason": state.close_reason.value if state.close_reason else None,
                }
            return None
        
        channel = self.active_channels[channel_id]
        metrics = self.channel_metrics.get(channel_id)
        
        info = channel.get_channel_info()
        if metrics:
            info["metrics"] = metrics.to_dict()
        
        # Add security information
        security = self.security_manager.get_channel_security(channel_id)
        info["security_events"] = len(security.security_events)
        info["fraud_proofs"] = len(security.fraud_proofs)
        
        # Add dispute information
        disputes = self.dispute_manager.contract.get_channel_disputes(channel_id)
        info["disputes"] = len(disputes)
        info["active_disputes"] = len([d for d in disputes if d.status.value in ["pending", "evidence_period", "challenge_period"]])
        
        return info
    
    def get_participant_channels(self, participant: str) -> List[ChannelId]:
        """Get all channels for a participant."""
        return list(self.participant_channels.get(participant, set()))
    
    def get_channel_metrics(self, channel_id: ChannelId) -> Optional[ChannelMetrics]:
        """Get metrics for a channel."""
        return self.channel_metrics.get(channel_id)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global metrics across all channels."""
        total_channels = len(self.active_channels)
        total_updates = sum(m.total_updates for m in self.channel_metrics.values())
        total_volume = sum(m.total_volume for m in self.channel_metrics.values())
        total_security_events = sum(m.security_events for m in self.channel_metrics.values())
        total_disputes = sum(m.disputes_initiated for m in self.channel_metrics.values())
        
        # Calculate average success rate
        success_rates = [m.get_success_rate() for m in self.channel_metrics.values()]
        average_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        # Calculate average update time
        update_times = [m.average_update_time for m in self.channel_metrics.values() if m.average_update_time > 0]
        average_update_time = sum(update_times) / len(update_times) if update_times else 0
        
        return {
            "total_channels": total_channels,
            "total_updates": total_updates,
            "total_volume": total_volume,
            "total_security_events": total_security_events,
            "total_disputes": total_disputes,
            "average_success_rate": average_success_rate,
            "average_update_time": average_update_time,
            "security_statistics": self.security_manager.get_global_security_statistics(),
            "dispute_statistics": self.dispute_manager.get_dispute_statistics(),
        }
    
    def add_event_handler(self, event: ChannelEvent, handler: callable) -> None:
        """Add an event handler."""
        self.event_handlers[event].append(handler)
    
    def remove_event_handler(self, event: ChannelEvent, handler: callable) -> None:
        """Remove an event handler."""
        if handler in self.event_handlers[event]:
            self.event_handlers[event].remove(handler)
    
    def _emit_event(self, event: ChannelEvent, channel_id: ChannelId) -> None:
        """Emit an event to all registered handlers."""
        for handler in self.event_handlers[event]:
            try:
                handler(event, channel_id)
            except Exception as e:
                print(f"Error in event handler for {event}: {e}")
    
    def _calculate_update_volume(self, update: StateUpdate) -> int:
        """Calculate the volume of a state update."""
        if update.update_type == StateUpdateType.TRANSFER:
            return update.state_data.get("amount", 0)
        elif update.update_type == StateUpdateType.MULTI_PARTY:
            transfers = update.state_data.get("transfers", [])
            return sum(transfer.get("amount", 0) for transfer in transfers)
        return 0
    
    def _cleanup_channel(self, channel_id: ChannelId) -> None:
        """Clean up resources for a closed channel."""
        # Remove from active channels
        if channel_id in self.active_channels:
            del self.active_channels[channel_id]
        
        # Clean up off-chain state
        self.off_chain_manager.cleanup_channel(channel_id)
        
        # Remove from participant tracking
        participants_to_remove = []
        for participant, channels in self.participant_channels.items():
            channels.discard(channel_id)
            if not channels:
                participants_to_remove.append(participant)
        
        for participant in participants_to_remove:
            del self.participant_channels[participant]
    
    async def monitor_channels(self) -> None:
        """Monitor all channels for timeouts, disputes, and security issues."""
        while self.performance_monitoring:
            try:
                # Monitor disputes
                self.dispute_manager.monitor_disputes()
                
                # Monitor security
                self.security_manager.monitor_security()
                
                # Check for expired channels
                current_time = int(time.time())
                expired_channels = []
                
                for channel_id, channel in self.active_channels.items():
                    channel_state = channel.get_latest_state()
                    if channel_state:
                        # Check if channel has expired
                        if self.config.enable_timeout_mechanism:
                            time_since_last = current_time - channel_state.last_update_timestamp
                            if time_since_last > self.config.timeout_blocks * 12:  # Assume 12s per block
                                expired_channels.append(channel_id)
                
                # Handle expired channels
                for channel_id in expired_channels:
                    channel = self.active_channels[channel_id]
                    channel.expire_channel()
                    self._cleanup_channel(channel_id)
                    self._emit_event(ChannelEvent.EXPIRED, channel_id)
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.monitoring_interval)
            
            except Exception as e:
                print(f"Error in channel monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def start_monitoring(self) -> None:
        """Start the monitoring loop."""
        if not self.executor:
            self.executor = asyncio.new_event_loop()
        
        # Run monitoring in background
        asyncio.create_task(self.monitor_channels())
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self.performance_monitoring = False
        if self.executor:
            self.executor.close()
            self.executor = None
    
    def export_channel_state(self, channel_id: ChannelId) -> Optional[Dict[str, Any]]:
        """Export channel state for backup."""
        if channel_id not in self.active_channels:
            return None
        
        return self.off_chain_manager.export_state(channel_id)
    
    def import_channel_state(self, state_data: Dict[str, Any]) -> bool:
        """Import channel state from backup."""
        return self.off_chain_manager.import_state(state_data)
    
    def get_channel_statistics(self) -> Dict[str, Any]:
        """Get comprehensive channel statistics."""
        return {
            "active_channels": len(self.active_channels),
            "total_participants": len(self.participant_channels),
            "global_metrics": self.get_global_metrics(),
            "config": self.config.to_dict(),
        }
