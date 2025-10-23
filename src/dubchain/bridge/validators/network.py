"""
Bridge Validator Network with BFT Consensus

This module provides a Byzantine Fault Tolerant validator network for bridge operations including:
- Validator registration and management
- BFT consensus for bridge decisions
- Slashing mechanisms for malicious validators
- Emergency pause functionality
- Governance integration
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from decimal import Decimal
import secrets
from enum import Enum
import threading

from ...errors import BridgeError, ClientError
from ...logging import get_logger
from ..universal import UniversalBridge, BridgeConfig, ChainType, UniversalTransaction

logger = get_logger(__name__)


class ValidatorStatus(Enum):
    """Validator status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SLASHED = "slashed"
    PENDING = "pending"


class ConsensusPhase(Enum):
    """Consensus phases."""
    PREPARE = "prepare"
    PRE_COMMIT = "pre_commit"
    COMMIT = "commit"
    DECIDE = "decide"


@dataclass
class Validator:
    """Validator information."""
    address: str
    public_key: str
    stake: int
    status: ValidatorStatus
    commission_rate: float  # 0.0 to 1.0
    last_heartbeat: float = field(default_factory=time.time)
    slashing_count: int = 0
    total_rewards: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class ConsensusMessage:
    """Consensus message."""
    message_id: str
    validator_address: str
    phase: ConsensusPhase
    proposal_id: str
    vote: bool
    signature: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class BridgeProposal:
    """Bridge proposal for consensus."""
    proposal_id: str
    transaction: UniversalTransaction
    proposer: str
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, approved, rejected, executed
    votes: Dict[str, bool] = field(default_factory=dict)
    consensus_reached: bool = False


@dataclass
class ValidatorConfig:
    """Configuration for validator network."""
    min_validators: int = 4
    max_validators: int = 100
    min_stake: int = 1000000000000000000000  # 1000 tokens
    max_stake: int = 100000000000000000000000  # 100000 tokens
    consensus_threshold: float = 0.67  # 2/3 majority
    heartbeat_interval: float = 30.0  # seconds
    heartbeat_timeout: float = 120.0  # seconds
    consensus_timeout: float = 300.0  # seconds
    slashing_threshold: int = 3  # strikes before slashing
    enable_emergency_pause: bool = True
    emergency_pause_threshold: float = 0.5  # 50% of validators


class ValidatorManager:
    """Manages validator registration and status."""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.validators: Dict[str, Validator] = {}
        self.validator_lock = threading.Lock()
    
    def register_validator(self, address: str, public_key: str, stake: int) -> bool:
        """Register a new validator."""
        with self.validator_lock:
            if address in self.validators:
                return False
            
            if stake < self.config.min_stake:
                return False
            
            if len(self.validators) >= self.config.max_validators:
                return False
            
            validator = Validator(
                address=address,
                public_key=public_key,
                stake=stake,
                status=ValidatorStatus.PENDING
            )
            
            self.validators[address] = validator
            logger.info(f"Validator {address} registered with stake {stake}")
            return True
    
    def activate_validator(self, address: str) -> bool:
        """Activate a validator."""
        with self.validator_lock:
            if address not in self.validators:
                return False
            
            validator = self.validators[address]
            if validator.status != ValidatorStatus.PENDING:
                return False
            
            validator.status = ValidatorStatus.ACTIVE
            logger.info(f"Validator {address} activated")
            return True
    
    def slash_validator(self, address: str, reason: str) -> bool:
        """Slash a validator for malicious behavior."""
        with self.validator_lock:
            if address not in self.validators:
                return False
            
            validator = self.validators[address]
            validator.slashing_count += 1
            
            if validator.slashing_count >= self.config.slashing_threshold:
                validator.status = ValidatorStatus.SLASHED
                logger.warning(f"Validator {address} slashed: {reason}")
                return True
            
            logger.warning(f"Validator {address} strike {validator.slashing_count}: {reason}")
            return False
    
    def update_heartbeat(self, address: str) -> bool:
        """Update validator heartbeat."""
        with self.validator_lock:
            if address not in self.validators:
                return False
            
            validator = self.validators[address]
            validator.last_heartbeat = time.time()
            return True
    
    def get_active_validators(self) -> List[Validator]:
        """Get list of active validators."""
        with self.validator_lock:
            return [v for v in self.validators.values() if v.status == ValidatorStatus.ACTIVE]
    
    def is_validator_active(self, address: str) -> bool:
        """Check if validator is active."""
        with self.validator_lock:
            if address not in self.validators:
                return False
            
            validator = self.validators[address]
            return validator.status == ValidatorStatus.ACTIVE
    
    def get_validator_stake(self, address: str) -> int:
        """Get validator stake."""
        with self.validator_lock:
            if address not in self.validators:
                return 0
            
            return self.validators[address].stake


class BFTConsensus:
    """Byzantine Fault Tolerant consensus implementation."""
    
    def __init__(self, validator_manager: ValidatorManager, config: ValidatorConfig):
        self.validator_manager = validator_manager
        self.config = config
        self.active_proposals: Dict[str, BridgeProposal] = {}
        self.consensus_messages: Dict[str, List[ConsensusMessage]] = {}
        self.consensus_lock = threading.Lock()
    
    def propose_transaction(self, transaction: UniversalTransaction, proposer: str) -> str:
        """Propose a transaction for consensus."""
        if not self.validator_manager.is_validator_active(proposer):
            raise BridgeError("Only active validators can propose transactions")
        
        proposal_id = f"proposal_{secrets.token_hex(16)}"
        
        proposal = BridgeProposal(
            proposal_id=proposal_id,
            transaction=transaction,
            proposer=proposer
        )
        
        with self.consensus_lock:
            self.active_proposals[proposal_id] = proposal
            self.consensus_messages[proposal_id] = []
        
        logger.info(f"Transaction proposal {proposal_id} created by {proposer}")
        return proposal_id
    
    def vote_on_proposal(self, proposal_id: str, validator_address: str, 
                        vote: bool, signature: str) -> bool:
        """Vote on a proposal."""
        if not self.validator_manager.is_validator_active(validator_address):
            return False
        
        with self.consensus_lock:
            if proposal_id not in self.active_proposals:
                return False
            
            proposal = self.active_proposals[proposal_id]
            
            # Check if already voted
            if validator_address in proposal.votes:
                return False
            
            # Record vote
            proposal.votes[validator_address] = vote
            
            # Create consensus message
            message = ConsensusMessage(
                message_id=f"msg_{secrets.token_hex(8)}",
                validator_address=validator_address,
                phase=ConsensusPhase.PRE_COMMIT,
                proposal_id=proposal_id,
                vote=vote,
                signature=signature
            )
            
            self.consensus_messages[proposal_id].append(message)
            
            # Check consensus
            self._check_consensus(proposal_id)
            
            return True
    
    def _check_consensus(self, proposal_id: str) -> None:
        """Check if consensus has been reached."""
        proposal = self.active_proposals[proposal_id]
        active_validators = self.validator_manager.get_active_validators()
        
        if len(active_validators) == 0:
            return
        
        # Count votes
        yes_votes = sum(1 for vote in proposal.votes.values() if vote)
        no_votes = sum(1 for vote in proposal.votes.values() if not vote)
        total_votes = len(proposal.votes)
        
        # Check if we have enough votes
        required_votes = int(len(active_validators) * self.config.consensus_threshold)
        
        if total_votes >= required_votes:
            if yes_votes > no_votes:
                proposal.status = "approved"
                proposal.consensus_reached = True
                logger.info(f"Proposal {proposal_id} approved by consensus")
            else:
                proposal.status = "rejected"
                proposal.consensus_reached = True
                logger.info(f"Proposal {proposal_id} rejected by consensus")
    
    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get proposal status."""
        with self.consensus_lock:
            if proposal_id not in self.active_proposals:
                return None
            
            proposal = self.active_proposals[proposal_id]
            active_validators = self.validator_manager.get_active_validators()
            
            return {
                "proposal_id": proposal_id,
                "status": proposal.status,
                "consensus_reached": proposal.consensus_reached,
                "votes": proposal.votes,
                "total_votes": len(proposal.votes),
                "required_votes": int(len(active_validators) * self.config.consensus_threshold),
                "created_at": proposal.created_at
            }
    
    def get_pending_proposals(self) -> List[BridgeProposal]:
        """Get pending proposals."""
        with self.consensus_lock:
            return [p for p in self.active_proposals.values() if p.status == "pending"]


class EmergencyManager:
    """Manages emergency pause functionality."""
    
    def __init__(self, validator_manager: ValidatorManager, config: ValidatorConfig):
        self.validator_manager = validator_manager
        self.config = config
        self.emergency_paused: bool = False
        self.emergency_votes: Dict[str, bool] = {}
        self.emergency_lock = threading.Lock()
    
    def request_emergency_pause(self, validator_address: str, reason: str) -> bool:
        """Request emergency pause."""
        if not self.validator_manager.is_validator_active(validator_address):
            return False
        
        with self.emergency_lock:
            if self.emergency_paused:
                return False
            
            # Record vote
            self.emergency_votes[validator_address] = True
            
            # Check if threshold reached
            active_validators = self.validator_manager.get_active_validators()
            required_votes = int(len(active_validators) * self.config.emergency_pause_threshold)
            
            if len(self.emergency_votes) >= required_votes:
                self.emergency_paused = True
                logger.critical(f"EMERGENCY PAUSE ACTIVATED: {reason}")
                return True
            
            logger.warning(f"Emergency pause vote by {validator_address}: {reason}")
            return True
    
    def request_emergency_resume(self, validator_address: str) -> bool:
        """Request emergency resume."""
        if not self.validator_manager.is_validator_active(validator_address):
            return False
        
        with self.emergency_lock:
            if not self.emergency_paused:
                return False
            
            # Record vote
            self.emergency_votes[validator_address] = False
            
            # Check if threshold reached
            active_validators = self.validator_manager.get_active_validators()
            required_votes = int(len(active_validators) * self.config.consensus_threshold)
            
            resume_votes = sum(1 for vote in self.emergency_votes.values() if not vote)
            
            if resume_votes >= required_votes:
                self.emergency_paused = False
                self.emergency_votes.clear()
                logger.info("Emergency pause lifted")
                return True
            
            return True
    
    def is_emergency_paused(self) -> bool:
        """Check if emergency pause is active."""
        with self.emergency_lock:
            return self.emergency_paused


class BridgeValidatorNetwork:
    """Main bridge validator network implementation."""
    
    def __init__(self, universal_bridge: UniversalBridge, config: ValidatorConfig):
        self.universal_bridge = universal_bridge
        self.config = config
        self.validator_manager = ValidatorManager(config)
        self.consensus = BFTConsensus(self.validator_manager, config)
        self.emergency_manager = EmergencyManager(self.validator_manager, config)
        
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._consensus_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the validator network."""
        if self._running:
            return
        
        self._running = True
        
        # Start heartbeat monitoring
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        # Start consensus processing
        self._consensus_task = asyncio.create_task(self._consensus_processor())
        
        logger.info("Bridge validator network started")
    
    async def stop(self) -> None:
        """Stop the validator network."""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._consensus_task:
            self._consensus_task.cancel()
            try:
                await self._consensus_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Bridge validator network stopped")
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor validator heartbeats."""
        while self._running:
            try:
                current_time = time.time()
                active_validators = self.validator_manager.get_active_validators()
                
                for validator in active_validators:
                    if current_time - validator.last_heartbeat > self.config.heartbeat_timeout:
                        logger.warning(f"Validator {validator.address} heartbeat timeout")
                        # Could implement slashing here
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _consensus_processor(self) -> None:
        """Process consensus proposals."""
        while self._running:
            try:
                # Check for emergency pause
                if self.emergency_manager.is_emergency_paused():
                    await asyncio.sleep(10)
                    continue
                
                # Process approved proposals
                pending_proposals = self.consensus.get_pending_proposals()
                
                for proposal in pending_proposals:
                    # Check timeout
                    if time.time() - proposal.created_at > self.config.consensus_timeout:
                        proposal.status = "rejected"
                        logger.warning(f"Proposal {proposal.proposal_id} timed out")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in consensus processor: {e}")
                await asyncio.sleep(5)
    
    def register_validator(self, address: str, public_key: str, stake: int) -> bool:
        """Register a new validator."""
        return self.validator_manager.register_validator(address, public_key, stake)
    
    def activate_validator(self, address: str) -> bool:
        """Activate a validator."""
        return self.validator_manager.activate_validator(address)
    
    def propose_transaction(self, transaction: UniversalTransaction, proposer: str) -> str:
        """Propose a transaction for consensus."""
        if self.emergency_manager.is_emergency_paused():
            raise BridgeError("Bridge is in emergency pause")
        
        return self.consensus.propose_transaction(transaction, proposer)
    
    def vote_on_proposal(self, proposal_id: str, validator_address: str, 
                        vote: bool, signature: str) -> bool:
        """Vote on a proposal."""
        if self.emergency_manager.is_emergency_paused():
            return False
        
        return self.consensus.vote_on_proposal(proposal_id, validator_address, vote, signature)
    
    def request_emergency_pause(self, validator_address: str, reason: str) -> bool:
        """Request emergency pause."""
        return self.emergency_manager.request_emergency_pause(validator_address, reason)
    
    def request_emergency_resume(self, validator_address: str) -> bool:
        """Request emergency resume."""
        return self.emergency_manager.request_emergency_resume(validator_address)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        active_validators = self.validator_manager.get_active_validators()
        total_stake = sum(v.stake for v in active_validators)
        
        return {
            "total_validators": len(self.validator_manager.validators),
            "active_validators": len(active_validators),
            "total_stake": total_stake,
            "min_stake": self.config.min_stake,
            "max_stake": self.config.max_stake,
            "consensus_threshold": self.config.consensus_threshold,
            "emergency_paused": self.emergency_manager.is_emergency_paused(),
            "pending_proposals": len(self.consensus.get_pending_proposals()),
            "running": self._running
        }
    
    def get_validator_info(self, address: str) -> Optional[Dict[str, Any]]:
        """Get validator information."""
        if address not in self.validator_manager.validators:
            return None
        
        validator = self.validator_manager.validators[address]
        return {
            "address": validator.address,
            "stake": validator.stake,
            "status": validator.status.value,
            "commission_rate": validator.commission_rate,
            "slashing_count": validator.slashing_count,
            "total_rewards": validator.total_rewards,
            "last_heartbeat": validator.last_heartbeat,
            "created_at": validator.created_at
        }
