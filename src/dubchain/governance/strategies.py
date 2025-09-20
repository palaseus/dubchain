"""
Voting strategies for governance proposals.

This module implements various voting strategies including token-weighted,
quadratic voting, conviction voting, and snapshot-based voting.
"""

import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..errors.exceptions import ValidationError, GovernanceError
from .core import Vote, VoteChoice, VotingPower, Proposal


class VotingStrategy(ABC):
    """Abstract base class for voting strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize voting strategy with configuration."""
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate_voting_power(
        self,
        voter_address: str,
        token_balance: int,
        delegated_power: int = 0,
        **kwargs
    ) -> VotingPower:
        """Calculate voting power for a voter."""
        pass
    
    @abstractmethod
    def validate_vote(self, vote: Vote, proposal: Proposal) -> bool:
        """Validate a vote according to this strategy."""
        pass
    
    @abstractmethod
    def calculate_proposal_result(
        self,
        proposal: Proposal,
        votes: List[Vote]
    ) -> Dict[str, Any]:
        """Calculate the result of a proposal using this strategy."""
        pass
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this voting strategy."""
        return {
            "name": self.name,
            "config": self.config,
            "description": self.__doc__ or "No description available",
        }


class TokenWeightedStrategy(VotingStrategy):
    """Token-weighted voting strategy (1 token = 1 vote)."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize token-weighted strategy."""
        super().__init__(config or {})
        self.min_token_balance = self.config.get("min_token_balance", 1)
        self.max_voting_power = self.config.get("max_voting_power", None)
    
    def calculate_voting_power(
        self,
        voter_address: str,
        token_balance: int,
        delegated_power: int = 0,
        **kwargs
    ) -> VotingPower:
        """Calculate voting power based on token balance."""
        if token_balance < self.min_token_balance:
            power = 0
        else:
            power = token_balance
        
        # Apply maximum voting power limit if configured
        if self.max_voting_power:
            power = min(power, self.max_voting_power)
        
        return VotingPower(
            voter_address=voter_address,
            power=power,
            token_balance=token_balance,
            delegated_power=delegated_power,
        )
    
    def validate_vote(self, vote: Vote, proposal: Proposal) -> bool:
        """Validate a vote for token-weighted strategy."""
        # Check if voter has minimum token balance
        if vote.voting_power.token_balance < self.min_token_balance:
            return False
        
        # Check if voting power is within limits
        if self.max_voting_power and vote.voting_power.power > self.max_voting_power:
            return False
        
        return True
    
    def calculate_proposal_result(
        self,
        proposal: Proposal,
        votes: List[Vote]
    ) -> Dict[str, Any]:
        """Calculate proposal result using token-weighted voting."""
        total_power = 0
        for_power = 0
        against_power = 0
        abstain_power = 0
        
        for vote in votes:
            power = vote.voting_power.total_power()
            total_power += power
            
            if vote.choice == VoteChoice.FOR:
                for_power += power
            elif vote.choice == VoteChoice.AGAINST:
                against_power += power
            elif vote.choice == VoteChoice.ABSTAIN:
                abstain_power += power
        
        return {
            "strategy": "token_weighted",
            "total_voting_power": total_power,
            "for_power": for_power,
            "against_power": against_power,
            "abstain_power": abstain_power,
            "total_votes": len(votes),
            "quorum_met": total_power >= proposal.quorum_threshold,
            "approval_percentage": for_power / max(total_power, 1),
            "approved": (
                total_power >= proposal.quorum_threshold and 
                for_power / max(total_power, 1) >= proposal.approval_threshold
            ),
        }


class QuadraticVotingStrategy(VotingStrategy):
    """Quadratic voting strategy (voting power = sqrt(token_balance))."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize quadratic voting strategy."""
        super().__init__(config or {})
        self.min_token_balance = self.config.get("min_token_balance", 1)
        self.max_votes_per_proposal = self.config.get("max_votes_per_proposal", 100)
        self.vote_cost_multiplier = self.config.get("vote_cost_multiplier", 1.0)
    
    def calculate_voting_power(
        self,
        voter_address: str,
        token_balance: int,
        delegated_power: int = 0,
        **kwargs
    ) -> VotingPower:
        """Calculate voting power using quadratic formula."""
        if token_balance < self.min_token_balance:
            power = 0
        else:
            # Quadratic voting: power = sqrt(token_balance)
            power = int(math.sqrt(token_balance))
        
        return VotingPower(
            voter_address=voter_address,
            power=power,
            token_balance=token_balance,
            delegated_power=delegated_power,
        )
    
    def validate_vote(self, vote: Vote, proposal: Proposal) -> bool:
        """Validate a vote for quadratic voting strategy."""
        # Check if voter has minimum token balance
        if vote.voting_power.token_balance < self.min_token_balance:
            return False
        
        # Check if voting power is within limits
        if vote.voting_power.power > self.max_votes_per_proposal:
            return False
        
        return True
    
    def calculate_proposal_result(
        self,
        proposal: Proposal,
        votes: List[Vote]
    ) -> Dict[str, Any]:
        """Calculate proposal result using quadratic voting."""
        total_power = 0
        for_power = 0
        against_power = 0
        abstain_power = 0
        
        for vote in votes:
            power = vote.voting_power.total_power()
            total_power += power
            
            if vote.choice == VoteChoice.FOR:
                for_power += power
            elif vote.choice == VoteChoice.AGAINST:
                against_power += power
            elif vote.choice == VoteChoice.ABSTAIN:
                abstain_power += power
        
        return {
            "strategy": "quadratic_voting",
            "total_voting_power": total_power,
            "for_power": for_power,
            "against_power": against_power,
            "abstain_power": abstain_power,
            "total_votes": len(votes),
            "quorum_met": total_power >= proposal.quorum_threshold,
            "approval_percentage": for_power / max(total_power, 1),
            "approved": (
                total_power >= proposal.quorum_threshold and 
                for_power / max(total_power, 1) >= proposal.approval_threshold
            ),
        }


@dataclass
class ConvictionVotingConfig:
    """Configuration for conviction voting."""
    
    max_conviction: float = 1.0
    conviction_growth_rate: float = 0.1
    conviction_decay_rate: float = 0.05
    min_conviction_threshold: float = 0.1
    max_voting_period: int = 1000  # blocks


class ConvictionVotingStrategy(VotingStrategy):
    """Conviction voting strategy with time-based conviction accumulation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize conviction voting strategy."""
        super().__init__(config or {})
        self.conviction_config = ConvictionVotingConfig(**self.config)
        self.conviction_snapshots: Dict[str, Dict[str, float]] = {}  # voter -> proposal -> conviction
    
    def calculate_voting_power(
        self,
        voter_address: str,
        token_balance: int,
        delegated_power: int = 0,
        conviction: float = 0.0,
        **kwargs
    ) -> VotingPower:
        """Calculate voting power based on conviction."""
        # Base power is token balance
        base_power = token_balance
        
        # Apply conviction multiplier
        conviction_multiplier = 1.0 + conviction
        power = int(base_power * conviction_multiplier)
        
        return VotingPower(
            voter_address=voter_address,
            power=power,
            token_balance=token_balance,
            delegated_power=delegated_power,
        )
    
    def update_conviction(
        self,
        voter_address: str,
        proposal_id: str,
        current_block: int,
        has_voted: bool
    ) -> float:
        """Update conviction for a voter on a proposal."""
        if voter_address not in self.conviction_snapshots:
            self.conviction_snapshots[voter_address] = {}
        
        if proposal_id not in self.conviction_snapshots[voter_address]:
            self.conviction_snapshots[voter_address][proposal_id] = 0.0
        
        current_conviction = self.conviction_snapshots[voter_address][proposal_id]
        
        if has_voted:
            # Conviction grows when actively voting
            new_conviction = min(
                current_conviction + self.conviction_config.conviction_growth_rate,
                self.conviction_config.max_conviction
            )
        else:
            # Conviction decays when not voting
            new_conviction = max(
                current_conviction - self.conviction_config.conviction_decay_rate,
                0.0
            )
        
        self.conviction_snapshots[voter_address][proposal_id] = new_conviction
        return round(new_conviction, 10)  # Round to avoid floating point precision issues
    
    def validate_vote(self, vote: Vote, proposal: Proposal) -> bool:
        """Validate a vote for conviction voting strategy."""
        # Get current conviction
        conviction = self.conviction_snapshots.get(vote.voter_address, {}).get(proposal.proposal_id, 0.0)
        
        # Check minimum conviction threshold
        if conviction < self.conviction_config.min_conviction_threshold:
            return False
        
        return True
    
    def calculate_proposal_result(
        self,
        proposal: Proposal,
        votes: List[Vote]
    ) -> Dict[str, Any]:
        """Calculate proposal result using conviction voting."""
        total_power = 0
        for_power = 0
        against_power = 0
        abstain_power = 0
        
        for vote in votes:
            power = vote.voting_power.total_power()
            total_power += power
            
            if vote.choice == VoteChoice.FOR:
                for_power += power
            elif vote.choice == VoteChoice.AGAINST:
                against_power += power
            elif vote.choice == VoteChoice.ABSTAIN:
                abstain_power += power
        
        return {
            "strategy": "conviction_voting",
            "total_voting_power": total_power,
            "for_power": for_power,
            "against_power": against_power,
            "abstain_power": abstain_power,
            "total_votes": len(votes),
            "quorum_met": total_power >= proposal.quorum_threshold,
            "approval_percentage": for_power / max(total_power, 1),
            "approved": (
                total_power >= proposal.quorum_threshold and 
                for_power / max(total_power, 1) >= proposal.approval_threshold
            ),
        }


@dataclass
class SnapshotVotingConfig:
    """Configuration for snapshot voting."""
    
    snapshot_block_offset: int = 100  # Blocks before proposal start
    merkle_tree_depth: int = 20
    signature_verification: bool = True
    off_chain_voting_enabled: bool = True


class SnapshotVotingStrategy(VotingStrategy):
    """Snapshot-based voting strategy with off-chain voting support."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize snapshot voting strategy."""
        super().__init__(config or {})
        self.snapshot_config = SnapshotVotingConfig(**self.config)
        self.snapshots: Dict[int, Dict[str, int]] = {}  # block -> address -> balance
        self.merkle_trees: Dict[int, str] = {}  # block -> merkle root
    
    def create_snapshot(self, block_height: int, balances: Dict[str, int]) -> str:
        """Create a voting power snapshot at a specific block."""
        self.snapshots[block_height] = balances.copy()
        
        # Create Merkle tree for snapshot
        merkle_root = self._create_merkle_tree(balances)
        self.merkle_trees[block_height] = merkle_root
        
        return merkle_root
    
    def _create_merkle_tree(self, balances: Dict[str, int]) -> str:
        """Create Merkle tree from balance snapshot."""
        # Sort addresses for deterministic ordering
        sorted_addresses = sorted(balances.keys())
        
        # Create leaf nodes (address + balance hash)
        leaves = []
        for address in sorted_addresses:
            balance = balances[address]
            leaf_data = f"{address}:{balance}"
            leaf_hash = SHA256Hasher.hash(leaf_data.encode())
            leaves.append(str(leaf_hash))
        
        # Build Merkle tree
        if not leaves:
            return str(SHA256Hasher.hash(b"empty"))
        
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = SHA256Hasher.hash((left + right).encode())
                next_level.append(str(combined))
            current_level = next_level
        
        return current_level[0]
    
    def generate_merkle_proof(
        self,
        block_height: int,
        address: str
    ) -> Optional[Dict[str, Any]]:
        """Generate Merkle proof for an address in a snapshot."""
        if block_height not in self.snapshots:
            return None
        
        balances = self.snapshots[block_height]
        if address not in balances:
            return None
        
        # Create proof path
        sorted_addresses = sorted(balances.keys())
        address_index = sorted_addresses.index(address)
        
        # Build proof path
        proof_path = []
        current_level = []
        
        # Create leaf nodes
        for addr in sorted_addresses:
            balance = balances[addr]
            leaf_data = f"{addr}:{balance}"
            leaf_hash = SHA256Hasher.hash(leaf_data.encode())
            current_level.append(leaf_hash)
        
        # Build tree and collect proof
        level_index = address_index
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = SHA256Hasher.hash((str(left) + str(right)).encode())
                next_level.append(combined)
                
                # Add to proof if this is our path
                if i == level_index or i + 1 == level_index:
                    sibling = right if i == level_index else left
                    proof_path.append(sibling)
            
            level_index = level_index // 2
            current_level = next_level
        
        return {
            "block_height": block_height,
            "address": address,
            "balance": balances[address],
            "merkle_root": self.merkle_trees[block_height],
            "proof_path": proof_path,
            "leaf_index": address_index,
        }
    
    def verify_merkle_proof(self, proof: Dict[str, Any]) -> bool:
        """Verify a Merkle proof."""
        address = proof["address"]
        balance = proof["balance"]
        merkle_root = proof["merkle_root"]
        proof_path = proof["proof_path"]
        leaf_index = proof["leaf_index"]
        
        # Recreate leaf hash
        leaf_data = f"{address}:{balance}"
        leaf_hash = SHA256Hasher.hash(leaf_data.encode())
        
        # Verify proof path
        current_hash = leaf_hash
        current_index = leaf_index
        
        for sibling_hash in proof_path:
            if current_index % 2 == 0:
                # Current is left, sibling is right
                current_hash = SHA256Hasher.hash((str(current_hash) + str(sibling_hash)).encode())
            else:
                # Current is right, sibling is left
                current_hash = SHA256Hasher.hash((str(sibling_hash) + str(current_hash)).encode())
            current_index = current_index // 2
        
        return str(current_hash) == merkle_root
    
    def calculate_voting_power(
        self,
        voter_address: str,
        token_balance: int,
        delegated_power: int = 0,
        snapshot_block: Optional[int] = None,
        **kwargs
    ) -> VotingPower:
        """Calculate voting power based on snapshot."""
        if snapshot_block and snapshot_block in self.snapshots:
            # Use snapshot balance
            snapshot_balance = self.snapshots[snapshot_block].get(voter_address, 0)
            power = snapshot_balance
        else:
            # Use current balance
            power = token_balance
        
        return VotingPower(
            voter_address=voter_address,
            power=power,
            token_balance=token_balance,
            delegated_power=delegated_power,
        )
    
    def validate_vote(self, vote: Vote, proposal: Proposal) -> bool:
        """Validate a vote for snapshot voting strategy."""
        # For snapshot voting, we need to verify the snapshot proof
        # This would typically be done with off-chain signatures
        return True
    
    def calculate_proposal_result(
        self,
        proposal: Proposal,
        votes: List[Vote]
    ) -> Dict[str, Any]:
        """Calculate proposal result using snapshot voting."""
        total_power = 0
        for_power = 0
        against_power = 0
        abstain_power = 0
        
        for vote in votes:
            power = vote.voting_power.total_power()
            total_power += power
            
            if vote.choice == VoteChoice.FOR:
                for_power += power
            elif vote.choice == VoteChoice.AGAINST:
                against_power += power
            elif vote.choice == VoteChoice.ABSTAIN:
                abstain_power += power
        
        return {
            "strategy": "snapshot_voting",
            "total_voting_power": total_power,
            "for_power": for_power,
            "against_power": against_power,
            "abstain_power": abstain_power,
            "total_votes": len(votes),
            "quorum_met": total_power >= proposal.quorum_threshold,
            "approval_percentage": for_power / max(total_power, 1),
            "approved": (
                total_power >= proposal.quorum_threshold and 
                for_power / max(total_power, 1) >= proposal.approval_threshold
            ),
        }
    
    def create_vote_merkle_tree(self, proposal_id: str, votes: List[Vote]) -> str:
        """Create Merkle tree for proposal votes."""
        # Create vote data for Merkle tree
        vote_data = {}
        for vote in votes:
            vote_key = f"{vote.voter_address}:{vote.choice.value}:{vote.voting_power.total_power()}"
            vote_data[vote_key] = vote.voting_power.total_power()
        
        merkle_root = self._create_merkle_tree(vote_data)
        # Store the merkle tree for this proposal
        self.merkle_trees[f"votes_{proposal_id}"] = merkle_root
        return merkle_root
    
    def generate_vote_proof(self, proposal_id: str, votes: List[Vote], target_vote: Vote) -> Dict[str, Any]:
        """Generate Merkle proof for a specific vote."""
        # Create vote data for Merkle tree
        vote_data = {}
        for vote in votes:
            vote_key = f"{vote.voter_address}:{vote.choice.value}:{vote.voting_power.total_power()}"
            vote_data[vote_key] = vote.voting_power.total_power()
        
        # Create a temporary snapshot for this vote data
        temp_block_height = 999999  # Use a high number to avoid conflicts
        self.snapshots[temp_block_height] = vote_data
        merkle_root = self._create_merkle_tree(vote_data)
        self.merkle_trees[temp_block_height] = merkle_root
        
        target_key = f"{target_vote.voter_address}:{target_vote.choice.value}:{target_vote.voting_power.total_power()}"
        proof = self.generate_merkle_proof(temp_block_height, target_key)
        
        # Clean up temporary data
        del self.snapshots[temp_block_height]
        del self.merkle_trees[temp_block_height]
        
        return proof


class StrategyFactory:
    """Factory for creating voting strategies."""
    
    _strategies = {
        "token_weighted": TokenWeightedStrategy,
        "quadratic_voting": QuadraticVotingStrategy,
        "conviction_voting": ConvictionVotingStrategy,
        "snapshot_voting": SnapshotVotingStrategy,
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any] = None) -> VotingStrategy:
        """Create a voting strategy by name."""
        if strategy_name not in cls._strategies:
            raise ValidationError(f"Unknown voting strategy: {strategy_name}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config or {})
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available voting strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """Register a new voting strategy."""
        if not issubclass(strategy_class, VotingStrategy):
            raise ValidationError("Strategy must inherit from VotingStrategy")
        
        cls._strategies[name] = strategy_class
