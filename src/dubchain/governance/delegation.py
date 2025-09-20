"""
Vote delegation system for governance.

This module implements vote delegation with support for delegation chains,
delegation strategies, and security measures against delegation attacks.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.signatures import PrivateKey, PublicKey
from ..errors.exceptions import ValidationError, GovernanceError
from .core import VotingPower, GovernanceConfig


@dataclass
class Delegation:
    """A vote delegation."""
    
    delegator_address: str
    delegatee_address: str
    delegation_power: int
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_active: bool = True
    signature: Optional[str] = None
    
    def __post_init__(self):
        """Validate delegation after initialization."""
        if self.delegator_address == self.delegatee_address:
            raise ValidationError("Cannot delegate to self")
        
        if self.delegation_power <= 0:
            raise ValidationError("Delegation power must be positive")
    
    def is_expired(self) -> bool:
        """Check if delegation has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if delegation is valid."""
        return self.is_active and not self.is_expired()


@dataclass
class DelegationChain:
    """A chain of delegations."""
    
    chain_id: str
    delegator_address: str
    delegatee_address: str
    chain: List[str] = field(default_factory=list)  # List of addresses in chain
    total_power: int = 0
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize delegation chain."""
        if not self.chain:
            self.chain = [self.delegator_address, self.delegatee_address]
    
    def add_delegation(self, address: str) -> None:
        """Add a delegation to the chain."""
        if address not in self.chain:
            self.chain.append(address)
    
    def get_chain_length(self) -> int:
        """Get the length of the delegation chain."""
        return len(self.chain)
    
    def contains_cycle(self) -> bool:
        """Check if the chain contains a cycle."""
        return len(self.chain) != len(set(self.chain))


class DelegationStrategy(ABC):
    """Abstract base class for delegation strategies."""
    
    @abstractmethod
    def calculate_delegated_power(
        self,
        delegator_address: str,
        delegatee_address: str,
        base_power: int,
        delegation_chain: Optional[DelegationChain] = None
    ) -> int:
        """Calculate delegated power based on strategy."""
        pass
    
    @abstractmethod
    def validate_delegation(
        self,
        delegation: Delegation,
        existing_delegations: List[Delegation]
    ) -> bool:
        """Validate a delegation according to this strategy."""
        pass


class SimpleDelegationStrategy(DelegationStrategy):
    """Simple delegation strategy (1:1 power transfer)."""
    
    def calculate_delegated_power(
        self,
        delegator_address: str,
        delegatee_address: str,
        base_power: int,
        delegation_chain: Optional[DelegationChain] = None
    ) -> int:
        """Calculate delegated power (1:1 transfer)."""
        return base_power
    
    def validate_delegation(
        self,
        delegation: Delegation,
        existing_delegations: List[Delegation]
    ) -> bool:
        """Validate delegation (no special rules)."""
        return True


class DecayDelegationStrategy(DelegationStrategy):
    """Delegation strategy with power decay over chain length."""
    
    def __init__(self, decay_factor: float = 0.9):
        """Initialize decay delegation strategy."""
        self.decay_factor = decay_factor
    
    def calculate_delegated_power(
        self,
        delegator_address: str,
        delegatee_address: str,
        base_power: int,
        delegation_chain: Optional[DelegationChain] = None
    ) -> int:
        """Calculate delegated power with decay."""
        if delegation_chain is None:
            return base_power
        
        chain_length = delegation_chain.get_chain_length()
        decayed_power = base_power * (self.decay_factor ** (chain_length - 1))
        return int(decayed_power)
    
    def validate_delegation(
        self,
        delegation: Delegation,
        existing_delegations: List[Delegation]
    ) -> bool:
        """Validate delegation with decay rules."""
        return True


class DelegationManager:
    """Manages vote delegations and delegation chains."""
    
    def __init__(self, config: GovernanceConfig):
        """Initialize delegation manager."""
        self.config = config
        self.delegations: Dict[str, List[Delegation]] = {}  # delegatee -> delegations
        self.delegation_chains: Dict[str, DelegationChain] = {}
        self.delegation_strategy = SimpleDelegationStrategy()
        
        # Security tracking
        self.delegation_history: Dict[str, List[Delegation]] = {}  # address -> history
        self.circular_delegation_detector = CircularDelegationDetector()
    
    def set_delegation_strategy(self, strategy: DelegationStrategy) -> None:
        """Set the delegation strategy."""
        self.delegation_strategy = strategy
    
    def create_delegation(
        self,
        delegator_address: str,
        delegatee_address: str,
        delegation_power: int,
        expires_at: Optional[float] = None,
        signature: Optional[str] = None
    ) -> Delegation:
        """Create a new delegation."""
        if delegator_address == delegatee_address:
            raise ValidationError("Cannot delegate to self")
        
        # Check for circular delegations
        if self.circular_delegation_detector.would_create_cycle(
            delegator_address, delegatee_address, self.delegations
        ):
            raise ValidationError("Delegation would create a circular dependency")
        
        # Create delegation
        delegation = Delegation(
            delegator_address=delegator_address,
            delegatee_address=delegatee_address,
            delegation_power=delegation_power,
            expires_at=expires_at,
            signature=signature,
        )
        
        # Validate delegation
        existing_delegations = self.delegations.get(delegatee_address, [])
        if not self.delegation_strategy.validate_delegation(delegation, existing_delegations):
            raise ValidationError("Delegation validation failed")
        
        # Add delegation
        if delegatee_address not in self.delegations:
            self.delegations[delegatee_address] = []
        self.delegations[delegatee_address].append(delegation)
        
        # Update delegation history
        if delegator_address not in self.delegation_history:
            self.delegation_history[delegator_address] = []
        self.delegation_history[delegator_address].append(delegation)
        
        # Update delegation chains
        self._update_delegation_chains(delegation)
        
        return delegation
    
    def revoke_delegation(
        self,
        delegator_address: str,
        delegatee_address: str
    ) -> bool:
        """Revoke a delegation."""
        if delegatee_address not in self.delegations:
            return False
        
        delegations = self.delegations[delegatee_address]
        for i, delegation in enumerate(delegations):
            if (delegation.delegator_address == delegator_address and 
                delegation.is_valid()):
                delegations[i].is_active = False
                return True
        
        return False
    
    def get_delegated_power(
        self,
        delegatee_address: str,
        current_block: int
    ) -> int:
        """Get total delegated power for an address."""
        if delegatee_address not in self.delegations:
            return 0
        
        total_power = 0
        for delegation in self.delegations[delegatee_address]:
            if delegation.is_valid():
                # Get delegation chain
                chain = self._get_delegation_chain(delegation)
                
                # Calculate delegated power using strategy
                delegated_power = self.delegation_strategy.calculate_delegated_power(
                    delegation.delegator_address,
                    delegation.delegatee_address,
                    delegation.delegation_power,
                    chain
                )
                total_power += delegated_power
        
        return total_power
    
    def get_delegation_chain(
        self,
        delegator_address: str,
        delegatee_address: str
    ) -> Optional[DelegationChain]:
        """Get delegation chain between two addresses."""
        # First check for direct chain
        chain_key = f"{delegator_address}:{delegatee_address}"
        direct_chain = self.delegation_chains.get(chain_key)
        if direct_chain:
            return direct_chain
        
        # If no direct chain, try to find transitive chain
        return self._find_transitive_delegation_chain(delegator_address, delegatee_address)
    
    def _find_transitive_delegation_chain(
        self,
        delegator_address: str,
        delegatee_address: str
    ) -> Optional[DelegationChain]:
        """Find transitive delegation chain using BFS."""
        if delegator_address == delegatee_address:
            return None
        
        # Build adjacency list for valid delegations
        adjacency = {}
        for delegatee, delegations in self.delegations.items():
            for delegation in delegations:
                if delegation.is_valid():
                    if delegation.delegator_address not in adjacency:
                        adjacency[delegation.delegator_address] = []
                    adjacency[delegation.delegator_address].append(delegation.delegatee_address)
        
        # BFS to find path
        queue = [(delegator_address, [delegator_address])]
        visited = {delegator_address}
        
        while queue:
            current_address, path = queue.pop(0)
            
            if current_address in adjacency:
                for next_address in adjacency[current_address]:
                    if next_address == delegatee_address:
                        # Found the path, create delegation chain
                        chain = DelegationChain(
                            chain_id=f"transitive_{delegator_address}_{delegatee_address}",
                            delegator_address=delegator_address,
                            delegatee_address=delegatee_address,
                            chain=path + [delegatee_address]
                        )
                        return chain
                    
                    if next_address not in visited:
                        visited.add(next_address)
                        queue.append((next_address, path + [next_address]))
        
        return None
    
    def get_all_delegations_for_address(self, address: str) -> List[Delegation]:
        """Get all delegations for an address (both as delegator and delegatee)."""
        delegations = []
        
        # Delegations where address is the delegatee
        if address in self.delegations:
            delegations.extend(self.delegations[address])
        
        # Delegations where address is the delegator
        for delegatee, dels in self.delegations.items():
            for delegation in dels:
                if delegation.delegator_address == address:
                    delegations.append(delegation)
        
        return delegations
    
    def _update_delegation_chains(self, delegation: Delegation) -> None:
        """Update delegation chains after adding a delegation."""
        chain_key = f"{delegation.delegator_address}:{delegation.delegatee_address}"
        
        # Create or update delegation chain
        if chain_key not in self.delegation_chains:
            chain = DelegationChain(
                chain_id=chain_key,
                delegator_address=delegation.delegator_address,
                delegatee_address=delegation.delegatee_address,
            )
            self.delegation_chains[chain_key] = chain
        else:
            chain = self.delegation_chains[chain_key]
        
        # Update chain power
        chain.total_power += delegation.delegation_power
    
    def _get_delegation_chain(self, delegation: Delegation) -> Optional[DelegationChain]:
        """Get delegation chain for a delegation."""
        chain_key = f"{delegation.delegator_address}:{delegation.delegatee_address}"
        return self.delegation_chains.get(chain_key)
    
    def cleanup_expired_delegations(self) -> int:
        """Clean up expired delegations."""
        cleaned_count = 0
        
        for delegatee, delegations in self.delegations.items():
            for delegation in delegations:
                if delegation.is_expired():
                    delegation.is_active = False
                    cleaned_count += 1
        
        return cleaned_count
    
    def get_delegation_statistics(self) -> Dict[str, Any]:
        """Get delegation statistics."""
        total_delegations = 0
        active_delegations = 0
        expired_delegations = 0
        total_delegated_power = 0
        
        for delegations in self.delegations.values():
            for delegation in delegations:
                total_delegations += 1
                if delegation.is_valid():
                    active_delegations += 1
                    total_delegated_power += delegation.delegation_power
                elif delegation.is_expired():
                    expired_delegations += 1
        
        return {
            "total_delegations": total_delegations,
            "active_delegations": active_delegations,
            "expired_delegations": expired_delegations,
            "total_delegated_power": total_delegated_power,
            "unique_delegators": len(set(
                d.delegator_address for delegations in self.delegations.values()
                for d in delegations
            )),
            "unique_delegatees": len(self.delegations),
        }


class CircularDelegationDetector:
    """Detects circular delegations to prevent infinite loops."""
    
    def would_create_cycle(
        self,
        delegator_address: str,
        delegatee_address: str,
        existing_delegations: Dict[str, List[Delegation]]
    ) -> bool:
        """Check if adding a delegation would create a cycle."""
        # Handle self-delegation case
        if delegator_address == delegatee_address:
            return True
        
        # Create a copy of existing delegations to avoid modifying the original
        temp_delegations = {}
        for addr, dels in existing_delegations.items():
            temp_delegations[addr] = dels.copy()
        
        # Add the new delegation temporarily
        # The delegations dict is keyed by delegatee, so we add to the delegatee's list
        if delegatee_address not in temp_delegations:
            temp_delegations[delegatee_address] = []
        
        # Create a mock delegation object for cycle detection
        class MockDelegation:
            def __init__(self, delegator, delegatee):
                self.delegator_address = delegator
                self.delegatee_address = delegatee
        
        temp_delegation = MockDelegation(delegator_address, delegatee_address)
        temp_delegations[delegatee_address].append(temp_delegation)
        
        # Build a reverse mapping for cycle detection (delegator -> list of delegatees)
        delegator_to_delegatees = {}
        for delegatee, delegations_list in temp_delegations.items():
            for delegation in delegations_list:
                delegator = delegation.delegator_address
                if delegator not in delegator_to_delegatees:
                    delegator_to_delegatees[delegator] = []
                delegator_to_delegatees[delegator].append(delegation)
        
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(address: str) -> bool:
            if address in rec_stack:
                return True
            if address in visited:
                return False
                
            visited.add(address)
            rec_stack.add(address)
            
            # Check all delegations from this address using the reverse mapping
            if address in delegator_to_delegatees:
                for delegation in delegator_to_delegatees[address]:
                    if hasattr(delegation, 'is_valid') and delegation.is_valid():
                        next_address = delegation.delegatee_address
                        if has_cycle(next_address):
                            return True
                    elif hasattr(delegation, 'delegatee_address') and not hasattr(delegation, 'is_valid'):
                        # Handle mock delegation objects (they don't have is_valid method)
                        next_address = delegation.delegatee_address
                        if has_cycle(next_address):
                            return True
            
            rec_stack.remove(address)
            return False
        
        # Check for cycles starting from the delegator
        return has_cycle(delegator_address)
