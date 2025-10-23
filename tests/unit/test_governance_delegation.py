"""
Unit tests for governance delegation system.

This module tests delegation management, delegation chains,
delegation strategies, and circular delegation detection.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
from unittest.mock import Mock, patch

from dubchain.governance.delegation import (
    Delegation,
    DelegationChain,
    DelegationStrategy,
    SimpleDelegationStrategy,
    DecayDelegationStrategy,
    DelegationManager,
    CircularDelegationDetector,
)
from dubchain.governance.core import GovernanceConfig
from dubchain.errors.exceptions import ValidationError


class TestDelegation:
    """Test Delegation class."""
    
    def test_delegation_creation(self):
        """Test creating a delegation."""
        delegation = Delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        assert delegation.delegator_address == "0x123"
        assert delegation.delegatee_address == "0x456"
        assert delegation.delegation_power == 1000
        assert delegation.is_active is True
        assert delegation.expires_at is None
        assert delegation.is_valid() is True
    
    def test_delegation_creation_with_expiry(self):
        """Test creating a delegation with expiry."""
        expiry_time = time.time() + 3600  # 1 hour from now
        
        delegation = Delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000,
            expires_at=expiry_time
        )
        
        assert delegation.expires_at == expiry_time
        assert delegation.is_expired() is False
        assert delegation.is_valid() is True
    
    def test_delegation_validation_self_delegation(self):
        """Test that self-delegation raises validation error."""
        with pytest.raises(ValidationError):
            Delegation(
                delegator_address="0x123",
                delegatee_address="0x123",
                delegation_power=1000
            )
    
    def test_delegation_validation_negative_power(self):
        """Test that negative delegation power raises validation error."""
        with pytest.raises(ValidationError):
            Delegation(
                delegator_address="0x123",
                delegatee_address="0x456",
                delegation_power=-100
            )
    
    def test_delegation_expiry(self):
        """Test delegation expiry functionality."""
        past_time = time.time() - 3600  # 1 hour ago
        
        delegation = Delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000,
            expires_at=past_time
        )
        
        assert delegation.is_expired() is True
        assert delegation.is_valid() is False
    
    def test_delegation_inactive(self):
        """Test inactive delegation."""
        delegation = Delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        delegation.is_active = False
        assert delegation.is_valid() is False


class TestDelegationChain:
    """Test DelegationChain class."""
    
    def test_delegation_chain_creation(self):
        """Test creating a delegation chain."""
        chain = DelegationChain(
            chain_id="chain_123",
            delegator_address="0x123",
            delegatee_address="0x456"
        )
        
        assert chain.chain_id == "chain_123"
        assert chain.delegator_address == "0x123"
        assert chain.delegatee_address == "0x456"
        assert chain.chain == ["0x123", "0x456"]
        assert chain.total_power == 0
        assert chain.get_chain_length() == 2
    
    def test_delegation_chain_add_delegation(self):
        """Test adding delegations to a chain."""
        chain = DelegationChain(
            chain_id="chain_123",
            delegator_address="0x123",
            delegatee_address="0x456"
        )
        
        chain.add_delegation("0x789")
        assert chain.chain == ["0x123", "0x456", "0x789"]
        assert chain.get_chain_length() == 3
    
    def test_delegation_chain_duplicate_address(self):
        """Test adding duplicate address to chain."""
        chain = DelegationChain(
            chain_id="chain_123",
            delegator_address="0x123",
            delegatee_address="0x456"
        )
        
        chain.add_delegation("0x123")  # Duplicate
        assert chain.chain == ["0x123", "0x456"]  # Should not add duplicate
        assert chain.get_chain_length() == 2
    
    def test_delegation_chain_contains_cycle(self):
        """Test cycle detection in delegation chain."""
        chain = DelegationChain(
            chain_id="chain_123",
            delegator_address="0x123",
            delegatee_address="0x456"
        )
        
        # No cycle initially
        assert chain.contains_cycle() is False
        
        # Add addresses to create a cycle: 0x123 -> 0x456 -> 0x789 -> 0x123
        chain.add_delegation("0x789")
        # Manually add duplicate to create cycle (bypassing add_delegation's duplicate check)
        chain.chain.append("0x123")
        assert chain.contains_cycle() is True


class TestSimpleDelegationStrategy:
    """Test SimpleDelegationStrategy class."""
    
    def test_simple_delegation_strategy_creation(self):
        """Test creating simple delegation strategy."""
        strategy = SimpleDelegationStrategy()
        assert isinstance(strategy, DelegationStrategy)
    
    def test_calculate_delegated_power(self):
        """Test calculating delegated power with simple strategy."""
        strategy = SimpleDelegationStrategy()
        
        power = strategy.calculate_delegated_power(
            delegator_address="0x123",
            delegatee_address="0x456",
            base_power=1000
        )
        
        assert power == 1000  # 1:1 transfer
    
    def test_calculate_delegated_power_with_chain(self):
        """Test calculating delegated power with delegation chain."""
        strategy = SimpleDelegationStrategy()
        
        chain = DelegationChain(
            chain_id="chain_123",
            delegator_address="0x123",
            delegatee_address="0x456"
        )
        chain.add_delegation("0x789")
        
        power = strategy.calculate_delegated_power(
            delegator_address="0x123",
            delegatee_address="0x456",
            base_power=1000,
            delegation_chain=chain
        )
        
        assert power == 1000  # Still 1:1 transfer regardless of chain length
    
    def test_validate_delegation(self):
        """Test validating delegation with simple strategy."""
        strategy = SimpleDelegationStrategy()
        
        delegation = Delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        assert strategy.validate_delegation(delegation, []) is True


class TestDecayDelegationStrategy:
    """Test DecayDelegationStrategy class."""
    
    def test_decay_delegation_strategy_creation(self):
        """Test creating decay delegation strategy."""
        strategy = DecayDelegationStrategy(decay_factor=0.8)
        assert isinstance(strategy, DelegationStrategy)
        assert strategy.decay_factor == 0.8
    
    def test_calculate_delegated_power_no_chain(self):
        """Test calculating delegated power without delegation chain."""
        strategy = DecayDelegationStrategy(decay_factor=0.9)
        
        power = strategy.calculate_delegated_power(
            delegator_address="0x123",
            delegatee_address="0x456",
            base_power=1000
        )
        
        assert power == 1000  # No decay without chain
    
    def test_calculate_delegated_power_with_chain(self):
        """Test calculating delegated power with delegation chain."""
        strategy = DecayDelegationStrategy(decay_factor=0.9)
        
        chain = DelegationChain(
            chain_id="chain_123",
            delegator_address="0x123",
            delegatee_address="0x456"
        )
        chain.add_delegation("0x789")
        chain.add_delegation("0xabc")
        
        power = strategy.calculate_delegated_power(
            delegator_address="0x123",
            delegatee_address="0x456",
            base_power=1000,
            delegation_chain=chain
        )
        
        # Chain length is 4, so decay is 0.9^(4-1) = 0.9^3 = 0.729
        expected_power = int(1000 * (0.9 ** 3))
        assert power == expected_power
    
    def test_calculate_delegated_power_strong_decay(self):
        """Test calculating delegated power with strong decay."""
        strategy = DecayDelegationStrategy(decay_factor=0.5)
        
        chain = DelegationChain(
            chain_id="chain_123",
            delegator_address="0x123",
            delegatee_address="0x456"
        )
        chain.add_delegation("0x789")
        
        power = strategy.calculate_delegated_power(
            delegator_address="0x123",
            delegatee_address="0x456",
            base_power=1000,
            delegation_chain=chain
        )
        
        # Chain length is 3, so decay is 0.5^(3-1) = 0.5^2 = 0.25
        expected_power = int(1000 * (0.5 ** 2))
        assert power == expected_power


class TestDelegationManager:
    """Test DelegationManager class."""
    
    def test_delegation_manager_creation(self):
        """Test creating delegation manager."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        assert manager.config == config
        assert isinstance(manager.delegation_strategy, SimpleDelegationStrategy)
        assert len(manager.delegations) == 0
        assert len(manager.delegation_chains) == 0
    
    def test_set_delegation_strategy(self):
        """Test setting delegation strategy."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        decay_strategy = DecayDelegationStrategy(decay_factor=0.8)
        manager.set_delegation_strategy(decay_strategy)
        
        assert manager.delegation_strategy == decay_strategy
    
    def test_create_delegation(self):
        """Test creating a delegation."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        delegation = manager.create_delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        assert delegation.delegator_address == "0x123"
        assert delegation.delegatee_address == "0x456"
        assert delegation.delegation_power == 1000
        
        # Check that delegation was added
        assert "0x456" in manager.delegations
        assert len(manager.delegations["0x456"]) == 1
        assert manager.delegations["0x456"][0] == delegation
        
        # Check delegation history
        assert "0x123" in manager.delegation_history
        assert len(manager.delegation_history["0x123"]) == 1
    
    def test_create_delegation_self_delegation(self):
        """Test that self-delegation raises validation error."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        with pytest.raises(ValidationError):
            manager.create_delegation(
                delegator_address="0x123",
                delegatee_address="0x123",
                delegation_power=1000
            )
    
    def test_revoke_delegation(self):
        """Test revoking a delegation."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        # Create delegation
        delegation = manager.create_delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        # Revoke delegation
        success = manager.revoke_delegation("0x123", "0x456")
        
        assert success is True
        assert delegation.is_active is False
    
    def test_revoke_nonexistent_delegation(self):
        """Test revoking nonexistent delegation."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        success = manager.revoke_delegation("0x123", "0x456")
        assert success is False
    
    def test_get_delegated_power(self):
        """Test getting delegated power for an address."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        # Create delegation
        manager.create_delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        # Get delegated power
        power = manager.get_delegated_power("0x456", 100)
        
        assert power == 1000
    
    def test_get_delegated_power_no_delegations(self):
        """Test getting delegated power with no delegations."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        power = manager.get_delegated_power("0x456", 100)
        assert power == 0
    
    def test_get_delegated_power_expired_delegation(self):
        """Test getting delegated power with expired delegation."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        # Create delegation with expiry
        past_time = time.time() - 3600
        delegation = Delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000,
            expires_at=past_time
        )
        
        manager.delegations["0x456"] = [delegation]
        
        power = manager.get_delegated_power("0x456", 100)
        assert power == 0  # Expired delegation should not count
    
    def test_get_delegation_chain(self):
        """Test getting delegation chain."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        # Create delegation
        manager.create_delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000
        )
        
        chain = manager.get_delegation_chain("0x123", "0x456")
        
        assert chain is not None
        assert chain.delegator_address == "0x123"
        assert chain.delegatee_address == "0x456"
    
    def test_get_all_delegations_for_address(self):
        """Test getting all delegations for an address."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        # Create delegations
        manager.create_delegation("0x123", "0x456", 1000)
        manager.create_delegation("0x123", "0x789", 500)
        manager.create_delegation("0x456", "0xabc", 200)  # Changed to avoid cycle
        
        # Get delegations for 0x123
        delegations = manager.get_all_delegations_for_address("0x123")
        
        assert len(delegations) == 2  # 2 as delegator, 0 as delegatee
    
    def test_cleanup_expired_delegations(self):
        """Test cleaning up expired delegations."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        # Create delegation with expiry
        past_time = time.time() - 3600
        delegation = Delegation(
            delegator_address="0x123",
            delegatee_address="0x456",
            delegation_power=1000,
            expires_at=past_time
        )
        
        manager.delegations["0x456"] = [delegation]
        
        # Cleanup
        cleaned_count = manager.cleanup_expired_delegations()
        
        assert cleaned_count == 1
        assert delegation.is_active is False
    
    def test_get_delegation_statistics(self):
        """Test getting delegation statistics."""
        config = GovernanceConfig()
        manager = DelegationManager(config)
        
        # Create delegations
        manager.create_delegation("0x123", "0x456", 1000)
        manager.create_delegation("0x789", "0x456", 500)
        
        # Create expired delegation
        past_time = time.time() - 3600
        expired_delegation = Delegation(
            delegator_address="0xabc",
            delegatee_address="0xdef",
            delegation_power=200,
            expires_at=past_time
        )
        manager.delegations["0xdef"] = [expired_delegation]
        
        stats = manager.get_delegation_statistics()
        
        assert stats["total_delegations"] == 3
        assert stats["active_delegations"] == 2
        assert stats["expired_delegations"] == 1
        assert stats["total_delegated_power"] == 1500
        assert stats["unique_delegators"] == 3
        assert stats["unique_delegatees"] == 2


class TestCircularDelegationDetector:
    """Test CircularDelegationDetector class."""
    
    def test_circular_delegation_detector_creation(self):
        """Test creating circular delegation detector."""
        detector = CircularDelegationDetector()
        assert isinstance(detector, CircularDelegationDetector)
    
    def test_no_cycle_detection(self):
        """Test that no cycle is detected in simple delegation."""
        detector = CircularDelegationDetector()
        
        delegations = {
            "0x123": [Delegation("0x123", "0x456", 1000)],
            "0x456": [Delegation("0x456", "0x789", 500)]
        }
        
        would_create_cycle = detector.would_create_cycle("0x789", "0xabc", delegations)
        assert would_create_cycle is False
    
    def test_simple_cycle_detection(self):
        """Test detecting a simple cycle."""
        detector = CircularDelegationDetector()
        
        delegations = {
            "0x123": [Delegation("0x123", "0x456", 1000)],
            "0x456": [Delegation("0x456", "0x789", 500)]
        }
        
        # This would create a cycle: 0x789 -> 0x123 -> 0x456 -> 0x789
        would_create_cycle = detector.would_create_cycle("0x789", "0x123", delegations)
        assert would_create_cycle is True
    
    def test_complex_cycle_detection(self):
        """Test detecting a complex cycle."""
        detector = CircularDelegationDetector()
        
        delegations = {
            "0x123": [Delegation("0x123", "0x456", 1000)],
            "0x456": [Delegation("0x456", "0x789", 500)],
            "0x789": [Delegation("0x789", "0xabc", 200)],
            "0xabc": [Delegation("0xabc", "0xdef", 100)]
        }
        
        # This would create a cycle: 0xdef -> 0x123 -> 0x456 -> 0x789 -> 0xabc -> 0xdef
        would_create_cycle = detector.would_create_cycle("0xdef", "0x123", delegations)
        assert would_create_cycle is True
    
    def test_self_cycle_detection(self):
        """Test detecting self-delegation cycle."""
        detector = CircularDelegationDetector()
        
        delegations = {}
        
        # Self-delegation should create a cycle
        would_create_cycle = detector.would_create_cycle("0x123", "0x123", delegations)
        assert would_create_cycle is True
    
    def test_cycle_detection_with_invalid_delegations(self):
        """Test cycle detection with invalid delegations."""
        detector = CircularDelegationDetector()
        
        delegations = {
            "0x123": [Delegation("0x123", "0x456", 1000)],
            "0x456": [Delegation("0x456", "0x789", 500)]
        }
        
        # Make delegation invalid
        delegations["0x456"][0].is_active = False
        
        # Should not create cycle with invalid delegation
        would_create_cycle = detector.would_create_cycle("0x789", "0x123", delegations)
        assert would_create_cycle is False
