"""
Unit tests for atomic swap functionality.
"""

import hashlib
import time
from unittest.mock import Mock, patch

import pytest

from dubchain.bridge.atomic_swap import (
    AtomicSwap,
    SwapExecution,
    SwapProposal,
    SwapValidator,
)
from dubchain.bridge.bridge_types import BridgeStatus, BridgeType


class TestSwapProposal:
    """Test the SwapProposal class."""

    def test_swap_proposal_creation(self):
        """Test creating a swap proposal."""
        proposal = SwapProposal(
            proposal_id="test_swap_1",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,  # 64 character hash
            timeout=3600,
        )

        assert proposal.proposal_id == "test_swap_1"
        assert proposal.initiator == "alice"
        assert proposal.counterparty == "bob"
        assert proposal.source_chain == "dubchain_mainnet"
        assert proposal.target_chain == "ethereum_mainnet"
        assert proposal.source_asset == "DUB"
        assert proposal.target_asset == "ETH"
        assert proposal.source_amount == 1000
        assert proposal.target_amount == 1
        assert proposal.secret_hash == "a" * 64
        assert proposal.timeout == 3600
        assert proposal.status == "pending"

    def test_swap_proposal_expiration(self):
        """Test swap proposal expiration."""
        # Create proposal with short timeout
        proposal = SwapProposal(
            proposal_id="test_swap_1",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,
            timeout=1,  # 1 second timeout
        )

        # Initially not expired
        assert not proposal.is_expired()

        # Wait for expiration
        time.sleep(1.1)
        assert proposal.is_expired()

    def test_swap_proposal_hash(self):
        """Test swap proposal hash calculation."""
        proposal = SwapProposal(
            proposal_id="test_swap_1",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,
            timeout=3600,
        )

        hash_value = proposal.calculate_hash()
        assert hash_value is not None
        assert len(hash_value) == 64  # SHA256 hash length

    def test_swap_proposal_serialization(self):
        """Test swap proposal serialization."""
        proposal = SwapProposal(
            proposal_id="test_swap_1",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,
            timeout=3600,
        )

        # Test to_dict
        data = proposal.to_dict()
        assert isinstance(data, dict)
        assert data["proposal_id"] == "test_swap_1"
        assert data["initiator"] == "alice"
        assert data["source_amount"] == 1000

        # Test from_dict
        deserialized = SwapProposal.from_dict(data)
        assert deserialized.proposal_id == proposal.proposal_id
        assert deserialized.initiator == proposal.initiator
        assert deserialized.source_amount == proposal.source_amount


class TestSwapExecution:
    """Test the SwapExecution class."""

    def test_swap_execution_creation(self):
        """Test creating a swap execution."""
        execution = SwapExecution(proposal_id="test_swap_1")

        assert execution.proposal_id == "test_swap_1"
        assert execution.execution_phase == "initiated"
        assert not execution.initiator_locked
        assert not execution.counterparty_locked
        assert not execution.secret_revealed
        assert execution.lock_timeout == 3600
        assert execution.reveal_timeout == 1800

    def test_swap_execution_phases(self):
        """Test swap execution phase transitions."""
        execution = SwapExecution(proposal_id="test_swap_1")

        # Initially can't proceed to reveal
        assert not execution.can_proceed_to_reveal()
        assert not execution.can_complete()

        # Lock both parties
        execution.initiator_locked = True
        execution.counterparty_locked = True

        # Now can proceed to reveal
        assert execution.can_proceed_to_reveal()
        assert not execution.can_complete()

        # Reveal secret
        execution.secret_revealed = True

        # Now can complete
        assert execution.can_complete()

    def test_swap_execution_timeouts(self):
        """Test swap execution timeouts."""
        execution = SwapExecution(
            proposal_id="test_swap_1",
            lock_timeout=1,  # 1 second
            reveal_timeout=1,  # 1 second
        )

        # Initially not expired
        assert not execution.is_lock_expired()
        assert not execution.is_reveal_expired()

        # Wait for expiration
        time.sleep(1.1)
        assert execution.is_lock_expired()
        assert execution.is_reveal_expired()


class TestSwapValidator:
    """Test the SwapValidator class."""

    def test_swap_validator_creation(self):
        """Test creating a swap validator."""
        validator = SwapValidator()

        assert validator is not None
        assert isinstance(validator.validation_rules, dict)
        assert isinstance(validator.validation_cache, dict)

    def test_validate_proposal(self):
        """Test validating a swap proposal."""
        validator = SwapValidator()

        # Valid proposal
        proposal = SwapProposal(
            proposal_id="test_swap_1",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,
            timeout=3600,
        )

        result = validator.validate_proposal(proposal)
        assert result is True

        # Invalid proposal - same initiator and counterparty
        invalid_proposal = SwapProposal(
            proposal_id="test_swap_2",
            initiator="alice",
            counterparty="alice",  # Same as initiator
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,
            timeout=3600,
        )

        result = validator.validate_proposal(invalid_proposal)
        assert result is False

        # Invalid proposal - same chains
        invalid_proposal2 = SwapProposal(
            proposal_id="test_swap_3",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="dubchain_mainnet",  # Same chain
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,
            timeout=3600,
        )

        result = validator.validate_proposal(invalid_proposal2)
        assert result is False

        # Invalid proposal - negative amounts
        invalid_proposal3 = SwapProposal(
            proposal_id="test_swap_4",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=-1000,  # Negative amount
            target_amount=1,
            secret_hash="a" * 64,
            timeout=3600,
        )

        result = validator.validate_proposal(invalid_proposal3)
        assert result is False

        # Invalid proposal - invalid secret hash length
        invalid_proposal4 = SwapProposal(
            proposal_id="test_swap_5",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="short",  # Too short
            timeout=3600,
        )

        result = validator.validate_proposal(invalid_proposal4)
        assert result is False

    def test_validate_secret(self):
        """Test validating a secret against its hash."""
        validator = SwapValidator()

        # Valid secret
        secret = "my_secret"
        secret_hash = hashlib.sha256(secret.encode()).hexdigest()

        result = validator.validate_secret(secret, secret_hash)
        assert result is True

        # Invalid secret
        result = validator.validate_secret("wrong_secret", secret_hash)
        assert result is False

        # Empty secret
        result = validator.validate_secret("", secret_hash)
        assert result is False

        # Empty hash
        result = validator.validate_secret(secret, "")
        assert result is False

    def test_validate_execution(self):
        """Test validating a swap execution."""
        validator = SwapValidator()

        # Valid execution
        execution = SwapExecution(
            proposal_id="test_swap_1", lock_timeout=3600  # 1 hour
        )
        execution.initiator_locked = True
        execution.counterparty_locked = True

        proposal = SwapProposal(
            proposal_id="test_swap_1",
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            secret_hash="a" * 64,
            timeout=3600,
        )

        result = validator.validate_execution(execution, proposal)
        assert result is True

        # Invalid execution - not locked
        execution2 = SwapExecution(proposal_id="test_swap_2")
        result = validator.validate_execution(execution2, proposal)
        assert result is False

        # Invalid execution - expired
        execution3 = SwapExecution(
            proposal_id="test_swap_3", lock_timeout=1  # 1 second
        )
        execution3.initiator_locked = True
        execution3.counterparty_locked = True

        time.sleep(1.1)  # Wait for expiration
        result = validator.validate_execution(execution3, proposal)
        assert result is False


class TestAtomicSwap:
    """Test the AtomicSwap class."""

    @pytest.fixture
    def atomic_swap(self):
        """Create an atomic swap instance."""
        return AtomicSwap()

    def test_atomic_swap_creation(self, atomic_swap):
        """Test creating an atomic swap instance."""
        assert atomic_swap is not None
        assert len(atomic_swap.proposals) == 0
        assert len(atomic_swap.executions) == 0
        assert atomic_swap.validator is not None
        assert "proposals_created" in atomic_swap.swap_metrics

    def test_create_proposal(self, atomic_swap):
        """Test creating a swap proposal."""
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        assert proposal is not None
        assert proposal.proposal_id is not None
        assert proposal.initiator == "alice"
        assert proposal.counterparty == "bob"
        assert proposal.source_amount == 1000
        assert proposal.target_amount == 1
        assert proposal.proposal_id in atomic_swap.proposals
        assert atomic_swap.swap_metrics["proposals_created"] == 1

    def test_accept_proposal(self, atomic_swap):
        """Test accepting a swap proposal."""
        # Create proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        # Accept proposal
        execution = atomic_swap.accept_proposal(proposal.proposal_id, "bob")

        assert execution is not None
        assert execution.proposal_id == proposal.proposal_id
        assert execution.execution_phase == "initiated"
        assert proposal.proposal_id in atomic_swap.executions
        assert proposal.status == "accepted"
        assert atomic_swap.swap_metrics["proposals_accepted"] == 1

    def test_lock_funds(self, atomic_swap):
        """Test locking funds for a swap."""
        # Create and accept proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        execution = atomic_swap.accept_proposal(proposal.proposal_id, "bob")

        # Lock initiator funds
        result = atomic_swap.lock_funds(proposal.proposal_id, "alice", "init_tx_hash")
        assert result is True
        assert execution.initiator_locked is True

        # Lock counterparty funds
        result = atomic_swap.lock_funds(proposal.proposal_id, "bob", "counter_tx_hash")
        assert result is True
        assert execution.counterparty_locked is True

    def test_reveal_secret(self, atomic_swap):
        """Test revealing the secret for a swap."""
        # Create and accept proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        execution = atomic_swap.accept_proposal(proposal.proposal_id, "bob")

        # Lock funds first
        atomic_swap.lock_funds(proposal.proposal_id, "alice", "init_tx_hash")
        atomic_swap.lock_funds(proposal.proposal_id, "bob", "counter_tx_hash")

        # Reveal secret
        secret = proposal.secret
        result = atomic_swap.reveal_secret(proposal.proposal_id, secret)
        assert result is True
        assert execution.secret_revealed is True
        assert proposal.secret == secret

    def test_complete_swap(self, atomic_swap):
        """Test completing a swap."""
        # Create and accept proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        execution = atomic_swap.accept_proposal(proposal.proposal_id, "bob")

        # Lock funds and reveal secret
        atomic_swap.lock_funds(proposal.proposal_id, "alice", "init_tx_hash")
        atomic_swap.lock_funds(proposal.proposal_id, "bob", "counter_tx_hash")
        atomic_swap.reveal_secret(proposal.proposal_id, proposal.secret)

        # Complete swap
        result = atomic_swap.complete_swap(proposal.proposal_id)
        assert result is True
        assert proposal.status == "completed"
        assert atomic_swap.swap_metrics["swaps_completed"] == 1

    def test_cancel_proposal(self, atomic_swap):
        """Test canceling a swap proposal."""
        # Create proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        # Cancel proposal
        result = atomic_swap.cancel_proposal(proposal.proposal_id, "alice")
        assert result is True
        assert proposal.status == "cancelled"

    def test_get_proposal(self, atomic_swap):
        """Test getting a swap proposal."""
        # Create proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        # Get proposal
        retrieved_proposal = atomic_swap.get_proposal(proposal.proposal_id)
        assert retrieved_proposal is not None
        assert retrieved_proposal.proposal_id == proposal.proposal_id
        assert retrieved_proposal.initiator == "alice"

    def test_get_proposal_not_found(self, atomic_swap):
        """Test getting non-existent proposal."""
        proposal = atomic_swap.get_proposal("non_existent_proposal")
        assert proposal is None

    def test_get_execution(self, atomic_swap):
        """Test getting a swap execution."""
        # Create and accept proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        execution = atomic_swap.accept_proposal(proposal.proposal_id, "bob")

        # Get execution
        retrieved_execution = atomic_swap.get_execution(execution.proposal_id)
        assert retrieved_execution is not None
        assert retrieved_execution.proposal_id == execution.proposal_id
        assert retrieved_execution.execution_phase == "initiated"

    def test_get_execution_not_found(self, atomic_swap):
        """Test getting non-existent execution."""
        execution = atomic_swap.get_execution("non_existent_execution")
        assert execution is None

    def test_cleanup_expired_proposals(self, atomic_swap):
        """Test cleaning up expired proposals."""
        # Create expired proposal
        proposal = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=1,  # 1 second timeout
        )

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup expired proposals
        cleaned_count = atomic_swap.cleanup_expired_proposals()
        assert cleaned_count >= 0
        # The proposal should be removed
        assert proposal.proposal_id not in atomic_swap.proposals

    def test_get_statistics(self, atomic_swap):
        """Test getting swap statistics."""
        # Create some proposals
        proposal1 = atomic_swap.create_proposal(
            initiator="alice",
            counterparty="bob",
            source_chain="dubchain_mainnet",
            target_chain="ethereum_mainnet",
            source_asset="DUB",
            target_asset="ETH",
            source_amount=1000,
            target_amount=1,
            timeout=3600,
        )

        execution1 = atomic_swap.accept_proposal(proposal1.proposal_id, "bob")

        # Lock funds and reveal secret
        atomic_swap.lock_funds(proposal1.proposal_id, "alice", "init_tx_hash")
        atomic_swap.lock_funds(proposal1.proposal_id, "bob", "counter_tx_hash")
        atomic_swap.reveal_secret(proposal1.proposal_id, proposal1.secret)
        atomic_swap.complete_swap(proposal1.proposal_id)

        # Get statistics
        stats = atomic_swap.get_statistics()
        assert stats is not None
        assert "metrics" in stats
        metrics = stats["metrics"]
        assert "proposals_created" in metrics
        assert "proposals_accepted" in metrics
        assert "swaps_completed" in metrics
        assert "swaps_failed" in metrics
        assert metrics["proposals_created"] >= 1
        assert metrics["proposals_accepted"] >= 1
        assert metrics["swaps_completed"] >= 1
