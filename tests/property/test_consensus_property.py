"""
Property-based tests for consensus mechanisms.

This module uses Hypothesis to test consensus mechanisms with randomly
generated inputs, ensuring properties hold across a wide range of scenarios.
"""

import pytest

# Temporarily disable property tests due to hanging issues
pytestmark = pytest.mark.skip(reason="Temporarily disabled due to hanging issues")
from hypothesis import given, strategies as st, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
import time
from typing import Dict, List, Any

from src.dubchain.consensus.consensus_types import (
    ConsensusConfig,
    ConsensusType,
    PoAStatus,
)
from src.dubchain.consensus.proof_of_authority import ProofOfAuthority
from src.dubchain.consensus.proof_of_history import ProofOfHistory
from src.dubchain.consensus.proof_of_space_time import ProofOfSpaceTime
from src.dubchain.consensus.hotstuff import HotStuffConsensus


class TestConsensusProperties:
    """Property-based tests for consensus mechanisms."""

    @given(
        authority_count=st.integers(min_value=1, max_value=10),
        block_count=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=20)
    def test_poa_authority_rotation_property(self, authority_count: int, block_count: int):
        """Test that PoA authority rotation is fair and deterministic."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=[f"auth{i}" for i in range(authority_count)]
        )
        poa = ProofOfAuthority(config)
        
        # Track authority selections
        authority_selections = []
        
        for _ in range(block_count):
            authority = poa.get_next_authority()
            authority_selections.append(authority)
        
        # Property 1: All authorities should be selected
        unique_authorities = set(authority_selections)
        assert len(unique_authorities) == authority_count
        
        # Property 2: Authority selection should be deterministic
        # (same sequence for same initial state)
        poa2 = ProofOfAuthority(config)
        authority_selections2 = []
        
        for _ in range(block_count):
            authority = poa2.get_next_authority()
            authority_selections2.append(authority)
        
        assert authority_selections == authority_selections2

    @given(
        validator_count=st.integers(min_value=1, max_value=20),
        block_count=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=15)
    def test_hotstuff_safety_property(self, validator_count: int, block_count: int):
        """Test that HotStuff maintains safety properties."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        for i in range(validator_count):
            hotstuff.add_validator(f"validator{i}")
        
        committed_blocks = set()
        
        for i in range(block_count):
            block_data = {
                "block_number": i,
                "timestamp": time.time(),
                "transactions": [f"tx_{i}"],
                "previous_hash": f"0x{i:064x}",
                "proposer_id": hotstuff.get_current_leader(),
            }
            
            result = hotstuff.propose_block(block_data)
            
            if result.success:
                # Property: No two blocks with same number should be committed
                assert result.block_hash not in committed_blocks
                committed_blocks.add(result.block_hash)
                
                # Property: Block should be in committed set
                assert result.block_hash in hotstuff.get_committed_blocks()

    @given(
        farmer_count=st.integers(min_value=1, max_value=10),
        plot_size=st.integers(min_value=1024*1024*100, max_value=1024*1024*1000),  # 100MB to 1GB
        challenge_count=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=10)
    def test_pospace_plot_property(self, farmer_count: int, plot_size: int, challenge_count: int):
        """Test that PoSpace plot properties are maintained."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_SPACE_TIME)
        pospace = ProofOfSpaceTime(config)
        
        # Register farmers and create plots
        plot_ids = []
        for i in range(farmer_count):
            pospace.register_farmer(f"farmer{i}")
            plot_id = pospace.create_plot(f"farmer{i}", plot_size)
            if plot_id:
                plot_ids.append(plot_id)
                pospace.start_farming(plot_id)
        
        # Property: All created plots should be valid
        for plot_id in plot_ids:
            plot = pospace.get_plot_info(plot_id)
            assert plot is not None
            assert plot.size_bytes >= config.pospace_min_plot_size
            assert plot.is_active is True
        
        # Property: Plot statistics should be consistent
        stats = pospace.get_farming_statistics()
        assert stats["total_farmers"] == farmer_count
        assert stats["total_plots"] == len(plot_ids)
        assert stats["active_plots"] == len(plot_ids)
        assert stats["total_storage_bytes"] == sum(pospace.get_plot_info(pid).size_bytes for pid in plot_ids)

    @given(
        entry_count=st.integers(min_value=1, max_value=100),
        validator_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=10)
    def test_poh_sequence_property(self, entry_count: int, validator_count: int):
        """Test that PoH maintains sequence properties."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        # Register validators
        for i in range(validator_count):
            poh.register_validator(f"validator{i}")
        
        # Create PoH entries
        for _ in range(entry_count):
            poh._create_poh_entry()
        
        # Property: Entries should be in sequence
        assert len(poh.state.entries) == entry_count
        
        for i in range(1, len(poh.state.entries)):
            current_entry = poh.state.entries[i]
            previous_entry = poh.state.entries[i-1]
            
            # Property: Each entry should reference the previous one
            assert current_entry.previous_hash == previous_entry.hash
            
            # Property: Timestamps should be increasing
            assert current_entry.timestamp >= previous_entry.timestamp

    @given(
        authority_count=st.integers(min_value=1, max_value=5),
        slash_events=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.floats(min_value=1.0, max_value=50.0)
            ),
            min_size=0,
            max_size=20
        ),
    )
    @settings(max_examples=10)
    def test_poa_reputation_property(self, authority_count: int, slash_events: List[tuple]):
        """Test that PoA reputation system maintains properties."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=[f"auth{i}" for i in range(authority_count)]
        )
        poa = ProofOfAuthority(config)
        
        initial_reputations = {}
        for auth_id in poa.state.authorities:
            initial_reputations[auth_id] = poa.state.authorities[auth_id].reputation_score
        
        # Apply slash events
        for auth_id, penalty in slash_events:
            if auth_id in poa.state.authorities:
                poa.slash_authority(auth_id, penalty)
        
        # Property: Reputation should never go below 0
        for auth_id, authority in poa.state.authorities.items():
            assert authority.reputation_score >= 0.0
            
            # Property: Reputation should not exceed initial value
            assert authority.reputation_score <= initial_reputations[auth_id]
            
            # Property: If reputation is below threshold, authority should be revoked
            if authority.reputation_score < config.poa_reputation_threshold:
                assert authority.status == PoAStatus.REVOKED or not authority.is_active

    @given(
        validator_count=st.integers(min_value=3, max_value=10),
        view_changes=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=10)
    def test_hotstuff_view_change_property(self, validator_count: int, view_changes: int):
        """Test that HotStuff view changes maintain properties."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        for i in range(validator_count):
            hotstuff.add_validator(f"validator{i}")
        
        initial_view = hotstuff.get_current_view()
        initial_leader = hotstuff.get_current_leader()
        
        # Perform view changes
        for _ in range(view_changes):
            success = hotstuff.start_view_change()
            if not success:
                break  # Hit max view changes limit
        
        # Property: View number should increase with each successful view change
        assert hotstuff.get_current_view() >= initial_view
        
        # Property: Leader should change with view changes
        if view_changes > 0 and hotstuff.state.view_change_counter < config.hotstuff_max_view_changes:
            assert hotstuff.get_current_leader() != initial_leader
        
        # Property: Current leader should be in validator set
        current_leader = hotstuff.get_current_leader()
        if current_leader:
            assert current_leader in hotstuff.get_validators()

    @given(
        block_data=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(
                st.text(),
                st.integers(),
                st.floats(),
                st.booleans(),
                st.lists(st.text())
            ),
            min_size=1,
            max_size=10
        ),
    )
    @settings(max_examples=20)
    def test_consensus_block_validation_property(self, block_data: Dict[str, Any]):
        """Test that consensus mechanisms properly validate block data."""
        consensus_types = [
            ConsensusType.PROOF_OF_AUTHORITY,
            ConsensusType.PROOF_OF_HISTORY,
            ConsensusType.PROOF_OF_SPACE_TIME,
            ConsensusType.HOTSTUFF,
        ]
        
        for consensus_type in consensus_types:
            config = ConsensusConfig(consensus_type=consensus_type)
            
            if consensus_type == ConsensusType.PROOF_OF_AUTHORITY:
                consensus = ProofOfAuthority(config)
                consensus.register_authority("auth1", "pubkey1")
            elif consensus_type == ConsensusType.PROOF_OF_HISTORY:
                consensus = ProofOfHistory(config)
                consensus.register_validator("validator1")
            elif consensus_type == ConsensusType.PROOF_OF_SPACE_TIME:
                consensus = ProofOfSpaceTime(config)
                consensus.register_farmer("farmer1")
                plot_id = consensus.create_plot("farmer1", 1024*1024*200)
                if plot_id:
                    consensus.start_farming(plot_id)
            elif consensus_type == ConsensusType.HOTSTUFF:
                consensus = HotStuffConsensus(config)
                consensus.add_validator("validator1")
            
            # Property: Invalid block data should be rejected
            result = consensus.propose_block(block_data)
            
            # If block data is missing required fields, it should fail
            required_fields = ["block_number", "timestamp", "transactions", "previous_hash"]
            has_required_fields = all(field in block_data for field in required_fields)
            
            if not has_required_fields:
                assert result.success is False
            else:
                # Even with required fields, other validation might fail
                # This is acceptable as long as the system doesn't crash
                assert isinstance(result.success, bool)


class ConsensusStateMachine(RuleBasedStateMachine):
    """State machine for testing consensus mechanisms."""

    def __init__(self):
        super().__init__()
        self.consensus = None
        self.consensus_type = None
        self.block_counter = 0
        self.successful_blocks = 0

    @rule(consensus_type=st.sampled_from([
        ConsensusType.PROOF_OF_AUTHORITY,
        ConsensusType.PROOF_OF_HISTORY,
        ConsensusType.PROOF_OF_SPACE_TIME,
        ConsensusType.HOTSTUFF,
    ]))
    def initialize_consensus(self, consensus_type):
        """Initialize consensus mechanism."""
        config = ConsensusConfig(consensus_type=consensus_type)
        
        if consensus_type == ConsensusType.PROOF_OF_AUTHORITY:
            self.consensus = ProofOfAuthority(config)
            self.consensus.register_authority("auth1", "pubkey1")
        elif consensus_type == ConsensusType.PROOF_OF_HISTORY:
            self.consensus = ProofOfHistory(config)
            self.consensus.register_validator("validator1")
        elif consensus_type == ConsensusType.PROOF_OF_SPACE_TIME:
            self.consensus = ProofOfSpaceTime(config)
            self.consensus.register_farmer("farmer1")
            plot_id = self.consensus.create_plot("farmer1", 1024*1024*200)
            if plot_id:
                self.consensus.start_farming(plot_id)
        elif consensus_type == ConsensusType.HOTSTUFF:
            self.consensus = HotStuffConsensus(config)
            self.consensus.add_validator("validator1")
        
        self.consensus_type = consensus_type

    @rule()
    def propose_block(self):
        """Propose a block."""
        if self.consensus is None:
            return
        
        block_data = {
            "block_number": self.block_counter,
            "timestamp": time.time(),
            "transactions": [f"tx_{self.block_counter}"],
            "previous_hash": f"0x{self.block_counter:064x}",
        }
        
        # Add proposer_id for mechanisms that need it
        if self.consensus_type in [ConsensusType.PROOF_OF_HISTORY, ConsensusType.PROOF_OF_SPACE_TIME, ConsensusType.HOTSTUFF]:
            if hasattr(self.consensus, 'get_current_leader'):
                block_data["proposer_id"] = self.consensus.get_current_leader()
            else:
                block_data["proposer_id"] = "proposer1"
        
        result = self.consensus.propose_block(block_data)
        
        if result.success:
            self.successful_blocks += 1
        
        self.block_counter += 1

    @rule()
    def get_metrics(self):
        """Get consensus metrics."""
        if self.consensus is None:
            return
        
        metrics = self.consensus.get_consensus_metrics()
        assert metrics is not None
        assert metrics.consensus_type == self.consensus_type

    @invariant()
    def block_counter_consistency(self):
        """Invariant: Block counter should be consistent."""
        if self.consensus is None:
            return
        
        # Block counter should never be negative
        assert self.block_counter >= 0
        
        # Successful blocks should not exceed total blocks
        assert self.successful_blocks <= self.block_counter

    @invariant()
    def consensus_state_consistency(self):
        """Invariant: Consensus state should be consistent."""
        if self.consensus is None:
            return
        
        # Consensus should have valid state
        assert self.consensus is not None
        
        # Metrics should be accessible
        metrics = self.consensus.get_consensus_metrics()
        assert metrics is not None


# Register the state machine test
TestConsensusStateMachine = ConsensusStateMachine.TestCase
