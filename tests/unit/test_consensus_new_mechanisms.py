"""
Unit tests for new consensus mechanisms.

This module tests the new consensus mechanisms: Proof-of-Authority,
Proof-of-History, Proof-of-Space/Time, and HotStuff.
"""

import logging

logger = logging.getLogger(__name__)
import pytest
import time
from unittest.mock import Mock, patch

from src.dubchain.consensus.consensus_types import (
    ConsensusConfig,
    ConsensusType,
    PoAStatus,
    PoHStatus,
    PoSpaceStatus,
    HotStuffPhase,
)
from src.dubchain.consensus.proof_of_authority import ProofOfAuthority
from src.dubchain.consensus.proof_of_history import ProofOfHistory
from src.dubchain.consensus.proof_of_space_time import ProofOfSpaceTime
from src.dubchain.consensus.hotstuff import HotStuffConsensus


class TestProofOfAuthority:
    """Test Proof-of-Authority consensus mechanism."""

    def test_poa_initialization(self):
        """Test PoA initialization."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1", "auth2", "auth3"]
        )
        poa = ProofOfAuthority(config)
        
        assert len(poa.state.authorities) == 3
        assert "auth1" in poa.state.authorities
        assert "auth2" in poa.state.authorities
        assert "auth3" in poa.state.authorities

    def test_register_authority(self):
        """Test authority registration."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_AUTHORITY)
        poa = ProofOfAuthority(config)
        
        # Register new authority
        success = poa.register_authority("new_auth", "pubkey123")
        assert success is True
        assert "new_auth" in poa.state.authorities
        assert poa.state.authorities["new_auth"].status == PoAStatus.CANDIDATE

    def test_revoke_authority(self):
        """Test authority revocation."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1"]
        )
        poa = ProofOfAuthority(config)
        
        # Revoke authority
        success = poa.revoke_authority("auth1", "misbehavior")
        assert success is True
        assert poa.state.authorities["auth1"].status == PoAStatus.REVOKED
        assert not poa.state.authorities["auth1"].is_active

    def test_slash_authority(self):
        """Test authority slashing."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1"]
        )
        poa = ProofOfAuthority(config)
        
        # Slash authority
        success = poa.slash_authority("auth1", 20.0)
        assert success is True
        assert poa.state.authorities["auth1"].reputation_score == 80.0

    def test_get_next_authority(self):
        """Test authority selection."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1", "auth2", "auth3"]
        )
        poa = ProofOfAuthority(config)
        
        # Get next authority (round-robin)
        auth1 = poa.get_next_authority()
        auth2 = poa.get_next_authority()
        auth3 = poa.get_next_authority()
        auth4 = poa.get_next_authority()
        
        assert auth1 in ["auth1", "auth2", "auth3"]
        assert auth2 in ["auth1", "auth2", "auth3"]
        assert auth3 in ["auth1", "auth2", "auth3"]
        assert auth4 == auth1  # Round-robin

    def test_propose_block_success(self):
        """Test successful block proposal."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1"],
            block_time=0.01  # Very short block time for testing
        )
        poa = ProofOfAuthority(config)
        
        # Wait to ensure block time constraint is met
        time.sleep(0.02)
        
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123"
        }
        
        result = poa.propose_block(block_data)
        assert result.success is True
        assert result.validator_id == "auth1"
        assert result.consensus_type == ConsensusType.PROOF_OF_AUTHORITY

    def test_propose_block_invalid_data(self):
        """Test block proposal with invalid data."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1"]
        )
        poa = ProofOfAuthority(config)
        
        # Missing required fields
        block_data = {"block_number": 1}
        
        result = poa.propose_block(block_data)
        assert result.success is False
        assert "Invalid block data" in result.error_message

    def test_authority_rotation(self):
        """Test authority rotation."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1"],
            poa_rotation_period=1  # Short period for testing
        )
        poa = ProofOfAuthority(config)
        
        # Force rotation
        poa.state.rotation_counter = 1
        poa._rotate_authorities()
        
        # Should reset counter
        assert poa.state.rotation_counter == 0


class TestProofOfHistory:
    """Test Proof-of-History consensus mechanism."""

    def test_poh_initialization(self):
        """Test PoH initialization."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        assert len(poh.state.entries) == 0
        assert poh.state.current_leader is None
        assert not poh.state.poh_clock_running

    def test_register_validator(self):
        """Test validator registration."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        # Register validator
        success = poh.register_validator("validator1")
        assert success is True
        assert "validator1" in poh.state.validators
        assert poh.state.current_leader == "validator1"

    def test_poh_clock_control(self):
        """Test PoH clock start/stop."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        # Start clock
        poh.start_poh_clock()
        assert poh.state.poh_clock_running is True
        
        # Stop clock
        poh.stop_poh_clock()
        assert poh.state.poh_clock_running is False

    def test_create_poh_entry(self):
        """Test PoH entry creation."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        # Create entry
        poh._create_poh_entry()
        
        assert len(poh.state.entries) == 1
        entry = poh.state.entries[0]
        assert entry.entry_id == "poh_0"
        assert entry.validator_id == "system"

    def test_leader_rotation(self):
        """Test leader rotation."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_HISTORY,
            poh_leader_rotation=2
        )
        poh = ProofOfHistory(config)
        
        # Add validators
        poh.register_validator("validator1")
        poh.register_validator("validator2")
        
        # Force leader rotation
        poh.state.leader_rotation_counter = 2
        poh._rotate_leader()
        
        # Leader should rotate to the next validator
        assert poh.state.current_leader in ["validator1", "validator2"]

    def test_propose_block_success(self):
        """Test successful block proposal."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        # Register validator and create PoH entry
        poh.register_validator("validator1")
        poh._create_poh_entry()
        
        # Mock VDF verification to always succeed
        with patch.object(poh, '_validate_poh_entry', return_value=True):
            block_data = {
                "block_number": 1,
                "timestamp": time.time(),
                "transactions": [],
                "previous_hash": "0x123",
                "proposer_id": "validator1"
            }
            
            result = poh.propose_block(block_data)
            assert result.success is True
            assert result.validator_id == "validator1"

    def test_propose_block_not_leader(self):
        """Test block proposal by non-leader."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        # Register validator
        poh.register_validator("validator1")
        
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "proposer_id": "validator2"  # Not the leader
        }
        
        result = poh.propose_block(block_data)
        assert result.success is False
        assert "Not current leader" in result.error_message

    def test_verify_poh_sequence(self):
        """Test PoH sequence verification."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        poh = ProofOfHistory(config)
        
        # Create multiple entries
        for _ in range(3):
            poh._create_poh_entry()
        
        # Mock VDF verification to always succeed
        with patch.object(poh, '_validate_poh_entry', return_value=True):
            # Verify sequence
            is_valid = poh.verify_poh_sequence(0, 2)
            assert is_valid is True


class TestProofOfSpaceTime:
    """Test Proof-of-Space/Time consensus mechanism."""

    def test_pospace_initialization(self):
        """Test PoSpace initialization."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        assert len(pospace.state.plots) == 0
        assert len(pospace.state.farmers) == 0
        assert pospace.state.current_difficulty == 1000

    def test_register_farmer(self):
        """Test farmer registration."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        # Register farmer
        success = pospace.register_farmer("farmer1")
        assert success is True
        assert "farmer1" in pospace.state.farmers

    def test_create_plot(self):
        """Test plot creation."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        # Register farmer first
        pospace.register_farmer("farmer1")
        
        # Create plot (reduced size for faster testing)
        plot_id = pospace.create_plot("farmer1", 1024 * 1024)  # 1MB instead of 200MB
        assert plot_id is not None
        assert plot_id in pospace.state.plots

    def test_create_plot_too_small(self):
        """Test plot creation with insufficient size."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        # Register farmer first
        pospace.register_farmer("farmer1")
        
        # Try to create plot that's too small
        plot_id = pospace.create_plot("farmer1", 1024)  # 1KB
        assert plot_id is None

    def test_start_stop_farming(self):
        """Test starting and stopping farming."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        # Register farmer and create plot
        pospace.register_farmer("farmer1")
        plot_id = pospace.create_plot("farmer1", 1024 * 1024)  # 1MB instead of 200MB
        
        # Start farming
        success = pospace.start_farming(plot_id)
        assert success is True
        assert pospace.state.plots[plot_id].is_active is True
        
        # Stop farming
        success = pospace.stop_farming(plot_id)
        assert success is True
        assert pospace.state.plots[plot_id].is_active is False

    def test_propose_block_success(self):
        """Test successful block proposal."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        # Register farmer and create plot
        pospace.register_farmer("farmer1")
        plot_id = pospace.create_plot("farmer1", 1024 * 1024)  # 1MB instead of 200MB
        pospace.start_farming(plot_id)
        
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "proposer_id": "farmer1"
        }
        
        # Mock challenge solving to always succeed
        with patch.object(pospace.challenge_manager, 'solve_challenge', return_value="proof123"):
            result = pospace.propose_block(block_data)
            assert result.success is True
            assert result.validator_id == "farmer1"

    def test_propose_block_no_plots(self):
        """Test block proposal without active plots."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        # Register farmer but don't create plots
        pospace.register_farmer("farmer1")
        
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "proposer_id": "farmer1"
        }
        
        result = pospace.propose_block(block_data)
        assert result.success is False
        assert "No active plots" in result.error_message

    def test_difficulty_adjustment(self):
        """Test difficulty adjustment."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        original_difficulty = pospace.state.current_difficulty
        
        # Adjust difficulty
        pospace._adjust_difficulty()
        
        # Difficulty should change
        assert pospace.state.current_difficulty != original_difficulty

    def test_get_farming_statistics(self):
        """Test farming statistics."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_SPACE_TIME,
            pospace_min_plot_size=1024 * 1024  # 1MB minimum for testing
        )
        pospace = ProofOfSpaceTime(config)
        
        # Register farmer and create plot
        pospace.register_farmer("farmer1")
        plot_id = pospace.create_plot("farmer1", 1024 * 1024)  # 1MB instead of 200MB
        pospace.start_farming(plot_id)
        
        stats = pospace.get_farming_statistics()
        assert stats["total_farmers"] == 1
        assert stats["total_plots"] == 1
        assert stats["active_plots"] == 1
        assert stats["total_storage_bytes"] == 1024 * 1024  # 1MB instead of 200MB


class TestHotStuffConsensus:
    """Test HotStuff consensus mechanism."""

    def test_hotstuff_initialization(self):
        """Test HotStuff initialization."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        assert hotstuff.state.current_view == 0
        assert hotstuff.state.current_leader is None
        assert len(hotstuff.state.validators) == 0

    def test_add_validator(self):
        """Test validator addition."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validator
        success = hotstuff.add_validator("validator1")
        assert success is True
        assert "validator1" in hotstuff.state.validators
        assert hotstuff.state.current_leader == "validator1"

    def test_remove_validator(self):
        """Test validator removal."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        hotstuff.add_validator("validator1")
        hotstuff.add_validator("validator2")
        
        # Remove validator
        success = hotstuff.remove_validator("validator1")
        assert success is True
        assert "validator1" not in hotstuff.state.validators
        assert hotstuff.state.current_leader == "validator2"

    def test_leader_rotation(self):
        """Test leader rotation."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        hotstuff.add_validator("validator1")
        hotstuff.add_validator("validator2")
        hotstuff.add_validator("validator3")
        
        # Rotate leader
        hotstuff._rotate_leader()
        assert hotstuff.state.current_leader == "validator2"
        assert hotstuff.state.current_view == 1

    def test_safety_threshold(self):
        """Test safety threshold calculation."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        for i in range(7):  # 7 validators
            hotstuff.add_validator(f"validator{i}")
        
        # Safety threshold should be 2f+1 = 5 for 7 validators
        threshold = hotstuff._get_safety_threshold()
        assert threshold == 5

    def test_propose_block_success(self):
        """Test successful block proposal."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        hotstuff.add_validator("validator1")
        hotstuff.add_validator("validator2")
        hotstuff.add_validator("validator3")
        hotstuff.add_validator("validator4")
        
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "proposer_id": "validator1"  # Current leader
        }
        
        result = hotstuff.propose_block(block_data)
        assert result.success is True
        assert result.validator_id == "validator1"

    def test_propose_block_not_leader(self):
        """Test block proposal by non-leader."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        hotstuff.add_validator("validator1")
        hotstuff.add_validator("validator2")
        
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "proposer_id": "validator2"  # Not the leader
        }
        
        result = hotstuff.propose_block(block_data)
        assert result.success is False
        assert "Not current leader" in result.error_message

    def test_view_change(self):
        """Test view change."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        hotstuff.add_validator("validator1")
        hotstuff.add_validator("validator2")
        
        original_view = hotstuff.state.current_view
        
        # Start view change
        success = hotstuff.start_view_change()
        assert success is True
        assert hotstuff.state.current_view > original_view

    def test_max_view_changes(self):
        """Test maximum view changes limit."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.HOTSTUFF,
            hotstuff_max_view_changes=2
        )
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        hotstuff.add_validator("validator1")
        hotstuff.add_validator("validator2")
        
        # Exceed max view changes
        hotstuff.state.view_change_counter = 2
        
        # Try to start another view change
        success = hotstuff.start_view_change()
        assert success is False

    def test_hotstuff_phases(self):
        """Test HotStuff consensus phases."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        for i in range(4):  # Need at least 4 for safety threshold
            hotstuff.add_validator(f"validator{i}")
        
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "proposer_id": "validator0"
        }
        
        # Mock the protocol execution to test phases
        with patch.object(hotstuff, '_execute_hotstuff_protocol', return_value={"success": True, "phase": "decide"}):
            result = hotstuff.propose_block(block_data)
            assert result.success is True
            assert result.metadata["phase"] == "decide"

    def test_get_hotstuff_statistics(self):
        """Test HotStuff statistics."""
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        hotstuff = HotStuffConsensus(config)
        
        # Add validators
        hotstuff.add_validator("validator1")
        hotstuff.add_validator("validator2")
        
        stats = hotstuff.get_hotstuff_statistics()
        assert stats["current_view"] == 0
        assert stats["current_leader"] == "validator1"
        assert stats["validator_count"] == 2
        assert stats["safety_threshold"] == 1  # 2f+1 for 2 validators


class TestConsensusIntegration:
    """Test integration between consensus mechanisms and engine."""

    def test_consensus_engine_poa(self):
        """Test consensus engine with PoA."""
        from src.dubchain.consensus.consensus_engine import ConsensusEngine
        
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_AUTHORITY,
            poa_authority_set=["auth1", "auth2"]
        )
        engine = ConsensusEngine(config)
        
        assert isinstance(engine.consensus_mechanism, ProofOfAuthority)
        assert len(engine.consensus_mechanism.state.authorities) == 2

    def test_consensus_engine_poh(self):
        """Test consensus engine with PoH."""
        from src.dubchain.consensus.consensus_engine import ConsensusEngine
        
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_HISTORY)
        engine = ConsensusEngine(config)
        
        assert isinstance(engine.consensus_mechanism, ProofOfHistory)

    def test_consensus_engine_pospace(self):
        """Test consensus engine with PoSpace."""
        from src.dubchain.consensus.consensus_engine import ConsensusEngine
        
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_SPACE_TIME)
        engine = ConsensusEngine(config)
        
        assert isinstance(engine.consensus_mechanism, ProofOfSpaceTime)

    def test_consensus_engine_hotstuff(self):
        """Test consensus engine with HotStuff."""
        from src.dubchain.consensus.consensus_engine import ConsensusEngine
        
        config = ConsensusConfig(consensus_type=ConsensusType.HOTSTUFF)
        engine = ConsensusEngine(config)
        
        assert isinstance(engine.consensus_mechanism, HotStuffConsensus)

    def test_consensus_switching(self):
        """Test switching between consensus mechanisms."""
        from src.dubchain.consensus.consensus_engine import ConsensusEngine
        
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_AUTHORITY)
        engine = ConsensusEngine(config)
        
        # Switch to HotStuff
        success = engine.switch_consensus(ConsensusType.HOTSTUFF)
        assert success is True
        assert isinstance(engine.consensus_mechanism, HotStuffConsensus)
        assert engine.config.consensus_type == ConsensusType.HOTSTUFF
