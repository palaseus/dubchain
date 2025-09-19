"""
Unit tests for hybrid consensus module.
"""

import time
from unittest.mock import Mock, patch

import pytest

from dubchain.consensus.consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusState,
    ConsensusType,
)
from dubchain.consensus.hybrid_consensus import (
    ConsensusSelector,
    ConsensusSwitcher,
    HybridConsensus,
)
from dubchain.consensus.validator import Validator, ValidatorInfo


class TestConsensusSelector:
    """Test ConsensusSelector class."""

    @pytest.fixture
    def consensus_selector(self):
        """Create a ConsensusSelector instance for testing."""
        return ConsensusSelector()

    def test_consensus_selector_creation(self, consensus_selector):
        """Test ConsensusSelector creation."""
        assert consensus_selector.network_size_threshold == 50
        assert consensus_selector.latency_threshold == 100.0
        assert consensus_selector.fault_tolerance_requirement == 0.33
        assert consensus_selector.energy_efficiency_weight == 0.3
        assert consensus_selector.security_weight == 0.4
        assert consensus_selector.performance_weight == 0.3

    def test_consensus_selector_select_consensus_small_network(
        self, consensus_selector
    ):
        """Test consensus selection for small network."""
        network_conditions = {
            "network_size": 5,
            "average_latency": 50.0,
            "fault_tolerance": 0.5,
        }

        result = consensus_selector.select_consensus(network_conditions)
        assert result == ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE

    def test_consensus_selector_select_consensus_medium_network(
        self, consensus_selector
    ):
        """Test consensus selection for medium network."""
        network_conditions = {
            "network_size": 25,
            "average_latency": 50.0,
            "fault_tolerance": 0.3,
        }

        result = consensus_selector.select_consensus(network_conditions)
        assert result == ConsensusType.DELEGATED_PROOF_OF_STAKE

    def test_consensus_selector_select_consensus_large_network(
        self, consensus_selector
    ):
        """Test consensus selection for large network."""
        network_conditions = {
            "network_size": 100,
            "average_latency": 50.0,
            "fault_tolerance": 0.2,
        }

        result = consensus_selector.select_consensus(network_conditions)
        assert result == ConsensusType.PROOF_OF_STAKE

    def test_consensus_selector_select_consensus_very_large_network(
        self, consensus_selector
    ):
        """Test consensus selection for very large network."""
        network_conditions = {
            "network_size": 300,
            "average_latency": 50.0,
            "fault_tolerance": 0.1,
        }

        result = consensus_selector.select_consensus(network_conditions)
        assert result == ConsensusType.HYBRID

    def test_consensus_selector_calculate_consensus_score_pos(self, consensus_selector):
        """Test calculating score for Proof of Stake."""
        network_conditions = {"network_size": 100, "average_latency": 50.0}

        score = consensus_selector.calculate_consensus_score(
            ConsensusType.PROOF_OF_STAKE, network_conditions
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_consensus_selector_calculate_consensus_score_dpos(
        self, consensus_selector
    ):
        """Test calculating score for Delegated Proof of Stake."""
        network_conditions = {"network_size": 25, "average_latency": 30.0}

        score = consensus_selector.calculate_consensus_score(
            ConsensusType.DELEGATED_PROOF_OF_STAKE, network_conditions
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_consensus_selector_calculate_consensus_score_pbft(
        self, consensus_selector
    ):
        """Test calculating score for PBFT."""
        network_conditions = {"network_size": 10, "fault_tolerance": 0.4}

        score = consensus_selector.calculate_consensus_score(
            ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE, network_conditions
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_consensus_selector_calculate_consensus_score_unknown_type(
        self, consensus_selector
    ):
        """Test calculating score for unknown consensus type."""
        network_conditions = {"network_size": 50}

        score = consensus_selector.calculate_consensus_score(
            ConsensusType.HYBRID, network_conditions
        )
        assert score == 0.0


class TestConsensusSwitcher:
    """Test ConsensusSwitcher class."""

    @pytest.fixture
    def consensus_switcher(self):
        """Create a ConsensusSwitcher instance for testing."""
        return ConsensusSwitcher()

    def test_consensus_switcher_creation(self, consensus_switcher):
        """Test ConsensusSwitcher creation."""
        assert consensus_switcher.switch_cooldown == 300.0
        assert consensus_switcher.performance_window == 100
        assert consensus_switcher.performance_history == []

    def test_consensus_switcher_can_switch_immediately(self, consensus_switcher):
        """Test that consensus can switch immediately after creation."""
        # Set last_switch_time to past
        consensus_switcher.last_switch_time = time.time() - 400
        assert consensus_switcher.can_switch() is True

    def test_consensus_switcher_cannot_switch_during_cooldown(self, consensus_switcher):
        """Test that consensus cannot switch during cooldown period."""
        # Set last_switch_time to recent
        consensus_switcher.last_switch_time = time.time() - 100
        assert consensus_switcher.can_switch() is False

    def test_consensus_switcher_record_performance(self, consensus_switcher):
        """Test recording performance metrics."""
        consensus_switcher.record_performance(
            ConsensusType.PROOF_OF_STAKE, 1.5, True, 500000
        )

        assert len(consensus_switcher.performance_history) == 1
        performance_data = consensus_switcher.performance_history[0]
        assert performance_data["consensus_type"] == ConsensusType.PROOF_OF_STAKE
        assert performance_data["block_time"] == 1.5
        assert performance_data["success"] is True
        assert performance_data["gas_used"] == 500000

    def test_consensus_switcher_record_performance_history_limit(
        self, consensus_switcher
    ):
        """Test performance history size limit."""
        # Add more than performance_window records
        for i in range(consensus_switcher.performance_window + 50):
            consensus_switcher.record_performance(
                ConsensusType.PROOF_OF_STAKE, 1.0, True, 500000
            )

        assert (
            len(consensus_switcher.performance_history)
            == consensus_switcher.performance_window
        )

    def test_consensus_switcher_should_switch_consensus_during_cooldown(
        self, consensus_switcher
    ):
        """Test that consensus should not switch during cooldown."""
        consensus_switcher.last_switch_time = time.time() - 100  # Recent switch

        should_switch, new_consensus = consensus_switcher.should_switch_consensus(
            ConsensusType.PROOF_OF_STAKE, {"network_size": 50}
        )

        assert should_switch is False
        assert new_consensus == ConsensusType.PROOF_OF_STAKE

    def test_consensus_switcher_should_switch_consensus_no_history(
        self, consensus_switcher
    ):
        """Test that consensus should not switch with no performance history."""
        consensus_switcher.last_switch_time = time.time() - 400  # Past switch

        should_switch, new_consensus = consensus_switcher.should_switch_consensus(
            ConsensusType.PROOF_OF_STAKE, {"network_size": 50}
        )

        assert should_switch is False
        assert new_consensus == ConsensusType.PROOF_OF_STAKE

    def test_consensus_switcher_should_switch_consensus_poor_performance(
        self, consensus_switcher
    ):
        """Test that consensus should switch with poor performance."""
        consensus_switcher.last_switch_time = time.time() - 400  # Past switch

        # Add poor performance history - mostly failures (success rate < 80%)
        for i in range(60):
            consensus_switcher.record_performance(
                ConsensusType.PROOF_OF_STAKE,
                15.0,  # High block time (> 10.0 threshold)
                i % 5 == 0,  # 20% success rate (below 80% threshold)
                500000,
            )

        should_switch, new_consensus = consensus_switcher.should_switch_consensus(
            ConsensusType.PROOF_OF_STAKE,
            {"network_size": 25},  # This will select DPoS instead of PoS
        )

        assert should_switch is True
        assert new_consensus != ConsensusType.PROOF_OF_STAKE

    def test_consensus_switcher_should_switch_consensus_good_performance(
        self, consensus_switcher
    ):
        """Test that consensus should not switch with good performance."""
        consensus_switcher.last_switch_time = time.time() - 400  # Past switch

        # Add good performance history
        for i in range(60):
            consensus_switcher.record_performance(
                ConsensusType.PROOF_OF_STAKE,
                2.0,  # Low block time
                True,  # High success rate
                500000,
            )

        should_switch, new_consensus = consensus_switcher.should_switch_consensus(
            ConsensusType.PROOF_OF_STAKE, {"network_size": 50}
        )

        assert should_switch is False
        assert new_consensus == ConsensusType.PROOF_OF_STAKE


class TestHybridConsensus:
    """Test HybridConsensus class."""

    @pytest.fixture
    def consensus_config(self):
        """Create a consensus config for testing."""
        return ConsensusConfig(
            consensus_type=ConsensusType.HYBRID,
            max_validators=10,
            block_time=15.0,
            max_gas_per_block=1000000,
            enable_hybrid=True,
        )

    @pytest.fixture
    def hybrid_consensus(self, consensus_config):
        """Create a HybridConsensus instance for testing."""
        return HybridConsensus(consensus_config)

    def test_hybrid_consensus_creation(self, consensus_config):
        """Test HybridConsensus creation."""
        hybrid = HybridConsensus(consensus_config)

        assert hybrid.config == consensus_config
        assert hybrid.current_consensus_type == ConsensusType.HYBRID
        assert hybrid.consensus_state is not None
        assert len(hybrid.consensus_mechanisms) == 3  # PoS, DPoS, PBFT
        assert hybrid.selector is not None
        assert hybrid.switcher is not None
        assert hybrid.network_conditions == {}
        assert hybrid.switch_count == 0

    def test_hybrid_consensus_initialize_consensus_mechanisms(self, hybrid_consensus):
        """Test initialization of consensus mechanisms."""
        mechanisms = hybrid_consensus.consensus_mechanisms

        assert ConsensusType.PROOF_OF_STAKE in mechanisms
        assert ConsensusType.DELEGATED_PROOF_OF_STAKE in mechanisms
        assert ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE in mechanisms

        # Test that mechanisms are properly initialized
        for mechanism in mechanisms.values():
            assert mechanism is not None

    def test_hybrid_consensus_update_network_conditions(self, hybrid_consensus):
        """Test updating network conditions."""
        conditions = {
            "network_size": 50,
            "average_latency": 100.0,
            "fault_tolerance": 0.3,
        }

        hybrid_consensus.update_network_conditions(conditions)

        assert hybrid_consensus.network_conditions == conditions
        assert hybrid_consensus.last_condition_update > 0

    def test_hybrid_consensus_switch_consensus_same_type(self, hybrid_consensus):
        """Test switching to the same consensus type."""
        # First switch to a valid type, then test switching to the same type
        with patch.object(hybrid_consensus, "_migrate_validators", return_value=True):
            result = hybrid_consensus.switch_consensus(ConsensusType.PROOF_OF_STAKE)
            assert result is True

            # Now test switching to the same type (should return True immediately)
            result = hybrid_consensus.switch_consensus(ConsensusType.PROOF_OF_STAKE)
            assert result is True

    def test_hybrid_consensus_switch_consensus_invalid_type(self, hybrid_consensus):
        """Test switching to invalid consensus type."""
        result = hybrid_consensus.switch_consensus("INVALID_TYPE")
        assert result is False

    def test_hybrid_consensus_switch_consensus_valid_type(self, hybrid_consensus):
        """Test switching to valid consensus type."""
        with patch.object(
            hybrid_consensus, "_migrate_validators", return_value=True
        ) as mock_migrate:
            result = hybrid_consensus.switch_consensus(ConsensusType.PROOF_OF_STAKE)

            assert result is True
            assert (
                hybrid_consensus.current_consensus_type == ConsensusType.PROOF_OF_STAKE
            )
            assert hybrid_consensus.switch_count == 1
            mock_migrate.assert_called_once_with(ConsensusType.PROOF_OF_STAKE)

    def test_hybrid_consensus_switch_consensus_migration_failure(
        self, hybrid_consensus
    ):
        """Test switching consensus with migration failure."""
        with patch.object(hybrid_consensus, "_migrate_validators", return_value=False):
            result = hybrid_consensus.switch_consensus(ConsensusType.PROOF_OF_STAKE)
            assert result is False

    def test_hybrid_consensus_migrate_validators_success(self, hybrid_consensus):
        """Test successful validator migration."""
        # Mock validator set
        mock_validator_info = Mock()
        mock_validator_info.validator_id = "validator_1"
        mock_validator_info.total_stake = 1000

        # Set current consensus type to PoS
        hybrid_consensus.current_consensus_type = ConsensusType.PROOF_OF_STAKE

        # Add a validator to the current consensus
        pos_consensus = hybrid_consensus.consensus_mechanisms[
            ConsensusType.PROOF_OF_STAKE
        ]
        pos_consensus.validator_set.validators["validator_1"] = mock_validator_info

        # Mock the Validator constructor to avoid the private key requirement
        with patch("dubchain.consensus.validator.Validator") as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator

            # Mock the new consensus mechanism
            with patch.object(
                hybrid_consensus.consensus_mechanisms[
                    ConsensusType.DELEGATED_PROOF_OF_STAKE
                ],
                "register_delegate",
            ) as mock_register:
                result = hybrid_consensus._migrate_validators(
                    ConsensusType.DELEGATED_PROOF_OF_STAKE
                )

                # The migration should succeed when Validator constructor is mocked
                assert result is True
                mock_register.assert_called_once()

    def test_hybrid_consensus_migrate_validators_exception(self, hybrid_consensus):
        """Test validator migration with exception."""
        # Set current consensus type to PoS
        hybrid_consensus.current_consensus_type = ConsensusType.PROOF_OF_STAKE

        # Mock validator set to raise exception when accessing validators
        mock_validator_set = Mock()
        mock_validator_set.validators = Mock(side_effect=Exception("Migration failed"))

        with patch.object(
            hybrid_consensus.consensus_mechanisms[ConsensusType.PROOF_OF_STAKE],
            "validator_set",
            mock_validator_set,
        ):
            result = hybrid_consensus._migrate_validators(
                ConsensusType.DELEGATED_PROOF_OF_STAKE
            )
            assert result is False

    def test_hybrid_consensus_propose_block_hybrid_type(self, hybrid_consensus):
        """Test block proposal with hybrid consensus type."""
        block_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
        }

        # The hybrid consensus type will return "Unknown consensus type" error
        result = hybrid_consensus.propose_block(block_data)

        assert result.success is False
        assert result.error_message == "Unknown consensus type"

    def test_hybrid_consensus_propose_block_no_mechanisms(self, hybrid_consensus):
        """Test block proposal with no consensus mechanisms."""
        hybrid_consensus.consensus_mechanisms = {}
        block_data = {"block_number": 1}

        result = hybrid_consensus.propose_block(block_data)

        assert result.success is False
        assert result.error_message == "No consensus mechanisms available"

    def test_hybrid_consensus_propose_block_proof_of_stake(self, hybrid_consensus):
        """Test block proposal with Proof of Stake."""
        hybrid_consensus.current_consensus_type = ConsensusType.PROOF_OF_STAKE
        block_data = {"block_number": 1}

        mock_result = ConsensusResult(
            success=True, consensus_type=ConsensusType.PROOF_OF_STAKE
        )

        with patch.object(
            hybrid_consensus.consensus_mechanisms[ConsensusType.PROOF_OF_STAKE],
            "select_proposer",
            return_value="validator_1",
        ):
            with patch.object(
                hybrid_consensus.consensus_mechanisms[ConsensusType.PROOF_OF_STAKE],
                "finalize_block",
                return_value=mock_result,
            ):
                result = hybrid_consensus.propose_block(block_data)

                assert result == mock_result

    def test_hybrid_consensus_propose_block_no_proposer(self, hybrid_consensus):
        """Test block proposal with no proposer available."""
        hybrid_consensus.current_consensus_type = ConsensusType.PROOF_OF_STAKE
        block_data = {"block_number": 1}

        with patch.object(
            hybrid_consensus.consensus_mechanisms[ConsensusType.PROOF_OF_STAKE],
            "select_proposer",
            return_value=None,
        ):
            result = hybrid_consensus.propose_block(block_data)

            assert result.success is False
            assert result.error_message == "No proposer available"

    def test_hybrid_consensus_propose_block_delegated_proof_of_stake(
        self, hybrid_consensus
    ):
        """Test block proposal with Delegated Proof of Stake."""
        hybrid_consensus.current_consensus_type = ConsensusType.DELEGATED_PROOF_OF_STAKE
        block_data = {"block_number": 1}

        mock_result = ConsensusResult(
            success=True, consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE
        )

        with patch.object(
            hybrid_consensus.consensus_mechanisms[
                ConsensusType.DELEGATED_PROOF_OF_STAKE
            ],
            "produce_block",
            return_value=mock_result,
        ):
            result = hybrid_consensus.propose_block(block_data)

            assert result == mock_result

    def test_hybrid_consensus_propose_block_pbft(self, hybrid_consensus):
        """Test block proposal with PBFT."""
        hybrid_consensus.current_consensus_type = (
            ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
        )
        block_data = {"block_number": 1}

        mock_result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE,
        )

        with patch.object(
            hybrid_consensus.consensus_mechanisms[
                ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
            ],
            "start_consensus",
            return_value=mock_result,
        ):
            result = hybrid_consensus.propose_block(block_data)

            assert result == mock_result

    def test_hybrid_consensus_propose_block_unknown_type(self, hybrid_consensus):
        """Test block proposal with unknown consensus type."""
        # Set to a type that's not in the mechanisms dict
        hybrid_consensus.current_consensus_type = ConsensusType.HYBRID
        # Remove all mechanisms to simulate unknown type
        hybrid_consensus.consensus_mechanisms = {}
        block_data = {"block_number": 1}

        result = hybrid_consensus.propose_block(block_data)

        assert result.success is False
        assert result.error_message == "No consensus mechanisms available"

    def test_hybrid_consensus_get_consensus_info(self, hybrid_consensus):
        """Test getting consensus information."""
        info = hybrid_consensus.get_consensus_info()

        assert isinstance(info, dict)
        assert "current_consensus" in info
        assert "switch_count" in info
        assert "can_switch" in info
        assert "network_conditions" in info
        assert "performance_history_length" in info

    def test_hybrid_consensus_get_consensus_info_no_mechanisms(self, hybrid_consensus):
        """Test getting consensus info with no mechanisms."""
        hybrid_consensus.consensus_mechanisms = {}

        info = hybrid_consensus.get_consensus_info()

        assert "error" in info
        assert info["error"] == "No consensus mechanisms available"

    def test_hybrid_consensus_get_consensus_metrics(self, hybrid_consensus):
        """Test getting consensus metrics."""
        # Mock metrics from mechanisms
        mock_metrics = ConsensusMetrics(consensus_type=ConsensusType.PROOF_OF_STAKE)
        mock_metrics.total_blocks = 10
        mock_metrics.successful_blocks = 8
        mock_metrics.validator_count = 5
        mock_metrics.active_validators = 3

        with patch.object(
            hybrid_consensus.consensus_mechanisms[ConsensusType.PROOF_OF_STAKE],
            "get_consensus_metrics",
            return_value=mock_metrics,
        ):
            with patch.object(
                hybrid_consensus.consensus_mechanisms[
                    ConsensusType.DELEGATED_PROOF_OF_STAKE
                ],
                "get_consensus_metrics",
                return_value=mock_metrics,
            ):
                with patch.object(
                    hybrid_consensus.consensus_mechanisms[
                        ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE
                    ],
                    "get_consensus_metrics",
                    return_value=mock_metrics,
                ):
                    metrics = hybrid_consensus.get_consensus_metrics()

                    assert metrics.total_blocks == 30  # 10 * 3 mechanisms
                    assert metrics.successful_blocks == 24  # 8 * 3 mechanisms
                    assert metrics.validator_count == 15  # 5 * 3 mechanisms
                    assert metrics.active_validators == 9  # 3 * 3 mechanisms
                    assert metrics.consensus_type == ConsensusType.HYBRID

    def test_hybrid_consensus_to_dict(self, hybrid_consensus):
        """Test converting to dictionary."""
        data = hybrid_consensus.to_dict()

        assert isinstance(data, dict)
        assert "config" in data
        assert "current_consensus_type" in data
        assert "consensus_state" in data
        assert "switcher" in data
        assert "network_conditions" in data
        assert "switch_count" in data
        assert "metrics" in data

    def test_hybrid_consensus_from_dict(self, hybrid_consensus):
        """Test creating from dictionary."""
        data = hybrid_consensus.to_dict()

        with patch.object(ConsensusConfig, "from_dict") as mock_config_from_dict:
            new_hybrid = HybridConsensus.from_dict(data)

            mock_config_from_dict.assert_called_once_with(data["config"])
            assert new_hybrid.current_consensus_type == ConsensusType(
                data["current_consensus_type"]
            )
            assert new_hybrid.switch_count == data["switch_count"]
