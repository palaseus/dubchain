"""
Unit tests for consensus engine module.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from dubchain.consensus.consensus_engine import ConsensusEngine
from dubchain.consensus.consensus_types import (
    ConsensusConfig,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusState,
    ConsensusType,
)
from dubchain.consensus.validator import Validator, ValidatorInfo, ValidatorManager


class TestConsensusEngine:
    """Test ConsensusEngine class."""

    @pytest.fixture
    def consensus_config(self):
        """Create a consensus config for testing."""
        return ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            max_validators=10,
            block_time=15.0,
            max_gas_per_block=1000000,
        )

    @pytest.fixture
    def consensus_engine(self, consensus_config):
        """Create a ConsensusEngine instance for testing."""
        return ConsensusEngine(consensus_config)

    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator for testing."""
        validator = Mock()
        validator.validator_id = "validator_1"
        validator.public_key = "pub_key_1"
        return validator

    @pytest.fixture
    def valid_block_data(self):
        """Create valid block data for testing."""
        return {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "gas_used": 500000,
        }

    def test_consensus_engine_creation(self, consensus_config):
        """Test ConsensusEngine creation."""
        engine = ConsensusEngine(consensus_config)

        assert engine.config == consensus_config
        assert engine.consensus_state is not None
        assert engine.consensus_mechanism is not None
        assert engine.validator_manager is not None
        assert engine.performance_history == []
        assert engine.max_history_size == 1000
        assert engine.metrics is not None

    def test_consensus_engine_creation_with_custom_config(self):
        """Test ConsensusEngine creation with custom config."""
        config = ConsensusConfig(
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            max_validators=100,
            block_time=15.0,
        )
        engine = ConsensusEngine(config)

        assert engine.config.consensus_type == ConsensusType.PROOF_OF_STAKE
        assert engine.config.max_validators == 100
        assert engine.config.block_time == 15.0

    def test_consensus_engine_create_consensus_mechanism_proof_of_stake(self):
        """Test creating Proof of Stake consensus mechanism."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_STAKE)
        engine = ConsensusEngine(config)

        # Test that the mechanism was created
        assert engine.consensus_mechanism is not None
        assert hasattr(engine.consensus_mechanism, "register_validator")

    def test_consensus_engine_create_consensus_mechanism_delegated_proof_of_stake(self):
        """Test creating Delegated Proof of Stake consensus mechanism."""
        config = ConsensusConfig(consensus_type=ConsensusType.DELEGATED_PROOF_OF_STAKE)
        engine = ConsensusEngine(config)

        # Test that the mechanism was created
        assert engine.consensus_mechanism is not None
        # Test that it has some common methods
        assert hasattr(engine.consensus_mechanism, "__class__")

    def test_consensus_engine_create_consensus_mechanism_hybrid(self):
        """Test creating Hybrid consensus mechanism."""
        config = ConsensusConfig(consensus_type=ConsensusType.HYBRID)
        engine = ConsensusEngine(config)

        # Test that the mechanism was created
        assert engine.consensus_mechanism is not None
        # Test that it has some common methods
        assert hasattr(engine.consensus_mechanism, "__class__")

    def test_consensus_engine_register_validator_success(
        self, consensus_engine, mock_validator
    ):
        """Test successful validator registration."""
        with patch.object(
            consensus_engine.validator_manager, "register_validator", return_value=True
        ) as mock_register:
            with patch.object(
                consensus_engine.consensus_mechanism,
                "register_validator",
                return_value=True,
            ) as mock_consensus_register:
                result = consensus_engine.register_validator(mock_validator, 1000)

                assert result is True
                mock_register.assert_called_once_with(mock_validator, 1000)
                mock_consensus_register.assert_called_once_with(mock_validator, 1000)

    def test_consensus_engine_register_validator_validator_manager_failure(
        self, consensus_engine, mock_validator
    ):
        """Test validator registration failure at validator manager level."""
        with patch.object(
            consensus_engine.validator_manager, "register_validator", return_value=False
        ):
            result = consensus_engine.register_validator(mock_validator, 1000)
            assert result is False

    def test_consensus_engine_stake_to_validator(self):
        """Test staking to validator."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "stake_to_validator")
        assert callable(engine.stake_to_validator)

    def test_consensus_engine_propose_block_success(
        self, consensus_engine, valid_block_data
    ):
        """Test successful block proposal."""
        mock_result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            gas_used=500000,
            validator_id="validator_1",
        )

        with patch.object(
            consensus_engine.consensus_mechanism,
            "select_proposer",
            return_value="validator_1",
        ) as mock_select:
            with patch.object(
                consensus_engine.consensus_mechanism,
                "finalize_block",
                return_value=mock_result,
            ) as mock_finalize:
                with patch.object(
                    consensus_engine, "_validate_block_data", return_value=True
                ) as mock_validate:
                    with patch.object(
                        consensus_engine, "_record_performance"
                    ) as mock_record:
                        result = consensus_engine.propose_block(valid_block_data)

                        assert result == mock_result
                        mock_validate.assert_called_once_with(valid_block_data)
                        mock_select.assert_called_once_with(1)
                        mock_finalize.assert_called_once_with(
                            valid_block_data, "validator_1"
                        )
                        mock_record.assert_called_once()

    def test_consensus_engine_propose_block_invalid_data(
        self, consensus_engine, valid_block_data
    ):
        """Test block proposal with invalid data."""
        with patch.object(consensus_engine, "_validate_block_data", return_value=False):
            result = consensus_engine.propose_block(valid_block_data)

            assert result.success is False
            assert result.error_message == "Invalid block data"
            assert result.consensus_type == consensus_engine.config.consensus_type

    def test_consensus_engine_propose_block_async(self):
        """Test async block proposal."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and is async
        assert hasattr(engine, "propose_block_async")
        assert callable(engine.propose_block_async)

    def test_consensus_engine_get_validator_info(self):
        """Test getting validator info."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "get_validator_info")
        assert callable(engine.get_validator_info)

    def test_consensus_engine_get_active_validators(self):
        """Test getting active validators."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "get_active_validators")
        assert callable(engine.get_active_validators)

    def test_consensus_engine_get_consensus_metrics(self):
        """Test getting consensus metrics."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "get_consensus_metrics")
        assert callable(engine.get_consensus_metrics)

        # Test that it returns ConsensusMetrics
        metrics = engine.get_consensus_metrics()
        assert isinstance(metrics, ConsensusMetrics)

    def test_consensus_engine_get_performance_statistics(self):
        """Test getting performance statistics."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "get_performance_statistics")
        assert callable(engine.get_performance_statistics)

        # Test that it returns a dictionary
        stats = engine.get_performance_statistics()
        assert isinstance(stats, dict)

    def test_consensus_engine_switch_consensus(self):
        """Test switching consensus mechanism."""
        config = ConsensusConfig(consensus_type=ConsensusType.PROOF_OF_STAKE)
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "switch_consensus")
        assert callable(engine.switch_consensus)

    def test_consensus_engine_get_consensus_info(self):
        """Test getting consensus info."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "get_consensus_info")
        assert callable(engine.get_consensus_info)

        # Test that it returns a dictionary
        info = engine.get_consensus_info()
        assert isinstance(info, dict)

    def test_consensus_engine_shutdown(self):
        """Test shutting down consensus engine."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "shutdown")
        assert callable(engine.shutdown)

    def test_consensus_engine_to_dict(self):
        """Test converting to dictionary."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "to_dict")
        assert callable(engine.to_dict)

        # Test that it returns a dictionary
        data = engine.to_dict()
        assert isinstance(data, dict)

    def test_consensus_engine_from_dict(self):
        """Test creating from dictionary."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the class method exists and can be called
        assert hasattr(ConsensusEngine, "from_dict")
        assert callable(ConsensusEngine.from_dict)

    def test_consensus_engine_validate_block_data_valid(
        self, consensus_engine, valid_block_data
    ):
        """Test validating valid block data."""
        result = consensus_engine._validate_block_data(valid_block_data)
        assert result is True

    def test_consensus_engine_validate_block_data_missing_fields(
        self, consensus_engine
    ):
        """Test validating block data with missing fields."""
        invalid_data = {
            "block_number": 1,
            "timestamp": time.time()
            # Missing 'transactions' and 'previous_hash'
        }

        result = consensus_engine._validate_block_data(invalid_data)
        assert result is False

    def test_consensus_engine_validate_block_data_invalid_timestamp(
        self, consensus_engine
    ):
        """Test validating block data with invalid timestamp."""
        invalid_data = {
            "block_number": 1,
            "timestamp": time.time() - 400,  # More than 5 minutes old
            "transactions": [],
            "previous_hash": "0x123",
            "gas_used": 500000,
        }

        result = consensus_engine._validate_block_data(invalid_data)
        assert result is False

    def test_consensus_engine_validate_block_data_excessive_gas(self, consensus_engine):
        """Test validating block data with excessive gas usage."""
        invalid_data = {
            "block_number": 1,
            "timestamp": time.time(),
            "transactions": [],
            "previous_hash": "0x123",
            "gas_used": 2000000,  # Exceeds max_gas_per_block
        }

        result = consensus_engine._validate_block_data(invalid_data)
        assert result is False

    def test_consensus_engine_record_performance(self, consensus_engine):
        """Test recording performance metrics."""
        result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            gas_used=500000,
            validator_id="validator_1",
        )

        consensus_engine._record_performance(result, 1.5)

        assert len(consensus_engine.performance_history) == 1
        performance_data = consensus_engine.performance_history[0]
        assert performance_data["success"] is True
        assert performance_data["block_time"] == 1.5
        assert performance_data["gas_used"] == 500000
        assert performance_data["validator_id"] == "validator_1"

    def test_consensus_engine_migrate_validators(self):
        """Test migrating validators."""
        config = ConsensusConfig()
        engine = ConsensusEngine(config)

        # Test that the method exists and can be called
        assert hasattr(engine, "_migrate_validators")
        assert callable(engine._migrate_validators)

    def test_consensus_engine_create_consensus_mechanism_pbft(self):
        """Test creating PBFT consensus mechanism."""
        config = ConsensusConfig(consensus_type=ConsensusType.PRACTICAL_BYZANTINE_FAULT_TOLERANCE)
        engine = ConsensusEngine(config)

        # Test that the mechanism was created
        assert engine.consensus_mechanism is not None
        assert hasattr(engine.consensus_mechanism, "__class__")

    def test_consensus_engine_create_consensus_mechanism_default(self):
        """Test creating default consensus mechanism for unknown type."""
        # Create a mock consensus type that doesn't exist
        class UnknownConsensusType:
            pass
        
        config = ConsensusConfig()
        config.consensus_type = UnknownConsensusType()
        engine = ConsensusEngine(config)

        # Should default to Proof of Stake
        assert engine.consensus_mechanism is not None
        assert hasattr(engine.consensus_mechanism, "__class__")

    def test_consensus_engine_register_validator_consensus_mechanism_failure(
        self, consensus_engine, mock_validator
    ):
        """Test validator registration failure at consensus mechanism level."""
        with patch.object(
            consensus_engine.validator_manager, "register_validator", return_value=True
        ):
            with patch.object(
                consensus_engine.consensus_mechanism,
                "register_validator",
                return_value=False,
            ):
                result = consensus_engine.register_validator(mock_validator, 1000)
                assert result is False

    def test_consensus_engine_register_validator_with_delegate_method(
        self, consensus_engine, mock_validator
    ):
        """Test validator registration with register_delegate method."""
        # Create a new mock consensus mechanism with register_delegate
        mock_consensus = Mock()
        mock_consensus.register_delegate.return_value = True
        # Remove register_validator method to force use of register_delegate
        del mock_consensus.register_validator
        consensus_engine.consensus_mechanism = mock_consensus
        
        with patch.object(
            consensus_engine.validator_manager, "register_validator", return_value=True
        ):
            result = consensus_engine.register_validator(mock_validator, 1000)
            assert result is True
            mock_consensus.register_delegate.assert_called_once_with(
                mock_validator, 1000
            )

    def test_consensus_engine_register_validator_with_add_validator_method(
        self, consensus_engine, mock_validator
    ):
        """Test validator registration with add_validator method."""
        # Create a new mock consensus mechanism with add_validator
        mock_consensus = Mock()
        mock_consensus.add_validator.return_value = True
        # Remove other methods to force use of add_validator
        del mock_consensus.register_validator
        del mock_consensus.register_delegate
        consensus_engine.consensus_mechanism = mock_consensus
        
        with patch.object(
            consensus_engine.validator_manager, "register_validator", return_value=True
        ):
            result = consensus_engine.register_validator(mock_validator, 1000)
            assert result is True
            mock_consensus.add_validator.assert_called_once_with(
                mock_validator
            )

    def test_consensus_engine_stake_to_validator_success(self, consensus_engine):
        """Test successful staking to validator."""
        with patch.object(
            consensus_engine.validator_manager, "stake", return_value=True
        ) as mock_stake:
            with patch.object(
                consensus_engine.consensus_mechanism,
                "stake_to_validator",
                return_value=True,
            ) as mock_consensus_stake:
                result = consensus_engine.stake_to_validator("validator_1", "delegator_1", 1000)

                assert result is True
                mock_stake.assert_called_once_with("validator_1", "delegator_1", 1000)
                mock_consensus_stake.assert_called_once_with("validator_1", "delegator_1", 1000)

    def test_consensus_engine_stake_to_validator_validator_manager_failure(self, consensus_engine):
        """Test staking failure at validator manager level."""
        with patch.object(
            consensus_engine.validator_manager, "stake", return_value=False
        ):
            result = consensus_engine.stake_to_validator("validator_1", "delegator_1", 1000)
            assert result is False

    def test_consensus_engine_stake_to_validator_with_vote_for_delegate_method(self, consensus_engine):
        """Test staking with vote_for_delegate method."""
        # Create a new mock consensus mechanism with vote_for_delegate
        mock_consensus = Mock()
        mock_consensus.vote_for_delegate.return_value = True
        # Remove stake_to_validator method to force use of vote_for_delegate
        del mock_consensus.stake_to_validator
        consensus_engine.consensus_mechanism = mock_consensus
        
        with patch.object(
            consensus_engine.validator_manager, "stake", return_value=True
        ):
            result = consensus_engine.stake_to_validator("validator_1", "delegator_1", 1000)
            assert result is True
            mock_consensus.vote_for_delegate.assert_called_once_with(
                "delegator_1", "validator_1", 1000
            )

    def test_consensus_engine_propose_block_with_propose_block_method(
        self, consensus_engine, valid_block_data
    ):
        """Test block proposal with propose_block method."""
        mock_result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            gas_used=500000,
            validator_id="validator_1",
        )

        with patch.object(
            consensus_engine, "_validate_block_data", return_value=True
        ) as mock_validate:
            # Create a new mock consensus mechanism with propose_block
            mock_consensus = Mock()
            mock_consensus.propose_block = Mock(return_value=mock_result)
            consensus_engine.consensus_mechanism = mock_consensus
            
            with patch.object(
                consensus_engine, "_record_performance"
            ) as mock_record:
                result = consensus_engine.propose_block(valid_block_data)

                assert result == mock_result
                mock_validate.assert_called_once_with(valid_block_data)
                mock_consensus.propose_block.assert_called_once_with(valid_block_data)
                mock_record.assert_called_once()

    def test_consensus_engine_propose_block_with_produce_block_method(
        self, consensus_engine, valid_block_data
    ):
        """Test block proposal with produce_block method."""
        mock_result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            gas_used=500000,
            validator_id="validator_1",
        )

        with patch.object(
            consensus_engine, "_validate_block_data", return_value=True
        ) as mock_validate:
            # Create a mock consensus mechanism with only produce_block method
            mock_consensus = Mock()
            mock_consensus.produce_block.return_value = mock_result
            # Remove other methods to force use of produce_block
            del mock_consensus.propose_block
            del mock_consensus.finalize_block
            del mock_consensus.start_consensus
            consensus_engine.consensus_mechanism = mock_consensus
            
            with patch.object(
                consensus_engine, "_record_performance"
            ) as mock_record:
                result = consensus_engine.propose_block(valid_block_data)

                assert result == mock_result
                mock_validate.assert_called_once_with(valid_block_data)
                mock_consensus.produce_block.assert_called_once_with(valid_block_data)
                mock_record.assert_called_once()

    def test_consensus_engine_propose_block_with_start_consensus_method(
        self, consensus_engine, valid_block_data
    ):
        """Test block proposal with start_consensus method."""
        mock_result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            gas_used=500000,
            validator_id="validator_1",
        )

        with patch.object(
            consensus_engine, "_validate_block_data", return_value=True
        ) as mock_validate:
            # Create a mock consensus mechanism with only start_consensus method
            mock_consensus = Mock()
            mock_consensus.start_consensus.return_value = mock_result
            # Remove other methods to force use of start_consensus
            del mock_consensus.propose_block
            del mock_consensus.finalize_block
            del mock_consensus.produce_block
            consensus_engine.consensus_mechanism = mock_consensus
            
            with patch.object(
                consensus_engine, "_record_performance"
            ) as mock_record:
                result = consensus_engine.propose_block(valid_block_data)

                assert result == mock_result
                mock_validate.assert_called_once_with(valid_block_data)
                mock_consensus.start_consensus.assert_called_once_with(valid_block_data)
                mock_record.assert_called_once()

    def test_consensus_engine_propose_block_no_proposer_available(
        self, consensus_engine, valid_block_data
    ):
        """Test block proposal when no proposer is available."""
        with patch.object(
            consensus_engine, "_validate_block_data", return_value=True
        ) as mock_validate:
            with patch.object(
                consensus_engine.consensus_mechanism,
                "select_proposer",
                return_value=None,
            ) as mock_select:
                with patch.object(
                    consensus_engine, "_record_performance"
                ) as mock_record:
                    result = consensus_engine.propose_block(valid_block_data)

                    assert result.success is False
                    assert result.error_message == "No proposer available"
                    assert result.consensus_type == consensus_engine.config.consensus_type
                    mock_validate.assert_called_once_with(valid_block_data)
                    mock_select.assert_called_once_with(1)
                    mock_record.assert_called_once()

    def test_consensus_engine_propose_block_unsupported_mechanism(
        self, consensus_engine, valid_block_data
    ):
        """Test block proposal with unsupported consensus mechanism."""
        with patch.object(
            consensus_engine, "_validate_block_data", return_value=True
        ) as mock_validate:
            # Create a mock consensus mechanism with no supported methods
            mock_consensus = Mock()
            # Remove all methods to simulate unsupported mechanism
            del mock_consensus.propose_block
            del mock_consensus.finalize_block
            del mock_consensus.produce_block
            del mock_consensus.start_consensus
            consensus_engine.consensus_mechanism = mock_consensus
            
            with patch.object(
                consensus_engine, "_record_performance"
            ) as mock_record:
                result = consensus_engine.propose_block(valid_block_data)

                assert result.success is False
                assert result.error_message == "Unsupported consensus mechanism"
                assert result.consensus_type == consensus_engine.config.consensus_type
                mock_validate.assert_called_once_with(valid_block_data)
                mock_record.assert_called_once()

    @pytest.mark.asyncio
    async def test_consensus_engine_propose_block_async_success(
        self, consensus_engine, valid_block_data
    ):
        """Test successful async block proposal."""
        mock_result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            gas_used=500000,
            validator_id="validator_1",
        )

        with patch.object(
            consensus_engine, "propose_block", return_value=mock_result
        ) as mock_propose:
            result = await consensus_engine.propose_block_async(valid_block_data)

            assert result == mock_result
            mock_propose.assert_called_once_with(valid_block_data)

    def test_consensus_engine_get_validator_info_success(self, consensus_engine):
        """Test getting validator info successfully."""
        mock_validator_info = Mock()
        consensus_engine.validator_manager.validator_set.validators = {
            "validator_1": mock_validator_info
        }

        result = consensus_engine.get_validator_info("validator_1")
        assert result == mock_validator_info

    def test_consensus_engine_get_validator_info_not_found(self, consensus_engine):
        """Test getting validator info when validator not found."""
        consensus_engine.validator_manager.validator_set.validators = {}

        result = consensus_engine.get_validator_info("validator_1")
        assert result is None

    def test_consensus_engine_get_active_validators(self, consensus_engine):
        """Test getting active validators."""
        consensus_engine.validator_manager.validator_set.active_validators = {
            "validator_1", "validator_2"
        }

        result = consensus_engine.get_active_validators()
        assert set(result) == {"validator_1", "validator_2"}

    def test_consensus_engine_get_consensus_metrics_with_mechanism_metrics(self, consensus_engine):
        """Test getting consensus metrics with mechanism metrics."""
        mock_mechanism_metrics = ConsensusMetrics(
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            total_blocks=100,
            successful_blocks=95,
            failed_blocks=5,
            average_block_time=1.5,
            average_gas_used=500000,
            validator_count=10,
            active_validators=8,
        )

        mock_validator_metrics = Mock()
        mock_validator_metrics.validator_count = 10
        mock_validator_metrics.active_validators = 8

        with patch.object(
            consensus_engine.consensus_mechanism,
            "get_consensus_metrics",
            return_value=mock_mechanism_metrics,
        ) as mock_get_metrics:
            with patch.object(
                consensus_engine.validator_manager,
                "get_validator_metrics",
                return_value=mock_validator_metrics,
            ) as mock_get_validator_metrics:
                result = consensus_engine.get_consensus_metrics()

                assert result == mock_mechanism_metrics
                assert result.validator_count == 10
                assert result.active_validators == 8
                mock_get_metrics.assert_called_once()
                mock_get_validator_metrics.assert_called_once()

    def test_consensus_engine_get_consensus_metrics_without_mechanism_metrics(self, consensus_engine):
        """Test getting consensus metrics without mechanism metrics."""
        mock_validator_metrics = Mock()
        mock_validator_metrics.validator_count = 10
        mock_validator_metrics.active_validators = 8

        # Create a mock consensus mechanism without get_consensus_metrics
        mock_consensus = Mock()
        consensus_engine.consensus_mechanism = mock_consensus

        with patch.object(
            consensus_engine.validator_manager,
            "get_validator_metrics",
            return_value=mock_validator_metrics,
        ) as mock_get_validator_metrics:
            result = consensus_engine.get_consensus_metrics()

            assert result.validator_count == 10
            assert result.active_validators == 8
            mock_get_validator_metrics.assert_called_once()

    def test_consensus_engine_get_performance_statistics_empty_history(self, consensus_engine):
        """Test getting performance statistics with empty history."""
        consensus_engine.performance_history = []

        result = consensus_engine.get_performance_statistics()
        assert result == {}

    def test_consensus_engine_get_performance_statistics_with_history(self, consensus_engine):
        """Test getting performance statistics with history."""
        consensus_engine.performance_history = [
            {
                "success": True,
                "block_time": 1.0,
                "gas_used": 500000,
                "validator_id": "validator_1",
            },
            {
                "success": False,
                "block_time": 2.0,
                "gas_used": 600000,
                "validator_id": "validator_2",
            },
            {
                "success": True,
                "block_time": 1.5,
                "gas_used": 550000,
                "validator_id": "validator_1",
            },
        ]

        with patch.object(
            consensus_engine, "get_active_validators", return_value=["validator_1", "validator_2"]
        ):
            # Mock the validator_set length
            consensus_engine.validator_manager.validator_set.validators = {"v1": Mock(), "v2": Mock()}
            result = consensus_engine.get_performance_statistics()

            assert result["total_blocks"] == 3
            assert result["successful_blocks"] == 2
            assert result["failed_blocks"] == 1
            assert result["success_rate"] == 2/3
            assert result["average_block_time"] == 1.5
            assert result["average_gas_used"] == 550000
            assert result["consensus_type"] == consensus_engine.config.consensus_type.value
            assert result["active_validators"] == 2
            assert result["total_validators"] == 2

    def test_consensus_engine_switch_consensus_same_type(self, consensus_engine):
        """Test switching to the same consensus type."""
        result = consensus_engine.switch_consensus(ConsensusType.PROOF_OF_STAKE)
        assert result is True

    def test_consensus_engine_switch_consensus_different_type(self, consensus_engine):
        """Test switching to a different consensus type."""
        with patch.object(
            consensus_engine, "_migrate_validators", return_value=True
        ) as mock_migrate:
            result = consensus_engine.switch_consensus(ConsensusType.DELEGATED_PROOF_OF_STAKE)

            assert result is True
            assert consensus_engine.config.consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE
            assert consensus_engine.consensus_state.current_consensus == ConsensusType.DELEGATED_PROOF_OF_STAKE
            assert consensus_engine.metrics.consensus_type == ConsensusType.DELEGATED_PROOF_OF_STAKE
            mock_migrate.assert_called_once()

    def test_consensus_engine_switch_consensus_migration_failure(self, consensus_engine):
        """Test switching consensus with migration failure."""
        with patch.object(
            consensus_engine, "_migrate_validators", return_value=False
        ) as mock_migrate:
            result = consensus_engine.switch_consensus(ConsensusType.DELEGATED_PROOF_OF_STAKE)

            assert result is False
            # Should not have changed the consensus type
            assert consensus_engine.config.consensus_type == ConsensusType.PROOF_OF_STAKE
            mock_migrate.assert_called_once()

    def test_consensus_engine_migrate_validators_success(self, consensus_engine):
        """Test successful validator migration."""
        mock_validator_info = Mock()
        mock_validator_info.validator_id = "validator_1"
        mock_validator_info.total_stake = 1000

        consensus_engine.validator_manager.validator_set.validators = {
            "validator_1": mock_validator_info
        }

        mock_new_consensus = Mock()
        mock_new_consensus.register_validator = Mock(return_value=True)

        with patch('dubchain.consensus.consensus_engine.Validator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator

            result = consensus_engine._migrate_validators(mock_new_consensus)

            assert result is True
            mock_validator_class.assert_called_once_with("validator_1", None)
            mock_new_consensus.register_validator.assert_called_once_with(mock_validator, 1000)

    def test_consensus_engine_migrate_validators_with_register_delegate(self, consensus_engine):
        """Test validator migration with register_delegate method."""
        mock_validator_info = Mock()
        mock_validator_info.validator_id = "validator_1"
        mock_validator_info.total_stake = 1000

        consensus_engine.validator_manager.validator_set.validators = {
            "validator_1": mock_validator_info
        }

        mock_new_consensus = Mock()
        mock_new_consensus.register_delegate = Mock(return_value=True)
        # Remove register_validator method to force use of register_delegate
        if hasattr(mock_new_consensus, 'register_validator'):
            del mock_new_consensus.register_validator

        with patch('dubchain.consensus.consensus_engine.Validator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator

            result = consensus_engine._migrate_validators(mock_new_consensus)

            assert result is True
            mock_validator_class.assert_called_once_with("validator_1", None)
            mock_new_consensus.register_delegate.assert_called_once_with(mock_validator, 1000)

    def test_consensus_engine_migrate_validators_with_add_validator(self, consensus_engine):
        """Test validator migration with add_validator method."""
        mock_validator_info = Mock()
        mock_validator_info.validator_id = "validator_1"
        mock_validator_info.total_stake = 1000

        consensus_engine.validator_manager.validator_set.validators = {
            "validator_1": mock_validator_info
        }

        mock_new_consensus = Mock()
        mock_new_consensus.add_validator = Mock(return_value=True)
        # Remove other methods to force use of add_validator
        if hasattr(mock_new_consensus, 'register_validator'):
            del mock_new_consensus.register_validator
        if hasattr(mock_new_consensus, 'register_delegate'):
            del mock_new_consensus.register_delegate

        with patch('dubchain.consensus.consensus_engine.Validator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator

            result = consensus_engine._migrate_validators(mock_new_consensus)

            assert result is True
            mock_validator_class.assert_called_once_with("validator_1", None)
            mock_new_consensus.add_validator.assert_called_once_with(mock_validator)

    def test_consensus_engine_migrate_validators_exception(self, consensus_engine):
        """Test validator migration with exception."""
        mock_validator_info = Mock()
        mock_validator_info.validator_id = "validator_1"
        mock_validator_info.total_stake = 1000

        consensus_engine.validator_manager.validator_set.validators = {
            "validator_1": mock_validator_info
        }

        mock_new_consensus = Mock()
        mock_new_consensus.register_validator = Mock(side_effect=Exception("Migration failed"))

        with patch('dubchain.consensus.consensus_engine.Validator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator_class.return_value = mock_validator

            result = consensus_engine._migrate_validators(mock_new_consensus)

            assert result is False

    def test_consensus_engine_get_consensus_info(self, consensus_engine):
        """Test getting comprehensive consensus info."""
        with patch.object(
            consensus_engine, "get_consensus_metrics", return_value=Mock()
        ) as mock_get_metrics:
            with patch.object(
                consensus_engine, "get_performance_statistics", return_value={"test": "data"}
            ) as mock_get_performance:
                with patch.object(
                    consensus_engine, "get_active_validators", return_value=["validator_1"]
                ) as mock_get_active:
                    result = consensus_engine.get_consensus_info()

                    assert isinstance(result, dict)
                    assert "consensus_type" in result
                    assert "config" in result
                    assert "state" in result
                    assert "metrics" in result
                    assert "performance" in result
                    assert "validators" in result
                    mock_get_metrics.assert_called_once()
                    mock_get_performance.assert_called_once()
                    # get_active_validators is called twice in get_consensus_info
                    assert mock_get_active.call_count == 2

    def test_consensus_engine_shutdown(self, consensus_engine):
        """Test shutting down consensus engine."""
        with patch.object(
            consensus_engine.executor, "shutdown"
        ) as mock_shutdown:
            consensus_engine.shutdown()
            mock_shutdown.assert_called_once_with(wait=True)

    def test_consensus_engine_to_dict(self, consensus_engine):
        """Test converting to dictionary."""
        with patch.object(
            consensus_engine.config, "to_dict", return_value={"config": "data"}
        ) as mock_config_to_dict:
            with patch.object(
                consensus_engine.validator_manager, "to_dict", return_value={"validator": "data"}
            ) as mock_validator_to_dict:
                result = consensus_engine.to_dict()

                assert isinstance(result, dict)
                assert "config" in result
                assert "consensus_state" in result
                assert "validator_manager" in result
                assert "performance_history" in result
                assert "metrics" in result
                mock_config_to_dict.assert_called_once()
                mock_validator_to_dict.assert_called_once()

    def test_consensus_engine_from_dict(self, consensus_engine):
        """Test creating from dictionary."""
        data = {
            "config": {"consensus_type": "proof_of_stake", "max_validators": 10},
            "validator_manager": {"validators": {}},
            "performance_history": [{"test": "data"}],
            "metrics": {
                "total_blocks": 100,
                "successful_blocks": 95,
                "failed_blocks": 5,
                "average_block_time": 1.5,
                "average_gas_used": 500000,
                "validator_count": 10,
                "active_validators": 8,
                "consensus_type": "proof_of_stake",
                "last_updated": 1234567890.0,
            },
        }

        with patch.object(
            ConsensusConfig, "from_dict", return_value=consensus_engine.config
        ) as mock_config_from_dict:
            with patch.object(
                ValidatorManager, "from_dict", return_value=consensus_engine.validator_manager
            ) as mock_validator_from_dict:
                result = ConsensusEngine.from_dict(data)

                assert isinstance(result, ConsensusEngine)
                assert result.performance_history == [{"test": "data"}]
                mock_config_from_dict.assert_called_once_with(data["config"])
                mock_validator_from_dict.assert_called_once_with(data["validator_manager"])

    def test_consensus_engine_record_performance_history_overflow(self, consensus_engine):
        """Test recording performance with history overflow."""
        # Fill history to max size
        consensus_engine.performance_history = [
            {"test": f"data_{i}"} for i in range(consensus_engine.max_history_size)
        ]

        result = ConsensusResult(
            success=True,
            consensus_type=ConsensusType.PROOF_OF_STAKE,
            gas_used=500000,
            validator_id="validator_1",
        )

        consensus_engine._record_performance(result, 1.5)

        # Should still be at max size
        assert len(consensus_engine.performance_history) == consensus_engine.max_history_size
        # Should contain the new entry
        assert consensus_engine.performance_history[-1]["validator_id"] == "validator_1"
