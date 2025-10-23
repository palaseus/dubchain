"""
Unit tests for VM gas meter.
"""

import logging

logger = logging.getLogger(__name__)
from unittest.mock import Mock, patch

import pytest

from dubchain.vm.gas_meter import (
    GasCost,
    GasCostConfig,
    GasCostType,
    GasMeter,
    GasOptimizer,
)


class TestGasCost:
    """Test GasCost class."""

    def test_gas_cost_values(self):
        """Test gas cost values."""
        assert GasCost.BASE.value == 2
        assert GasCost.VERY_LOW.value == 3
        assert GasCost.LOW.value == 5
        assert GasCost.MID.value == 8
        assert GasCost.HIGH.value == 10
        assert GasCost.EXTCODE.value == 700
        assert GasCost.BALANCE.value == 400
        assert GasCost.SLOAD.value == 200
        assert GasCost.JUMPDEST.value == 1
        assert GasCost.SSET.value == 20000
        assert GasCost.SRESET.value == 5000
        assert GasCost.RSCLEAR.value == 15000
        assert GasCost.SELFDESTRUCT.value == 5000
        assert GasCost.CREATE.value == 32000
        assert GasCost.CALL.value == 700
        assert GasCost.CALLVALUE.value == 9000
        assert GasCost.CALLSTIPEND.value == 2300
        assert GasCost.NEWACCOUNT.value == 25000
        assert GasCost.EXP.value == 10
        assert GasCost.MEMORY.value == 3
        assert GasCost.TXDATAZERO.value == 4
        assert GasCost.TXDATANONZERO.value == 68
        assert GasCost.TRANSACTION.value == 21000
        assert GasCost.LOG.value == 375
        assert GasCost.LOGDATA.value == 8
        assert GasCost.LOGTOPIC.value == 375
        assert GasCost.SHA3.value == 30
        assert GasCost.SHA3WORD.value == 6
        assert GasCost.COPY.value == 3
        assert GasCost.BLOCKHASH.value == 20
        assert GasCost.QUADDIVISOR.value == 512

    def test_gas_cost_types(self):
        """Test gas cost types."""
        for cost in GasCost:
            assert isinstance(cost.value, int)
            assert cost.value >= 0


class TestGasMeter:
    """Test GasMeter class."""

    def test_gas_meter_creation(self):
        """Test gas meter creation."""
        gas_meter = GasMeter(1000000, 1000000)

        assert gas_meter.initial_gas == 1000000
        assert gas_meter.gas_used == 0
        assert gas_meter.remaining_gas == 1000000

    def test_gas_meter_creation_zero_limit(self):
        """Test gas meter creation with zero limit."""
        gas_meter = GasMeter(0, 0)

        assert gas_meter.gas_limit == 0
        assert gas_meter.gas_used == 0
        assert gas_meter.gas_remaining == 0

    def test_gas_meter_creation_negative_limit(self):
        """Test gas meter creation with negative limit."""
        with pytest.raises(ValueError, match="Gas limit must be non-negative"):
            GasMeter(-1, -1)

    def test_consume_gas_success(self):
        """Test successful gas consumption."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.consume_gas(100)
        assert gas_meter.gas_used == 100
        assert gas_meter.gas_remaining == 900

    def test_consume_gas_multiple(self):
        """Test multiple gas consumption."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.consume_gas(100)
        assert gas_meter.consume_gas(200)
        assert gas_meter.consume_gas(300)

        assert gas_meter.gas_used == 600
        assert gas_meter.gas_remaining == 400

    def test_consume_gas_exact_limit(self):
        """Test gas consumption up to exact limit."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.consume_gas(1000)
        assert gas_meter.gas_used == 1000
        assert gas_meter.gas_remaining == 0

    def test_consume_gas_exceed_limit(self):
        """Test gas consumption exceeding limit."""
        gas_meter = GasMeter(1000, 1000)

        assert not gas_meter.consume_gas(1001)
        assert gas_meter.gas_used == 0
        assert gas_meter.gas_remaining == 1000

    def test_consume_gas_negative(self):
        """Test gas consumption with negative amount."""
        gas_meter = GasMeter(1000, 1000)

        with pytest.raises(ValueError, match="Gas amount must be non-negative"):
            gas_meter.consume_gas(-1)

    def test_consume_gas_zero(self):
        """Test gas consumption with zero amount."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.consume_gas(0)
        assert gas_meter.gas_used == 0
        assert gas_meter.gas_remaining == 1000

    def test_get_gas_used(self):
        """Test getting gas used."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.get_gas_used() == 0

        gas_meter.consume_gas(300)
        assert gas_meter.get_gas_used() == 300

    def test_get_gas_remaining(self):
        """Test getting gas remaining."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.get_gas_remaining() == 1000

        gas_meter.consume_gas(300)
        assert gas_meter.get_gas_remaining() == 700

    def test_get_gas_limit(self):
        """Test getting gas limit."""
        gas_meter = GasMeter(1000, 1000)
        assert gas_meter.get_gas_limit() == 1000

    def test_is_out_of_gas_false(self):
        """Test is_out_of_gas when gas is available."""
        gas_meter = GasMeter(1000, 1000)

        assert not gas_meter.is_out_of_gas()

        gas_meter.consume_gas(500)
        assert not gas_meter.is_out_of_gas()

    def test_is_out_of_gas_true(self):
        """Test is_out_of_gas when gas is exhausted."""
        gas_meter = GasMeter(1000, 1000)

        gas_meter.consume_gas(1000)
        assert gas_meter.is_out_of_gas()

    def test_reset(self):
        """Test gas meter reset."""
        gas_meter = GasMeter(1000, 1000)

        gas_meter.consume_gas(500)
        assert gas_meter.gas_used == 500
        assert gas_meter.gas_remaining == 500

        gas_meter.reset()
        assert gas_meter.gas_used == 0
        assert gas_meter.gas_remaining == 1000

    def test_reset_with_new_limit(self):
        """Test gas meter reset with new limit."""
        gas_meter = GasMeter(1000, 1000)

        gas_meter.consume_gas(500)
        gas_meter.reset(2000)

        assert gas_meter.gas_limit == 2000
        assert gas_meter.gas_used == 0
        assert gas_meter.gas_remaining == 2000

    def test_reset_with_negative_limit(self):
        """Test gas meter reset with negative limit."""
        gas_meter = GasMeter(1000, 1000)

        with pytest.raises(ValueError, match="Gas limit must be non-negative"):
            gas_meter.reset(-1)

    def test_gas_usage_percentage(self):
        """Test gas usage percentage calculation."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.get_gas_usage_percentage() == 0.0

        gas_meter.consume_gas(250)
        assert gas_meter.get_gas_usage_percentage() == 25.0

        gas_meter.consume_gas(250)
        assert gas_meter.get_gas_usage_percentage() == 50.0

        gas_meter.consume_gas(500)
        assert gas_meter.get_gas_usage_percentage() == 100.0

    def test_gas_usage_percentage_zero_limit(self):
        """Test gas usage percentage with zero limit."""
        gas_meter = GasMeter(1000, 0)

        assert gas_meter.get_gas_usage_percentage() == 0.0

    def test_gas_efficiency(self):
        """Test gas efficiency calculation."""
        gas_meter = GasMeter(1000, 1000)

        assert gas_meter.get_gas_efficiency() == 100.0

        gas_meter.consume_gas(250)
        assert gas_meter.get_gas_efficiency() == 75.0

        gas_meter.consume_gas(250)
        assert gas_meter.get_gas_efficiency() == 50.0

        gas_meter.consume_gas(500)
        assert gas_meter.get_gas_efficiency() == 0.0

    def test_gas_efficiency_zero_limit(self):
        """Test gas efficiency with zero limit."""
        gas_meter = GasMeter(0, 0)

        assert gas_meter.get_gas_efficiency() == 100.0

    def test_string_representation(self):
        """Test gas meter string representation."""
        gas_meter = GasMeter(1000, 1000)

        assert str(gas_meter) == "GasMeter(used=0, remaining=1000, refunds=0)"

        gas_meter.consume_gas(300)
        assert str(gas_meter) == "GasMeter(used=300, remaining=700, refunds=0)"

    def test_repr_representation(self):
        """Test gas meter repr representation."""
        gas_meter = GasMeter(1000, 1000)

        assert (
            repr(gas_meter)
            == "GasMeter(initial=1000, used=0, remaining=1000, refunds=0, utilization=0.0%)"
        )

        gas_meter.consume_gas(300)
        assert (
            repr(gas_meter)
            == "GasMeter(initial=1000, used=300, remaining=700, refunds=0, utilization=30.0%)"
        )

    def test_gas_meter_equality(self):
        """Test gas meter equality."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 1000)
        gas_meter3 = GasMeter(1000, 2000)

        assert gas_meter1 == gas_meter2
        assert gas_meter1 != gas_meter3

        gas_meter1.consume_gas(100)
        assert gas_meter1 != gas_meter2

    def test_gas_meter_hash(self):
        """Test gas meter hash."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 1000)
        gas_meter3 = GasMeter(1000, 2000)

        assert hash(gas_meter1) == hash(gas_meter2)
        assert hash(gas_meter1) != hash(gas_meter3)

    def test_gas_meter_comparison(self):
        """Test gas meter comparison."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 2000)

        assert gas_meter1 < gas_meter2
        assert gas_meter2 > gas_meter1
        assert gas_meter1 <= gas_meter2
        assert gas_meter2 >= gas_meter1

    def test_gas_meter_comparison_same_limit(self):
        """Test gas meter comparison with same limit."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 1000)

        assert gas_meter1 <= gas_meter2
        assert gas_meter1 >= gas_meter2
        assert not (gas_meter1 < gas_meter2)
        assert not (gas_meter1 > gas_meter2)

    def test_gas_meter_comparison_after_usage(self):
        """Test gas meter comparison after usage."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 1000)

        gas_meter1.consume_gas(100)

        assert gas_meter1 < gas_meter2
        assert gas_meter2 > gas_meter1

    def test_gas_meter_comparison_different_limits(self):
        """Test gas meter comparison with different limits."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 2000)

        gas_meter1.consume_gas(500)
        gas_meter2.consume_gas(1000)

        assert gas_meter1 < gas_meter2
        assert gas_meter2 > gas_meter1

    def test_gas_meter_comparison_same_usage(self):
        """Test gas meter comparison with same usage."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 2000)

        gas_meter1.consume_gas(500)
        gas_meter2.consume_gas(500)

        assert gas_meter1 < gas_meter2
        assert gas_meter2 > gas_meter1

    def test_gas_meter_comparison_same_usage_same_limit(self):
        """Test gas meter comparison with same usage and limit."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 1000)

        gas_meter1.consume_gas(500)
        gas_meter2.consume_gas(500)

        assert gas_meter1 <= gas_meter2
        assert gas_meter1 >= gas_meter2
        assert not (gas_meter1 < gas_meter2)
        assert not (gas_meter1 > gas_meter2)

    def test_gas_meter_comparison_same_usage_different_limit(self):
        """Test gas meter comparison with same usage but different limit."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 2000)

        gas_meter1.consume_gas(500)
        gas_meter2.consume_gas(500)

        assert gas_meter1 < gas_meter2
        assert gas_meter2 > gas_meter1

    def test_gas_meter_comparison_same_usage_different_limit_reversed(self):
        """Test gas meter comparison with same usage but different limit (reversed)."""
        gas_meter1 = GasMeter(1000, 2000)
        gas_meter2 = GasMeter(1000, 1000)

        gas_meter1.consume_gas(500)
        gas_meter2.consume_gas(500)

        assert gas_meter1 > gas_meter2
        assert gas_meter2 < gas_meter1

    def test_gas_meter_comparison_same_usage_different_limit_equal(self):
        """Test gas meter comparison with same usage but different limit (equal)."""
        gas_meter1 = GasMeter(1000, 1000)
        gas_meter2 = GasMeter(1000, 2000)

        gas_meter1.consume_gas(500)
        gas_meter2.consume_gas(500)

        assert gas_meter1 < gas_meter2
        assert gas_meter2 > gas_meter1

    def test_gas_meter_comparison_same_usage_different_limit_equal_reversed(self):
        """Test gas meter comparison with same usage but different limit (equal, reversed)."""
        gas_meter1 = GasMeter(1000, 2000)
        gas_meter2 = GasMeter(1000, 1000)

        gas_meter1.consume_gas(500)
        gas_meter2.consume_gas(500)

        assert gas_meter1 > gas_meter2
        assert gas_meter2 < gas_meter1


class TestGasCostConfig:
    """Test GasCostConfig class."""

    def test_gas_cost_config_creation(self):
        """Test gas cost config creation."""
        config = GasCostConfig(
            base_cost=100,
            cost_type=GasCostType.FIXED,
            dynamic_factor=1.5,
            max_cost=1000,
            min_cost=10,
        )

        assert config.base_cost == 100
        assert config.cost_type == GasCostType.FIXED
        assert config.dynamic_factor == 1.5
        assert config.max_cost == 1000
        assert config.min_cost == 10

    def test_gas_cost_config_calculate_cost_fixed(self):
        """Test gas cost calculation for fixed cost type."""
        config = GasCostConfig(base_cost=100, cost_type=GasCostType.FIXED)

        cost = config.calculate_cost()
        assert cost == 100


class TestGasOptimizer:
    """Test GasOptimizer class."""

    def test_gas_optimizer_creation(self):
        """Test gas optimizer creation."""
        optimizer = GasOptimizer()
        assert optimizer is not None

    def test_optimize_contract(self):
        """Test contract optimization."""
        optimizer = GasOptimizer()
        bytecode = b"\x60\x01\x60\x02\x01"  # Simple ADD operation
        gas_limit = 100000

        result = optimizer.optimize_contract(bytecode, gas_limit)

        assert isinstance(result, dict)
        # Check for actual keys returned by the method
        assert (
            "stack_optimization" in result
            or "memory_optimization" in result
            or "storage_optimization" in result
        )

    def test_estimate_gas_usage(self):
        """Test gas usage estimation."""
        optimizer = GasOptimizer()
        bytecode = b"\x60\x01\x60\x02\x01"  # Simple ADD operation

        result = optimizer.estimate_gas_usage(bytecode)

        assert isinstance(result, dict)
        # Check for actual keys returned by the method
        assert "estimated_total_gas" in result
        assert "bytecode_size" in result
        assert "operation_counts" in result

        assert result["estimated_total_gas"] >= 0
        assert result["bytecode_size"] >= 0
        assert isinstance(result["operation_counts"], dict)
