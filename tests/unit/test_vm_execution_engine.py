"""
Unit tests for VM execution engine.
"""

import logging

logger = logging.getLogger(__name__)
import time

import pytest

from dubchain.crypto.hashing import Hash, SHA256Hasher
from dubchain.vm.contract import ContractState, ContractType, SmartContract
from dubchain.vm.execution_engine import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionResult,
    ExecutionState,
)
from dubchain.vm.gas_meter import GasMeter


class TestExecutionContext:
    """Test ExecutionContext class."""

    def test_execution_context_creation(self):
        """Test execution context creation."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {"block_number": 123}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        assert context.contract == contract
        assert context.caller == "0xcaller"
        assert context.value == 1000
        assert context.data == b"call_data"
        assert context.gas_limit == 100000
        assert context.gas_meter == gas_meter
        assert context.block_context == block_context
        assert context.pc == 0
        assert len(context.stack) == 0
        assert len(context.memory) == 0
        assert context.return_data == b""
        assert context.call_depth == 0
        assert context.max_call_depth == 1024
        assert context.state == ExecutionState.RUNNING
        assert context.error_message == ""

    def test_execution_context_creation_invalid_gas_limit(self):
        """Test execution context creation with invalid gas limit."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(0, 0)
        block_context = {}

        with pytest.raises(ValueError, match="Gas limit must be positive"):
            ExecutionContext(
                contract=contract,
                caller="0xcaller",
                value=1000,
                data=b"call_data",
                gas_limit=0,
                gas_meter=gas_meter,
                block_context=block_context,
            )

    def test_execution_context_creation_invalid_call_depth(self):
        """Test execution context creation with invalid call depth."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        with pytest.raises(ValueError, match="Call depth must be non-negative"):
            ExecutionContext(
                contract=contract,
                caller="0xcaller",
                value=1000,
                data=b"call_data",
                gas_limit=100000,
                gas_meter=gas_meter,
                block_context=block_context,
                call_depth=-1,
            )

    def test_execution_context_creation_invalid_max_call_depth(self):
        """Test execution context creation with invalid max call depth."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        with pytest.raises(ValueError, match="Max call depth must be positive"):
            ExecutionContext(
                contract=contract,
                caller="0xcaller",
                value=1000,
                data=b"call_data",
                gas_limit=100000,
                gas_meter=gas_meter,
                block_context=block_context,
                max_call_depth=0,
            )

    def test_push_stack(self):
        """Test pushing to stack."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        context.push_stack(123)
        assert len(context.stack) == 1
        assert context.stack[0] == 123

        context.push_stack(456)
        assert len(context.stack) == 2
        assert context.stack[1] == 456

    def test_push_stack_overflow(self):
        """Test stack overflow."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        # Fill stack to limit
        for i in range(1024):
            context.push_stack(i)

        # Try to push one more
        context.push_stack(9999)

        assert context.state == ExecutionState.STACK_OVERFLOW
        assert context.error_message == "Stack overflow"

    def test_pop_stack(self):
        """Test popping from stack."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        context.push_stack(123)
        context.push_stack(456)

        value = context.pop_stack()
        assert value == 456
        assert len(context.stack) == 1

        value = context.pop_stack()
        assert value == 123
        assert len(context.stack) == 0

    def test_pop_stack_underflow(self):
        """Test stack underflow."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        value = context.pop_stack()

        assert context.state == ExecutionState.STACK_UNDERFLOW
        assert context.error_message == "Stack underflow"
        assert value == 0

    def test_peek_stack(self):
        """Test peeking at stack."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        context.push_stack(123)
        context.push_stack(456)

        value = context.peek_stack(0)
        assert value == 456

        value = context.peek_stack(1)
        assert value == 123

    def test_peek_stack_underflow(self):
        """Test peeking at empty stack."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        value = context.peek_stack(0)

        assert context.state == ExecutionState.STACK_UNDERFLOW
        assert context.error_message == "Stack underflow"
        assert value == 0

    def test_get_memory(self):
        """Test getting memory."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        # Set some memory
        context.set_memory(0, b"test_data")

        # Get memory
        data = context.get_memory(0, 9)
        assert data == b"test_data"

        # Get memory beyond current size (should extend with zeros)
        data = context.get_memory(0, 20)
        assert data == b"test_data" + b"\x00" * 11

    def test_get_memory_invalid_offset(self):
        """Test getting memory with invalid offset."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        data = context.get_memory(-1, 10)

        assert context.state == ExecutionState.INVALID_MEMORY_ACCESS
        assert context.error_message == "Invalid memory access"
        assert data == b""

    def test_get_memory_invalid_size(self):
        """Test getting memory with invalid size."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        data = context.get_memory(0, -1)

        assert context.state == ExecutionState.INVALID_MEMORY_ACCESS
        assert context.error_message == "Invalid memory access"
        assert data == b""

    def test_set_memory(self):
        """Test setting memory."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        context.set_memory(0, b"test_data")
        assert context.memory == bytearray(b"test_data")

        context.set_memory(10, b"more_data")
        assert context.memory == bytearray(b"test_data" + b"\x00" + b"more_data")

    def test_set_memory_invalid_offset(self):
        """Test setting memory with invalid offset."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        context.set_memory(-1, b"data")

        assert context.state == ExecutionState.INVALID_MEMORY_ACCESS
        assert context.error_message == "Invalid memory access"

    def test_consume_gas_success(self):
        """Test successful gas consumption."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        assert context.consume_gas(1000)
        assert context.gas_meter.get_gas_used() == 1000

    def test_consume_gas_failure(self):
        """Test failed gas consumption."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(1000, 1000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=1000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        assert not context.consume_gas(1001)
        assert context.state == ExecutionState.OUT_OF_GAS
        assert context.error_message == "Out of gas"

    def test_revert(self):
        """Test reverting execution."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        context.revert("Test error")

        assert context.state == ExecutionState.REVERTED
        assert context.error_message == "Test error"

    def test_stop(self):
        """Test stopping execution."""
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode
        gas_meter = GasMeter(100000, 100000)
        block_context = {}

        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context=block_context,
        )

        context.stop()

        assert context.state == ExecutionState.STOPPED


class TestExecutionResult:
    """Test ExecutionResult class."""

    def test_execution_result_creation(self):
        """Test execution result creation."""
        result = ExecutionResult(
            success=True,
            gas_used=1000,
            return_data=b"return_data",
            events=[],
            storage_changes={},
            error_message="",
            execution_time=0.1,
        )

        assert result.success is True
        assert result.gas_used == 1000
        assert result.return_data == b"return_data"
        assert result.events == []
        assert result.storage_changes == {}
        assert result.error_message == ""
        assert result.execution_time == 0.1

    def test_execution_result_to_dict(self):
        """Test execution result to dictionary conversion."""
        result = ExecutionResult(
            success=True,
            gas_used=1000,
            return_data=b"return_data",
            events=[],
            storage_changes={},
            error_message="",
            execution_time=0.1,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["gas_used"] == 1000
        assert result_dict["return_data"] == b"return_data".hex()
        assert result_dict["events"] == []
        assert result_dict["storage_changes"] == {}
        assert result_dict["error_message"] == ""
        assert result_dict["execution_time"] == 0.1


class TestExecutionEngine:
    """Test ExecutionEngine class."""

    def test_execution_engine_creation(self):
        """Test execution engine creation."""
        engine = ExecutionEngine()

        assert engine.max_call_depth == 1024
        assert engine.max_memory_size == 2**32
        assert len(engine.execution_history) == 0
        assert "total_executions" in engine.performance_metrics

    def test_execution_engine_creation_custom_params(self):
        """Test execution engine creation with custom parameters."""
        engine = ExecutionEngine(max_call_depth=512, max_memory_size=1024)

        assert engine.max_call_depth == 512
        assert engine.max_memory_size == 1024

    def test_execute_contract_success(self):
        """Test successful contract execution."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.gas_used == 0
        assert result.return_data == b""
        assert result.events == []
        assert result.storage_changes == {}
        assert result.error_message == ""
        assert result.execution_time >= 0

    def test_execute_contract_exception(self):
        """Test contract execution with exception."""
        engine = ExecutionEngine()
        contract = SmartContract(
            address="0x123", bytecode=b"invalid"
        )  # Invalid bytecode

        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert "Invalid PUSH opcode" in result.error_message

    def test_performance_metrics_update(self):
        """Test performance metrics update."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute contract
        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        metrics = engine.get_performance_metrics()

        assert metrics["total_executions"] == 1
        assert metrics["total_gas_used"] == 0
        assert metrics["total_execution_time"] >= 0
        assert metrics["average_gas_per_execution"] == 0
        assert metrics["average_execution_time"] >= 0
        assert metrics["success_rate"] == 1.0

    def test_execution_history_recording(self):
        """Test execution history recording."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute contract
        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        history = engine.get_execution_history()

        assert len(history) == 1
        assert history[0]["contract_address"] == "0x123"
        assert history[0]["caller"] == "0xcaller"
        assert history[0]["success"] is True
        assert history[0]["gas_used"] == 0
        assert history[0]["execution_time"] >= 0

    def test_execution_history_limit(self):
        """Test execution history limit."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute many contracts
        for i in range(1001):
            result = engine.execute_contract(
                contract=contract,
                caller="0xcaller",
                value=1000,
                data=b"call_data",
                gas_limit=100000,
                block_context={"block_number": 123},
            )

        history = engine.get_execution_history()

        # Should keep only last 1000 executions
        assert len(history) == 1000

    def test_get_execution_history_with_limit(self):
        """Test getting execution history with limit."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute multiple contracts
        for i in range(5):
            result = engine.execute_contract(
                contract=contract,
                caller="0xcaller",
                value=1000,
                data=b"call_data",
                gas_limit=100000,
                block_context={"block_number": 123},
            )

        history = engine.get_execution_history(limit=3)

        assert len(history) == 3

    def test_reset_metrics(self):
        """Test resetting performance metrics."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute contract
        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        # Reset metrics
        engine.reset_metrics()

        metrics = engine.get_performance_metrics()
        history = engine.get_execution_history()

        assert metrics["total_executions"] == 0
        assert metrics["total_gas_used"] == 0
        assert metrics["total_execution_time"] == 0.0
        assert len(history) == 0

    def test_get_opcode_usage_stats(self):
        """Test getting opcode usage statistics."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute contract
        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        stats = engine.get_opcode_usage_stats()

        # Should have STOP opcode executed
        assert len(stats) == 1
        assert "STOP" in stats
        assert stats["STOP"]["count"] == 1

    def test_get_error_analysis(self):
        """Test getting error analysis."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute contract
        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        analysis = engine.get_error_analysis()

        # Should be empty since no errors occurred
        assert len(analysis) == 0

    def test_execution_engine_string_representation(self):
        """Test execution engine string representation."""
        engine = ExecutionEngine()

        # Test that we can get performance metrics
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "total_executions" in metrics

    def test_execution_engine_performance_tracking(self):
        """Test execution engine performance tracking."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")  # STOP opcode

        # Execute multiple contracts
        for i in range(3):
            result = engine.execute_contract(
                contract=contract,
                caller="0xcaller",
                value=1000,
                data=b"call_data",
                gas_limit=100000,
                block_context={"block_number": 123},
            )

        metrics = engine.get_performance_metrics()

        assert metrics["total_executions"] == 3
        assert metrics["success_rate"] == 1.0
        assert metrics["average_gas_per_execution"] == 0
        assert metrics["average_execution_time"] >= 0

    def test_execution_engine_error_tracking(self):
        """Test execution engine error tracking."""
        engine = ExecutionEngine()
        contract = SmartContract(
            address="0x123", bytecode=b"invalid"
        )  # Invalid bytecode

        # Execute contract
        result = engine.execute_contract(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            block_context={"block_number": 123},
        )

        metrics = engine.get_performance_metrics()
        analysis = engine.get_error_analysis()

        assert metrics["total_executions"] == 1
        assert metrics["success_rate"] == 0.0
        assert len(analysis) > 0

    def test_get_opcode_gas_cost(self):
        """Test getting opcode gas cost."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test basic opcode
        opcode = Opcode(
            name="ADD",
            opcode=0x01,
            gas_cost=3,
            type=OpcodeType.ARITHMETIC,
            description="Addition operation",
        )
        cost = engine._get_opcode_gas_cost(opcode, context)
        assert cost == 3

        # Test SSTORE opcode
        opcode = Opcode(
            name="SSTORE",
            opcode=0x55,
            gas_cost=20000,
            type=OpcodeType.STORAGE,
            description="Store to storage",
        )
        cost = engine._get_opcode_gas_cost(opcode, context)
        assert cost == 40000  # base_cost + 20000

        # Test SLOAD opcode
        opcode = Opcode(
            name="SLOAD",
            opcode=0x54,
            gas_cost=200,
            type=OpcodeType.STORAGE,
            description="Load from storage",
        )
        cost = engine._get_opcode_gas_cost(opcode, context)
        assert cost == 400  # base_cost + 200

    def test_execute_arithmetic_opcode(self):
        """Test executing arithmetic opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test ADD
        context.push_stack(10)
        context.push_stack(20)
        opcode = Opcode(
            name="ADD",
            opcode=0x01,
            gas_cost=3,
            type=OpcodeType.ARITHMETIC,
            description="Addition operation",
        )
        engine._execute_arithmetic_opcode(context, opcode)
        assert context.pop_stack() == 30

        # Test SUB
        context.push_stack(10)
        context.push_stack(20)
        opcode = Opcode(
            name="SUB",
            opcode=0x03,
            gas_cost=3,
            type=OpcodeType.ARITHMETIC,
            description="Subtraction operation",
        )
        engine._execute_arithmetic_opcode(context, opcode)
        assert context.pop_stack() == 10

        # Test MUL
        context.push_stack(5)
        context.push_stack(6)
        opcode = Opcode(
            name="MUL",
            opcode=0x02,
            gas_cost=5,
            type=OpcodeType.ARITHMETIC,
            description="Multiplication operation",
        )
        engine._execute_arithmetic_opcode(context, opcode)
        assert context.pop_stack() == 30

        # Test DIV
        context.push_stack(4)
        context.push_stack(20)
        opcode = Opcode(
            name="DIV",
            opcode=0x04,
            gas_cost=5,
            type=OpcodeType.ARITHMETIC,
            description="Division operation",
        )
        engine._execute_arithmetic_opcode(context, opcode)
        assert context.pop_stack() == 5

        # Test DIV by zero
        context.push_stack(20)
        context.push_stack(0)
        opcode = Opcode(
            name="DIV",
            opcode=0x04,
            gas_cost=5,
            type=OpcodeType.ARITHMETIC,
            description="Division operation",
        )
        engine._execute_arithmetic_opcode(context, opcode)
        assert context.pop_stack() == 0

        # Test MOD
        context.push_stack(5)
        context.push_stack(23)
        opcode = Opcode(
            name="MOD",
            opcode=0x06,
            gas_cost=5,
            type=OpcodeType.ARITHMETIC,
            description="Modulo operation",
        )
        engine._execute_arithmetic_opcode(context, opcode)
        assert context.pop_stack() == 3

        # Test MOD by zero
        context.push_stack(23)
        context.push_stack(0)
        opcode = Opcode(
            name="MOD",
            opcode=0x06,
            gas_cost=5,
            type=OpcodeType.ARITHMETIC,
            description="Modulo operation",
        )
        engine._execute_arithmetic_opcode(context, opcode)
        assert context.pop_stack() == 0

    def test_execute_comparison_opcode(self):
        """Test executing comparison opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test LT (less than)
        context.push_stack(20)
        context.push_stack(10)
        opcode = Opcode(
            name="LT",
            opcode=0x10,
            gas_cost=3,
            type=OpcodeType.COMPARISON,
            description="Less than comparison",
        )
        engine._execute_comparison_opcode(context, opcode)
        assert context.pop_stack() == 1  # 10 < 20 is True

        context.push_stack(10)
        context.push_stack(20)
        engine._execute_comparison_opcode(context, opcode)
        assert context.pop_stack() == 0  # 20 < 10 is False

        # Test GT (greater than)
        context.push_stack(10)
        context.push_stack(20)
        opcode = Opcode(
            name="GT",
            opcode=0x11,
            gas_cost=3,
            type=OpcodeType.COMPARISON,
            description="Greater than comparison",
        )
        engine._execute_comparison_opcode(context, opcode)
        assert context.pop_stack() == 1  # 20 > 10 is True

        context.push_stack(20)
        context.push_stack(10)
        engine._execute_comparison_opcode(context, opcode)
        assert context.pop_stack() == 0  # 10 > 20 is False

        # Test EQ (equal)
        context.push_stack(10)
        context.push_stack(10)
        opcode = Opcode(
            name="EQ",
            opcode=0x14,
            gas_cost=3,
            type=OpcodeType.COMPARISON,
            description="Equality comparison",
        )
        engine._execute_comparison_opcode(context, opcode)
        assert context.pop_stack() == 1  # 10 == 10 is True

        context.push_stack(10)
        context.push_stack(20)
        engine._execute_comparison_opcode(context, opcode)
        assert context.pop_stack() == 0  # 10 == 20 is False

    def test_execute_bitwise_opcode(self):
        """Test executing bitwise opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test AND
        context.push_stack(0b1010)  # 10
        context.push_stack(0b1100)  # 12
        opcode = Opcode(
            name="AND",
            opcode=0x16,
            gas_cost=3,
            type=OpcodeType.BITWISE,
            description="Bitwise AND",
        )
        engine._execute_bitwise_opcode(context, opcode)
        assert context.pop_stack() == 0b1000  # 8

        # Test OR
        context.push_stack(0b1010)  # 10
        context.push_stack(0b1100)  # 12
        opcode = Opcode(
            name="OR",
            opcode=0x17,
            gas_cost=3,
            type=OpcodeType.BITWISE,
            description="Bitwise OR",
        )
        engine._execute_bitwise_opcode(context, opcode)
        assert context.pop_stack() == 0b1110  # 14

        # Test XOR
        context.push_stack(0b1010)  # 10
        context.push_stack(0b1100)  # 12
        opcode = Opcode(
            name="XOR",
            opcode=0x18,
            gas_cost=3,
            type=OpcodeType.BITWISE,
            description="Bitwise XOR",
        )
        engine._execute_bitwise_opcode(context, opcode)
        assert context.pop_stack() == 0b0110  # 6

    def test_execute_sha3_opcode(self):
        """Test executing SHA3 opcode."""
        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Set some memory data
        context.set_memory(0, b"test_data")

        # Push offset and size
        context.push_stack(0)  # offset
        context.push_stack(9)  # size

        engine._execute_sha3_opcode(context)

        # Should have a hash result on the stack
        result = context.pop_stack()
        assert isinstance(result, int)
        assert result > 0

    def test_execute_environment_opcode(self):
        """Test executing environment opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test ADDRESS
        opcode = Opcode(
            name="ADDRESS",
            opcode=0x30,
            gas_cost=2,
            type=OpcodeType.ENVIRONMENT,
            description="Get contract address",
        )
        engine._execute_environment_opcode(context, opcode)
        address_result = context.pop_stack()
        assert isinstance(address_result, int)
        assert address_result > 0

        # Test CALLER
        opcode = Opcode(
            name="CALLER",
            opcode=0x33,
            gas_cost=2,
            type=OpcodeType.ENVIRONMENT,
            description="Get caller address",
        )
        engine._execute_environment_opcode(context, opcode)
        caller_result = context.pop_stack()
        assert isinstance(caller_result, int)
        assert caller_result > 0

        # Test CALLVALUE
        opcode = Opcode(
            name="CALLVALUE",
            opcode=0x34,
            gas_cost=2,
            type=OpcodeType.ENVIRONMENT,
            description="Get call value",
        )
        engine._execute_environment_opcode(context, opcode)
        value_result = context.pop_stack()
        assert value_result == 1000

    def test_execute_block_opcode(self):
        """Test executing block opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={"timestamp": 1234567890},
        )

        # Test BLOCKHASH
        context.push_stack(123)
        opcode = Opcode(
            name="BLOCKHASH",
            opcode=0x40,
            gas_cost=20,
            type=OpcodeType.BLOCK,
            description="Get block hash",
        )
        engine._execute_block_opcode(context, opcode)
        block_hash = context.pop_stack()
        assert isinstance(block_hash, int)
        assert block_hash > 0

        # Test TIMESTAMP
        opcode = Opcode(
            name="TIMESTAMP",
            opcode=0x42,
            gas_cost=2,
            type=OpcodeType.BLOCK,
            description="Get block timestamp",
        )
        engine._execute_block_opcode(context, opcode)
        timestamp = context.pop_stack()
        assert timestamp == 1234567890

    def test_execute_storage_opcode(self):
        """Test executing storage opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test SSTORE
        context.push_stack(456)  # value
        context.push_stack(123)  # key
        opcode = Opcode(
            name="SSTORE",
            opcode=0x55,
            gas_cost=20000,
            type=OpcodeType.STORAGE,
            description="Store to storage",
        )
        engine._execute_storage_opcode(context, opcode)

        # Test SLOAD
        context.push_stack(123)  # key
        opcode = Opcode(
            name="SLOAD",
            opcode=0x54,
            gas_cost=200,
            type=OpcodeType.STORAGE,
            description="Load from storage",
        )
        engine._execute_storage_opcode(context, opcode)
        value = context.pop_stack()
        assert value == 456

    def test_execute_memory_opcode(self):
        """Test executing memory opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test MSTORE
        context.push_stack(0x1234567890ABCDEF)  # value
        context.push_stack(0)  # offset
        opcode = Opcode(
            name="MSTORE",
            opcode=0x52,
            gas_cost=3,
            type=OpcodeType.MEMORY,
            description="Store to memory",
        )
        engine._execute_memory_opcode(context, opcode)

        # Test MLOAD
        context.push_stack(0)  # offset
        opcode = Opcode(
            name="MLOAD",
            opcode=0x51,
            gas_cost=3,
            type=OpcodeType.MEMORY,
            description="Load from memory",
        )
        engine._execute_memory_opcode(context, opcode)
        value = context.pop_stack()
        assert value == 0x1234567890ABCDEF

    def test_execute_stack_opcode(self):
        """Test executing stack opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test POP
        context.push_stack(123)
        context.push_stack(456)
        opcode = Opcode(
            name="POP",
            opcode=0x50,
            gas_cost=2,
            type=OpcodeType.STACK,
            description="Pop from stack",
        )
        engine._execute_stack_opcode(context, opcode)

        # Should have removed the top element
        assert context.pop_stack() == 123

    def test_execute_dup_opcode(self):
        """Test executing DUP opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test DUP1
        context.push_stack(123)
        context.push_stack(456)
        opcode = Opcode(
            name="DUP1",
            opcode=0x80,
            gas_cost=3,
            type=OpcodeType.DUP,
            description="Duplicate stack item",
        )
        engine._execute_dup_opcode(context, opcode)

        # Should have duplicated the top element
        assert context.pop_stack() == 456
        assert context.pop_stack() == 456
        assert context.pop_stack() == 123

    def test_execute_swap_opcode(self):
        """Test executing SWAP opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Test SWAP1
        context.push_stack(123)
        context.push_stack(456)
        context.push_stack(789)
        opcode = Opcode(
            name="SWAP1",
            opcode=0x90,
            gas_cost=3,
            type=OpcodeType.SWAP,
            description="Swap stack elements",
        )
        engine._execute_swap_opcode(context, opcode)

        # Should have swapped top two elements
        assert context.pop_stack() == 456
        assert context.pop_stack() == 789
        assert context.pop_stack() == 123

    def test_execute_log_opcode(self):
        """Test executing LOG opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={"block_number": 123, "transaction_hash": Hash.zero()},
        )

        # Set some memory data
        context.set_memory(0, b"log_data")

        # Test LOG0
        context.push_stack(0)  # offset
        context.push_stack(8)  # size
        opcode = Opcode(
            name="LOG0",
            opcode=0xA0,
            gas_cost=375,
            type=OpcodeType.LOG,
            description="Log event",
        )
        engine._execute_log_opcode(context, opcode)

        # Should have created an event
        assert len(contract.events) == 1
        event = contract.events[0]
        assert event.address == "0x123"
        assert event.data == b"log_data"
        assert event.block_number == 123

    def test_execute_system_opcode(self):
        """Test executing system opcodes."""
        from dubchain.vm.opcodes import Opcode, OpcodeType

        engine = ExecutionEngine()
        contract = SmartContract(address="0x123", bytecode=b"\x00")
        gas_meter = GasMeter(100000, 100000)
        context = ExecutionContext(
            contract=contract,
            caller="0xcaller",
            value=1000,
            data=b"call_data",
            gas_limit=100000,
            gas_meter=gas_meter,
            block_context={},
        )

        # Set some memory data
        context.set_memory(0, b"return_data")

        # Test RETURN
        context.push_stack(11)  # size
        context.push_stack(0)  # offset
        opcode = Opcode(
            name="RETURN",
            opcode=0xF3,
            gas_cost=0,
            type=OpcodeType.SYSTEM,
            description="Return from execution",
        )
        engine._execute_system_opcode(context, opcode)

        assert context.return_data == b"return_data"
        assert context.state == ExecutionState.STOPPED

        # Test REVERT
        context.state = ExecutionState.RUNNING
        context.set_memory(0, b"revert_data")
        context.push_stack(11)  # size
        context.push_stack(0)  # offset
        opcode = Opcode(
            name="REVERT",
            opcode=0xFD,
            gas_cost=0,
            type=OpcodeType.SYSTEM,
            description="Revert execution",
        )
        engine._execute_system_opcode(context, opcode)

        assert context.return_data == b"revert_data"
        assert context.state == ExecutionState.REVERTED
        assert context.error_message == "Contract reverted"
