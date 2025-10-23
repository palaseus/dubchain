"""
Unit tests for VM opcodes.
"""

import logging

logger = logging.getLogger(__name__)
import pytest

from dubchain.vm.opcodes import OPCODES, Opcode, OpcodeType


class TestOpcode:
    """Test Opcode class."""

    def test_opcode_creation(self):
        """Test opcode creation."""
        opcode = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation",
        )

        assert opcode.name == "ADD"
        assert opcode.opcode == 0x01
        assert opcode.type == OpcodeType.ARITHMETIC
        assert opcode.gas_cost == 3
        assert opcode.description == "Addition operation"

    def test_opcode_string_representation(self):
        """Test opcode string representation."""
        opcode = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation",
        )

        assert str(opcode) == "ADD(0x01)"
        assert (
            repr(opcode)
            == "Opcode(name='ADD', opcode=0x01, type=OpcodeType.ARITHMETIC, gas_cost=3)"
        )

    def test_opcode_equality(self):
        """Test opcode equality."""
        opcode1 = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation",
        )

        opcode2 = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation",
        )

        opcode3 = Opcode(
            name="SUB",
            opcode=0x02,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Subtraction operation",
        )

        assert opcode1 == opcode2
        assert opcode1 != opcode3
        assert hash(opcode1) == hash(opcode2)
        assert hash(opcode1) != hash(opcode3)


class TestOpcodeType:
    """Test OpcodeType enum."""

    def test_opcode_type_values(self):
        """Test opcode type values."""
        assert OpcodeType.STOP.value == "stop"
        assert OpcodeType.ARITHMETIC.value == "arithmetic"
        assert OpcodeType.COMPARISON.value == "comparison"
        assert OpcodeType.BITWISE.value == "bitwise"
        assert OpcodeType.SHA3.value == "sha3"
        assert OpcodeType.ENVIRONMENT.value == "environment"
        assert OpcodeType.BLOCK.value == "block"
        assert OpcodeType.STORAGE.value == "storage"
        assert OpcodeType.MEMORY.value == "memory"
        assert OpcodeType.STACK.value == "stack"
        assert OpcodeType.DUP.value == "dup"
        assert OpcodeType.SWAP.value == "swap"
        assert OpcodeType.LOG.value == "log"
        assert OpcodeType.SYSTEM.value == "system"


class TestOPCODES:
    """Test OPCODES dictionary."""

    def test_opcodes_dictionary_structure(self):
        """Test OPCODES dictionary structure."""
        assert isinstance(OPCODES, dict)
        assert len(OPCODES) > 0

        # Check that all values are Opcode instances
        for opcode in OPCODES.values():
            assert isinstance(opcode, Opcode)

    def test_arithmetic_opcodes(self):
        """Test arithmetic opcodes."""
        arithmetic_opcodes = [
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "MOD",
            "ADDMOD",
            "MULMOD",
            "EXP",
        ]

        for opcode_name in arithmetic_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type == OpcodeType.ARITHMETIC

    def test_comparison_opcodes(self):
        """Test comparison opcodes."""
        comparison_opcodes = [
            "LT",
            "GT",
            "SLT",
            "SGT",
            "EQ",
            "ISZERO",
            "AND",
            "OR",
            "XOR",
            "NOT",
        ]

        for opcode_name in comparison_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type in [OpcodeType.COMPARISON, OpcodeType.BITWISE]

    def test_storage_opcodes(self):
        """Test storage opcodes."""
        storage_opcodes = ["SLOAD", "SSTORE"]

        for opcode_name in storage_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type == OpcodeType.STORAGE

    def test_memory_opcodes(self):
        """Test memory opcodes."""
        memory_opcodes = ["MLOAD", "MSTORE", "MSTORE8", "MSIZE"]

        for opcode_name in memory_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type == OpcodeType.MEMORY

    def test_stack_opcodes(self):
        """Test stack opcodes."""
        stack_opcodes = [
            "POP",
            "PUSH1",
            "PUSH2",
            "PUSH32",
            "DUP1",
            "DUP2",
            "SWAP1",
            "SWAP2",
        ]

        for opcode_name in stack_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type in [
                    OpcodeType.STACK,
                    OpcodeType.DUP,
                    OpcodeType.SWAP,
                ]

    def test_system_opcodes(self):
        """Test system opcodes."""
        system_opcodes = ["STOP", "RETURN", "REVERT", "INVALID", "SELFDESTRUCT"]

        for opcode_name in system_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type in [OpcodeType.STOP, OpcodeType.SYSTEM]

    def test_environment_opcodes(self):
        """Test environment opcodes."""
        environment_opcodes = [
            "ADDRESS",
            "BALANCE",
            "ORIGIN",
            "CALLER",
            "CALLVALUE",
            "CALLDATALOAD",
            "CALLDATASIZE",
            "CALLDATACOPY",
            "CODESIZE",
            "CODECOPY",
            "GASPRICE",
            "EXTCODESIZE",
            "EXTCODECOPY",
            "RETURNDATASIZE",
            "RETURNDATACOPY",
            "EXTCODEHASH",
        ]

        for opcode_name in environment_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type == OpcodeType.ENVIRONMENT

    def test_block_opcodes(self):
        """Test block opcodes."""
        block_opcodes = [
            "BLOCKHASH",
            "COINBASE",
            "TIMESTAMP",
            "NUMBER",
            "DIFFICULTY",
            "GASLIMIT",
        ]

        for opcode_name in block_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type == OpcodeType.BLOCK

    def test_log_opcodes(self):
        """Test log opcodes."""
        log_opcodes = ["LOG0", "LOG1", "LOG2", "LOG3", "LOG4"]

        for opcode_name in log_opcodes:
            if opcode_name in [opcode.name for opcode in OPCODES.values()]:
                opcode = next(
                    opcode for opcode in OPCODES.values() if opcode.name == opcode_name
                )
                assert opcode.type == OpcodeType.LOG

    def test_opcode_gas_costs(self):
        """Test opcode gas costs."""
        for opcode in OPCODES.values():
            assert opcode.gas_cost >= 0
            assert isinstance(opcode.gas_cost, int)

    def test_opcode_descriptions(self):
        """Test opcode descriptions."""
        for opcode in OPCODES.values():
            assert isinstance(opcode.description, str)
            assert len(opcode.description) > 0

    def test_opcode_names(self):
        """Test opcode names."""
        for opcode in OPCODES.values():
            assert isinstance(opcode.name, str)
            assert len(opcode.name) > 0
            assert opcode.name.isupper()

    def test_opcode_codes(self):
        """Test opcode codes."""
        for opcode in OPCODES.values():
            assert isinstance(opcode.opcode, int)
            assert 0 <= opcode.opcode <= 255

    def test_opcode_types(self):
        """Test opcode types."""
        for opcode in OPCODES.values():
            assert isinstance(opcode.type, OpcodeType)

    def test_opcode_uniqueness(self):
        """Test opcode uniqueness."""
        opcode_codes = [opcode.opcode for opcode in OPCODES.values()]
        opcode_names = [opcode.name for opcode in OPCODES.values()]

        assert len(opcode_codes) == len(set(opcode_codes))
        assert len(opcode_names) == len(set(opcode_names))

    def test_opcode_coverage(self):
        """Test opcode coverage."""
        # Test that we have opcodes for all major categories
        opcode_types = set(opcode.type for opcode in OPCODES.values())

        expected_types = {
            OpcodeType.STOP,
            OpcodeType.ARITHMETIC,
            OpcodeType.COMPARISON,
            OpcodeType.BITWISE,
            OpcodeType.SHA3,
            OpcodeType.ENVIRONMENT,
            OpcodeType.BLOCK,
            OpcodeType.STORAGE,
            OpcodeType.MEMORY,
            OpcodeType.STACK,
            OpcodeType.DUP,
            OpcodeType.SWAP,
            OpcodeType.LOG,
            OpcodeType.SYSTEM,
        }

        assert opcode_types.issuperset(expected_types)

    def test_opcode_consistency(self):
        """Test opcode consistency."""
        for opcode in OPCODES.values():
            # Check that opcode name matches its type
            if opcode.type == OpcodeType.ARITHMETIC:
                assert opcode.name in [
                    "ADD",
                    "SUB",
                    "MUL",
                    "DIV",
                    "MOD",
                    "ADDMOD",
                    "MULMOD",
                    "EXP",
                ]
            elif opcode.type == OpcodeType.COMPARISON:
                assert opcode.name in ["LT", "GT", "SLT", "SGT", "EQ", "ISZERO"]
            elif opcode.type == OpcodeType.BITWISE:
                assert opcode.name in ["AND", "OR", "XOR", "NOT"]
            elif opcode.type == OpcodeType.STORAGE:
                assert opcode.name in ["SLOAD", "SSTORE"]
            elif opcode.type == OpcodeType.MEMORY:
                assert opcode.name in ["MLOAD", "MSTORE", "MSTORE8", "MSIZE"]
            elif opcode.type == OpcodeType.SYSTEM:
                assert opcode.name in [
                    "STOP",
                    "RETURN",
                    "REVERT",
                    "INVALID",
                    "SELFDESTRUCT",
                    "CREATE",
                    "CALL",
                    "CALLCODE",
                    "DELEGATECALL",
                    "CREATE2",
                    "STATICCALL",
                ]

    def test_opcode_gas_cost_ranges(self):
        """Test opcode gas cost ranges."""
        for opcode in OPCODES.values():
            if opcode.type == OpcodeType.STOP:
                assert opcode.gas_cost == 0
            elif opcode.type == OpcodeType.ARITHMETIC:
                assert 3 <= opcode.gas_cost <= 10
            elif opcode.type == OpcodeType.STORAGE:
                assert opcode.gas_cost >= 200
            elif opcode.type == OpcodeType.MEMORY:
                assert 2 <= opcode.gas_cost <= 6
            elif opcode.type == OpcodeType.SYSTEM:
                assert opcode.gas_cost >= 0
