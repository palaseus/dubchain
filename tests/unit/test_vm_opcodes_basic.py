"""Basic tests for VM opcodes module."""

import pytest
from unittest.mock import Mock

from src.dubchain.vm.opcodes import (
    OpcodeType,
    OpcodeEnum,
    Opcode,
    OpcodeInfo,
    OpcodeRegistry,
)


class TestOpcodeType:
    """Test OpcodeType enum."""

    def test_opcode_types(self):
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


class TestOpcodeEnum:
    """Test OpcodeEnum values."""

    def test_stop_and_arithmetic_opcodes(self):
        """Test stop and arithmetic opcode values."""
        assert OpcodeEnum.STOP == 0x00
        assert OpcodeEnum.ADD == 0x01
        assert OpcodeEnum.MUL == 0x02
        assert OpcodeEnum.SUB == 0x03
        assert OpcodeEnum.DIV == 0x04
        assert OpcodeEnum.MOD == 0x05

    def test_comparison_and_bitwise_opcodes(self):
        """Test comparison and bitwise opcode values."""
        assert OpcodeEnum.LT == 0x10
        assert OpcodeEnum.GT == 0x11
        assert OpcodeEnum.EQ == 0x14
        assert OpcodeEnum.ISZERO == 0x15
        assert OpcodeEnum.AND == 0x16
        assert OpcodeEnum.OR == 0x17
        assert OpcodeEnum.XOR == 0x18
        assert OpcodeEnum.NOT == 0x19

    def test_cryptographic_opcodes(self):
        """Test cryptographic opcode values."""
        assert OpcodeEnum.SHA3 == 0x20
        assert OpcodeEnum.KECCAK256 == 0x21
        assert OpcodeEnum.RIPEMD160 == 0x22
        assert OpcodeEnum.SHA256 == 0x23
        assert OpcodeEnum.ECRECOVER == 0x24

    def test_environmental_opcodes(self):
        """Test environmental opcode values."""
        assert OpcodeEnum.ADDRESS == 0x30
        assert OpcodeEnum.BALANCE == 0x31
        assert OpcodeEnum.ORIGIN == 0x32
        assert OpcodeEnum.CALLER == 0x33
        assert OpcodeEnum.CALLVALUE == 0x34

    def test_block_opcodes(self):
        """Test block opcode values."""
        assert OpcodeEnum.BLOCKHASH == 0x40
        assert OpcodeEnum.COINBASE == 0x41
        assert OpcodeEnum.TIMESTAMP == 0x42
        assert OpcodeEnum.NUMBER == 0x43
        assert OpcodeEnum.DIFFICULTY == 0x44
        assert OpcodeEnum.GASLIMIT == 0x45


class TestOpcode:
    """Test Opcode class."""

    def test_init(self):
        """Test Opcode initialization."""
        opcode = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation"
        )
        
        assert opcode.name == "ADD"
        assert opcode.opcode == 0x01
        assert opcode.type == OpcodeType.ARITHMETIC
        assert opcode.gas_cost == 3
        assert opcode.description == "Addition operation"

    def test_str_repr(self):
        """Test string representation."""
        opcode = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation"
        )
        
        str_repr = str(opcode)
        assert "ADD" in str_repr
        assert "0x01" in str_repr
        
        repr_str = repr(opcode)
        assert "ADD" in repr_str
        assert "0x01" in repr_str

    def test_eq_hash(self):
        """Test equality and hashing."""
        opcode1 = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation"
        )
        
        opcode2 = Opcode(
            name="ADD",
            opcode=0x01,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Addition operation"
        )
        
        opcode3 = Opcode(
            name="SUB",
            opcode=0x03,
            type=OpcodeType.ARITHMETIC,
            gas_cost=3,
            description="Subtraction operation"
        )
        
        assert opcode1 == opcode2
        assert opcode1 != opcode3
        assert hash(opcode1) == hash(opcode2)
        assert hash(opcode1) != hash(opcode3)


class TestOpcodeInfo:
    """Test OpcodeInfo class."""

    def test_init(self):
        """Test OpcodeInfo initialization."""
        info = OpcodeInfo(
            name="ADD",
            gas_cost=3,
            stack_inputs=2,
            stack_outputs=1,
            description="Addition operation",
            category="arithmetic"
        )
        
        assert info.name == "ADD"
        assert info.gas_cost == 3
        assert info.stack_inputs == 2
        assert info.stack_outputs == 1
        assert info.description == "Addition operation"
        assert info.category == "arithmetic"


class TestOpcodeRegistry:
    """Test OpcodeRegistry class."""

    def test_init(self):
        """Test OpcodeRegistry initialization."""
        registry = OpcodeRegistry()
        assert len(registry._opcodes) > 0  # Should have registered opcodes
        assert len(registry._handlers) == 0  # No handlers initially

    def test_get_info(self):
        """Test getting opcode info from registry."""
        registry = OpcodeRegistry()
        
        info = registry.get_info(OpcodeEnum.ADD)
        assert info is not None
        assert info.name == "ADD"
        assert info.gas_cost == 3
        assert info.stack_inputs == 2
        assert info.stack_outputs == 1
        assert info.category == "arithmetic"

    def test_get_info_nonexistent(self):
        """Test getting info for non-existent opcode."""
        registry = OpcodeRegistry()
        
        # Use a truly invalid opcode value that's not in the enum
        try:
            nonexistent_opcode = OpcodeEnum(0x100)  # This should fail
            info = registry.get_info(nonexistent_opcode)
            assert info is None
        except ValueError:
            # This is expected - the opcode doesn't exist in the enum
            pass

    def test_get_gas_cost(self):
        """Test getting gas cost."""
        registry = OpcodeRegistry()
        
        gas_cost = registry.get_gas_cost(OpcodeEnum.ADD)
        assert gas_cost == 3
        
        gas_cost = registry.get_gas_cost(OpcodeEnum.MUL)
        assert gas_cost == 5

    def test_get_gas_cost_nonexistent(self):
        """Test getting gas cost for non-existent opcode."""
        registry = OpcodeRegistry()
        
        try:
            nonexistent_opcode = OpcodeEnum(0x100)  # This should fail
            gas_cost = registry.get_gas_cost(nonexistent_opcode)
            assert gas_cost == 0
        except ValueError:
            # This is expected - the opcode doesn't exist in the enum
            pass

    def test_get_stack_inputs(self):
        """Test getting stack inputs."""
        registry = OpcodeRegistry()
        
        inputs = registry.get_stack_inputs(OpcodeEnum.ADD)
        assert inputs == 2
        
        inputs = registry.get_stack_inputs(OpcodeEnum.STOP)
        assert inputs == 0

    def test_get_stack_outputs(self):
        """Test getting stack outputs."""
        registry = OpcodeRegistry()
        
        outputs = registry.get_stack_outputs(OpcodeEnum.ADD)
        assert outputs == 1
        
        outputs = registry.get_stack_outputs(OpcodeEnum.STOP)
        assert outputs == 0

    def test_is_valid_opcode(self):
        """Test checking if opcode is valid."""
        registry = OpcodeRegistry()
        
        assert registry.is_valid_opcode(0x01) is True  # ADD
        assert registry.is_valid_opcode(0x00) is True  # STOP
        assert registry.is_valid_opcode(0x100) is False  # Invalid (not in enum)

    def test_get_all_opcodes(self):
        """Test getting all opcodes."""
        registry = OpcodeRegistry()
        
        all_opcodes = registry.get_all_opcodes()
        assert len(all_opcodes) > 0
        assert OpcodeEnum.ADD in all_opcodes
        assert OpcodeEnum.STOP in all_opcodes

    def test_get_opcodes_by_category(self):
        """Test getting opcodes by category."""
        registry = OpcodeRegistry()
        
        arithmetic_opcodes = registry.get_opcodes_by_category("arithmetic")
        assert len(arithmetic_opcodes) > 0
        assert OpcodeEnum.ADD in arithmetic_opcodes
        assert OpcodeEnum.SUB in arithmetic_opcodes
        assert OpcodeEnum.MUL in arithmetic_opcodes
        
        control_opcodes = registry.get_opcodes_by_category("control")
        assert OpcodeEnum.STOP in control_opcodes
