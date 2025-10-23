"""
Advanced opcodes for the GodChain Virtual Machine.

This module defines a comprehensive set of opcodes for smart contract execution,
including arithmetic, memory, storage, control flow, and cryptographic operations.
"""

import logging

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PublicKey, Signature


class OpcodeType(Enum):
    """Types of opcodes."""

    STOP = "stop"
    ARITHMETIC = "arithmetic"
    COMPARISON = "comparison"
    BITWISE = "bitwise"
    SHA3 = "sha3"
    ENVIRONMENT = "environment"
    BLOCK = "block"
    STORAGE = "storage"
    MEMORY = "memory"
    STACK = "stack"
    DUP = "dup"
    SWAP = "swap"
    LOG = "log"
    SYSTEM = "system"


class OpcodeEnum(IntEnum):
    """Enumeration of all supported opcodes."""

    # Stop and arithmetic operations
    STOP = 0x00
    ADD = 0x01
    MUL = 0x02
    SUB = 0x03
    DIV = 0x04
    MOD = 0x05
    ADDMOD = 0x06
    MULMOD = 0x07
    EXP = 0x08
    SIGNEXTEND = 0x09

    # Comparison and bitwise operations
    LT = 0x10
    GT = 0x11
    SLT = 0x12
    SGT = 0x13
    EQ = 0x14
    ISZERO = 0x15
    AND = 0x16
    OR = 0x17
    XOR = 0x18
    NOT = 0x19
    BYTE = 0x1A
    SHL = 0x1B
    SHR = 0x1C
    SAR = 0x1D

    # SHA3 and cryptographic operations
    SHA3 = 0x20
    KECCAK256 = 0x21
    RIPEMD160 = 0x22
    SHA256 = 0x23
    ECRECOVER = 0x24
    ECRECOVER_PRECOMPILED = 0x25

    # Environmental information
    ADDRESS = 0x30
    BALANCE = 0x31
    ORIGIN = 0x32
    CALLER = 0x33
    CALLVALUE = 0x34
    CALLDATALOAD = 0x35
    CALLDATASIZE = 0x36
    CALLDATACOPY = 0x37
    CODESIZE = 0x38
    CODECOPY = 0x39
    GASPRICE = 0x3A
    EXTCODESIZE = 0x3B
    EXTCODECOPY = 0x3C
    RETURNDATASIZE = 0x3D
    RETURNDATACOPY = 0x3E
    EXTCODEHASH = 0x3F

    # Block information
    BLOCKHASH = 0x40
    COINBASE = 0x41
    TIMESTAMP = 0x42
    NUMBER = 0x43
    DIFFICULTY = 0x44
    GASLIMIT = 0x45
    CHAINID = 0x46
    SELFBALANCE = 0x47

    # Storage and memory operations
    POP = 0x50
    MLOAD = 0x51
    MSTORE = 0x52
    MSTORE8 = 0x53
    SLOAD = 0x54
    SSTORE = 0x55
    MSIZE = 0x56
    GAS = 0x57
    LOG0 = 0xA0
    LOG1 = 0xA1
    LOG2 = 0xA2
    LOG3 = 0xA3
    LOG4 = 0xA4

    # System operations
    CREATE = 0xF0
    CALL = 0xF1
    CALLCODE = 0xF2
    RETURN = 0xF3
    DELEGATECALL = 0xF4
    CREATE2 = 0xF5
    STATICCALL = 0xFA
    REVERT = 0xFD
    INVALID = 0xFE
    SELFDESTRUCT = 0xFF

    # Stack operations
    PUSH0 = 0x5F
    PUSH1 = 0x60
    PUSH2 = 0x61
    PUSH3 = 0x62
    PUSH4 = 0x63
    PUSH5 = 0x64
    PUSH6 = 0x65
    PUSH7 = 0x66
    PUSH8 = 0x67
    PUSH9 = 0x68
    PUSH10 = 0x69
    PUSH11 = 0x6A
    PUSH12 = 0x6B
    PUSH13 = 0x6C
    PUSH14 = 0x6D
    PUSH15 = 0x6E
    PUSH16 = 0x6F
    PUSH17 = 0x70
    PUSH18 = 0x71
    PUSH19 = 0x72
    PUSH20 = 0x73
    PUSH21 = 0x74
    PUSH22 = 0x75
    PUSH23 = 0x76
    PUSH24 = 0x77
    PUSH25 = 0x78
    PUSH26 = 0x79
    PUSH27 = 0x7A
    PUSH28 = 0x7B
    PUSH29 = 0x7C
    PUSH30 = 0x7D
    PUSH31 = 0x7E
    PUSH32 = 0x7F

    # Duplication operations
    DUP1 = 0x80
    DUP2 = 0x81
    DUP3 = 0x82
    DUP4 = 0x83
    DUP5 = 0x84
    DUP6 = 0x85
    DUP7 = 0x86
    DUP8 = 0x87
    DUP9 = 0x88
    DUP10 = 0x89
    DUP11 = 0x8A
    DUP12 = 0x8B
    DUP13 = 0x8C
    DUP14 = 0x8D
    DUP15 = 0x8E
    DUP16 = 0x8F

    # Exchange operations
    SWAP1 = 0x90
    SWAP2 = 0x91
    SWAP3 = 0x92
    SWAP4 = 0x93
    SWAP5 = 0x94
    SWAP6 = 0x95
    SWAP7 = 0x96
    SWAP8 = 0x97
    SWAP9 = 0x98
    SWAP10 = 0x99
    SWAP11 = 0x9A
    SWAP12 = 0x9B
    SWAP13 = 0x9C
    SWAP14 = 0x9D
    SWAP15 = 0x9E
    SWAP16 = 0x9F

    # Control flow operations
    JUMP = 0x56
    JUMPI = 0x57
    PC = 0x58
    JUMPDEST = 0x5B


@dataclass
class OpcodeInfo:
    """Information about an opcode."""

    name: str
    gas_cost: int
    stack_inputs: int
    stack_outputs: int
    description: str
    category: str


class OpcodeRegistry:
    """Registry for opcode information and execution handlers."""

    def __init__(self):
        self._opcodes: Dict[OpcodeEnum, OpcodeInfo] = {}
        self._handlers: Dict[OpcodeEnum, Callable] = {}
        self._register_opcodes()

    def _register_opcodes(self) -> None:
        """Register all opcodes with their information."""

        # Stop and arithmetic operations
        self._register(OpcodeEnum.STOP, "STOP", 0, 0, 0, "Halts execution", "control")
        self._register(
            OpcodeEnum.ADD, "ADD", 3, 2, 1, "Addition operation", "arithmetic"
        )
        self._register(
            OpcodeEnum.MUL, "MUL", 5, 2, 1, "Multiplication operation", "arithmetic"
        )
        self._register(
            OpcodeEnum.SUB, "SUB", 3, 2, 1, "Subtraction operation", "arithmetic"
        )
        self._register(
            OpcodeEnum.DIV, "DIV", 5, 2, 1, "Integer division operation", "arithmetic"
        )
        self._register(OpcodeEnum.MOD, "MOD", 5, 2, 1, "Modulo operation", "arithmetic")
        self._register(
            OpcodeEnum.ADDMOD, "ADDMOD", 8, 3, 1, "Modular addition", "arithmetic"
        )
        self._register(
            OpcodeEnum.MULMOD, "MULMOD", 8, 3, 1, "Modular multiplication", "arithmetic"
        )
        self._register(
            OpcodeEnum.EXP, "EXP", 10, 2, 1, "Exponential operation", "arithmetic"
        )

        # Comparison and bitwise operations
        self._register(
            OpcodeEnum.LT, "LT", 3, 2, 1, "Less than comparison", "comparison"
        )
        self._register(
            OpcodeEnum.GT, "GT", 3, 2, 1, "Greater than comparison", "comparison"
        )
        self._register(
            OpcodeEnum.SLT, "SLT", 3, 2, 1, "Signed less than comparison", "comparison"
        )
        self._register(
            OpcodeEnum.SGT,
            "SGT",
            3,
            2,
            1,
            "Signed greater than comparison",
            "comparison",
        )
        self._register(
            OpcodeEnum.EQ, "EQ", 3, 2, 1, "Equality comparison", "comparison"
        )
        self._register(
            OpcodeEnum.ISZERO, "ISZERO", 3, 1, 1, "Is zero check", "comparison"
        )
        self._register(OpcodeEnum.AND, "AND", 3, 2, 1, "Bitwise AND", "bitwise")
        self._register(OpcodeEnum.OR, "OR", 3, 2, 1, "Bitwise OR", "bitwise")
        self._register(OpcodeEnum.XOR, "XOR", 3, 2, 1, "Bitwise XOR", "bitwise")
        self._register(OpcodeEnum.NOT, "NOT", 3, 1, 1, "Bitwise NOT", "bitwise")
        self._register(
            OpcodeEnum.BYTE, "BYTE", 3, 2, 1, "Get byte from word", "bitwise"
        )
        self._register(OpcodeEnum.SHL, "SHL", 3, 2, 1, "Left shift", "bitwise")
        self._register(OpcodeEnum.SHR, "SHR", 3, 2, 1, "Right shift", "bitwise")
        self._register(
            OpcodeEnum.SAR, "SAR", 3, 2, 1, "Arithmetic right shift", "bitwise"
        )

        # Cryptographic operations
        self._register(OpcodeEnum.SHA3, "SHA3", 30, 2, 1, "Keccak-256 hash", "crypto")
        self._register(
            OpcodeEnum.KECCAK256, "KECCAK256", 30, 2, 1, "Keccak-256 hash", "crypto"
        )
        self._register(OpcodeEnum.SHA256, "SHA256", 60, 2, 1, "SHA-256 hash", "crypto")
        self._register(
            OpcodeEnum.RIPEMD160, "RIPEMD160", 600, 2, 1, "RIPEMD-160 hash", "crypto"
        )
        self._register(
            OpcodeEnum.ECRECOVER,
            "ECRECOVER",
            3000,
            4,
            1,
            "ECDSA signature recovery",
            "crypto",
        )

        # Environmental information
        self._register(
            OpcodeEnum.ADDRESS,
            "ADDRESS",
            2,
            0,
            1,
            "Get contract address",
            "environment",
        )
        self._register(
            OpcodeEnum.BALANCE,
            "BALANCE",
            100,
            1,
            1,
            "Get account balance",
            "environment",
        )
        self._register(
            OpcodeEnum.ORIGIN,
            "ORIGIN",
            2,
            0,
            1,
            "Get transaction origin",
            "environment",
        )
        self._register(
            OpcodeEnum.CALLER, "CALLER", 2, 0, 1, "Get caller address", "environment"
        )
        self._register(
            OpcodeEnum.CALLVALUE, "CALLVALUE", 2, 0, 1, "Get call value", "environment"
        )
        self._register(
            OpcodeEnum.CALLDATALOAD,
            "CALLDATALOAD",
            3,
            1,
            1,
            "Load call data",
            "environment",
        )
        self._register(
            OpcodeEnum.CALLDATASIZE,
            "CALLDATASIZE",
            2,
            0,
            1,
            "Get call data size",
            "environment",
        )
        self._register(
            OpcodeEnum.CALLDATACOPY,
            "CALLDATACOPY",
            3,
            3,
            0,
            "Copy call data",
            "environment",
        )
        self._register(
            OpcodeEnum.CODESIZE, "CODESIZE", 2, 0, 1, "Get code size", "environment"
        )
        self._register(
            OpcodeEnum.CODECOPY, "CODECOPY", 3, 3, 0, "Copy code", "environment"
        )
        self._register(
            OpcodeEnum.GASPRICE, "GASPRICE", 2, 0, 1, "Get gas price", "environment"
        )
        self._register(
            OpcodeEnum.EXTCODESIZE,
            "EXTCODESIZE",
            100,
            1,
            1,
            "Get external code size",
            "environment",
        )
        self._register(
            OpcodeEnum.EXTCODECOPY,
            "EXTCODECOPY",
            100,
            4,
            0,
            "Copy external code",
            "environment",
        )
        self._register(
            OpcodeEnum.EXTCODEHASH,
            "EXTCODEHASH",
            100,
            1,
            1,
            "Get external code hash",
            "environment",
        )

        # Block information
        self._register(
            OpcodeEnum.BLOCKHASH, "BLOCKHASH", 20, 1, 1, "Get block hash", "block"
        )
        self._register(
            OpcodeEnum.COINBASE, "COINBASE", 2, 0, 1, "Get block coinbase", "block"
        )
        self._register(
            OpcodeEnum.TIMESTAMP, "TIMESTAMP", 2, 0, 1, "Get block timestamp", "block"
        )
        self._register(
            OpcodeEnum.NUMBER, "NUMBER", 2, 0, 1, "Get block number", "block"
        )
        self._register(
            OpcodeEnum.DIFFICULTY,
            "DIFFICULTY",
            2,
            0,
            1,
            "Get block difficulty",
            "block",
        )
        self._register(
            OpcodeEnum.GASLIMIT, "GASLIMIT", 2, 0, 1, "Get block gas limit", "block"
        )
        self._register(OpcodeEnum.CHAINID, "CHAINID", 2, 0, 1, "Get chain ID", "block")
        self._register(
            OpcodeEnum.SELFBALANCE, "SELFBALANCE", 5, 0, 1, "Get self balance", "block"
        )

        # Storage and memory operations
        self._register(OpcodeEnum.POP, "POP", 2, 1, 0, "Remove top stack item", "stack")
        self._register(OpcodeEnum.MLOAD, "MLOAD", 3, 1, 1, "Load from memory", "memory")
        self._register(
            OpcodeEnum.MSTORE, "MSTORE", 3, 2, 0, "Store to memory", "memory"
        )
        self._register(
            OpcodeEnum.MSTORE8, "MSTORE8", 3, 2, 0, "Store byte to memory", "memory"
        )
        self._register(
            OpcodeEnum.SLOAD, "SLOAD", 100, 1, 1, "Load from storage", "storage"
        )
        self._register(
            OpcodeEnum.SSTORE, "SSTORE", 100, 2, 0, "Store to storage", "storage"
        )
        self._register(OpcodeEnum.MSIZE, "MSIZE", 2, 0, 1, "Get memory size", "memory")
        self._register(
            OpcodeEnum.GAS, "GAS", 2, 0, 1, "Get remaining gas", "environment"
        )

        # Logging operations
        self._register(
            OpcodeEnum.LOG0, "LOG0", 375, 2, 0, "Log with 0 topics", "logging"
        )
        self._register(
            OpcodeEnum.LOG1, "LOG1", 750, 3, 0, "Log with 1 topic", "logging"
        )
        self._register(
            OpcodeEnum.LOG2, "LOG2", 1125, 4, 0, "Log with 2 topics", "logging"
        )
        self._register(
            OpcodeEnum.LOG3, "LOG3", 1500, 5, 0, "Log with 3 topics", "logging"
        )
        self._register(
            OpcodeEnum.LOG4, "LOG4", 1875, 6, 0, "Log with 4 topics", "logging"
        )

        # System operations
        self._register(
            OpcodeEnum.CREATE, "CREATE", 32000, 3, 1, "Create new contract", "system"
        )
        self._register(
            OpcodeEnum.CALL, "CALL", 100, 7, 1, "Call external contract", "system"
        )
        self._register(
            OpcodeEnum.CALLCODE,
            "CALLCODE",
            100,
            7,
            1,
            "Call external contract (code)",
            "system",
        )
        self._register(OpcodeEnum.RETURN, "RETURN", 0, 2, 0, "Return data", "system")
        self._register(
            OpcodeEnum.DELEGATECALL,
            "DELEGATECALL",
            100,
            6,
            1,
            "Delegate call",
            "system",
        )
        self._register(
            OpcodeEnum.CREATE2,
            "CREATE2",
            32000,
            4,
            1,
            "Create new contract (deterministic)",
            "system",
        )
        self._register(
            OpcodeEnum.STATICCALL, "STATICCALL", 100, 6, 1, "Static call", "system"
        )
        self._register(
            OpcodeEnum.REVERT, "REVERT", 0, 2, 0, "Revert execution", "system"
        )
        self._register(
            OpcodeEnum.INVALID, "INVALID", 0, 0, 0, "Invalid instruction", "system"
        )
        self._register(
            OpcodeEnum.SELFDESTRUCT,
            "SELFDESTRUCT",
            5000,
            1,
            0,
            "Self destruct",
            "system",
        )

        # Stack operations
        for i in range(33):
            if i == 0:
                self._register(
                    OpcodeEnum.PUSH0, "PUSH0", 2, 0, 1, "Push 0 onto stack", "stack"
                )
            else:
                opcode = OpcodeEnum(OpcodeEnum.PUSH1 + i - 1)
                self._register(
                    opcode, f"PUSH{i}", 3, 0, 1, f"Push {i} bytes onto stack", "stack"
                )

        # Duplication operations
        for i in range(1, 17):
            opcode = OpcodeEnum(OpcodeEnum.DUP1 + i - 1)
            self._register(
                opcode, f"DUP{i}", 3, i, i + 1, f"Duplicate {i}th stack item", "stack"
            )

        # Exchange operations
        for i in range(1, 17):
            opcode = OpcodeEnum(OpcodeEnum.SWAP1 + i - 1)
            self._register(
                opcode,
                f"SWAP{i}",
                3,
                i + 1,
                i + 1,
                f"Swap top with {i}th stack item",
                "stack",
            )

        # Control flow operations
        self._register(
            OpcodeEnum.JUMP, "JUMP", 8, 1, 0, "Unconditional jump", "control"
        )
        self._register(
            OpcodeEnum.JUMPI, "JUMPI", 10, 2, 0, "Conditional jump", "control"
        )
        self._register(OpcodeEnum.PC, "PC", 2, 0, 1, "Get program counter", "control")
        self._register(
            OpcodeEnum.JUMPDEST, "JUMPDEST", 1, 0, 0, "Jump destination", "control"
        )

    def _register(
        self,
        opcode: OpcodeEnum,
        name: str,
        gas_cost: int,
        stack_inputs: int,
        stack_outputs: int,
        description: str,
        category: str,
    ) -> None:
        """Register an opcode with its information."""
        self._opcodes[opcode] = OpcodeInfo(
            name=name,
            gas_cost=gas_cost,
            stack_inputs=stack_inputs,
            stack_outputs=stack_outputs,
            description=description,
            category=category,
        )

    def get_info(self, opcode: OpcodeEnum) -> Optional[OpcodeInfo]:
        """Get information about an opcode."""
        return self._opcodes.get(opcode)

    def get_gas_cost(self, opcode: OpcodeEnum) -> int:
        """Get the gas cost of an opcode."""
        info = self.get_info(opcode)
        return info.gas_cost if info else 0

    def get_stack_inputs(self, opcode: OpcodeEnum) -> int:
        """Get the number of stack inputs required by an opcode."""
        info = self.get_info(opcode)
        return info.stack_inputs if info else 0

    def get_stack_outputs(self, opcode: OpcodeEnum) -> int:
        """Get the number of stack outputs produced by an opcode."""
        info = self.get_info(opcode)
        return info.stack_outputs if info else 0

    def is_valid_opcode(self, opcode: int) -> bool:
        """Check if an opcode is valid."""
        try:
            return OpcodeEnum(opcode) in self._opcodes
        except ValueError:
            return False

    def get_all_opcodes(self) -> List[OpcodeEnum]:
        """Get all registered opcodes."""
        return list(self._opcodes.keys())

    def get_opcodes_by_category(self, category: str) -> List[OpcodeEnum]:
        """Get all opcodes in a specific category."""
        return [
            opcode
            for opcode, info in self._opcodes.items()
            if info.category == category
        ]

    def register_handler(self, opcode: OpcodeEnum, handler: Callable) -> None:
        """Register a custom handler for an opcode."""
        self._handlers[opcode] = handler

    def get_handler(self, opcode: OpcodeEnum) -> Optional[Callable]:
        """Get the handler for an opcode."""
        return self._handlers.get(opcode)

    def __str__(self) -> str:
        """String representation of the registry."""
        return f"OpcodeRegistry({len(self._opcodes)} opcodes)"

    def __repr__(self) -> str:
        """Detailed representation of the registry."""
        categories = {}
        for info in self._opcodes.values():
            if info.category not in categories:
                categories[info.category] = 0
            categories[info.category] += 1

        return f"OpcodeRegistry({len(self._opcodes)} opcodes, {len(categories)} categories)"


# Create a simple Opcode dataclass for compatibility with tests
@dataclass
class Opcode:
    """Simple opcode representation for compatibility."""

    name: str
    opcode: int
    type: OpcodeType
    gas_cost: int
    description: str

    def __str__(self) -> str:
        return f"{self.name}(0x{self.opcode:02x})"

    def __repr__(self) -> str:
        return f"Opcode(name='{self.name}', opcode=0x{self.opcode:02x}, type={self.type}, gas_cost={self.gas_cost})"

    def __eq__(self, other):
        if not isinstance(other, Opcode):
            return False
        return self.opcode == other.opcode

    def __hash__(self):
        return hash(self.opcode)


# Create OPCODES dictionary for compatibility
OPCODES = {
    # Stop and arithmetic operations
    0x00: Opcode("STOP", 0x00, OpcodeType.STOP, 0, "Halts execution"),
    0x01: Opcode("ADD", 0x01, OpcodeType.ARITHMETIC, 3, "Addition operation"),
    0x02: Opcode("MUL", 0x02, OpcodeType.ARITHMETIC, 5, "Multiplication operation"),
    0x03: Opcode("SUB", 0x03, OpcodeType.ARITHMETIC, 3, "Subtraction operation"),
    0x04: Opcode("DIV", 0x04, OpcodeType.ARITHMETIC, 5, "Integer division operation"),
    0x05: Opcode("MOD", 0x05, OpcodeType.ARITHMETIC, 5, "Modulo operation"),
    0x06: Opcode("ADDMOD", 0x06, OpcodeType.ARITHMETIC, 8, "Modular addition"),
    0x07: Opcode("MULMOD", 0x07, OpcodeType.ARITHMETIC, 8, "Modular multiplication"),
    0x08: Opcode("EXP", 0x08, OpcodeType.ARITHMETIC, 10, "Exponential operation"),
    # Comparison and bitwise operations
    0x10: Opcode("LT", 0x10, OpcodeType.COMPARISON, 3, "Less than comparison"),
    0x11: Opcode("GT", 0x11, OpcodeType.COMPARISON, 3, "Greater than comparison"),
    0x12: Opcode("SLT", 0x12, OpcodeType.COMPARISON, 3, "Signed less than comparison"),
    0x13: Opcode(
        "SGT", 0x13, OpcodeType.COMPARISON, 3, "Signed greater than comparison"
    ),
    0x14: Opcode("EQ", 0x14, OpcodeType.COMPARISON, 3, "Equality comparison"),
    0x15: Opcode("ISZERO", 0x15, OpcodeType.COMPARISON, 3, "Is zero check"),
    0x16: Opcode("AND", 0x16, OpcodeType.BITWISE, 3, "Bitwise AND"),
    0x17: Opcode("OR", 0x17, OpcodeType.BITWISE, 3, "Bitwise OR"),
    0x18: Opcode("XOR", 0x18, OpcodeType.BITWISE, 3, "Bitwise XOR"),
    0x19: Opcode("NOT", 0x19, OpcodeType.BITWISE, 3, "Bitwise NOT"),
    # SHA3 and cryptographic operations
    0x20: Opcode("SHA3", 0x20, OpcodeType.SHA3, 30, "Keccak-256 hash"),
    # Environmental information
    0x30: Opcode("ADDRESS", 0x30, OpcodeType.ENVIRONMENT, 2, "Get contract address"),
    0x31: Opcode("BALANCE", 0x31, OpcodeType.ENVIRONMENT, 100, "Get account balance"),
    0x32: Opcode("ORIGIN", 0x32, OpcodeType.ENVIRONMENT, 2, "Get transaction origin"),
    0x33: Opcode("CALLER", 0x33, OpcodeType.ENVIRONMENT, 2, "Get caller address"),
    0x34: Opcode("CALLVALUE", 0x34, OpcodeType.ENVIRONMENT, 2, "Get call value"),
    0x35: Opcode("CALLDATALOAD", 0x35, OpcodeType.ENVIRONMENT, 3, "Load call data"),
    0x36: Opcode("CALLDATASIZE", 0x36, OpcodeType.ENVIRONMENT, 2, "Get call data size"),
    0x37: Opcode("CALLDATACOPY", 0x37, OpcodeType.ENVIRONMENT, 3, "Copy call data"),
    0x38: Opcode("CODESIZE", 0x38, OpcodeType.ENVIRONMENT, 2, "Get code size"),
    0x39: Opcode("CODECOPY", 0x39, OpcodeType.ENVIRONMENT, 3, "Copy code"),
    0x3A: Opcode("GASPRICE", 0x3A, OpcodeType.ENVIRONMENT, 2, "Get gas price"),
    0x3B: Opcode(
        "EXTCODESIZE", 0x3B, OpcodeType.ENVIRONMENT, 100, "Get external code size"
    ),
    0x3C: Opcode(
        "EXTCODECOPY", 0x3C, OpcodeType.ENVIRONMENT, 100, "Copy external code"
    ),
    0x3D: Opcode(
        "RETURNDATASIZE", 0x3D, OpcodeType.ENVIRONMENT, 2, "Get return data size"
    ),
    0x3E: Opcode("RETURNDATACOPY", 0x3E, OpcodeType.ENVIRONMENT, 3, "Copy return data"),
    0x3F: Opcode(
        "EXTCODEHASH", 0x3F, OpcodeType.ENVIRONMENT, 100, "Get external code hash"
    ),
    # Block information
    0x40: Opcode("BLOCKHASH", 0x40, OpcodeType.BLOCK, 20, "Get block hash"),
    0x41: Opcode("COINBASE", 0x41, OpcodeType.BLOCK, 2, "Get block coinbase"),
    0x42: Opcode("TIMESTAMP", 0x42, OpcodeType.BLOCK, 2, "Get block timestamp"),
    0x43: Opcode("NUMBER", 0x43, OpcodeType.BLOCK, 2, "Get block number"),
    0x44: Opcode("DIFFICULTY", 0x44, OpcodeType.BLOCK, 2, "Get block difficulty"),
    0x45: Opcode("GASLIMIT", 0x45, OpcodeType.BLOCK, 2, "Get block gas limit"),
    # Storage and memory operations
    0x50: Opcode("POP", 0x50, OpcodeType.STACK, 2, "Remove top stack item"),
    0x51: Opcode("MLOAD", 0x51, OpcodeType.MEMORY, 3, "Load from memory"),
    0x52: Opcode("MSTORE", 0x52, OpcodeType.MEMORY, 3, "Store to memory"),
    0x53: Opcode("MSTORE8", 0x53, OpcodeType.MEMORY, 3, "Store byte to memory"),
    0x54: Opcode("SLOAD", 0x54, OpcodeType.STORAGE, 200, "Load from storage"),
    0x55: Opcode("SSTORE", 0x55, OpcodeType.STORAGE, 20000, "Store to storage"),
    0x56: Opcode("MSIZE", 0x56, OpcodeType.MEMORY, 2, "Get memory size"),
    0x57: Opcode("GAS", 0x57, OpcodeType.ENVIRONMENT, 2, "Get remaining gas"),
    # Logging operations
    0xA0: Opcode("LOG0", 0xA0, OpcodeType.LOG, 375, "Log with 0 topics"),
    0xA1: Opcode("LOG1", 0xA1, OpcodeType.LOG, 750, "Log with 1 topic"),
    0xA2: Opcode("LOG2", 0xA2, OpcodeType.LOG, 1125, "Log with 2 topics"),
    0xA3: Opcode("LOG3", 0xA3, OpcodeType.LOG, 1500, "Log with 3 topics"),
    0xA4: Opcode("LOG4", 0xA4, OpcodeType.LOG, 1875, "Log with 4 topics"),
    # System operations
    0xF0: Opcode("CREATE", 0xF0, OpcodeType.SYSTEM, 32000, "Create new contract"),
    0xF1: Opcode("CALL", 0xF1, OpcodeType.SYSTEM, 100, "Call external contract"),
    0xF2: Opcode(
        "CALLCODE", 0xF2, OpcodeType.SYSTEM, 100, "Call external contract (code)"
    ),
    0xF3: Opcode("RETURN", 0xF3, OpcodeType.SYSTEM, 0, "Return data"),
    0xF4: Opcode("DELEGATECALL", 0xF4, OpcodeType.SYSTEM, 100, "Delegate call"),
    0xF5: Opcode(
        "CREATE2", 0xF5, OpcodeType.SYSTEM, 32000, "Create new contract (deterministic)"
    ),
    0xFA: Opcode("STATICCALL", 0xFA, OpcodeType.SYSTEM, 100, "Static call"),
    0xFD: Opcode("REVERT", 0xFD, OpcodeType.SYSTEM, 0, "Revert execution"),
    0xFE: Opcode("INVALID", 0xFE, OpcodeType.SYSTEM, 0, "Invalid instruction"),
    0xFF: Opcode("SELFDESTRUCT", 0xFF, OpcodeType.SYSTEM, 5000, "Self destruct"),
    # Stack operations
    0x5F: Opcode("PUSH0", 0x5F, OpcodeType.STACK, 2, "Push 0 onto stack"),
    0x60: Opcode("PUSH1", 0x60, OpcodeType.STACK, 3, "Push 1 byte onto stack"),
    0x61: Opcode("PUSH2", 0x61, OpcodeType.STACK, 3, "Push 2 bytes onto stack"),
    0x62: Opcode("PUSH3", 0x62, OpcodeType.STACK, 3, "Push 3 bytes onto stack"),
    0x63: Opcode("PUSH4", 0x63, OpcodeType.STACK, 3, "Push 4 bytes onto stack"),
    0x64: Opcode("PUSH5", 0x64, OpcodeType.STACK, 3, "Push 5 bytes onto stack"),
    0x65: Opcode("PUSH6", 0x65, OpcodeType.STACK, 3, "Push 6 bytes onto stack"),
    0x66: Opcode("PUSH7", 0x66, OpcodeType.STACK, 3, "Push 7 bytes onto stack"),
    0x67: Opcode("PUSH8", 0x67, OpcodeType.STACK, 3, "Push 8 bytes onto stack"),
    0x68: Opcode("PUSH9", 0x68, OpcodeType.STACK, 3, "Push 9 bytes onto stack"),
    0x69: Opcode("PUSH10", 0x69, OpcodeType.STACK, 3, "Push 10 bytes onto stack"),
    0x6A: Opcode("PUSH11", 0x6A, OpcodeType.STACK, 3, "Push 11 bytes onto stack"),
    0x6B: Opcode("PUSH12", 0x6B, OpcodeType.STACK, 3, "Push 12 bytes onto stack"),
    0x6C: Opcode("PUSH13", 0x6C, OpcodeType.STACK, 3, "Push 13 bytes onto stack"),
    0x6D: Opcode("PUSH14", 0x6D, OpcodeType.STACK, 3, "Push 14 bytes onto stack"),
    0x6E: Opcode("PUSH15", 0x6E, OpcodeType.STACK, 3, "Push 15 bytes onto stack"),
    0x6F: Opcode("PUSH16", 0x6F, OpcodeType.STACK, 3, "Push 16 bytes onto stack"),
    0x70: Opcode("PUSH17", 0x70, OpcodeType.STACK, 3, "Push 17 bytes onto stack"),
    0x71: Opcode("PUSH18", 0x71, OpcodeType.STACK, 3, "Push 18 bytes onto stack"),
    0x72: Opcode("PUSH19", 0x72, OpcodeType.STACK, 3, "Push 19 bytes onto stack"),
    0x73: Opcode("PUSH20", 0x73, OpcodeType.STACK, 3, "Push 20 bytes onto stack"),
    0x74: Opcode("PUSH21", 0x74, OpcodeType.STACK, 3, "Push 21 bytes onto stack"),
    0x75: Opcode("PUSH22", 0x75, OpcodeType.STACK, 3, "Push 22 bytes onto stack"),
    0x76: Opcode("PUSH23", 0x76, OpcodeType.STACK, 3, "Push 23 bytes onto stack"),
    0x77: Opcode("PUSH24", 0x77, OpcodeType.STACK, 3, "Push 24 bytes onto stack"),
    0x78: Opcode("PUSH25", 0x78, OpcodeType.STACK, 3, "Push 25 bytes onto stack"),
    0x79: Opcode("PUSH26", 0x79, OpcodeType.STACK, 3, "Push 26 bytes onto stack"),
    0x7A: Opcode("PUSH27", 0x7A, OpcodeType.STACK, 3, "Push 27 bytes onto stack"),
    0x7B: Opcode("PUSH28", 0x7B, OpcodeType.STACK, 3, "Push 28 bytes onto stack"),
    0x7C: Opcode("PUSH29", 0x7C, OpcodeType.STACK, 3, "Push 29 bytes onto stack"),
    0x7D: Opcode("PUSH30", 0x7D, OpcodeType.STACK, 3, "Push 30 bytes onto stack"),
    0x7E: Opcode("PUSH31", 0x7E, OpcodeType.STACK, 3, "Push 31 bytes onto stack"),
    0x7F: Opcode("PUSH32", 0x7F, OpcodeType.STACK, 3, "Push 32 bytes onto stack"),
    # Duplication operations
    0x80: Opcode("DUP1", 0x80, OpcodeType.DUP, 3, "Duplicate 1st stack item"),
    0x81: Opcode("DUP2", 0x81, OpcodeType.DUP, 3, "Duplicate 2nd stack item"),
    0x82: Opcode("DUP3", 0x82, OpcodeType.DUP, 3, "Duplicate 3rd stack item"),
    0x83: Opcode("DUP4", 0x83, OpcodeType.DUP, 3, "Duplicate 4th stack item"),
    0x84: Opcode("DUP5", 0x84, OpcodeType.DUP, 3, "Duplicate 5th stack item"),
    0x85: Opcode("DUP6", 0x85, OpcodeType.DUP, 3, "Duplicate 6th stack item"),
    0x86: Opcode("DUP7", 0x86, OpcodeType.DUP, 3, "Duplicate 7th stack item"),
    0x87: Opcode("DUP8", 0x87, OpcodeType.DUP, 3, "Duplicate 8th stack item"),
    0x88: Opcode("DUP9", 0x88, OpcodeType.DUP, 3, "Duplicate 9th stack item"),
    0x89: Opcode("DUP10", 0x89, OpcodeType.DUP, 3, "Duplicate 10th stack item"),
    0x8A: Opcode("DUP11", 0x8A, OpcodeType.DUP, 3, "Duplicate 11th stack item"),
    0x8B: Opcode("DUP12", 0x8B, OpcodeType.DUP, 3, "Duplicate 12th stack item"),
    0x8C: Opcode("DUP13", 0x8C, OpcodeType.DUP, 3, "Duplicate 13th stack item"),
    0x8D: Opcode("DUP14", 0x8D, OpcodeType.DUP, 3, "Duplicate 14th stack item"),
    0x8E: Opcode("DUP15", 0x8E, OpcodeType.DUP, 3, "Duplicate 15th stack item"),
    0x8F: Opcode("DUP16", 0x8F, OpcodeType.DUP, 3, "Duplicate 16th stack item"),
    # Exchange operations
    0x90: Opcode("SWAP1", 0x90, OpcodeType.SWAP, 3, "Swap top with 1st stack item"),
    0x91: Opcode("SWAP2", 0x91, OpcodeType.SWAP, 3, "Swap top with 2nd stack item"),
    0x92: Opcode("SWAP3", 0x92, OpcodeType.SWAP, 3, "Swap top with 3rd stack item"),
    0x93: Opcode("SWAP4", 0x93, OpcodeType.SWAP, 3, "Swap top with 4th stack item"),
    0x94: Opcode("SWAP5", 0x94, OpcodeType.SWAP, 3, "Swap top with 5th stack item"),
    0x95: Opcode("SWAP6", 0x95, OpcodeType.SWAP, 3, "Swap top with 6th stack item"),
    0x96: Opcode("SWAP7", 0x96, OpcodeType.SWAP, 3, "Swap top with 7th stack item"),
    0x97: Opcode("SWAP8", 0x97, OpcodeType.SWAP, 3, "Swap top with 8th stack item"),
    0x98: Opcode("SWAP9", 0x98, OpcodeType.SWAP, 3, "Swap top with 9th stack item"),
    0x99: Opcode("SWAP10", 0x99, OpcodeType.SWAP, 3, "Swap top with 10th stack item"),
    0x9A: Opcode("SWAP11", 0x9A, OpcodeType.SWAP, 3, "Swap top with 11th stack item"),
    0x9B: Opcode("SWAP12", 0x9B, OpcodeType.SWAP, 3, "Swap top with 12th stack item"),
    0x9C: Opcode("SWAP13", 0x9C, OpcodeType.SWAP, 3, "Swap top with 13th stack item"),
    0x9D: Opcode("SWAP14", 0x9D, OpcodeType.SWAP, 3, "Swap top with 14th stack item"),
    0x9E: Opcode("SWAP15", 0x9E, OpcodeType.SWAP, 3, "Swap top with 15th stack item"),
    0x9F: Opcode("SWAP16", 0x9F, OpcodeType.SWAP, 3, "Swap top with 16th stack item"),
}
