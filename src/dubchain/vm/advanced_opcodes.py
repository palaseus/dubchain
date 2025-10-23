"""
Advanced opcodes for DubChain Virtual Machine.

This module provides sophisticated opcodes including:
- Advanced cryptographic operations
- Optimized memory and storage operations
- Parallel execution support
- Gas optimization opcodes
- Custom contract operations
"""

import logging

logger = logging.getLogger(__name__)
import hashlib
import struct
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

from .opcodes import OpcodeEnum, OpcodeInfo, OpcodeType


# Extend OpcodeType with additional types for advanced opcodes
class AdvancedOpcodeType(Enum):
    """Extended opcode types for advanced functionality."""

    CONTROL = "control"
    CRYPTO = "crypto"
    PARALLEL = "parallel"
    OPTIMIZATION = "optimization"
    STRING = "string"
    BIGINT = "bigint"
    ARRAY = "array"
    JSON = "json"
    TIME = "time"
    RANDOM = "random"
    DEBUG = "debug"


class AdvancedOpcodeEnum(IntEnum):
    """Advanced opcodes for enhanced VM functionality."""

    # Advanced Cryptographic Operations (0x100-0x1FF)
    KECCAK256 = 0x100
    BLAKE2B = 0x101
    BLAKE2S = 0x102
    SHA3_256 = 0x103
    SHA3_512 = 0x104
    RIPEMD160 = 0x105
    ECDSA_RECOVER = 0x106
    ECDSA_VERIFY = 0x107
    ED25519_VERIFY = 0x108
    BLS_VERIFY = 0x109

    # Advanced Memory Operations (0x200-0x2FF)
    MEMCOPY = 0x200
    MEMCMP = 0x201
    MEMSET = 0x202
    MEMFIND = 0x203
    MEMREVERSE = 0x204
    MEMROTATE = 0x205
    MEMSHIFT = 0x206
    MEMSWAP = 0x207

    # Advanced Storage Operations (0x300-0x3FF)
    SSTORE_BATCH = 0x300
    SLOAD_BATCH = 0x301
    SSTORE_MAP = 0x302
    SLOAD_MAP = 0x303
    SSTORE_ARRAY = 0x304
    SLOAD_ARRAY = 0x305
    SSTORE_STRUCT = 0x306
    SLOAD_STRUCT = 0x307

    # Parallel Execution (0x400-0x4FF)
    PARALLEL_START = 0x400
    PARALLEL_END = 0x401
    PARALLEL_WAIT = 0x402
    PARALLEL_FORK = 0x403
    PARALLEL_JOIN = 0x404
    PARALLEL_SYNC = 0x405

    # Gas Optimization (0x500-0x5FF)
    GAS_SAVE = 0x500
    GAS_REFUND = 0x501
    GAS_TRANSFER = 0x502
    GAS_ESTIMATE = 0x503
    GAS_OPTIMIZE = 0x504

    # Custom Contract Operations (0x600-0x6FF)
    CONTRACT_CALL_OPTIMIZED = 0x600
    CONTRACT_DELEGATE_OPTIMIZED = 0x601
    CONTRACT_STATIC_CALL_OPTIMIZED = 0x602
    CONTRACT_CREATE2_OPTIMIZED = 0x603
    CONTRACT_SELFDESTRUCT_OPTIMIZED = 0x604

    # Advanced Math Operations (0x700-0x7FF)
    BIGINT_ADD = 0x700
    BIGINT_SUB = 0x701
    BIGINT_MUL = 0x702
    BIGINT_DIV = 0x703
    BIGINT_MOD = 0x704
    BIGINT_POW = 0x705
    BIGINT_SQRT = 0x706
    BIGINT_GCD = 0x707

    # String Operations (0x800-0x8FF)
    STRING_CONCAT = 0x800
    STRING_SPLIT = 0x801
    STRING_JOIN = 0x802
    STRING_REPLACE = 0x803
    STRING_FIND = 0x804
    STRING_SUBSTR = 0x805
    STRING_TO_UPPER = 0x806
    STRING_TO_LOWER = 0x807

    # Array Operations (0x900-0x9FF)
    ARRAY_PUSH = 0x900
    ARRAY_POP = 0x901
    ARRAY_INSERT = 0x902
    ARRAY_REMOVE = 0x903
    ARRAY_SORT = 0x904
    ARRAY_REVERSE = 0x905
    ARRAY_SLICE = 0x906
    ARRAY_CONCAT = 0x907

    # JSON Operations (0xA00-0xAFF)
    JSON_PARSE = 0xA00
    JSON_STRINGIFY = 0xA01
    JSON_GET = 0xA02
    JSON_SET = 0xA03
    JSON_DELETE = 0xA04
    JSON_MERGE = 0xA05
    JSON_VALIDATE = 0xA06

    # Time Operations (0xB00-0xBFF)
    TIMESTAMP = 0xB00
    BLOCK_NUMBER = 0xB01
    BLOCK_HASH = 0xB02
    DIFFICULTY = 0xB03
    GAS_LIMIT = 0xB04
    COINBASE = 0xB05

    # Random Operations (0xC00-0xCFF)
    RANDOM = 0xC00
    RANDOM_SEED = 0xC01
    RANDOM_RANGE = 0xC02
    RANDOM_BYTES = 0xC03

    # Debug Operations (0xD00-0xDFF)
    DEBUG_PRINT = 0xD00
    DEBUG_LOG = 0xD01
    DEBUG_TRACE = 0xD02
    DEBUG_BREAKPOINT = 0xD03


@dataclass
class AdvancedOpcodeInfo:
    """Information about advanced opcodes."""

    opcode: AdvancedOpcodeEnum
    name: str
    opcode_type: OpcodeType
    gas_cost: int
    stack_inputs: int
    stack_outputs: int
    description: str
    implementation: Optional[callable] = None


class AdvancedOpcodeRegistry:
    """Registry for advanced opcodes."""

    def __init__(self):
        """Initialize advanced opcode registry."""
        self.opcodes: Dict[AdvancedOpcodeEnum, AdvancedOpcodeInfo] = {}
        self._register_opcodes()

    def _register_opcodes(self) -> None:
        """Register all advanced opcodes."""
        # Advanced Cryptographic Operations
        self._register_opcode(
            AdvancedOpcodeEnum.KECCAK256,
            "KECCAK256",
            AdvancedOpcodeType.CRYPTO,
            30,
            1,
            1,
            "Keccak-256 hash",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BLAKE2B,
            "BLAKE2B",
            AdvancedOpcodeType.CRYPTO,
            25,
            1,
            1,
            "BLAKE2b hash",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BLAKE2S,
            "BLAKE2S",
            AdvancedOpcodeType.CRYPTO,
            20,
            1,
            1,
            "BLAKE2s hash",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SHA3_256,
            "SHA3_256",
            AdvancedOpcodeType.CRYPTO,
            30,
            1,
            1,
            "SHA3-256 hash",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SHA3_512,
            "SHA3_512",
            AdvancedOpcodeType.CRYPTO,
            50,
            1,
            1,
            "SHA3-512 hash",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.RIPEMD160,
            "RIPEMD160",
            AdvancedOpcodeType.CRYPTO,
            20,
            1,
            1,
            "RIPEMD-160 hash",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ECDSA_RECOVER,
            "ECDSA_RECOVER",
            AdvancedOpcodeType.CRYPTO,
            100,
            3,
            1,
            "ECDSA signature recovery",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ECDSA_VERIFY,
            "ECDSA_VERIFY",
            AdvancedOpcodeType.CRYPTO,
            100,
            4,
            1,
            "ECDSA signature verification",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ED25519_VERIFY,
            "ED25519_VERIFY",
            AdvancedOpcodeType.CRYPTO,
            60,
            3,
            1,
            "Ed25519 signature verification",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BLS_VERIFY,
            "BLS_VERIFY",
            AdvancedOpcodeType.CRYPTO,
            200,
            4,
            1,
            "BLS signature verification",
        )

        # Advanced Memory Operations
        self._register_opcode(
            AdvancedOpcodeEnum.MEMCOPY,
            "MEMCOPY",
            OpcodeType.MEMORY,
            10,
            3,
            0,
            "Copy memory region",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.MEMCMP,
            "MEMCMP",
            OpcodeType.MEMORY,
            8,
            3,
            1,
            "Compare memory regions",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.MEMSET,
            "MEMSET",
            OpcodeType.MEMORY,
            5,
            3,
            0,
            "Set memory region to value",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.MEMFIND,
            "MEMFIND",
            OpcodeType.MEMORY,
            12,
            3,
            1,
            "Find pattern in memory",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.MEMREVERSE,
            "MEMREVERSE",
            OpcodeType.MEMORY,
            8,
            2,
            0,
            "Reverse memory region",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.MEMROTATE,
            "MEMROTATE",
            OpcodeType.MEMORY,
            10,
            3,
            0,
            "Rotate memory region",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.MEMSHIFT,
            "MEMSHIFT",
            OpcodeType.MEMORY,
            8,
            3,
            0,
            "Shift memory region",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.MEMSWAP,
            "MEMSWAP",
            OpcodeType.MEMORY,
            6,
            2,
            0,
            "Swap memory regions",
        )

        # Advanced Storage Operations
        self._register_opcode(
            AdvancedOpcodeEnum.SSTORE_BATCH,
            "SSTORE_BATCH",
            OpcodeType.STORAGE,
            50,
            2,
            0,
            "Batch storage write",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SLOAD_BATCH,
            "SLOAD_BATCH",
            OpcodeType.STORAGE,
            30,
            1,
            1,
            "Batch storage read",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SSTORE_MAP,
            "SSTORE_MAP",
            OpcodeType.STORAGE,
            40,
            3,
            0,
            "Map storage write",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SLOAD_MAP,
            "SLOAD_MAP",
            OpcodeType.STORAGE,
            25,
            2,
            1,
            "Map storage read",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SSTORE_ARRAY,
            "SSTORE_ARRAY",
            OpcodeType.STORAGE,
            35,
            3,
            0,
            "Array storage write",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SLOAD_ARRAY,
            "SLOAD_ARRAY",
            OpcodeType.STORAGE,
            20,
            2,
            1,
            "Array storage read",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SSTORE_STRUCT,
            "SSTORE_STRUCT",
            OpcodeType.STORAGE,
            45,
            3,
            0,
            "Struct storage write",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.SLOAD_STRUCT,
            "SLOAD_STRUCT",
            OpcodeType.STORAGE,
            30,
            2,
            1,
            "Struct storage read",
        )

        # Parallel Execution
        self._register_opcode(
            AdvancedOpcodeEnum.PARALLEL_START,
            "PARALLEL_START",
            AdvancedOpcodeType.PARALLEL,
            20,
            0,
            0,
            "Start parallel execution",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.PARALLEL_END,
            "PARALLEL_END",
            AdvancedOpcodeType.PARALLEL,
            15,
            0,
            0,
            "End parallel execution",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.PARALLEL_WAIT,
            "PARALLEL_WAIT",
            AdvancedOpcodeType.PARALLEL,
            10,
            1,
            0,
            "Wait for parallel execution",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.PARALLEL_FORK,
            "PARALLEL_FORK",
            AdvancedOpcodeType.PARALLEL,
            25,
            1,
            1,
            "Fork parallel execution",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.PARALLEL_JOIN,
            "PARALLEL_JOIN",
            AdvancedOpcodeType.PARALLEL,
            20,
            1,
            0,
            "Join parallel execution",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.PARALLEL_SYNC,
            "PARALLEL_SYNC",
            AdvancedOpcodeType.PARALLEL,
            15,
            0,
            0,
            "Synchronize parallel execution",
        )

        # Gas Optimization
        self._register_opcode(
            AdvancedOpcodeEnum.GAS_SAVE,
            "GAS_SAVE",
            AdvancedOpcodeType.OPTIMIZATION,
            5,
            1,
            1,
            "Save gas for later use",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.GAS_REFUND,
            "GAS_REFUND",
            AdvancedOpcodeType.OPTIMIZATION,
            5,
            1,
            0,
            "Refund gas to caller",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.GAS_TRANSFER,
            "GAS_TRANSFER",
            AdvancedOpcodeType.OPTIMIZATION,
            10,
            2,
            0,
            "Transfer gas between accounts",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.GAS_ESTIMATE,
            "GAS_ESTIMATE",
            AdvancedOpcodeType.OPTIMIZATION,
            15,
            1,
            1,
            "Estimate gas for operation",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.GAS_OPTIMIZE,
            "GAS_OPTIMIZE",
            AdvancedOpcodeType.OPTIMIZATION,
            20,
            0,
            1,
            "Optimize gas usage",
        )

        # Custom Contract Operations
        self._register_opcode(
            AdvancedOpcodeEnum.CONTRACT_CALL_OPTIMIZED,
            "CONTRACT_CALL_OPTIMIZED",
            OpcodeType.SYSTEM,
            100,
            7,
            1,
            "Optimized contract call",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.CONTRACT_DELEGATE_OPTIMIZED,
            "CONTRACT_DELEGATE_OPTIMIZED",
            OpcodeType.SYSTEM,
            100,
            6,
            1,
            "Optimized delegate call",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.CONTRACT_STATIC_CALL_OPTIMIZED,
            "CONTRACT_STATIC_CALL_OPTIMIZED",
            OpcodeType.SYSTEM,
            100,
            6,
            1,
            "Optimized static call",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.CONTRACT_CREATE2_OPTIMIZED,
            "CONTRACT_CREATE2_OPTIMIZED",
            OpcodeType.SYSTEM,
            200,
            4,
            1,
            "Optimized CREATE2",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.CONTRACT_SELFDESTRUCT_OPTIMIZED,
            "CONTRACT_SELFDESTRUCT_OPTIMIZED",
            OpcodeType.SYSTEM,
            50,
            1,
            0,
            "Optimized selfdestruct",
        )

        # Advanced Math Operations
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_ADD,
            "BIGINT_ADD",
            OpcodeType.ARITHMETIC,
            20,
            2,
            1,
            "Big integer addition",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_SUB,
            "BIGINT_SUB",
            OpcodeType.ARITHMETIC,
            20,
            2,
            1,
            "Big integer subtraction",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_MUL,
            "BIGINT_MUL",
            OpcodeType.ARITHMETIC,
            30,
            2,
            1,
            "Big integer multiplication",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_DIV,
            "BIGINT_DIV",
            OpcodeType.ARITHMETIC,
            40,
            2,
            1,
            "Big integer division",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_MOD,
            "BIGINT_MOD",
            OpcodeType.ARITHMETIC,
            40,
            2,
            1,
            "Big integer modulo",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_POW,
            "BIGINT_POW",
            OpcodeType.ARITHMETIC,
            50,
            2,
            1,
            "Big integer power",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_SQRT,
            "BIGINT_SQRT",
            OpcodeType.ARITHMETIC,
            60,
            1,
            1,
            "Big integer square root",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BIGINT_GCD,
            "BIGINT_GCD",
            OpcodeType.ARITHMETIC,
            70,
            2,
            1,
            "Big integer GCD",
        )

        # String Operations
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_CONCAT,
            "STRING_CONCAT",
            AdvancedOpcodeType.STRING,
            15,
            2,
            1,
            "String concatenation",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_SPLIT,
            "STRING_SPLIT",
            AdvancedOpcodeType.STRING,
            20,
            2,
            1,
            "String split",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_JOIN,
            "STRING_JOIN",
            AdvancedOpcodeType.STRING,
            18,
            2,
            1,
            "String join",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_REPLACE,
            "STRING_REPLACE",
            AdvancedOpcodeType.STRING,
            25,
            3,
            1,
            "String replace",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_FIND,
            "STRING_FIND",
            AdvancedOpcodeType.STRING,
            12,
            2,
            1,
            "String find",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_SUBSTR,
            "STRING_SUBSTR",
            AdvancedOpcodeType.STRING,
            10,
            3,
            1,
            "String substring",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_TO_UPPER,
            "STRING_TO_UPPER",
            AdvancedOpcodeType.STRING,
            8,
            1,
            1,
            "String to uppercase",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.STRING_TO_LOWER,
            "STRING_TO_LOWER",
            AdvancedOpcodeType.STRING,
            8,
            1,
            1,
            "String to lowercase",
        )

        # Array Operations
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_PUSH,
            "ARRAY_PUSH",
            AdvancedOpcodeType.ARRAY,
            15,
            2,
            1,
            "Array push",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_POP,
            "ARRAY_POP",
            AdvancedOpcodeType.ARRAY,
            10,
            1,
            1,
            "Array pop",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_INSERT,
            "ARRAY_INSERT",
            AdvancedOpcodeType.ARRAY,
            20,
            3,
            1,
            "Array insert",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_REMOVE,
            "ARRAY_REMOVE",
            AdvancedOpcodeType.ARRAY,
            18,
            2,
            1,
            "Array remove",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_SORT,
            "ARRAY_SORT",
            AdvancedOpcodeType.ARRAY,
            30,
            1,
            1,
            "Array sort",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_REVERSE,
            "ARRAY_REVERSE",
            AdvancedOpcodeType.ARRAY,
            15,
            1,
            1,
            "Array reverse",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_SLICE,
            "ARRAY_SLICE",
            AdvancedOpcodeType.ARRAY,
            12,
            3,
            1,
            "Array slice",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.ARRAY_CONCAT,
            "ARRAY_CONCAT",
            AdvancedOpcodeType.ARRAY,
            18,
            2,
            1,
            "Array concatenation",
        )

        # JSON Operations
        self._register_opcode(
            AdvancedOpcodeEnum.JSON_PARSE,
            "JSON_PARSE",
            AdvancedOpcodeType.JSON,
            25,
            1,
            1,
            "JSON parse",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.JSON_STRINGIFY,
            "JSON_STRINGIFY",
            AdvancedOpcodeType.JSON,
            20,
            1,
            1,
            "JSON stringify",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.JSON_GET,
            "JSON_GET",
            AdvancedOpcodeType.JSON,
            15,
            2,
            1,
            "JSON get value",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.JSON_SET,
            "JSON_SET",
            AdvancedOpcodeType.JSON,
            20,
            3,
            1,
            "JSON set value",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.JSON_DELETE,
            "JSON_DELETE",
            AdvancedOpcodeType.JSON,
            15,
            2,
            1,
            "JSON delete value",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.JSON_MERGE,
            "JSON_MERGE",
            AdvancedOpcodeType.JSON,
            25,
            2,
            1,
            "JSON merge",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.JSON_VALIDATE,
            "JSON_VALIDATE",
            AdvancedOpcodeType.JSON,
            10,
            1,
            1,
            "JSON validate",
        )

        # Time Operations
        self._register_opcode(
            AdvancedOpcodeEnum.TIMESTAMP,
            "TIMESTAMP",
            AdvancedOpcodeType.TIME,
            2,
            0,
            1,
            "Get block timestamp",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BLOCK_NUMBER,
            "BLOCK_NUMBER",
            OpcodeType.BLOCK,
            2,
            0,
            1,
            "Get block number",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.BLOCK_HASH,
            "BLOCK_HASH",
            OpcodeType.BLOCK,
            20,
            1,
            1,
            "Get block hash",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.DIFFICULTY,
            "DIFFICULTY",
            OpcodeType.BLOCK,
            2,
            0,
            1,
            "Get block difficulty",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.GAS_LIMIT,
            "GAS_LIMIT",
            OpcodeType.BLOCK,
            2,
            0,
            1,
            "Get block gas limit",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.COINBASE,
            "COINBASE",
            OpcodeType.BLOCK,
            2,
            0,
            1,
            "Get block coinbase",
        )

        # Random Operations
        self._register_opcode(
            AdvancedOpcodeEnum.RANDOM,
            "RANDOM",
            AdvancedOpcodeType.RANDOM,
            10,
            0,
            1,
            "Generate random number",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.RANDOM_SEED,
            "RANDOM_SEED",
            AdvancedOpcodeType.RANDOM,
            5,
            1,
            0,
            "Set random seed",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.RANDOM_RANGE,
            "RANDOM_RANGE",
            AdvancedOpcodeType.RANDOM,
            15,
            2,
            1,
            "Random number in range",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.RANDOM_BYTES,
            "RANDOM_BYTES",
            AdvancedOpcodeType.RANDOM,
            20,
            1,
            1,
            "Generate random bytes",
        )

        # Debug Operations
        self._register_opcode(
            AdvancedOpcodeEnum.DEBUG_PRINT,
            "DEBUG_PRINT",
            AdvancedOpcodeType.DEBUG,
            5,
            1,
            0,
            "Debug print",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.DEBUG_LOG,
            "DEBUG_LOG",
            AdvancedOpcodeType.DEBUG,
            5,
            1,
            0,
            "Debug log",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.DEBUG_TRACE,
            "DEBUG_TRACE",
            AdvancedOpcodeType.DEBUG,
            10,
            0,
            0,
            "Debug trace",
        )
        self._register_opcode(
            AdvancedOpcodeEnum.DEBUG_BREAKPOINT,
            "DEBUG_BREAKPOINT",
            AdvancedOpcodeType.DEBUG,
            5,
            0,
            0,
            "Debug breakpoint",
        )

    def _register_opcode(
        self,
        opcode: AdvancedOpcodeEnum,
        name: str,
        opcode_type: OpcodeType,
        gas_cost: int,
        stack_inputs: int,
        stack_outputs: int,
        description: str,
    ) -> None:
        """Register an advanced opcode."""
        info = AdvancedOpcodeInfo(
            opcode=opcode,
            name=name,
            opcode_type=opcode_type,
            gas_cost=gas_cost,
            stack_inputs=stack_inputs,
            stack_outputs=stack_outputs,
            description=description,
        )
        self.opcodes[opcode] = info

    def get_opcode_info(
        self, opcode: AdvancedOpcodeEnum
    ) -> Optional[AdvancedOpcodeInfo]:
        """Get opcode information."""
        return self.opcodes.get(opcode)

    def get_all_opcodes(self) -> Dict[AdvancedOpcodeEnum, AdvancedOpcodeInfo]:
        """Get all registered opcodes."""
        return self.opcodes.copy()

    def get_opcodes_by_type(
        self, opcode_type: OpcodeType
    ) -> Dict[AdvancedOpcodeEnum, AdvancedOpcodeInfo]:
        """Get opcodes by type."""
        return {
            opcode: info
            for opcode, info in self.opcodes.items()
            if info.opcode_type == opcode_type
        }

    def is_valid_opcode(self, opcode: int) -> bool:
        """Check if opcode is valid."""
        return AdvancedOpcodeEnum(opcode) in self.opcodes


# Global registry instance
advanced_opcode_registry = AdvancedOpcodeRegistry()
