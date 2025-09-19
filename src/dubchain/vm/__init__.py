"""
GodChain Virtual Machine Package.

This package provides a sophisticated virtual machine for executing smart contracts
with advanced features including gas metering, opcode execution, and contract management.
"""

from .advanced_execution_engine import (
    AdvancedExecutionEngine,
    ExecutionMetrics,
    ParallelExecutionContext,
)
from .advanced_opcodes import (
    AdvancedOpcodeEnum,
    AdvancedOpcodeInfo,
    advanced_opcode_registry,
)
from .contract import (
    ContractEvent,
    ContractMemory,
    ContractState,
    ContractStorage,
    ContractType,
    SmartContract,
)
from .execution_engine import (
    ExecutionContext,
    ExecutionEngine,
    ExecutionResult,
    ExecutionState,
)
from .gas_meter import GasCost, GasCostConfig, GasMeter
from .opcodes import OPCODES, OpcodeEnum, OpcodeInfo, OpcodeRegistry, OpcodeType

__all__ = [
    # Opcodes
    "OpcodeEnum",
    "OpcodeInfo",
    "OpcodeRegistry",
    "OpcodeType",
    "OPCODES",
    # Advanced Opcodes
    "AdvancedOpcodeEnum",
    "AdvancedOpcodeInfo",
    "advanced_opcode_registry",
    # Gas metering
    "GasMeter",
    "GasCost",
    "GasCostConfig",
    # Contracts
    "SmartContract",
    "ContractEvent",
    "ContractType",
    "ContractState",
    "ContractStorage",
    "ContractMemory",
    # Execution engine
    "ExecutionEngine",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionState",
    # Advanced execution engine
    "AdvancedExecutionEngine",
    "ExecutionMetrics",
    "ParallelExecutionContext",
]
