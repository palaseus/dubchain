"""
Execution engine for the GodChain virtual machine.

This module provides the core execution engine that runs smart contracts
with advanced features like gas metering, call depth limits, and state management.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PublicKey
from .contract import ContractEvent, ContractState, ContractType, SmartContract
from .gas_meter import GasCost, GasMeter
from .opcodes import OPCODES, Opcode, OpcodeType


class ExecutionState(Enum):
    """Execution states."""

    RUNNING = "running"
    STOPPED = "stopped"
    REVERTED = "reverted"
    OUT_OF_GAS = "out_of_gas"
    INVALID_OPCODE = "invalid_opcode"
    STACK_UNDERFLOW = "stack_underflow"
    STACK_OVERFLOW = "stack_overflow"
    INVALID_JUMP = "invalid_jump"
    CALL_DEPTH_EXCEEDED = "call_depth_exceeded"
    INVALID_MEMORY_ACCESS = "invalid_memory_access"


@dataclass
class ExecutionContext:
    """Execution context for a contract call."""

    contract: SmartContract
    caller: str
    value: int
    data: bytes
    gas_limit: int
    gas_meter: GasMeter
    block_context: Dict[str, Any]

    # Execution state
    pc: int = 0  # Program counter
    stack: List[int] = field(default_factory=list)
    memory: bytearray = field(default_factory=bytearray)
    return_data: bytes = b""

    # Call context
    call_depth: int = 0
    max_call_depth: int = 1024

    # State tracking
    state: ExecutionState = ExecutionState.RUNNING
    error_message: str = ""

    def __post_init__(self):
        if self.gas_limit <= 0:
            raise ValueError("Gas limit must be positive")

        if self.call_depth < 0:
            raise ValueError("Call depth must be non-negative")

        if self.max_call_depth <= 0:
            raise ValueError("Max call depth must be positive")

    def push_stack(self, value: int) -> None:
        """Push value onto stack."""
        if len(self.stack) >= 1024:  # EVM stack limit
            self.state = ExecutionState.STACK_OVERFLOW
            self.error_message = "Stack overflow"
            return

        self.stack.append(value)

    def pop_stack(self) -> int:
        """Pop value from stack."""
        if not self.stack:
            self.state = ExecutionState.STACK_UNDERFLOW
            self.error_message = "Stack underflow"
            return 0

        return self.stack.pop()

    def peek_stack(self, index: int = 0) -> int:
        """Peek at stack value without popping."""
        if index >= len(self.stack):
            self.state = ExecutionState.STACK_UNDERFLOW
            self.error_message = "Stack underflow"
            return 0

        return self.stack[-(index + 1)]

    def get_memory(self, offset: int, size: int) -> bytes:
        """Get data from memory."""
        if offset < 0 or size < 0:
            self.state = ExecutionState.INVALID_MEMORY_ACCESS
            self.error_message = "Invalid memory access"
            return b""

        if offset + size > len(self.memory):
            # Extend memory with zeros
            self.memory.extend(b"\x00" * (offset + size - len(self.memory)))

        return bytes(self.memory[offset : offset + size])

    def set_memory(self, offset: int, data: bytes) -> None:
        """Set data in memory."""
        if offset < 0:
            self.state = ExecutionState.INVALID_MEMORY_ACCESS
            self.error_message = "Invalid memory access"
            return

        # Extend memory if needed to cover the full range
        required_length = offset + len(data)
        if required_length > len(self.memory):
            self.memory.extend(b"\x00" * (required_length - len(self.memory)))

        # Set the data
        self.memory[offset : offset + len(data)] = data

    def is_valid_jump_destination(self, destination: int) -> bool:
        """Check if jump destination is valid."""
        if destination < 0 or destination >= len(self.contract.bytecode):
            return False

        # Check if destination is at the beginning of an opcode
        # This is a simplified check - in reality, we'd need to parse the bytecode
        return True

    def consume_gas(self, amount: int) -> bool:
        """Consume gas and check if execution should continue."""
        if not self.gas_meter.consume_gas(amount):
            self.state = ExecutionState.OUT_OF_GAS
            self.error_message = "Out of gas"
            return False

        return True

    def revert(self, message: str = "") -> None:
        """Revert execution."""
        self.state = ExecutionState.REVERTED
        self.error_message = message

    def stop(self) -> None:
        """Stop execution."""
        self.state = ExecutionState.STOPPED


@dataclass
class ExecutionResult:
    """Result of contract execution."""

    success: bool
    gas_used: int
    return_data: bytes
    events: List[ContractEvent]
    storage_changes: Dict[Hash, bytes]
    error_message: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "gas_used": self.gas_used,
            "return_data": self.return_data.hex(),
            "events": [event.to_dict() for event in self.events],
            "storage_changes": {
                key.to_hex(): value.hex() for key, value in self.storage_changes.items()
            },
            "error_message": self.error_message,
            "execution_time": self.execution_time,
        }


class ExecutionEngine:
    """Advanced execution engine for smart contracts."""

    def __init__(self, max_call_depth: int = 1024, max_memory_size: int = 2**32):
        self.max_call_depth = max_call_depth
        self.max_memory_size = max_memory_size
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}

        # CUDA acceleration (lazy loading to avoid circular imports)
        self._cuda_accelerator = None

        # Initialize performance tracking
        self.performance_metrics = {
            "total_executions": 0,
            "total_gas_used": 0,
            "total_execution_time": 0.0,
            "average_gas_per_execution": 0.0,
            "average_execution_time": 0.0,
            "success_rate": 0.0,
            "opcode_usage": {},
            "error_counts": {},
        }

    @property
    def cuda_accelerator(self):
        """Get CUDA accelerator with lazy loading."""
        if self._cuda_accelerator is None:
            from .cuda_vm import get_global_cuda_vm_accelerator

            self._cuda_accelerator = get_global_cuda_vm_accelerator()
        return self._cuda_accelerator

    def execute_contract(
        self,
        contract: SmartContract,
        caller: str,
        value: int,
        data: bytes,
        gas_limit: int,
        block_context: Dict[str, Any],
    ) -> ExecutionResult:
        """Execute a smart contract."""
        start_time = time.time()

        # Create execution context
        gas_meter = GasMeter(gas_limit, gas_limit)
        context = ExecutionContext(
            contract=contract,
            caller=caller,
            value=value,
            data=data,
            gas_limit=gas_limit,
            gas_meter=gas_meter,
            block_context=block_context,
            call_depth=0,
            max_call_depth=self.max_call_depth,
        )

        # Execute the contract
        try:
            result = self._execute_bytecode(context)
        except Exception as e:
            result = ExecutionResult(
                success=False,
                gas_used=gas_meter.get_gas_used(),
                return_data=b"",
                events=[],
                storage_changes={},
                error_message=str(e),
            )

        # Calculate execution time
        execution_time = time.time() - start_time
        result.execution_time = execution_time

        # Update performance metrics
        self._update_performance_metrics(result, execution_time)

        # Record execution history
        self._record_execution(contract, caller, result, execution_time)

        return result

    def _execute_bytecode(self, context: ExecutionContext) -> ExecutionResult:
        """Execute contract bytecode."""
        bytecode = context.contract.bytecode
        events = []
        storage_changes = {}

        while context.state == ExecutionState.RUNNING and context.pc < len(bytecode):
            # Get current opcode
            if context.pc >= len(bytecode):
                break

            opcode_byte = bytecode[context.pc]

            # Handle PUSH opcodes (1-32 bytes of data)
            if 0x60 <= opcode_byte <= 0x7F:
                push_size = opcode_byte - 0x5F
                if context.pc + push_size >= len(bytecode):
                    context.state = ExecutionState.INVALID_OPCODE
                    context.error_message = "Invalid PUSH opcode"
                    break

                # Extract push data
                push_data = bytecode[context.pc + 1 : context.pc + 1 + push_size]
                value = int.from_bytes(push_data, "big")
                context.push_stack(value)

                # Consume gas for PUSH
                if not context.consume_gas(GasCost.BASE.value):
                    break

                context.pc += 1 + push_size
                continue

            # Get opcode
            opcode = OPCODES.get(opcode_byte)
            if not opcode:
                context.state = ExecutionState.INVALID_OPCODE
                context.error_message = f"Invalid opcode: 0x{opcode_byte:02x}"
                break

            # Execute opcode
            try:
                self._execute_opcode(context, opcode)
            except Exception as e:
                context.state = ExecutionState.REVERTED
                context.error_message = str(e)
                break

            # Update performance metrics
            self.performance_metrics["opcode_usage"][opcode.name] = (
                self.performance_metrics["opcode_usage"].get(opcode.name, 0) + 1
            )

            # Move to next instruction
            context.pc += 1

        # Create result
        result = ExecutionResult(
            success=context.state == ExecutionState.STOPPED,
            gas_used=context.gas_meter.get_gas_used(),
            return_data=context.return_data,
            events=events,
            storage_changes=storage_changes,
            error_message=context.error_message,
        )

        return result

    def _execute_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute a single opcode."""
        # Consume gas for opcode
        gas_cost = self._get_opcode_gas_cost(opcode, context)
        if not context.consume_gas(gas_cost):
            return

        # Execute opcode based on type
        if opcode.type == OpcodeType.STOP:
            context.stop()

        elif opcode.type == OpcodeType.ARITHMETIC:
            self._execute_arithmetic_opcode(context, opcode)

        elif opcode.type == OpcodeType.COMPARISON:
            self._execute_comparison_opcode(context, opcode)

        elif opcode.type == OpcodeType.BITWISE:
            self._execute_bitwise_opcode(context, opcode)

        elif opcode.type == OpcodeType.SHA3:
            self._execute_sha3_opcode(context)

        elif opcode.type == OpcodeType.ENVIRONMENT:
            self._execute_environment_opcode(context, opcode)

        elif opcode.type == OpcodeType.BLOCK:
            self._execute_block_opcode(context, opcode)

        elif opcode.type == OpcodeType.STORAGE:
            self._execute_storage_opcode(context, opcode)

        elif opcode.type == OpcodeType.MEMORY:
            self._execute_memory_opcode(context, opcode)

        elif opcode.type == OpcodeType.STACK:
            self._execute_stack_opcode(context, opcode)

        elif opcode.type == OpcodeType.DUP:
            self._execute_dup_opcode(context, opcode)

        elif opcode.type == OpcodeType.SWAP:
            self._execute_swap_opcode(context, opcode)

        elif opcode.type == OpcodeType.LOG:
            self._execute_log_opcode(context, opcode)

        elif opcode.type == OpcodeType.SYSTEM:
            self._execute_system_opcode(context, opcode)

        else:
            context.state = ExecutionState.INVALID_OPCODE
            context.error_message = f"Unsupported opcode type: {opcode.type}"

    def _get_opcode_gas_cost(self, opcode: Opcode, context: ExecutionContext) -> int:
        """Get gas cost for opcode execution."""
        base_cost = opcode.gas_cost

        # Add dynamic gas costs
        if opcode.name == "SSTORE":
            # Storage operations have dynamic costs
            return base_cost + 20000  # Simplified

        elif opcode.name == "SLOAD":
            return base_cost + 200  # Simplified

        elif opcode.name == "SHA3":
            # SHA3 cost depends on data size
            if len(context.stack) >= 2:
                offset = context.peek_stack(1)
                size = context.peek_stack(0)
                return base_cost + (size // 32) * 6

        elif opcode.name in ["CALL", "DELEGATECALL", "STATICCALL"]:
            # Call operations have additional costs
            return base_cost + 700  # Simplified

        return base_cost

    def _execute_arithmetic_opcode(
        self, context: ExecutionContext, opcode: Opcode
    ) -> None:
        """Execute arithmetic opcodes."""
        if opcode.name == "ADD":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = (a + b) % (2**256)
                context.push_stack(result)

        elif opcode.name == "SUB":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = (a - b) % (2**256)
                context.push_stack(result)

        elif opcode.name == "MUL":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = (a * b) % (2**256)
                context.push_stack(result)

        elif opcode.name == "DIV":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                if b == 0:
                    result = 0
                else:
                    result = a // b
                context.push_stack(result)

        elif opcode.name == "MOD":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                if b == 0:
                    result = 0
                else:
                    result = a % b
                context.push_stack(result)

    def _execute_comparison_opcode(
        self, context: ExecutionContext, opcode: Opcode
    ) -> None:
        """Execute comparison opcodes."""
        if opcode.name == "LT":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = 1 if a < b else 0
                context.push_stack(result)

        elif opcode.name == "GT":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = 1 if a > b else 0
                context.push_stack(result)

        elif opcode.name == "EQ":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = 1 if a == b else 0
                context.push_stack(result)

    def _execute_bitwise_opcode(
        self, context: ExecutionContext, opcode: Opcode
    ) -> None:
        """Execute bitwise opcodes."""
        if opcode.name == "AND":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = a & b
                context.push_stack(result)

        elif opcode.name == "OR":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = a | b
                context.push_stack(result)

        elif opcode.name == "XOR":
            if len(context.stack) >= 2:
                a = context.pop_stack()
                b = context.pop_stack()
                result = a ^ b
                context.push_stack(result)

    def _execute_sha3_opcode(self, context: ExecutionContext) -> None:
        """Execute SHA3 opcode."""
        if len(context.stack) >= 2:
            offset = context.pop_stack()
            size = context.pop_stack()

            data = context.get_memory(offset, size)
            hash_result = SHA256Hasher.hash(data)
            result = hash_result.to_int()
            context.push_stack(result)

    def _execute_environment_opcode(
        self, context: ExecutionContext, opcode: Opcode
    ) -> None:
        """Execute environment opcodes."""
        if opcode.name == "ADDRESS":
            # Push contract address (simplified)
            address_hash = SHA256Hasher.hash(context.contract.address.encode())
            address_int = int.from_bytes(address_hash.value[:4], "big")
            context.push_stack(address_int)

        elif opcode.name == "CALLER":
            # Push caller address (simplified)
            caller_hash = SHA256Hasher.hash(context.caller.encode())
            caller_int = int.from_bytes(caller_hash.value[:4], "big")
            context.push_stack(caller_int)

        elif opcode.name == "CALLVALUE":
            context.push_stack(context.value)

    def _execute_block_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute block opcodes."""
        if opcode.name == "BLOCKHASH":
            if len(context.stack) >= 1:
                block_number = context.pop_stack()
                # Simplified block hash retrieval
                block_hash = SHA256Hasher.hash(f"block_{block_number}".encode())
                hash_int = int.from_bytes(block_hash.value[:4], "big")
                context.push_stack(hash_int)

        elif opcode.name == "TIMESTAMP":
            timestamp = context.block_context.get("timestamp", int(time.time()))
            context.push_stack(timestamp)

    def _execute_storage_opcode(
        self, context: ExecutionContext, opcode: Opcode
    ) -> None:
        """Execute storage opcodes."""
        if opcode.name == "SLOAD":
            if len(context.stack) >= 1:
                key_int = context.pop_stack()
                key = Hash.from_int(key_int)
                value = context.contract.get_storage_value(key)
                value_int = int.from_bytes(value, "big")
                context.push_stack(value_int)

        elif opcode.name == "SSTORE":
            if len(context.stack) >= 2:
                key_int = context.pop_stack()
                value_int = context.pop_stack()
                key = Hash.from_int(key_int)
                value = value_int.to_bytes(32, "big")
                context.contract.set_storage_value(key, value)

    def _execute_memory_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute memory opcodes."""
        if opcode.name == "MSTORE":
            if len(context.stack) >= 2:
                offset = context.pop_stack()
                value = context.pop_stack()
                data = value.to_bytes(32, "big")
                context.set_memory(offset, data)

        elif opcode.name == "MLOAD":
            if len(context.stack) >= 1:
                offset = context.pop_stack()
                data = context.get_memory(offset, 32)
                value = int.from_bytes(data, "big")
                context.push_stack(value)

    def _execute_stack_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute stack opcodes."""
        if opcode.name == "POP":
            if context.stack:
                context.pop_stack()

    def _execute_dup_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute DUP opcodes."""
        if opcode.name.startswith("DUP"):
            dup_num = int(opcode.name[3:])
            if len(context.stack) >= dup_num:
                value = context.stack[-(dup_num)]
                context.push_stack(value)

    def _execute_swap_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute SWAP opcodes."""
        if opcode.name.startswith("SWAP"):
            swap_num = int(opcode.name[4:])
            if len(context.stack) >= swap_num + 1:
                # Swap top element with element at position swap_num
                top = context.stack[-1]
                context.stack[-1] = context.stack[-(swap_num + 1)]
                context.stack[-(swap_num + 1)] = top

    def _execute_log_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute LOG opcodes."""
        if opcode.name.startswith("LOG"):
            log_num = int(opcode.name[3:])
            if len(context.stack) >= log_num + 2:
                # Pop topics
                topics = []
                for _ in range(log_num):
                    topic_int = context.pop_stack()
                    topic = Hash.from_int(topic_int)
                    topics.append(topic)

                # Pop offset and size
                size = context.pop_stack()
                offset = context.pop_stack()

                # Get data
                data = context.get_memory(offset, size)

                # Create event
                event = ContractEvent(
                    address=context.contract.address,
                    topics=topics,
                    data=data,
                    block_number=context.block_context.get("block_number", 0),
                    transaction_hash=context.block_context.get(
                        "transaction_hash", Hash.zero()
                    ),
                    log_index=len(context.contract.events),
                )

                context.contract.events.append(event)

    def _execute_system_opcode(self, context: ExecutionContext, opcode: Opcode) -> None:
        """Execute system opcodes."""
        if opcode.name == "RETURN":
            if len(context.stack) >= 2:
                offset = context.pop_stack()
                size = context.pop_stack()
                context.return_data = context.get_memory(offset, size)
                context.stop()

        elif opcode.name == "REVERT":
            if len(context.stack) >= 2:
                offset = context.pop_stack()
                size = context.pop_stack()
                context.return_data = context.get_memory(offset, size)
                context.revert("Contract reverted")

    def _update_performance_metrics(
        self, result: ExecutionResult, execution_time: float
    ) -> None:
        """Update performance metrics."""
        self.performance_metrics["total_executions"] += 1
        self.performance_metrics["total_gas_used"] += result.gas_used
        self.performance_metrics["total_execution_time"] += execution_time

        # Calculate averages
        total_executions = self.performance_metrics["total_executions"]
        self.performance_metrics["average_gas_per_execution"] = (
            self.performance_metrics["total_gas_used"] / total_executions
        )
        self.performance_metrics["average_execution_time"] = (
            self.performance_metrics["total_execution_time"] / total_executions
        )

        # Update success rate
        if result.success:
            successful_executions = (
                self.performance_metrics.get("successful_executions", 0) + 1
            )
            self.performance_metrics["successful_executions"] = successful_executions
            self.performance_metrics["success_rate"] = (
                successful_executions / total_executions
            )

        # Update error counts
        if not result.success and result.error_message:
            error_type = (
                result.error_message.split(":")[0]
                if ":" in result.error_message
                else result.error_message
            )
            self.performance_metrics["error_counts"][error_type] = (
                self.performance_metrics["error_counts"].get(error_type, 0) + 1
            )

    def _record_execution(
        self,
        contract: SmartContract,
        caller: str,
        result: ExecutionResult,
        execution_time: float,
    ) -> None:
        """Record execution in history."""
        execution_record = {
            "timestamp": time.time(),
            "contract_address": contract.address,
            "caller": caller,
            "success": result.success,
            "gas_used": result.gas_used,
            "execution_time": execution_time,
            "error_message": result.error_message,
            "event_count": len(result.events),
            "storage_changes": len(result.storage_changes),
        }

        self.execution_history.append(execution_record)

        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()

    def get_execution_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get execution history."""
        history = self.execution_history
        if limit is not None:
            history = history[-limit:]

        return history

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_executions": 0,
            "total_gas_used": 0,
            "total_execution_time": 0.0,
            "average_gas_per_execution": 0.0,
            "average_execution_time": 0.0,
            "success_rate": 0.0,
            "opcode_usage": {},
            "error_counts": {},
        }
        self.execution_history.clear()

    def get_opcode_usage_stats(self) -> Dict[str, Any]:
        """Get opcode usage statistics."""
        total_opcodes = sum(self.performance_metrics["opcode_usage"].values())
        if total_opcodes == 0:
            return {}

        usage_stats = {}
        for opcode, count in self.performance_metrics["opcode_usage"].items():
            usage_stats[opcode] = {
                "count": count,
                "percentage": (count / total_opcodes) * 100,
            }

        return usage_stats

    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis."""
        total_errors = sum(self.performance_metrics["error_counts"].values())
        if total_errors == 0:
            return {}

        error_analysis = {}
        for error_type, count in self.performance_metrics["error_counts"].items():
            error_analysis[error_type] = {
                "count": count,
                "percentage": (count / total_errors) * 100,
            }

        return error_analysis

    def execute_contracts_batch(
        self, contracts: List[SmartContract], execution_data: List[Dict[str, Any]]
    ) -> List[ExecutionResult]:
        """
        Execute multiple contracts in parallel using CUDA acceleration.

        Args:
            contracts: List of contracts to execute
            execution_data: List of execution data for each contract

        Returns:
            List of execution results
        """
        return self.cuda_accelerator.execute_contracts_batch(contracts, execution_data)

    def process_bytecode_batch(
        self, bytecode_list: List[bytes], optimization_level: int = 1
    ) -> List[bytes]:
        """
        Process multiple bytecode sequences in parallel using CUDA acceleration.

        Args:
            bytecode_list: List of bytecode sequences to process
            optimization_level: Level of optimization to apply

        Returns:
            List of processed bytecode sequences
        """
        return self.cuda_accelerator.process_bytecode_batch(
            bytecode_list, optimization_level
        )

    def execute_operations_batch(
        self, operations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple VM operations in parallel using CUDA acceleration.

        Args:
            operations: List of VM operations to execute

        Returns:
            List of operation results
        """
        return self.cuda_accelerator.execute_operations_batch(operations)

    def optimize_bytecode_batch(
        self, bytecode_list: List[bytes], optimization_rules: List[str]
    ) -> List[bytes]:
        """
        Optimize multiple bytecode sequences in parallel using CUDA acceleration.

        Args:
            bytecode_list: List of bytecode sequences to optimize
            optimization_rules: List of optimization rules to apply

        Returns:
            List of optimized bytecode sequences
        """
        return self.cuda_accelerator.optimize_bytecode_batch(
            bytecode_list, optimization_rules
        )

    def benchmark_vm_performance(
        self, test_data: List[Dict[str, Any]], num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark VM performance with CUDA acceleration.

        Args:
            test_data: Test data for benchmarking
            num_iterations: Number of benchmark iterations

        Returns:
            Benchmark results
        """
        return self.cuda_accelerator.benchmark_vm_operations(test_data, num_iterations)

    def get_cuda_performance_metrics(self) -> Dict[str, Any]:
        """Get CUDA performance metrics."""
        return self.cuda_accelerator.get_performance_metrics()
