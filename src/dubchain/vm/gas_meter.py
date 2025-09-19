"""
Advanced gas metering and optimization for the GodChain Virtual Machine.

This module provides sophisticated gas management including:
- Dynamic gas cost calculation
- Gas optimization strategies
- Memory and storage cost tracking
- Transaction gas estimation
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GasCost(Enum):
    """Simple gas cost constants for compatibility."""

    BASE = 2
    VERY_LOW = 3
    LOW = 5
    MID = 8
    HIGH = 10
    EXTCODE = 700
    BALANCE = 400
    SLOAD = 200
    JUMPDEST = 1
    SSET = 20000
    SRESET = 5000
    RSCLEAR = 15000
    SELFDESTRUCT = 5000
    CREATE = 32000
    CALL = 700
    CALLVALUE = 9000
    CALLSTIPEND = 2300
    NEWACCOUNT = 25000
    EXP = 10
    MEMORY = 3
    TXDATAZERO = 4
    TXDATANONZERO = 68
    TRANSACTION = 21000
    LOG = 375
    LOGDATA = 8
    LOGTOPIC = 375
    SHA3 = 30
    SHA3WORD = 6
    COPY = 3
    BLOCKHASH = 20
    QUADDIVISOR = 512


class GasCostType(Enum):
    """Types of gas costs."""

    FIXED = "fixed"
    DYNAMIC = "dynamic"
    MEMORY = "memory"
    STORAGE = "storage"
    CALL = "call"
    CRYPTO = "crypto"


@dataclass
class GasCostConfig:
    """Gas cost configuration."""

    base_cost: int
    cost_type: GasCostType
    dynamic_factor: Optional[float] = None
    max_cost: Optional[int] = None
    min_cost: Optional[int] = None

    def calculate_cost(self, **kwargs) -> int:
        """Calculate the actual gas cost based on parameters."""
        cost = self.base_cost

        if self.cost_type == GasCostType.DYNAMIC and self.dynamic_factor:
            # Calculate dynamic cost based on parameters
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    cost += int(value * self.dynamic_factor)

        elif self.cost_type == GasCostType.MEMORY:
            # Memory cost calculation
            memory_size = kwargs.get("memory_size", 0)
            cost += self._calculate_memory_cost(memory_size)

        elif self.cost_type == GasCostType.STORAGE:
            # Storage cost calculation
            storage_operation = kwargs.get("storage_operation", "read")
            if storage_operation == "write":
                cost += 20000  # SSTORE cost
            elif storage_operation == "delete":
                cost += 5000  # Refund for deletion

        elif self.cost_type == GasCostType.CALL:
            # Call cost calculation
            call_type = kwargs.get("call_type", "call")
            if call_type == "create":
                cost += 32000
            elif call_type == "delegatecall":
                cost += 100
            else:
                cost += 100

        elif self.cost_type == GasCostType.CRYPTO:
            # Cryptographic operation cost
            crypto_type = kwargs.get("crypto_type", "sha3")
            if crypto_type == "sha3":
                cost += 30
            elif crypto_type == "ecdsa":
                cost += 3000
            elif crypto_type == "sha256":
                cost += 60

        # Apply bounds
        if self.min_cost is not None:
            cost = max(cost, self.min_cost)
        if self.max_cost is not None:
            cost = min(cost, self.max_cost)

        return cost

    def _calculate_memory_cost(self, memory_size: int) -> int:
        """Calculate memory expansion cost."""
        if memory_size == 0:
            return 0

        # Memory cost is quadratic: cost = memory_size^2 / 512
        return int(memory_size * memory_size / 512)


class GasMeter:
    """Advanced gas meter for VM execution."""

    def __init__(
        self,
        initial_gas: int,
        remaining_gas: int,
        gas_used: int = 0,
        refunds: int = 0,
        memory_cost: int = 0,
        storage_cost: int = 0,
        call_cost: int = 0,
        crypto_cost: int = 0,
    ):
        """Initialize gas meter."""
        if initial_gas < 0:
            raise ValueError("Gas limit must be non-negative")

        self.initial_gas = initial_gas
        self.remaining_gas = remaining_gas
        self.gas_used = gas_used
        self.refunds = refunds
        self.memory_cost = memory_cost
        self.storage_cost = storage_cost
        self.call_cost = call_cost
        self.crypto_cost = crypto_cost
        self.gas_costs: Dict[str, GasCostConfig] = {}

        self._setup_default_costs()

    @property
    def gas_limit(self) -> int:
        """Get gas limit."""
        return self.initial_gas

    @property
    def gas_remaining(self) -> int:
        """Get remaining gas."""
        return self.remaining_gas

    def get_gas_used(self) -> int:
        """Get gas used."""
        return self.gas_used

    def get_gas_remaining(self) -> int:
        """Get remaining gas."""
        return self.remaining_gas

    def get_gas_limit(self) -> int:
        """Get gas limit."""
        return self.initial_gas

    def get_gas_usage_percentage(self) -> float:
        """Get gas usage percentage."""
        if self.initial_gas == 0:
            return 0.0
        return (self.gas_used / self.initial_gas) * 100.0

    def get_gas_efficiency(self) -> float:
        """Get gas efficiency percentage."""
        if self.initial_gas == 0:
            return 100.0
        return (self.remaining_gas / self.initial_gas) * 100.0

    def is_out_of_gas(self) -> bool:
        """Check if out of gas."""
        return self.remaining_gas == 0

    def reset(self, new_limit: Optional[int] = None) -> None:
        """Reset gas meter."""
        if new_limit is not None:
            if new_limit < 0:
                raise ValueError("Gas limit must be non-negative")
            self.initial_gas = new_limit
        self.remaining_gas = self.initial_gas
        self.gas_used = 0
        self.refunds = 0
        self.memory_cost = 0
        self.storage_cost = 0
        self.call_cost = 0
        self.crypto_cost = 0

    def __str__(self) -> str:
        """String representation."""
        return f"GasMeter({self.initial_gas}, used={self.gas_used}, remaining={self.remaining_gas}, limit={self.initial_gas})"

    def __repr__(self) -> str:
        """Repr representation."""
        return f"GasMeter({self.initial_gas}, gas_limit={self.initial_gas}, gas_used={self.gas_used})"

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, GasMeter):
            return False
        return (
            self.initial_gas == other.initial_gas
            and self.gas_used == other.gas_used
            and self.remaining_gas == other.remaining_gas
        )

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.initial_gas, self.gas_used, self.remaining_gas))

    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if not isinstance(other, GasMeter):
            return NotImplemented
        return self.remaining_gas < other.remaining_gas

    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if not isinstance(other, GasMeter):
            return NotImplemented
        return self.remaining_gas <= other.remaining_gas

    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if not isinstance(other, GasMeter):
            return NotImplemented
        return self.remaining_gas > other.remaining_gas

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if not isinstance(other, GasMeter):
            return NotImplemented
        return self.remaining_gas >= other.remaining_gas

    def _setup_default_costs(self) -> None:
        """Setup default gas cost configurations."""
        self.gas_costs = {
            # Arithmetic operations
            "ADD": GasCostConfig(3, GasCostType.FIXED),
            "MUL": GasCostConfig(5, GasCostType.FIXED),
            "SUB": GasCostConfig(3, GasCostType.FIXED),
            "DIV": GasCostConfig(5, GasCostType.FIXED),
            "MOD": GasCostConfig(5, GasCostType.FIXED),
            "ADDMOD": GasCostConfig(8, GasCostType.FIXED),
            "MULMOD": GasCostConfig(8, GasCostType.FIXED),
            "EXP": GasCostConfig(10, GasCostType.DYNAMIC, dynamic_factor=10),
            # Comparison operations
            "LT": GasCostConfig(3, GasCostType.FIXED),
            "GT": GasCostConfig(3, GasCostType.FIXED),
            "EQ": GasCostConfig(3, GasCostType.FIXED),
            "ISZERO": GasCostConfig(3, GasCostType.FIXED),
            # Bitwise operations
            "AND": GasCostConfig(3, GasCostType.FIXED),
            "OR": GasCostConfig(3, GasCostType.FIXED),
            "XOR": GasCostConfig(3, GasCostType.FIXED),
            "NOT": GasCostConfig(3, GasCostType.FIXED),
            "BYTE": GasCostConfig(3, GasCostType.FIXED),
            "SHL": GasCostConfig(3, GasCostType.FIXED),
            "SHR": GasCostConfig(3, GasCostType.FIXED),
            "SAR": GasCostConfig(3, GasCostType.FIXED),
            # Memory operations
            "MLOAD": GasCostConfig(3, GasCostType.MEMORY),
            "MSTORE": GasCostConfig(3, GasCostType.MEMORY),
            "MSTORE8": GasCostConfig(3, GasCostType.MEMORY),
            "MSIZE": GasCostConfig(2, GasCostType.FIXED),
            # Storage operations
            "SLOAD": GasCostConfig(100, GasCostType.STORAGE),
            "SSTORE": GasCostConfig(100, GasCostType.STORAGE),
            # Cryptographic operations
            "SHA3": GasCostConfig(30, GasCostType.CRYPTO),
            "SHA256": GasCostConfig(60, GasCostType.CRYPTO),
            "RIPEMD160": GasCostConfig(600, GasCostType.CRYPTO),
            "ECRECOVER": GasCostConfig(3000, GasCostType.CRYPTO),
            # Environmental operations
            "ADDRESS": GasCostConfig(2, GasCostType.FIXED),
            "BALANCE": GasCostConfig(100, GasCostType.FIXED),
            "ORIGIN": GasCostConfig(2, GasCostType.FIXED),
            "CALLER": GasCostConfig(2, GasCostType.FIXED),
            "CALLVALUE": GasCostConfig(2, GasCostType.FIXED),
            "CALLDATALOAD": GasCostConfig(3, GasCostType.FIXED),
            "CALLDATASIZE": GasCostConfig(2, GasCostType.FIXED),
            "CALLDATACOPY": GasCostConfig(3, GasCostType.DYNAMIC, dynamic_factor=3),
            "CODESIZE": GasCostConfig(2, GasCostType.FIXED),
            "CODECOPY": GasCostConfig(3, GasCostType.DYNAMIC, dynamic_factor=3),
            "GASPRICE": GasCostConfig(2, GasCostType.FIXED),
            "EXTCODESIZE": GasCostConfig(100, GasCostType.FIXED),
            "EXTCODECOPY": GasCostConfig(100, GasCostType.DYNAMIC, dynamic_factor=3),
            "EXTCODEHASH": GasCostConfig(100, GasCostType.FIXED),
            # Block operations
            "BLOCKHASH": GasCostConfig(20, GasCostType.FIXED),
            "COINBASE": GasCostConfig(2, GasCostType.FIXED),
            "TIMESTAMP": GasCostConfig(2, GasCostType.FIXED),
            "NUMBER": GasCostConfig(2, GasCostType.FIXED),
            "DIFFICULTY": GasCostConfig(2, GasCostType.FIXED),
            "GASLIMIT": GasCostConfig(2, GasCostType.FIXED),
            "CHAINID": GasCostConfig(2, GasCostType.FIXED),
            "SELFBALANCE": GasCostConfig(5, GasCostType.FIXED),
            # Stack operations
            "POP": GasCostConfig(2, GasCostType.FIXED),
            "PUSH": GasCostConfig(3, GasCostType.FIXED),
            "DUP": GasCostConfig(3, GasCostType.FIXED),
            "SWAP": GasCostConfig(3, GasCostType.FIXED),
            # Control flow
            "JUMP": GasCostConfig(8, GasCostType.FIXED),
            "JUMPI": GasCostConfig(10, GasCostType.FIXED),
            "PC": GasCostConfig(2, GasCostType.FIXED),
            "JUMPDEST": GasCostConfig(1, GasCostType.FIXED),
            # System operations
            "CREATE": GasCostConfig(32000, GasCostType.CALL),
            "CALL": GasCostConfig(100, GasCostType.CALL),
            "CALLCODE": GasCostConfig(100, GasCostType.CALL),
            "DELEGATECALL": GasCostConfig(100, GasCostType.CALL),
            "STATICCALL": GasCostConfig(100, GasCostType.CALL),
            "CREATE2": GasCostConfig(32000, GasCostType.CALL),
            "RETURN": GasCostConfig(0, GasCostType.FIXED),
            "REVERT": GasCostConfig(0, GasCostType.FIXED),
            "SELFDESTRUCT": GasCostConfig(5000, GasCostType.FIXED),
            # Logging
            "LOG0": GasCostConfig(375, GasCostType.DYNAMIC, dynamic_factor=375),
            "LOG1": GasCostConfig(750, GasCostType.DYNAMIC, dynamic_factor=375),
            "LOG2": GasCostConfig(1125, GasCostType.DYNAMIC, dynamic_factor=375),
            "LOG3": GasCostConfig(1500, GasCostType.DYNAMIC, dynamic_factor=375),
            "LOG4": GasCostConfig(1875, GasCostType.DYNAMIC, dynamic_factor=375),
        }

    def consume_gas(self, amount: int, operation: str = "UNKNOWN", **kwargs) -> bool:
        """Consume gas for an operation."""
        if amount < 0:
            raise ValueError("Gas amount must be non-negative")

        if amount > self.remaining_gas:
            return False

        self.remaining_gas -= amount
        self.gas_used += amount

        # Track costs by category
        if operation in ["MLOAD", "MSTORE", "MSTORE8", "MSIZE"]:
            self.memory_cost += amount
        elif operation in ["SLOAD", "SSTORE"]:
            self.storage_cost += amount
        elif operation in [
            "CREATE",
            "CALL",
            "CALLCODE",
            "DELEGATECALL",
            "STATICCALL",
            "CREATE2",
        ]:
            self.call_cost += amount
        elif operation in ["SHA3", "SHA256", "RIPEMD160", "ECRECOVER"]:
            self.crypto_cost += amount

        return True

    def calculate_gas_cost(self, operation: str, **kwargs) -> int:
        """Calculate gas cost for an operation."""
        if operation not in self.gas_costs:
            return 0

        gas_cost = self.gas_costs[operation]
        return gas_cost.calculate_cost(**kwargs)

    def refund_gas(self, amount: int) -> None:
        """Refund gas (up to half of gas used)."""
        max_refund = self.gas_used // 2
        actual_refund = min(amount, max_refund)

        self.refunds += actual_refund
        self.remaining_gas += actual_refund

    def get_effective_gas_used(self) -> int:
        """Get effective gas used (gas used - refunds)."""
        return max(0, self.gas_used - self.refunds)

    def get_gas_utilization(self) -> float:
        """Get gas utilization percentage."""
        if self.initial_gas == 0:
            return 0.0
        return (self.gas_used / self.initial_gas) * 100

    def get_remaining_gas_percentage(self) -> float:
        """Get remaining gas percentage."""
        if self.initial_gas == 0:
            return 0.0
        return (self.remaining_gas / self.initial_gas) * 100

    def is_out_of_gas(self) -> bool:
        """Check if out of gas."""
        return self.remaining_gas <= 0

    def can_afford(self, operation: str, **kwargs) -> bool:
        """Check if we can afford an operation."""
        cost = self.calculate_gas_cost(operation, **kwargs)
        return cost <= self.remaining_gas

    def get_cost_breakdown(self) -> Dict[str, int]:
        """Get breakdown of gas costs by category."""
        return {
            "total_gas_used": self.gas_used,
            "memory_cost": self.memory_cost,
            "storage_cost": self.storage_cost,
            "call_cost": self.call_cost,
            "crypto_cost": self.crypto_cost,
            "other_cost": self.gas_used
            - self.memory_cost
            - self.storage_cost
            - self.call_cost
            - self.crypto_cost,
            "refunds": self.refunds,
            "effective_gas_used": self.get_effective_gas_used(),
            "remaining_gas": self.remaining_gas,
        }

    def __str__(self) -> str:
        """String representation of the gas meter."""
        return f"GasMeter(used={self.gas_used}, remaining={self.remaining_gas}, refunds={self.refunds})"

    def __repr__(self) -> str:
        """Detailed representation of the gas meter."""
        return (
            f"GasMeter(initial={self.initial_gas}, used={self.gas_used}, "
            f"remaining={self.remaining_gas}, refunds={self.refunds}, "
            f"utilization={self.get_gas_utilization():.1f}%)"
        )


class GasOptimizer:
    """Advanced gas optimization strategies."""

    def __init__(self):
        self.optimization_strategies = {
            "memory_optimization": self._optimize_memory_usage,
            "storage_optimization": self._optimize_storage_usage,
            "call_optimization": self._optimize_call_usage,
            "crypto_optimization": self._optimize_crypto_usage,
            "stack_optimization": self._optimize_stack_usage,
        }

    def optimize_contract(self, bytecode: bytes, gas_limit: int) -> Dict[str, Any]:
        """Optimize a contract for gas usage."""
        optimizations = {}

        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                result = strategy_func(bytecode, gas_limit)
                if result:
                    optimizations[strategy_name] = result
            except Exception:
                # Skip failed optimizations
                continue

        return optimizations

    def _optimize_memory_usage(
        self, bytecode: bytes, gas_limit: int
    ) -> Optional[Dict[str, Any]]:
        """Optimize memory usage patterns."""
        # Analyze memory access patterns
        memory_ops = []
        for i, byte in enumerate(bytecode):
            if byte in [0x51, 0x52, 0x53]:  # MLOAD, MSTORE, MSTORE8
                memory_ops.append((i, byte))

        if not memory_ops:
            return None

        # Suggest optimizations
        suggestions = []

        # Check for redundant memory operations
        for i in range(len(memory_ops) - 1):
            if memory_ops[i][1] == memory_ops[i + 1][1]:  # Same operation
                suggestions.append(
                    {
                        "type": "redundant_memory_operation",
                        "position": memory_ops[i][0],
                        "suggestion": "Consider combining memory operations",
                    }
                )

        return {
            "memory_operations": len(memory_ops),
            "suggestions": suggestions,
            "estimated_savings": len(suggestions) * 3,  # 3 gas per optimization
        }

    def _optimize_storage_usage(
        self, bytecode: bytes, gas_limit: int
    ) -> Optional[Dict[str, Any]]:
        """Optimize storage usage patterns."""
        storage_ops = []
        for i, byte in enumerate(bytecode):
            if byte in [0x54, 0x55]:  # SLOAD, SSTORE
                storage_ops.append((i, byte))

        if not storage_ops:
            return None

        # Analyze storage patterns
        sstore_count = sum(1 for _, op in storage_ops if op == 0x55)
        sload_count = sum(1 for _, op in storage_ops if op == 0x54)

        suggestions = []

        # Suggest packing multiple values into single storage slot
        if sstore_count > 1:
            suggestions.append(
                {
                    "type": "storage_packing",
                    "suggestion": "Consider packing multiple values into single storage slot",
                    "estimated_savings": (sstore_count - 1) * 20000,
                }
            )

        return {
            "storage_operations": len(storage_ops),
            "sstore_count": sstore_count,
            "sload_count": sload_count,
            "suggestions": suggestions,
            "estimated_savings": sum(
                s.get("estimated_savings", 0) for s in suggestions
            ),
        }

    def _optimize_call_usage(
        self, bytecode: bytes, gas_limit: int
    ) -> Optional[Dict[str, Any]]:
        """Optimize call usage patterns."""
        call_ops = []
        for i, byte in enumerate(bytecode):
            if byte in [
                0xF0,
                0xF1,
                0xF2,
                0xF4,
                0xF5,
                0xFA,
            ]:  # CREATE, CALL, CALLCODE, DELEGATECALL, CREATE2, STATICCALL
                call_ops.append((i, byte))

        if not call_ops:
            return None

        # Analyze call patterns
        create_count = sum(1 for _, op in call_ops if op in [0xF0, 0xF5])
        call_count = sum(1 for _, op in call_ops if op in [0xF1, 0xF2, 0xF4, 0xFA])

        suggestions = []

        # Suggest using STATICCALL when possible
        if call_count > 0:
            suggestions.append(
                {
                    "type": "static_call_optimization",
                    "suggestion": "Use STATICCALL for read-only operations",
                    "estimated_savings": call_count * 50,
                }
            )

        return {
            "call_operations": len(call_ops),
            "create_count": create_count,
            "call_count": call_count,
            "suggestions": suggestions,
            "estimated_savings": sum(
                s.get("estimated_savings", 0) for s in suggestions
            ),
        }

    def _optimize_crypto_usage(
        self, bytecode: bytes, gas_limit: int
    ) -> Optional[Dict[str, Any]]:
        """Optimize cryptographic operations."""
        crypto_ops = []
        for i, byte in enumerate(bytecode):
            if byte in [
                0x20,
                0x21,
                0x22,
                0x23,
                0x24,
            ]:  # SHA3, KECCAK256, RIPEMD160, SHA256, ECRECOVER
                crypto_ops.append((i, byte))

        if not crypto_ops:
            return None

        # Analyze crypto patterns
        sha3_count = sum(1 for _, op in crypto_ops if op in [0x20, 0x21])
        sha256_count = sum(1 for _, op in crypto_ops if op == 0x23)
        ecrecover_count = sum(1 for _, op in crypto_ops if op == 0x24)

        suggestions = []

        # Suggest using SHA3 instead of SHA256 when possible
        if sha256_count > 0:
            suggestions.append(
                {
                    "type": "hash_optimization",
                    "suggestion": "Consider using SHA3 instead of SHA256 for better performance",
                    "estimated_savings": sha256_count * 30,
                }
            )

        return {
            "crypto_operations": len(crypto_ops),
            "sha3_count": sha3_count,
            "sha256_count": sha256_count,
            "ecrecover_count": ecrecover_count,
            "suggestions": suggestions,
            "estimated_savings": sum(
                s.get("estimated_savings", 0) for s in suggestions
            ),
        }

    def _optimize_stack_usage(
        self, bytecode: bytes, gas_limit: int
    ) -> Optional[Dict[str, Any]]:
        """Optimize stack usage patterns."""
        stack_ops = []
        for i, byte in enumerate(bytecode):
            if 0x60 <= byte <= 0x7F:  # PUSH operations
                stack_ops.append((i, byte, byte - 0x60 + 1))
            elif 0x80 <= byte <= 0x8F:  # DUP operations
                stack_ops.append((i, byte, byte - 0x80 + 1))
            elif 0x90 <= byte <= 0x9F:  # SWAP operations
                stack_ops.append((i, byte, byte - 0x90 + 1))

        if not stack_ops:
            return None

        # Analyze stack patterns
        push_count = sum(1 for _, op, _ in stack_ops if 0x60 <= op <= 0x7F)
        dup_count = sum(1 for _, op, _ in stack_ops if 0x80 <= op <= 0x8F)
        swap_count = sum(1 for _, op, _ in stack_ops if 0x90 <= op <= 0x9F)

        suggestions = []

        # Suggest reducing stack depth
        if dup_count > push_count:
            suggestions.append(
                {
                    "type": "stack_depth_optimization",
                    "suggestion": "Consider reducing stack depth to minimize DUP operations",
                    "estimated_savings": (dup_count - push_count) * 3,
                }
            )

        return {
            "stack_operations": len(stack_ops),
            "push_count": push_count,
            "dup_count": dup_count,
            "swap_count": swap_count,
            "suggestions": suggestions,
            "estimated_savings": sum(
                s.get("estimated_savings", 0) for s in suggestions
            ),
        }

    def estimate_gas_usage(self, bytecode: bytes) -> Dict[str, int]:
        """Estimate gas usage for bytecode."""
        total_gas = 0
        operation_counts = {}

        for byte in bytecode:
            if byte == 0x00:  # STOP
                total_gas += 0
            elif byte == 0x01:  # ADD
                total_gas += 3
                operation_counts["ADD"] = operation_counts.get("ADD", 0) + 1
            elif byte == 0x02:  # MUL
                total_gas += 5
                operation_counts["MUL"] = operation_counts.get("MUL", 0) + 1
            elif byte == 0x54:  # SLOAD
                total_gas += 100
                operation_counts["SLOAD"] = operation_counts.get("SLOAD", 0) + 1
            elif byte == 0x55:  # SSTORE
                total_gas += 20000
                operation_counts["SSTORE"] = operation_counts.get("SSTORE", 0) + 1
            elif 0x60 <= byte <= 0x7F:  # PUSH operations
                total_gas += 3
                operation_counts["PUSH"] = operation_counts.get("PUSH", 0) + 1
            # Add more opcodes as needed

        return {
            "estimated_total_gas": total_gas,
            "operation_counts": operation_counts,
            "bytecode_size": len(bytecode),
        }
