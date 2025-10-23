# Execution Engine API

**Module:** `dubchain.vm.execution_engine`

## Classes

### ExecutionContext

Execution context for a contract call.

#### Methods

##### `consume_gas(self, amount)`

Consume gas and check if execution should continue.

**Parameters:**

- `self`: Any (required)
- `amount`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

##### `get_memory(self, offset, size)`

Get data from memory.

**Parameters:**

- `self`: Any (required)
- `offset`: <class 'int'> (required)
- `size`: <class 'int'> (required)

**Returns:** `<class 'bytes'>`

##### `is_valid_jump_destination(self, destination)`

Check if jump destination is valid.

**Parameters:**

- `self`: Any (required)
- `destination`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

##### `peek_stack(self, index)`

Peek at stack value without popping.

**Parameters:**

- `self`: Any (required)
- `index`: <class 'int'> = 0

**Returns:** `<class 'int'>`

##### `pop_stack(self)`

Pop value from stack.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `push_stack(self, value)`

Push value onto stack.

**Parameters:**

- `self`: Any (required)
- `value`: <class 'int'> (required)

**Returns:** `None`

##### `revert(self, message)`

Revert execution.

**Parameters:**

- `self`: Any (required)
- `message`: <class 'str'> = 

**Returns:** `None`

##### `set_memory(self, offset, data)`

Set data in memory.

**Parameters:**

- `self`: Any (required)
- `offset`: <class 'int'> (required)
- `data`: <class 'bytes'> (required)

**Returns:** `None`

##### `stop(self)`

Stop execution.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

### ExecutionEngine

Advanced execution engine for smart contracts.

#### Methods

##### `benchmark_vm_performance(self, test_data, num_iterations)`

Benchmark VM performance with CUDA acceleration.

Args:
    test_data: Test data for benchmarking
    num_iterations: Number of benchmark iterations

Returns:
    Benchmark results

**Parameters:**

- `self`: Any (required)
- `test_data`: typing.List[typing.Dict[str, typing.Any]] (required)
- `num_iterations`: <class 'int'> = 10

**Returns:** `typing.Dict[str, typing.Any]`

##### `execute_contract(self, contract, caller, value, data, gas_limit, block_context)`

Execute a smart contract.

**Parameters:**

- `self`: Any (required)
- `contract`: <class 'dubchain.vm.contract.SmartContract'> (required)
- `caller`: <class 'str'> (required)
- `value`: <class 'int'> (required)
- `data`: <class 'bytes'> (required)
- `gas_limit`: <class 'int'> (required)
- `block_context`: typing.Dict[str, typing.Any] (required)

**Returns:** `<class 'dubchain.vm.execution_engine.ExecutionResult'>`

##### `execute_contracts_batch(self, contracts, execution_data)`

Execute multiple contracts in parallel using CUDA acceleration.

Args:
    contracts: List of contracts to execute
    execution_data: List of execution data for each contract

Returns:
    List of execution results

**Parameters:**

- `self`: Any (required)
- `contracts`: typing.List[dubchain.vm.contract.SmartContract] (required)
- `execution_data`: typing.List[typing.Dict[str, typing.Any]] (required)

**Returns:** `typing.List[dubchain.vm.execution_engine.ExecutionResult]`

##### `execute_operations_batch(self, operations)`

Execute multiple VM operations in parallel using CUDA acceleration.

Args:
    operations: List of VM operations to execute

Returns:
    List of operation results

**Parameters:**

- `self`: Any (required)
- `operations`: typing.List[typing.Dict[str, typing.Any]] (required)

**Returns:** `typing.List[typing.Dict[str, typing.Any]]`

##### `get_cuda_performance_metrics(self)`

Get CUDA performance metrics.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_error_analysis(self)`

Get error analysis.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_execution_history(self, limit)`

Get execution history.

**Parameters:**

- `self`: Any (required)
- `limit`: typing.Optional[int] = None

**Returns:** `typing.List[typing.Dict[str, typing.Any]]`

##### `get_opcode_usage_stats(self)`

Get opcode usage statistics.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `get_performance_metrics(self)`

Get performance metrics.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

##### `optimize_bytecode_batch(self, bytecode_list, optimization_rules)`

Optimize multiple bytecode sequences in parallel using CUDA acceleration.

Args:
    bytecode_list: List of bytecode sequences to optimize
    optimization_rules: List of optimization rules to apply

Returns:
    List of optimized bytecode sequences

**Parameters:**

- `self`: Any (required)
- `bytecode_list`: typing.List[bytes] (required)
- `optimization_rules`: typing.List[str] (required)

**Returns:** `typing.List[bytes]`

##### `process_bytecode_batch(self, bytecode_list, optimization_level)`

Process multiple bytecode sequences in parallel using CUDA acceleration.

Args:
    bytecode_list: List of bytecode sequences to process
    optimization_level: Level of optimization to apply

Returns:
    List of processed bytecode sequences

**Parameters:**

- `self`: Any (required)
- `bytecode_list`: typing.List[bytes] (required)
- `optimization_level`: <class 'int'> = 1

**Returns:** `typing.List[bytes]`

##### `reset_metrics(self)`

Reset performance metrics.

**Parameters:**

- `self`: Any (required)

**Returns:** `None`

#### Properties

##### `cuda_accelerator`

Get CUDA accelerator with lazy loading.

### ExecutionResult

Result of contract execution.

#### Methods

##### `to_dict(self)`

Convert result to dictionary.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, typing.Any]`

### ExecutionState

Execution states.

**Inherits from:** Enum

## Usage Examples

```python
from dubchain.vm.execution_engine import *

# Create instance of ExecutionContext
executioncontext = ExecutionContext()

# Call method
result = executioncontext.consume_gas()
```
