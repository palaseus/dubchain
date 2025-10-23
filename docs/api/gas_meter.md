# Gas Meter API

**Module:** `dubchain.vm.gas_meter`

## Classes

### GasCost

Simple gas cost constants for compatibility.

**Inherits from:** Enum

### GasCostConfig

Gas cost configuration.

#### Methods

##### `calculate_cost(self, kwargs)`

Calculate the actual gas cost based on parameters.

**Parameters:**

- `self`: Any (required)
- `kwargs`: Any (required)

**Returns:** `<class 'int'>`

### GasCostType

Types of gas costs.

**Inherits from:** Enum

### GasMeter

Advanced gas meter for VM execution.

#### Methods

##### `calculate_gas_cost(self, operation, kwargs)`

Calculate gas cost for an operation.

**Parameters:**

- `self`: Any (required)
- `operation`: <class 'str'> (required)
- `kwargs`: Any (required)

**Returns:** `<class 'int'>`

##### `can_afford(self, operation, kwargs)`

Check if we can afford an operation.

**Parameters:**

- `self`: Any (required)
- `operation`: <class 'str'> (required)
- `kwargs`: Any (required)

**Returns:** `<class 'bool'>`

##### `consume_gas(self, amount, operation, kwargs)`

Consume gas for an operation.

**Parameters:**

- `self`: Any (required)
- `amount`: <class 'int'> (required)
- `operation`: <class 'str'> = UNKNOWN
- `kwargs`: Any (required)

**Returns:** `<class 'bool'>`

##### `get_cost_breakdown(self)`

Get breakdown of gas costs by category.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.Dict[str, int]`

##### `get_effective_gas_used(self)`

Get effective gas used (gas used - refunds).

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_gas_efficiency(self)`

Get gas efficiency percentage.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'float'>`

##### `get_gas_limit(self)`

Get gas limit.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_gas_remaining(self)`

Get remaining gas.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_gas_usage_percentage(self)`

Get gas usage percentage.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'float'>`

##### `get_gas_used(self)`

Get gas used.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'int'>`

##### `get_gas_utilization(self)`

Get gas utilization percentage.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'float'>`

##### `get_remaining_gas_percentage(self)`

Get remaining gas percentage.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'float'>`

##### `is_out_of_gas(self)`

Check if out of gas.

**Parameters:**

- `self`: Any (required)

**Returns:** `<class 'bool'>`

##### `refund_gas(self, amount)`

Refund gas (up to half of gas used).

**Parameters:**

- `self`: Any (required)
- `amount`: <class 'int'> (required)

**Returns:** `None`

##### `reset(self, new_limit)`

Reset gas meter.

**Parameters:**

- `self`: Any (required)
- `new_limit`: typing.Optional[int] = None

**Returns:** `None`

#### Properties

##### `gas_limit`

Get gas limit.

##### `gas_remaining`

Get remaining gas.

### GasOptimizer

Advanced gas optimization strategies.

#### Methods

##### `estimate_gas_usage(self, bytecode)`

Estimate gas usage for bytecode.

**Parameters:**

- `self`: Any (required)
- `bytecode`: <class 'bytes'> (required)

**Returns:** `typing.Dict[str, int]`

##### `optimize_contract(self, bytecode, gas_limit)`

Optimize a contract for gas usage.

**Parameters:**

- `self`: Any (required)
- `bytecode`: <class 'bytes'> (required)
- `gas_limit`: <class 'int'> (required)

**Returns:** `typing.Dict[str, typing.Any]`

## Usage Examples

```python
from dubchain.vm.gas_meter import *

# Create instance of GasCost
gascost = GasCost()

```
