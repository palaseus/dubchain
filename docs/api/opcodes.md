# Opcodes API

**Module:** `dubchain.vm.opcodes`

## Classes

### Opcode

Simple opcode representation for compatibility.

### OpcodeEnum

Enumeration of all supported opcodes.

**Inherits from:** IntEnum

### OpcodeInfo

Information about an opcode.

### OpcodeRegistry

Registry for opcode information and execution handlers.

#### Methods

##### `get_all_opcodes(self)`

Get all registered opcodes.

**Parameters:**

- `self`: Any (required)

**Returns:** `typing.List[dubchain.vm.opcodes.OpcodeEnum]`

##### `get_gas_cost(self, opcode)`

Get the gas cost of an opcode.

**Parameters:**

- `self`: Any (required)
- `opcode`: <enum 'OpcodeEnum'> (required)

**Returns:** `<class 'int'>`

##### `get_handler(self, opcode)`

Get the handler for an opcode.

**Parameters:**

- `self`: Any (required)
- `opcode`: <enum 'OpcodeEnum'> (required)

**Returns:** `typing.Optional[typing.Callable]`

##### `get_info(self, opcode)`

Get information about an opcode.

**Parameters:**

- `self`: Any (required)
- `opcode`: <enum 'OpcodeEnum'> (required)

**Returns:** `typing.Optional[dubchain.vm.opcodes.OpcodeInfo]`

##### `get_opcodes_by_category(self, category)`

Get all opcodes in a specific category.

**Parameters:**

- `self`: Any (required)
- `category`: <class 'str'> (required)

**Returns:** `typing.List[dubchain.vm.opcodes.OpcodeEnum]`

##### `get_stack_inputs(self, opcode)`

Get the number of stack inputs required by an opcode.

**Parameters:**

- `self`: Any (required)
- `opcode`: <enum 'OpcodeEnum'> (required)

**Returns:** `<class 'int'>`

##### `get_stack_outputs(self, opcode)`

Get the number of stack outputs produced by an opcode.

**Parameters:**

- `self`: Any (required)
- `opcode`: <enum 'OpcodeEnum'> (required)

**Returns:** `<class 'int'>`

##### `is_valid_opcode(self, opcode)`

Check if an opcode is valid.

**Parameters:**

- `self`: Any (required)
- `opcode`: <class 'int'> (required)

**Returns:** `<class 'bool'>`

##### `register_handler(self, opcode, handler)`

Register a custom handler for an opcode.

**Parameters:**

- `self`: Any (required)
- `opcode`: <enum 'OpcodeEnum'> (required)
- `handler`: typing.Callable (required)

**Returns:** `None`

### OpcodeType

Types of opcodes.

**Inherits from:** Enum

## Usage Examples

```python
from dubchain.vm.opcodes import *

# Create instance of Opcode
opcode = Opcode()

```
