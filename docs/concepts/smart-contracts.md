# Smart Contracts

This document explains smart contracts and the virtual machine implementation in DubChain.

## What are Smart Contracts?

Smart contracts are self-executing contracts with the terms of the agreement directly written into code. They automatically execute when predetermined conditions are met, without the need for intermediaries.

## Smart Contract Properties

### Autonomy
Contracts execute automatically without human intervention.

### Trustlessness
No need to trust a third party - the code is the law.

### Transparency
All contract code and execution is visible on the blockchain.

### Immutability
Once deployed, contract code cannot be changed (unless explicitly designed to be upgradeable).

## DubChain Virtual Machine

### Architecture
The DubChain VM is a stack-based virtual machine designed for executing smart contracts:

```python
class VirtualMachine:
    def __init__(self):
        self.stack = []
        self.memory = {}
        self.storage = {}
        self.gas_limit = 0
        self.gas_used = 0
        self.pc = 0  # Program counter
```

### Instruction Set
The VM supports a comprehensive set of instructions:

#### Stack Operations
- `PUSH`: Push value onto stack
- `POP`: Remove value from stack
- `DUP`: Duplicate stack element
- `SWAP`: Swap stack elements

#### Arithmetic Operations
- `ADD`: Addition
- `SUB`: Subtraction
- `MUL`: Multiplication
- `DIV`: Division
- `MOD`: Modulo

#### Comparison Operations
- `LT`: Less than
- `GT`: Greater than
- `EQ`: Equal
- `NE`: Not equal

#### Control Flow
- `JUMP`: Unconditional jump
- `JUMPI`: Conditional jump
- `PC`: Get program counter
- `STOP`: Stop execution

#### Memory Operations
- `MLOAD`: Load from memory
- `MSTORE`: Store to memory
- `MSIZE`: Get memory size

#### Storage Operations
- `SLOAD`: Load from storage
- `SSTORE`: Store to storage

#### Gas Operations
- `GAS`: Get remaining gas
- `GASPRICE`: Get gas price

### Gas Metering
Gas is used to prevent infinite loops and resource exhaustion:

```python
class GasMeter:
    def __init__(self, gas_limit):
        self.gas_limit = gas_limit
        self.gas_used = 0
        self.gas_price = 1
    
    def consume_gas(self, amount):
        if self.gas_used + amount > self.gas_limit:
            raise OutOfGasError()
        self.gas_used += amount
    
    def refund_gas(self, amount):
        self.gas_used = max(0, self.gas_used - amount)
```

### Contract Execution
```python
def execute_contract(bytecode, input_data, gas_limit):
    vm = VirtualMachine()
    vm.gas_limit = gas_limit
    
    # Load bytecode
    vm.load_bytecode(bytecode)
    
    # Execute instructions
    while vm.pc < len(vm.bytecode):
        instruction = vm.bytecode[vm.pc]
        vm.execute_instruction(instruction)
        vm.pc += 1
    
    return vm.get_result()
```

## Smart Contract Development

### Contract Structure
```python
class SmartContract:
    def __init__(self, code, address):
        self.code = code
        self.address = address
        self.storage = {}
        self.balance = 0
    
    def deploy(self, deployer, gas_limit):
        # Deploy contract to blockchain
        pass
    
    def call(self, caller, function_name, args, gas_limit):
        # Execute contract function
        pass
```

### Example Contract
```python
# Simple storage contract
contract_code = """
contract SimpleStorage {
    uint256 public storedData;
    
    function set(uint256 x) public {
        storedData = x;
    }
    
    function get() public view returns (uint256) {
        return storedData;
    }
}
"""
```

### Contract Compilation
```python
def compile_contract(source_code):
    # Parse contract source code
    ast = parse_contract(source_code)
    
    # Generate bytecode
    bytecode = generate_bytecode(ast)
    
    # Optimize bytecode
    optimized_bytecode = optimize_bytecode(bytecode)
    
    return optimized_bytecode
```

## Advanced Features

### Contract Inheritance
```python
contract Ownable {
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }
}

contract MyContract is Ownable {
    function restrictedFunction() public onlyOwner {
        // Only owner can call this
    }
}
```

### Events and Logging
```python
contract EventExample {
    event ValueSet(uint256 indexed value, address indexed setter);
    
    function setValue(uint256 value) public {
        emit ValueSet(value, msg.sender);
    }
}
```

### Error Handling
```python
contract ErrorHandling {
    error InsufficientBalance(uint256 available, uint256 required);
    
    function transfer(address to, uint256 amount) public {
        if (balance[msg.sender] < amount) {
            revert InsufficientBalance(balance[msg.sender], amount);
        }
        balance[msg.sender] -= amount;
        balance[to] += amount;
    }
}
```

## Security Considerations

### Common Vulnerabilities

**Reentrancy Attacks:**
```python
# Vulnerable contract
contract Vulnerable {
    mapping(address => uint256) public balances;
    
    function withdraw() public {
        uint256 amount = balances[msg.sender];
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        balances[msg.sender] = 0;  // Too late!
    }
}

# Secure contract
contract Secure {
    mapping(address => uint256) public balances;
    bool private locked;
    
    modifier noReentrancy() {
        require(!locked);
        locked = true;
        _;
        locked = false;
    }
    
    function withdraw() public noReentrancy {
        uint256 amount = balances[msg.sender];
        balances[msg.sender] = 0;  // Update state first
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
    }
}
```

**Integer Overflow/Underflow:**
```python
// Use SafeMath library
import "./SafeMath.sol";

contract SafeMathExample {
    using SafeMath for uint256;
    
    function safeAdd(uint256 a, uint256 b) public pure returns (uint256) {
        return a.add(b);  // Will revert on overflow
    }
}
```

**Access Control:**
```python
contract AccessControl {
    mapping(address => bool) public authorized;
    address public owner;
    
    modifier onlyAuthorized() {
        require(authorized[msg.sender] || msg.sender == owner);
        _;
    }
    
    function authorize(address account) public {
        require(msg.sender == owner);
        authorized[account] = true;
    }
}
```

### Security Best Practices

1. **Use established patterns** (OpenZeppelin contracts)
2. **Implement proper access controls**
3. **Handle all edge cases**
4. **Use formal verification**
5. **Conduct security audits**
6. **Test thoroughly**

## Performance Optimization

### Gas Optimization
```python
# Inefficient
function inefficient() public {
    for (uint i = 0; i < array.length; i++) {
        // Process array[i]
    }
}

# Efficient
function efficient() public {
    uint length = array.length;
    for (uint i = 0; i < length; i++) {
        // Process array[i]
    }
}
```

### Storage Optimization
```python
// Pack structs efficiently
struct PackedData {
    uint128 value1;  // 16 bytes
    uint128 value2;  // 16 bytes
    uint32 value3;   // 4 bytes
    uint32 value4;   // 4 bytes
    // Total: 40 bytes (fits in 2 storage slots)
}
```

### Memory Management
```python
// Use memory for temporary data
function processData(uint256[] memory data) public {
    // Process data in memory
    // Memory is cheaper than storage
}
```

## Testing Smart Contracts

### Unit Testing
```python
def test_contract_deployment():
    contract = SmartContract.compile(contract_code)
    result = contract.deploy(deployer, gas_limit=1000000)
    assert result.success
    assert contract.address is not None

def test_contract_function():
    contract = SmartContract.compile(contract_code)
    contract.deploy(deployer, gas_limit=1000000)
    
    result = contract.call(caller, "set", [42], gas_limit=100000)
    assert result.success
    
    result = contract.call(caller, "get", [], gas_limit=100000)
    assert result.success
    assert result.return_value == 42
```

### Integration Testing
```python
def test_contract_interaction():
    # Deploy multiple contracts
    contract1 = deploy_contract(contract1_code)
    contract2 = deploy_contract(contract2_code)
    
    # Test interaction between contracts
    result = contract1.call(caller, "interactWith", [contract2.address], gas_limit=1000000)
    assert result.success
```

## Further Reading

- [Blockchain Fundamentals](blockchain.md)
- [Consensus Mechanisms](consensus.md)
- [Cryptography](cryptography.md)
- [Performance Optimization](../../performance/README.md)
