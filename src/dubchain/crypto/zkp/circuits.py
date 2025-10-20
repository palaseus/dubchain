"""
ZKP Circuit definitions and constraint systems.

This module provides the circuit abstraction for defining zero-knowledge proofs,
including constraint systems, witnesses, and circuit builders.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ConstraintType(Enum):
    """Types of constraints in a circuit."""

    EQUALITY = "equality"
    INEQUALITY = "inequality"
    RANGE = "range"
    HASH = "hash"
    SIGNATURE = "signature"
    CUSTOM = "custom"


@dataclass
class Constraint:
    """Represents a constraint in a circuit."""

    constraint_id: str
    constraint_type: ConstraintType
    variables: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        """Validate constraint after initialization."""
        if not self.constraint_id:
            raise ValueError("constraint_id cannot be empty")
        if not self.variables:
            raise ValueError("constraint must have at least one variable")


@dataclass
class ConstraintSystem:
    """Represents a system of constraints for a circuit."""

    constraints: List[Constraint] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)  # variable_name -> type
    public_variables: List[str] = field(default_factory=list)
    private_variables: List[str] = field(default_factory=list)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the system."""
        # Validate all variables exist
        for var in constraint.variables:
            if var not in self.variables:
                raise ValueError(f"Variable {var} not defined in constraint system")

        self.constraints.append(constraint)

    def add_variable(self, name: str, var_type: str, is_public: bool = False) -> None:
        """Add a variable to the constraint system."""
        if name in self.variables:
            raise ValueError(f"Variable {name} already exists")

        self.variables[name] = var_type
        if is_public:
            self.public_variables.append(name)
        else:
            self.private_variables.append(name)

    def validate(self) -> bool:
        """Validate the constraint system."""
        # Check that all constraints reference valid variables
        for constraint in self.constraints:
            for var in constraint.variables:
                if var not in self.variables:
                    return False

        # Check that public and private variables don't overlap
        public_set = set(self.public_variables)
        private_set = set(self.private_variables)
        if public_set & private_set:
            return False

        return True

    def get_constraint_count(self) -> int:
        """Get the number of constraints."""
        return len(self.constraints)

    def get_variable_count(self) -> int:
        """Get the number of variables."""
        return len(self.variables)


@dataclass
class Witness:
    """Represents a witness (assignment of values to variables) for a circuit."""

    values: Dict[str, bytes] = field(default_factory=dict)
    public_values: Dict[str, bytes] = field(default_factory=dict)
    private_values: Dict[str, bytes] = field(default_factory=dict)

    def set_value(self, variable: str, value: bytes, is_public: bool = False) -> None:
        """Set a value for a variable."""
        self.values[variable] = value
        if is_public:
            self.public_values[variable] = value
        else:
            self.private_values[variable] = value

    def get_value(self, variable: str) -> Optional[bytes]:
        """Get the value of a variable."""
        return self.values.get(variable)

    def validate_against_system(self, system: ConstraintSystem) -> bool:
        """Validate witness against a constraint system."""
        # Check that all public variables have values
        for var in system.public_variables:
            if var not in self.public_values:
                return False

        # Check that all private variables have values
        for var in system.private_variables:
            if var not in self.private_values:
                return False

        # Check that all values are in the main values dict
        for var in system.variables:
            if var not in self.values:
                return False

        return True

    def to_bytes(self) -> bytes:
        """Serialize witness to bytes."""
        import json

        data = {
            "values": {k: v.hex() for k, v in self.values.items()},
            "public_values": {k: v.hex() for k, v in self.public_values.items()},
            "private_values": {k: v.hex() for k, v in self.private_values.items()},
        }
        return json.dumps(data, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Witness":
        """Deserialize witness from bytes."""
        import json

        parsed = json.loads(data.decode("utf-8"))
        return cls(
            values={k: bytes.fromhex(v) for k, v in parsed["values"].items()},
            public_values={
                k: bytes.fromhex(v) for k, v in parsed["public_values"].items()
            },
            private_values={
                k: bytes.fromhex(v) for k, v in parsed["private_values"].items()
            },
        )


@dataclass
class PublicInputs:
    """Represents public inputs to a circuit."""

    inputs: List[bytes] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)

    def add_input(self, name: str, value: bytes) -> None:
        """Add a public input."""
        self.inputs.append(value)
        self.input_names.append(name)

    def get_input(self, name: str) -> Optional[bytes]:
        """Get a public input by name."""
        try:
            index = self.input_names.index(name)
            return self.inputs[index]
        except ValueError:
            return None

    def to_bytes(self) -> bytes:
        """Serialize public inputs to bytes."""
        import json

        data = {
            "inputs": [inp.hex() for inp in self.inputs],
            "input_names": self.input_names,
        }
        return json.dumps(data, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "PublicInputs":
        """Deserialize public inputs from bytes."""
        import json

        parsed = json.loads(data.decode("utf-8"))
        return cls(
            inputs=[bytes.fromhex(inp) for inp in parsed["inputs"]],
            input_names=parsed["input_names"],
        )


@dataclass
class PrivateInputs:
    """Represents private inputs to a circuit."""

    inputs: List[bytes] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)

    def add_input(self, name: str, value: bytes) -> None:
        """Add a private input."""
        self.inputs.append(value)
        self.input_names.append(name)

    def get_input(self, name: str) -> Optional[bytes]:
        """Get a private input by name."""
        try:
            index = self.input_names.index(name)
            return self.inputs[index]
        except ValueError:
            return None

    def to_bytes(self) -> bytes:
        """Serialize private inputs to bytes."""
        import json

        data = {
            "inputs": [inp.hex() for inp in self.inputs],
            "input_names": self.input_names,
        }
        return json.dumps(data, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "PrivateInputs":
        """Deserialize private inputs from bytes."""
        import json

        parsed = json.loads(data.decode("utf-8"))
        return cls(
            inputs=[bytes.fromhex(inp) for inp in parsed["inputs"]],
            input_names=parsed["input_names"],
        )


class ZKCircuit(ABC):
    """Abstract base class for zero-knowledge circuits."""

    def __init__(self, circuit_id: str):
        self.circuit_id = circuit_id
        self.constraint_system = ConstraintSystem()
        self._built = False

    @abstractmethod
    def build(self) -> None:
        """Build the circuit by adding constraints and variables."""
        pass

    @abstractmethod
    def generate_witness(
        self, public_inputs: PublicInputs, private_inputs: PrivateInputs
    ) -> Witness:
        """Generate a witness for the given inputs."""
        pass

    @abstractmethod
    def verify_witness(self, witness: Witness) -> bool:
        """Verify that a witness satisfies all constraints."""
        pass

    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the circuit."""
        if not self._built:
            self.build()

        return {
            "circuit_id": self.circuit_id,
            "constraint_count": self.constraint_system.get_constraint_count(),
            "variable_count": self.constraint_system.get_variable_count(),
            "public_variables": self.constraint_system.public_variables,
            "private_variables": self.constraint_system.private_variables,
            "constraints": [
                {
                    "id": c.constraint_id,
                    "type": c.constraint_type.value,
                    "variables": c.variables,
                    "description": c.description,
                }
                for c in self.constraint_system.constraints
            ],
        }

    def validate(self) -> bool:
        """Validate the circuit."""
        if not self._built:
            self.build()

        return self.constraint_system.validate()

    @property
    def is_built(self) -> bool:
        """Check if circuit is built."""
        return self._built


class CircuitBuilder:
    """Builder for creating circuits programmatically."""

    def __init__(self, circuit_id: str):
        self.circuit_id = circuit_id
        self.constraint_system = ConstraintSystem()
        self._circuit_class: Optional[type] = None

    def add_variable(
        self, name: str, var_type: str, is_public: bool = False
    ) -> "CircuitBuilder":
        """Add a variable to the circuit."""
        self.constraint_system.add_variable(name, var_type, is_public)
        return self

    def add_constraint(self, constraint: Constraint) -> "CircuitBuilder":
        """Add a constraint to the circuit."""
        self.constraint_system.add_constraint(constraint)
        return self

    def add_equality_constraint(
        self, var1: str, var2: str, description: str = ""
    ) -> "CircuitBuilder":
        """Add an equality constraint."""
        constraint = Constraint(
            constraint_id=f"eq_{var1}_{var2}",
            constraint_type=ConstraintType.EQUALITY,
            variables=[var1, var2],
            description=description,
        )
        return self.add_constraint(constraint)

    def add_range_constraint(
        self, variable: str, min_val: int, max_val: int, description: str = ""
    ) -> "CircuitBuilder":
        """Add a range constraint."""
        constraint = Constraint(
            constraint_id=f"range_{variable}",
            constraint_type=ConstraintType.RANGE,
            variables=[variable],
            parameters={"min": min_val, "max": max_val},
            description=description,
        )
        return self.add_constraint(constraint)

    def add_hash_constraint(
        self,
        input_var: str,
        output_var: str,
        hash_algorithm: str = "sha256",
        description: str = "",
    ) -> "CircuitBuilder":
        """Add a hash constraint."""
        constraint = Constraint(
            constraint_id=f"hash_{input_var}_{output_var}",
            constraint_type=ConstraintType.HASH,
            variables=[input_var, output_var],
            parameters={"algorithm": hash_algorithm},
            description=description,
        )
        return self.add_constraint(constraint)

    def build(self) -> "BuiltCircuit":
        """Build the circuit."""
        if not self.constraint_system.validate():
            raise ValueError("Invalid constraint system")

        return BuiltCircuit(self.circuit_id, self.constraint_system)


class BuiltCircuit(ZKCircuit):
    """A circuit built using CircuitBuilder."""

    def __init__(self, circuit_id: str, constraint_system: ConstraintSystem):
        super().__init__(circuit_id)
        self.constraint_system = constraint_system
        self._built = True

    def build(self) -> None:
        """Circuit is already built."""
        pass

    def generate_witness(
        self, public_inputs: PublicInputs, private_inputs: PrivateInputs
    ) -> Witness:
        """Generate a witness for the given inputs."""
        witness = Witness()

        # Add public inputs
        for name, value in zip(public_inputs.input_names, public_inputs.inputs):
            witness.set_value(name, value, is_public=True)

        # Add private inputs
        for name, value in zip(private_inputs.input_names, private_inputs.inputs):
            witness.set_value(name, value, is_public=False)

        # Validate witness
        if not witness.validate_against_system(self.constraint_system):
            raise ValueError("Invalid witness for constraint system")

        return witness

    def verify_witness(self, witness: Witness) -> bool:
        """Verify that a witness satisfies all constraints."""
        if not witness.validate_against_system(self.constraint_system):
            return False

        # Verify each constraint
        for constraint in self.constraint_system.constraints:
            if not self._verify_constraint(constraint, witness):
                return False

        return True

    def _verify_constraint(self, constraint: Constraint, witness: Witness) -> bool:
        """Verify a single constraint."""
        if constraint.constraint_type == ConstraintType.EQUALITY:
            if len(constraint.variables) != 2:
                return False
            var1, var2 = constraint.variables
            return witness.get_value(var1) == witness.get_value(var2)

        elif constraint.constraint_type == ConstraintType.RANGE:
            if len(constraint.variables) != 1:
                return False
            var = constraint.variables[0]
            value = witness.get_value(var)
            if not value:
                return False

            # Convert bytes to int (assuming little-endian)
            int_value = int.from_bytes(value, "little")
            min_val = constraint.parameters.get("min", 0)
            max_val = constraint.parameters.get("max", 2**64 - 1)
            return min_val <= int_value <= max_val

        elif constraint.constraint_type == ConstraintType.HASH:
            if len(constraint.variables) != 2:
                return False
            input_var, output_var = constraint.variables
            input_value = witness.get_value(input_var)
            output_value = witness.get_value(output_var)

            if not input_value or not output_value:
                return False

            # Compute hash
            algorithm = constraint.parameters.get("algorithm", "sha256")
            if algorithm == "sha256":
                computed_hash = hashlib.sha256(input_value).digest()
            elif algorithm == "sha3_256":
                computed_hash = hashlib.sha3_256(input_value).digest()
            else:
                return False

            return computed_hash == output_value

        else:
            # Custom constraint - assume valid for now
            return True
