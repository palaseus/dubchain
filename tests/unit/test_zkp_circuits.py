"""
Unit tests for ZKP circuits and constraint systems.

This module tests the circuit definitions, constraint systems, witnesses,
and circuit builders.
"""

import pytest
import hashlib
from unittest.mock import Mock

from src.dubchain.crypto.zkp.circuits import (
    ConstraintType,
    Constraint,
    ConstraintSystem,
    Witness,
    PublicInputs,
    PrivateInputs,
    ZKCircuit,
    CircuitBuilder,
    BuiltCircuit,
)


class TestConstraint:
    """Test Constraint data structure."""
    
    def test_constraint_creation(self):
        """Test constraint creation."""
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "var2"],
            description="Test equality constraint"
        )
        
        assert constraint.constraint_id == "test_constraint"
        assert constraint.constraint_type == ConstraintType.EQUALITY
        assert constraint.variables == ["var1", "var2"]
        assert constraint.description == "Test equality constraint"
        assert constraint.parameters == {}
    
    def test_constraint_validation(self):
        """Test constraint validation."""
        # Valid constraint
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "var2"]
        )
        # Should not raise
        
        # Empty constraint ID
        with pytest.raises(ValueError, match="constraint_id cannot be empty"):
            Constraint(
                constraint_id="",
                constraint_type=ConstraintType.EQUALITY,
                variables=["var1", "var2"]
            )
        
        # Empty variables
        with pytest.raises(ValueError, match="constraint must have at least one variable"):
            Constraint(
                constraint_id="test_constraint",
                constraint_type=ConstraintType.EQUALITY,
                variables=[]
            )


class TestConstraintSystem:
    """Test ConstraintSystem."""
    
    def test_constraint_system_creation(self):
        """Test constraint system creation."""
        system = ConstraintSystem()
        
        assert system.constraints == []
        assert system.variables == {}
        assert system.public_variables == []
        assert system.private_variables == []
    
    def test_add_variable(self):
        """Test adding variables to constraint system."""
        system = ConstraintSystem()
        
        # Add public variable
        system.add_variable("public_var", "bytes", is_public=True)
        assert "public_var" in system.variables
        assert system.variables["public_var"] == "bytes"
        assert "public_var" in system.public_variables
        assert "public_var" not in system.private_variables
        
        # Add private variable
        system.add_variable("private_var", "int", is_public=False)
        assert "private_var" in system.variables
        assert system.variables["private_var"] == "int"
        assert "private_var" in system.private_variables
        assert "private_var" not in system.public_variables
    
    def test_add_duplicate_variable(self):
        """Test adding duplicate variable."""
        system = ConstraintSystem()
        system.add_variable("var1", "bytes", is_public=True)
        
        with pytest.raises(ValueError, match="Variable var1 already exists"):
            system.add_variable("var1", "int", is_public=False)
    
    def test_add_constraint(self):
        """Test adding constraints to system."""
        system = ConstraintSystem()
        
        # Add variables first
        system.add_variable("var1", "bytes", is_public=True)
        system.add_variable("var2", "bytes", is_public=False)
        
        # Add constraint
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "var2"]
        )
        
        system.add_constraint(constraint)
        assert len(system.constraints) == 1
        assert system.constraints[0] == constraint
    
    def test_add_constraint_with_undefined_variable(self):
        """Test adding constraint with undefined variable."""
        system = ConstraintSystem()
        
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["undefined_var"]
        )
        
        with pytest.raises(ValueError, match="Variable undefined_var not defined"):
            system.add_constraint(constraint)
    
    def test_validate_system(self):
        """Test constraint system validation."""
        system = ConstraintSystem()
        
        # Empty system is valid
        assert system.validate() is True
        
        # Add variables and constraints
        system.add_variable("var1", "bytes", is_public=True)
        system.add_variable("var2", "bytes", is_public=False)
        
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "var2"]
        )
        system.add_constraint(constraint)
        
        assert system.validate() is True
    
    def test_validate_system_with_undefined_variable(self):
        """Test validation with undefined variable in constraint."""
        system = ConstraintSystem()
        
        system.add_variable("var1", "bytes", is_public=True)
        
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "undefined_var"]
        )
        system.constraints.append(constraint)  # Add directly to bypass validation
        
        assert system.validate() is False
    
    def test_validate_system_with_overlapping_variables(self):
        """Test validation with overlapping public/private variables."""
        system = ConstraintSystem()
        
        system.add_variable("var1", "bytes", is_public=True)
        system.private_variables.append("var1")  # Add to private list manually
        
        assert system.validate() is False
    
    def test_get_counts(self):
        """Test getting constraint and variable counts."""
        system = ConstraintSystem()
        
        assert system.get_constraint_count() == 0
        assert system.get_variable_count() == 0
        
        system.add_variable("var1", "bytes", is_public=True)
        system.add_variable("var2", "int", is_public=False)
        
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "var2"]
        )
        system.add_constraint(constraint)
        
        assert system.get_constraint_count() == 1
        assert system.get_variable_count() == 2


class TestWitness:
    """Test Witness data structure."""
    
    def test_witness_creation(self):
        """Test witness creation."""
        witness = Witness()
        
        assert witness.values == {}
        assert witness.public_values == {}
        assert witness.private_values == {}
    
    def test_set_value(self):
        """Test setting witness values."""
        witness = Witness()
        
        # Set public value
        witness.set_value("public_var", b"public_data", is_public=True)
        assert witness.values["public_var"] == b"public_data"
        assert witness.public_values["public_var"] == b"public_data"
        assert "public_var" not in witness.private_values
        
        # Set private value
        witness.set_value("private_var", b"private_data", is_public=False)
        assert witness.values["private_var"] == b"private_data"
        assert witness.private_values["private_var"] == b"private_data"
        assert "private_var" not in witness.public_values
    
    def test_get_value(self):
        """Test getting witness values."""
        witness = Witness()
        witness.set_value("var1", b"data1", is_public=True)
        witness.set_value("var2", b"data2", is_public=False)
        
        assert witness.get_value("var1") == b"data1"
        assert witness.get_value("var2") == b"data2"
        assert witness.get_value("nonexistent") is None
    
    def test_validate_against_system(self):
        """Test witness validation against constraint system."""
        system = ConstraintSystem()
        system.add_variable("public_var", "bytes", is_public=True)
        system.add_variable("private_var", "bytes", is_public=False)
        
        witness = Witness()
        
        # Empty witness should be invalid
        assert witness.validate_against_system(system) is False
        
        # Add public value only
        witness.set_value("public_var", b"public_data", is_public=True)
        assert witness.validate_against_system(system) is False
        
        # Add private value
        witness.set_value("private_var", b"private_data", is_public=False)
        assert witness.validate_against_system(system) is True
    
    def test_witness_serialization(self):
        """Test witness serialization and deserialization."""
        witness = Witness()
        witness.set_value("public_var", b"public_data", is_public=True)
        witness.set_value("private_var", b"private_data", is_public=False)
        
        # Serialize
        serialized = witness.to_bytes()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize
        deserialized = Witness.from_bytes(serialized)
        
        assert deserialized.values == witness.values
        assert deserialized.public_values == witness.public_values
        assert deserialized.private_values == witness.private_values
    
    def test_witness_serialization_invalid_data(self):
        """Test witness deserialization with invalid data."""
        with pytest.raises(ValueError):
            Witness.from_bytes(b"invalid_json")


class TestPublicInputs:
    """Test PublicInputs data structure."""
    
    def test_public_inputs_creation(self):
        """Test public inputs creation."""
        inputs = PublicInputs()
        
        assert inputs.inputs == []
        assert inputs.input_names == []
    
    def test_add_input(self):
        """Test adding public inputs."""
        inputs = PublicInputs()
        
        inputs.add_input("input1", b"data1")
        inputs.add_input("input2", b"data2")
        
        assert inputs.inputs == [b"data1", b"data2"]
        assert inputs.input_names == ["input1", "input2"]
    
    def test_get_input(self):
        """Test getting public inputs by name."""
        inputs = PublicInputs()
        inputs.add_input("input1", b"data1")
        inputs.add_input("input2", b"data2")
        
        assert inputs.get_input("input1") == b"data1"
        assert inputs.get_input("input2") == b"data2"
        assert inputs.get_input("nonexistent") is None
    
    def test_public_inputs_serialization(self):
        """Test public inputs serialization and deserialization."""
        inputs = PublicInputs()
        inputs.add_input("input1", b"data1")
        inputs.add_input("input2", b"data2")
        
        # Serialize
        serialized = inputs.to_bytes()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize
        deserialized = PublicInputs.from_bytes(serialized)
        
        assert deserialized.inputs == inputs.inputs
        assert deserialized.input_names == inputs.input_names


class TestPrivateInputs:
    """Test PrivateInputs data structure."""
    
    def test_private_inputs_creation(self):
        """Test private inputs creation."""
        inputs = PrivateInputs()
        
        assert inputs.inputs == []
        assert inputs.input_names == []
    
    def test_add_input(self):
        """Test adding private inputs."""
        inputs = PrivateInputs()
        
        inputs.add_input("input1", b"data1")
        inputs.add_input("input2", b"data2")
        
        assert inputs.inputs == [b"data1", b"data2"]
        assert inputs.input_names == ["input1", "input2"]
    
    def test_get_input(self):
        """Test getting private inputs by name."""
        inputs = PrivateInputs()
        inputs.add_input("input1", b"data1")
        inputs.add_input("input2", b"data2")
        
        assert inputs.get_input("input1") == b"data1"
        assert inputs.get_input("input2") == b"data2"
        assert inputs.get_input("nonexistent") is None
    
    def test_private_inputs_serialization(self):
        """Test private inputs serialization and deserialization."""
        inputs = PrivateInputs()
        inputs.add_input("input1", b"data1")
        inputs.add_input("input2", b"data2")
        
        # Serialize
        serialized = inputs.to_bytes()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        # Deserialize
        deserialized = PrivateInputs.from_bytes(serialized)
        
        assert deserialized.inputs == inputs.inputs
        assert deserialized.input_names == inputs.input_names


class MockZKCircuit(ZKCircuit):
    """Mock ZK circuit for testing."""
    
    def __init__(self, circuit_id: str):
        super().__init__(circuit_id)
        self.build_called = False
        self.generate_witness_called = False
        self.verify_witness_called = False
    
    def build(self) -> None:
        self.build_called = True
        self._built = True
        # Add some mock variables and constraints
        self.constraint_system.add_variable("public_var", "bytes", is_public=True)
        self.constraint_system.add_variable("private_var", "bytes", is_public=False)
    
    def generate_witness(self, public_inputs: PublicInputs, private_inputs: PrivateInputs) -> Witness:
        self.generate_witness_called = True
        witness = Witness()
        
        # Add public inputs
        for name, value in zip(public_inputs.input_names, public_inputs.inputs):
            witness.set_value(name, value, is_public=True)
        
        # Add private inputs
        for name, value in zip(private_inputs.input_names, private_inputs.inputs):
            witness.set_value(name, value, is_public=False)
        
        return witness
    
    def verify_witness(self, witness: Witness) -> bool:
        self.verify_witness_called = True
        return witness.validate_against_system(self.constraint_system)


class TestZKCircuit:
    """Test ZKCircuit abstract base class."""
    
    def test_circuit_creation(self):
        """Test circuit creation."""
        circuit = MockZKCircuit("test_circuit")
        
        assert circuit.circuit_id == "test_circuit"
        assert not circuit.is_built
        assert not circuit._built
    
    def test_circuit_build(self):
        """Test circuit building."""
        circuit = MockZKCircuit("test_circuit")
        
        assert not circuit.build_called
        circuit.build()
        assert circuit.build_called
        assert circuit.is_built
    
    def test_circuit_info(self):
        """Test getting circuit info."""
        circuit = MockZKCircuit("test_circuit")
        
        info = circuit.get_circuit_info()
        
        assert info["circuit_id"] == "test_circuit"
        assert info["constraint_count"] == 0
        assert info["variable_count"] == 2  # public_var and private_var
        assert "public_var" in info["public_variables"]
        assert "private_var" in info["private_variables"]
    
    def test_circuit_validation(self):
        """Test circuit validation."""
        circuit = MockZKCircuit("test_circuit")
        
        # Should build automatically
        assert circuit.validate() is True
    
    def test_generate_witness(self):
        """Test witness generation."""
        circuit = MockZKCircuit("test_circuit")
        
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"public_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"private_data")
        
        witness = circuit.generate_witness(public_inputs, private_inputs)
        
        assert circuit.generate_witness_called
        assert witness.get_value("public_var") == b"public_data"
        assert witness.get_value("private_var") == b"private_data"
    
    def test_verify_witness(self):
        """Test witness verification."""
        circuit = MockZKCircuit("test_circuit")
        
        witness = Witness()
        witness.set_value("public_var", b"public_data", is_public=True)
        witness.set_value("private_var", b"private_data", is_public=False)
        
        assert circuit.verify_witness(witness) is True
        assert circuit.verify_witness_called


class TestCircuitBuilder:
    """Test CircuitBuilder."""
    
    def test_circuit_builder_creation(self):
        """Test circuit builder creation."""
        builder = CircuitBuilder("test_circuit")
        
        assert builder.circuit_id == "test_circuit"
        assert builder.constraint_system.get_variable_count() == 0
        assert builder.constraint_system.get_constraint_count() == 0
    
    def test_add_variable(self):
        """Test adding variables to builder."""
        builder = CircuitBuilder("test_circuit")
        
        builder.add_variable("var1", "bytes", is_public=True)
        builder.add_variable("var2", "int", is_public=False)
        
        assert builder.constraint_system.get_variable_count() == 2
        assert "var1" in builder.constraint_system.public_variables
        assert "var2" in builder.constraint_system.private_variables
    
    def test_add_constraint(self):
        """Test adding constraints to builder."""
        builder = CircuitBuilder("test_circuit")
        
        builder.add_variable("var1", "bytes", is_public=True)
        builder.add_variable("var2", "bytes", is_public=False)
        
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "var2"]
        )
        
        builder.add_constraint(constraint)
        assert builder.constraint_system.get_constraint_count() == 1
    
    def test_add_equality_constraint(self):
        """Test adding equality constraint."""
        builder = CircuitBuilder("test_circuit")
        
        builder.add_variable("var1", "bytes", is_public=True)
        builder.add_variable("var2", "bytes", is_public=False)
        
        builder.add_equality_constraint("var1", "var2", "Test equality")
        
        assert builder.constraint_system.get_constraint_count() == 1
        constraint = builder.constraint_system.constraints[0]
        assert constraint.constraint_type == ConstraintType.EQUALITY
        assert constraint.variables == ["var1", "var2"]
        assert constraint.description == "Test equality"
    
    def test_add_range_constraint(self):
        """Test adding range constraint."""
        builder = CircuitBuilder("test_circuit")
        
        builder.add_variable("var1", "int", is_public=True)
        
        builder.add_range_constraint("var1", 0, 100, "Test range")
        
        assert builder.constraint_system.get_constraint_count() == 1
        constraint = builder.constraint_system.constraints[0]
        assert constraint.constraint_type == ConstraintType.RANGE
        assert constraint.variables == ["var1"]
        assert constraint.parameters == {"min": 0, "max": 100}
        assert constraint.description == "Test range"
    
    def test_add_hash_constraint(self):
        """Test adding hash constraint."""
        builder = CircuitBuilder("test_circuit")
        
        builder.add_variable("input_var", "bytes", is_public=True)
        builder.add_variable("output_var", "bytes", is_public=False)
        
        builder.add_hash_constraint("input_var", "output_var", "sha256", "Test hash")
        
        assert builder.constraint_system.get_constraint_count() == 1
        constraint = builder.constraint_system.constraints[0]
        assert constraint.constraint_type == ConstraintType.HASH
        assert constraint.variables == ["input_var", "output_var"]
        assert constraint.parameters == {"algorithm": "sha256"}
        assert constraint.description == "Test hash"
    
    def test_build_circuit(self):
        """Test building circuit from builder."""
        builder = CircuitBuilder("test_circuit")
        
        builder.add_variable("var1", "bytes", is_public=True)
        builder.add_variable("var2", "bytes", is_public=False)
        builder.add_equality_constraint("var1", "var2")
        
        circuit = builder.build()
        
        assert isinstance(circuit, BuiltCircuit)
        assert circuit.circuit_id == "test_circuit"
        assert circuit.is_built
        assert circuit.constraint_system.get_variable_count() == 2
        assert circuit.constraint_system.get_constraint_count() == 1
    
    def test_build_invalid_circuit(self):
        """Test building invalid circuit."""
        builder = CircuitBuilder("test_circuit")
        
        # Add constraint with undefined variable
        constraint = Constraint(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["undefined_var"]
        )
        builder.constraint_system.constraints.append(constraint)
        
        with pytest.raises(ValueError, match="Invalid constraint system"):
            builder.build()


class TestBuiltCircuit:
    """Test BuiltCircuit."""
    
    def test_built_circuit_creation(self):
        """Test built circuit creation."""
        system = ConstraintSystem()
        system.add_variable("public_var", "bytes", is_public=True)
        system.add_variable("private_var", "bytes", is_public=False)
        
        circuit = BuiltCircuit("test_circuit", system)
        
        assert circuit.circuit_id == "test_circuit"
        assert circuit.is_built
        assert circuit.constraint_system == system
    
    def test_generate_witness(self):
        """Test witness generation."""
        system = ConstraintSystem()
        system.add_variable("public_var", "bytes", is_public=True)
        system.add_variable("private_var", "bytes", is_public=False)
        
        circuit = BuiltCircuit("test_circuit", system)
        
        public_inputs = PublicInputs()
        public_inputs.add_input("public_var", b"public_data")
        
        private_inputs = PrivateInputs()
        private_inputs.add_input("private_var", b"private_data")
        
        witness = circuit.generate_witness(public_inputs, private_inputs)
        
        assert witness.get_value("public_var") == b"public_data"
        assert witness.get_value("private_var") == b"private_data"
    
    def test_verify_witness(self):
        """Test witness verification."""
        system = ConstraintSystem()
        system.add_variable("public_var", "bytes", is_public=True)
        system.add_variable("private_var", "bytes", is_public=False)
        
        circuit = BuiltCircuit("test_circuit", system)
        
        # Valid witness
        witness = Witness()
        witness.set_value("public_var", b"public_data", is_public=True)
        witness.set_value("private_var", b"private_data", is_public=False)
        
        assert circuit.verify_witness(witness) is True
        
        # Invalid witness (missing variable)
        invalid_witness = Witness()
        invalid_witness.set_value("public_var", b"public_data", is_public=True)
        # Missing private_var
        
        assert circuit.verify_witness(invalid_witness) is False
    
    def test_verify_equality_constraint(self):
        """Test verifying equality constraint."""
        system = ConstraintSystem()
        system.add_variable("var1", "bytes", is_public=True)
        system.add_variable("var2", "bytes", is_public=False)
        
        constraint = Constraint(
            constraint_id="equality_constraint",
            constraint_type=ConstraintType.EQUALITY,
            variables=["var1", "var2"]
        )
        system.add_constraint(constraint)
        
        circuit = BuiltCircuit("test_circuit", system)
        
        # Valid witness (equal values)
        witness = Witness()
        witness.set_value("var1", b"same_data", is_public=True)
        witness.set_value("var2", b"same_data", is_public=False)
        
        assert circuit.verify_witness(witness) is True
        
        # Invalid witness (different values)
        invalid_witness = Witness()
        invalid_witness.set_value("var1", b"data1", is_public=True)
        invalid_witness.set_value("var2", b"data2", is_public=False)
        
        assert circuit.verify_witness(invalid_witness) is False
    
    def test_verify_range_constraint(self):
        """Test verifying range constraint."""
        system = ConstraintSystem()
        system.add_variable("var1", "int", is_public=True)
        
        constraint = Constraint(
            constraint_id="range_constraint",
            constraint_type=ConstraintType.RANGE,
            variables=["var1"],
            parameters={"min": 10, "max": 100}
        )
        system.add_constraint(constraint)
        
        circuit = BuiltCircuit("test_circuit", system)
        
        # Valid witness (within range)
        witness = Witness()
        witness.set_value("var1", (50).to_bytes(8, 'little'), is_public=True)
        
        assert circuit.verify_witness(witness) is True
        
        # Invalid witness (outside range)
        invalid_witness = Witness()
        invalid_witness.set_value("var1", (5).to_bytes(8, 'little'), is_public=True)
        
        assert circuit.verify_witness(invalid_witness) is False
    
    def test_verify_hash_constraint(self):
        """Test verifying hash constraint."""
        system = ConstraintSystem()
        system.add_variable("input_var", "bytes", is_public=True)
        system.add_variable("output_var", "bytes", is_public=False)
        
        constraint = Constraint(
            constraint_id="hash_constraint",
            constraint_type=ConstraintType.HASH,
            variables=["input_var", "output_var"],
            parameters={"algorithm": "sha256"}
        )
        system.add_constraint(constraint)
        
        circuit = BuiltCircuit("test_circuit", system)
        
        # Valid witness (correct hash)
        input_data = b"test_input"
        expected_hash = hashlib.sha256(input_data).digest()
        
        witness = Witness()
        witness.set_value("input_var", input_data, is_public=True)
        witness.set_value("output_var", expected_hash, is_public=False)
        
        assert circuit.verify_witness(witness) is True
        
        # Invalid witness (incorrect hash)
        invalid_witness = Witness()
        invalid_witness.set_value("input_var", input_data, is_public=True)
        invalid_witness.set_value("output_var", b"wrong_hash", is_public=False)
        
        assert circuit.verify_witness(invalid_witness) is False


if __name__ == "__main__":
    pytest.main([__file__])
