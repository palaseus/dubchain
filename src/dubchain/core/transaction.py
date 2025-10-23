"""
Transaction system for GodChain.

Implements a UTXO-based transaction model similar to Bitcoin, with support for
smart contracts and advanced features.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..crypto.hashing import Hash, SHA256Hasher
from ..crypto.signatures import PrivateKey, PublicKey, Signature


class TransactionType(Enum):
    """Types of transactions."""

    REGULAR = "regular"
    COINBASE = "coinbase"
    CONTRACT_DEPLOY = "contract_deploy"
    CONTRACT_CALL = "contract_call"
    MULTISIG = "multisig"


@dataclass(frozen=True)
class TransactionInput:
    """Input to a transaction (UTXO reference)."""

    previous_tx_hash: Hash
    output_index: int
    signature: Optional[Signature] = None
    public_key: Optional[PublicKey] = None
    script_sig: Optional[bytes] = None  # For custom scripts

    def __post_init__(self) -> None:
        if self.output_index < 0:
            raise ValueError("Output index must be non-negative")

    def to_bytes(self) -> bytes:
        """Serialize transaction input to bytes."""
        data = self.previous_tx_hash.value + self.output_index.to_bytes(
            4, byteorder="big"
        )

        if self.signature:
            data += self.signature.to_bytes()

        if self.public_key:
            data += self.public_key.to_bytes(compressed=True)

        if self.script_sig:
            data += len(self.script_sig).to_bytes(4, byteorder="big")
            data += self.script_sig

        return data

    def get_signature_hash(self, transaction: "Transaction", input_index: int) -> Hash:
        """Get the hash that should be signed for this input."""
        # Create a copy of the transaction with empty signatures
        tx_copy = transaction.copy_without_signatures()

        # Set the script_sig for this input
        tx_copy.inputs[input_index] = TransactionInput(
            previous_tx_hash=self.previous_tx_hash,
            output_index=self.output_index,
            script_sig=self.script_sig,
        )

        return tx_copy.get_hash()


@dataclass(frozen=True)
class TransactionOutput:
    """Output from a transaction (UTXO)."""

    amount: int
    recipient_address: str
    script_pubkey: Optional[bytes] = None  # For custom scripts
    contract_address: Optional[str] = None  # For contract calls
    data: Optional[bytes] = None  # For contract data

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Amount must be non-negative")

    def to_bytes(self) -> bytes:
        """Serialize transaction output to bytes."""
        data = (
            self.amount.to_bytes(8, byteorder="big")
            + len(self.recipient_address).to_bytes(4, byteorder="big")
            + self.recipient_address.encode("utf-8")
        )

        if self.script_pubkey:
            data += len(self.script_pubkey).to_bytes(4, byteorder="big")
            data += self.script_pubkey

        if self.contract_address:
            data += len(self.contract_address).to_bytes(4, byteorder="big")
            data += self.contract_address.encode("utf-8")

        if self.data:
            data += len(self.data).to_bytes(4, byteorder="big")
            data += self.data

        return data


@dataclass(frozen=True)
class UTXO:
    """Unspent Transaction Output."""

    tx_hash: Hash
    output_index: int
    amount: int
    recipient_address: str
    script_pubkey: Optional[bytes] = None
    contract_address: Optional[str] = None
    data: Optional[bytes] = None
    block_height: int = 0

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Amount must be non-negative")
        if self.output_index < 0:
            raise ValueError("Output index must be non-negative")
        if self.block_height < 0:
            raise ValueError("Block height must be non-negative")

    def to_transaction_output(self) -> TransactionOutput:
        """Convert UTXO to TransactionOutput."""
        return TransactionOutput(
            amount=self.amount,
            recipient_address=self.recipient_address,
            script_pubkey=self.script_pubkey,
            contract_address=self.contract_address,
            data=self.data,
        )

    def get_key(self) -> str:
        """Get unique key for this UTXO."""
        return f"{self.tx_hash.to_hex()}:{self.output_index}"


@dataclass(frozen=True)
class Transaction:
    """A blockchain transaction."""

    inputs: List[TransactionInput]
    outputs: List[TransactionOutput]
    transaction_type: TransactionType = TransactionType.REGULAR
    timestamp: int = field(default_factory=lambda: int(time.time()))
    nonce: int = 0
    gas_limit: int = 21000  # For contract transactions
    gas_price: int = 1  # In wei
    data: Optional[bytes] = None
    contract_code: Optional[bytes] = None  # For contract deployment

    def __post_init__(self) -> None:
        if not self.outputs:
            raise ValueError("Transaction must have at least one output")

        if self.transaction_type != TransactionType.COINBASE and not self.inputs:
            raise ValueError("Non-coinbase transactions must have inputs")

        if self.transaction_type == TransactionType.COINBASE and self.inputs:
            raise ValueError("Coinbase transactions cannot have inputs")

        if self.gas_limit < 0 or self.gas_price < 0:
            raise ValueError("Gas limit and price must be non-negative")

    def get_hash(self) -> Hash:
        """Get the hash of this transaction."""
        return SHA256Hasher.hash(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize transaction to bytes."""
        data = (
            self.transaction_type.value.encode("utf-8")
            + self.timestamp.to_bytes(8, byteorder="big")
            + self.nonce.to_bytes(8, byteorder="big")
            + self.gas_limit.to_bytes(8, byteorder="big")
            + self.gas_price.to_bytes(8, byteorder="big")
        )

        # Add inputs
        data += len(self.inputs).to_bytes(4, byteorder="big")
        for input_tx in self.inputs:
            data += input_tx.to_bytes()

        # Add outputs
        data += len(self.outputs).to_bytes(4, byteorder="big")
        for output in self.outputs:
            data += output.to_bytes()

        # Add optional data
        if self.data:
            data += len(self.data).to_bytes(4, byteorder="big")
            data += self.data

        if self.contract_code:
            data += len(self.contract_code).to_bytes(4, byteorder="big")
            data += self.contract_code

        return data

    def copy_without_signatures(self) -> "Transaction":
        """Create a copy of the transaction without signatures."""
        new_inputs = []
        for input_tx in self.inputs:
            new_input = TransactionInput(
                previous_tx_hash=input_tx.previous_tx_hash,
                output_index=input_tx.output_index,
                script_sig=input_tx.script_sig,
            )
            new_inputs.append(new_input)

        return Transaction(
            inputs=new_inputs,
            outputs=self.outputs,
            transaction_type=self.transaction_type,
            timestamp=self.timestamp,
            nonce=self.nonce,
            gas_limit=self.gas_limit,
            gas_price=self.gas_price,
            data=self.data,
            contract_code=self.contract_code,
        )

    def sign_input(self, input_index: int, private_key: PrivateKey) -> "Transaction":
        """Sign a specific input of the transaction."""
        if input_index >= len(self.inputs):
            raise ValueError("Input index out of range")

        input_tx = self.inputs[input_index]
        signature_hash = input_tx.get_signature_hash(self, input_index)
        signature = private_key.sign(signature_hash)
        public_key = private_key.get_public_key()

        # Create new input with signature
        new_input = TransactionInput(
            previous_tx_hash=input_tx.previous_tx_hash,
            output_index=input_tx.output_index,
            signature=signature,
            public_key=public_key,
            script_sig=input_tx.script_sig,
        )

        # Create new transaction with signed input
        new_inputs = list(self.inputs)
        new_inputs[input_index] = new_input

        return Transaction(
            inputs=new_inputs,
            outputs=self.outputs,
            transaction_type=self.transaction_type,
            timestamp=self.timestamp,
            nonce=self.nonce,
            gas_limit=self.gas_limit,
            gas_price=self.gas_price,
            data=self.data,
            contract_code=self.contract_code,
        )

    def verify_signature(self, input_index: int, utxo: UTXO) -> bool:
        """Verify the signature of a specific input."""
        if input_index >= len(self.inputs):
            return False

        input_tx = self.inputs[input_index]
        if not input_tx.signature or not input_tx.public_key:
            return False

        # Verify the signature
        signature_hash = input_tx.get_signature_hash(self, input_index)
        return input_tx.public_key.verify(input_tx.signature, signature_hash)

    def get_total_input_amount(self, utxos: Dict[str, UTXO]) -> int:
        """Get the total amount of all inputs."""
        total = 0
        for input_tx in self.inputs:
            utxo_key = f"{input_tx.previous_tx_hash.to_hex()}:{input_tx.output_index}"
            if utxo_key in utxos:
                total += utxos[utxo_key].amount
        return total

    def get_total_output_amount(self) -> int:
        """Get the total amount of all outputs."""
        return sum(output.amount for output in self.outputs)

    def get_fee(self, utxos: Dict[str, UTXO]) -> int:
        """Get the transaction fee."""
        if self.transaction_type == TransactionType.COINBASE:
            return 0

        input_amount = self.get_total_input_amount(utxos)
        output_amount = self.get_total_output_amount()
        return input_amount - output_amount

    def is_valid(self, utxos: Dict[str, UTXO]) -> bool:
        """Check if the transaction is valid."""
        try:
            # Check basic structure
            if not self.outputs:
                return False

            if self.transaction_type != TransactionType.COINBASE and not self.inputs:
                return False

            if self.transaction_type == TransactionType.COINBASE and self.inputs:
                return False

            # For non-coinbase transactions, verify signatures and amounts
            if self.transaction_type != TransactionType.COINBASE:
                # Check that all inputs exist
                for input_tx in self.inputs:
                    utxo_key = (
                        f"{input_tx.previous_tx_hash.to_hex()}:{input_tx.output_index}"
                    )
                    if utxo_key not in utxos:
                        return False

                # Verify all signatures
                for i, input_tx in enumerate(self.inputs):
                    utxo_key = (
                        f"{input_tx.previous_tx_hash.to_hex()}:{input_tx.output_index}"
                    )
                    utxo = utxos[utxo_key]

                    if not self.verify_signature(i, utxo):
                        return False

                # Check that input amount >= output amount
                input_amount = self.get_total_input_amount(utxos)
                output_amount = self.get_total_output_amount()

                if input_amount < output_amount:
                    return False

            return True

        except Exception:
            return False

    def get_utxos_created(self) -> List[UTXO]:
        """Get the UTXOs that this transaction creates."""
        utxos = []
        tx_hash = self.get_hash()

        for i, output in enumerate(self.outputs):
            utxo = UTXO(
                tx_hash=tx_hash,
                output_index=i,
                amount=output.amount,
                recipient_address=output.recipient_address,
                script_pubkey=output.script_pubkey,
                contract_address=output.contract_address,
                data=output.data,
            )
            utxos.append(utxo)

        return utxos

    def get_utxos_consumed(self) -> List[str]:
        """Get the keys of UTXOs that this transaction consumes."""
        return [
            f"{input_tx.previous_tx_hash.to_hex()}:{input_tx.output_index}"
            for input_tx in self.inputs
        ]

    @classmethod
    def create_coinbase(
        cls, recipient_address: str, amount: int, block_height: int
    ) -> "Transaction":
        """Create a coinbase transaction."""
        output = TransactionOutput(amount=amount, recipient_address=recipient_address)

        return cls(
            inputs=[],
            outputs=[output],
            transaction_type=TransactionType.COINBASE,
            data=f"Block {block_height}".encode("utf-8"),
        )

    @classmethod
    def create_transfer(
        cls,
        sender_private_key: PrivateKey,
        recipient_address: str,
        amount: int,
        utxos: List[UTXO],
        fee: int = 0,
    ) -> "Transaction":
        """Create a transfer transaction."""
        sender_public_key = sender_private_key.get_public_key()
        sender_address = sender_public_key.to_address()

        # Select UTXOs to spend
        selected_utxos = []
        total_amount = 0

        for utxo in utxos:
            if utxo.recipient_address == sender_address:
                selected_utxos.append(utxo)
                total_amount += utxo.amount
                if total_amount >= amount + fee:
                    break

        if total_amount < amount + fee:
            raise ValueError("Insufficient funds")

        # Create inputs
        inputs = []
        for utxo in selected_utxos:
            input_tx = TransactionInput(
                previous_tx_hash=utxo.tx_hash, output_index=utxo.output_index
            )
            inputs.append(input_tx)

        # Create outputs
        outputs = [
            TransactionOutput(amount=amount, recipient_address=recipient_address)
        ]

        # Add change output if needed
        change_amount = total_amount - amount - fee
        if change_amount > 0:
            outputs.append(
                TransactionOutput(
                    amount=change_amount, recipient_address=sender_address
                )
            )

        # Create transaction
        transaction = cls(
            inputs=inputs, outputs=outputs, transaction_type=TransactionType.REGULAR
        )

        # Sign all inputs
        for i in range(len(inputs)):
            transaction = transaction.sign_input(i, sender_private_key)

        return transaction

    def __str__(self) -> str:
        return f"Transaction({self.get_hash().to_hex()[:16]}...)"

    def __repr__(self) -> str:
        return f"Transaction(hash={self.get_hash().to_hex()}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"
