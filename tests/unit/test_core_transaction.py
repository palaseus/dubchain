"""
Unit tests for transaction system.
"""

import pytest

from dubchain.core.transaction import (
    UTXO,
    Transaction,
    TransactionInput,
    TransactionOutput,
    TransactionType,
)
from dubchain.crypto.hashing import Hash, SHA256Hasher
from dubchain.crypto.signatures import ECDSASigner, PrivateKey, PublicKey


class TestTransactionInput:
    """Test the TransactionInput class."""

    def test_transaction_input_creation(self):
        """Test creating a transaction input."""
        prev_hash = SHA256Hasher.hash("previous_tx")
        input_tx = TransactionInput(previous_tx_hash=prev_hash, output_index=0)

        assert input_tx.previous_tx_hash == prev_hash
        assert input_tx.output_index == 0
        assert input_tx.signature is None
        assert input_tx.public_key is None

    def test_transaction_input_negative_index(self):
        """Test that negative output index raises ValueError."""
        prev_hash = SHA256Hasher.hash("previous_tx")

        with pytest.raises(ValueError, match="Output index must be non-negative"):
            TransactionInput(previous_tx_hash=prev_hash, output_index=-1)

    def test_transaction_input_to_bytes(self):
        """Test serializing transaction input to bytes."""
        prev_hash = SHA256Hasher.hash("previous_tx")
        input_tx = TransactionInput(previous_tx_hash=prev_hash, output_index=1)

        data = input_tx.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_transaction_input_get_signature_hash(self):
        """Test getting signature hash for transaction input."""
        prev_hash = SHA256Hasher.hash("previous_tx")
        input_tx = TransactionInput(previous_tx_hash=prev_hash, output_index=0)

        # Create a simple transaction
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(inputs=[input_tx], outputs=[output])

        signature_hash = input_tx.get_signature_hash(transaction, 0)
        assert isinstance(signature_hash, Hash)


class TestTransactionOutput:
    """Test the TransactionOutput class."""

    def test_transaction_output_creation(self):
        """Test creating a transaction output."""
        output = TransactionOutput(amount=1000, recipient_address="recipient_address")

        assert output.amount == 1000
        assert output.recipient_address == "recipient_address"
        assert output.script_pubkey is None
        assert output.contract_address is None
        assert output.data is None

    def test_transaction_output_negative_amount(self):
        """Test that negative amount raises ValueError."""
        with pytest.raises(ValueError, match="Amount must be non-negative"):
            TransactionOutput(amount=-1000, recipient_address="recipient")

    def test_transaction_output_to_bytes(self):
        """Test serializing transaction output to bytes."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")

        data = output.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0


class TestUTXO:
    """Test the UTXO class."""

    def test_utxo_creation(self):
        """Test creating a UTXO."""
        tx_hash = SHA256Hasher.hash("transaction")
        utxo = UTXO(
            tx_hash=tx_hash, output_index=0, amount=1000, recipient_address="recipient"
        )

        assert utxo.tx_hash == tx_hash
        assert utxo.output_index == 0
        assert utxo.amount == 1000
        assert utxo.recipient_address == "recipient"
        assert utxo.block_height == 0

    def test_utxo_negative_values(self):
        """Test that negative values raise ValueError."""
        tx_hash = SHA256Hasher.hash("transaction")

        with pytest.raises(ValueError, match="Amount must be non-negative"):
            UTXO(
                tx_hash=tx_hash,
                output_index=0,
                amount=-1000,
                recipient_address="recipient",
            )

        with pytest.raises(ValueError, match="Output index must be non-negative"):
            UTXO(
                tx_hash=tx_hash,
                output_index=-1,
                amount=1000,
                recipient_address="recipient",
            )

        with pytest.raises(ValueError, match="Block height must be non-negative"):
            UTXO(
                tx_hash=tx_hash,
                output_index=0,
                amount=1000,
                recipient_address="recipient",
                block_height=-1,
            )

    def test_utxo_to_transaction_output(self):
        """Test converting UTXO to TransactionOutput."""
        tx_hash = SHA256Hasher.hash("transaction")
        utxo = UTXO(
            tx_hash=tx_hash, output_index=0, amount=1000, recipient_address="recipient"
        )

        output = utxo.to_transaction_output()
        assert isinstance(output, TransactionOutput)
        assert output.amount == utxo.amount
        assert output.recipient_address == utxo.recipient_address

    def test_utxo_get_key(self):
        """Test getting unique key for UTXO."""
        tx_hash = SHA256Hasher.hash("transaction")
        utxo = UTXO(
            tx_hash=tx_hash, output_index=1, amount=1000, recipient_address="recipient"
        )

        key = utxo.get_key()
        expected_key = f"{tx_hash.to_hex()}:1"
        assert key == expected_key


class TestTransaction:
    """Test the Transaction class."""

    def test_transaction_creation(self):
        """Test creating a transaction."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(
            inputs=[], outputs=[output], transaction_type=TransactionType.COINBASE
        )

        assert len(transaction.inputs) == 0
        assert len(transaction.outputs) == 1
        assert transaction.transaction_type == TransactionType.COINBASE

    def test_transaction_no_outputs(self):
        """Test that transaction without outputs raises ValueError."""
        with pytest.raises(
            ValueError, match="Transaction must have at least one output"
        ):
            Transaction(
                inputs=[], outputs=[], transaction_type=TransactionType.COINBASE
            )

    def test_transaction_non_coinbase_without_inputs(self):
        """Test that non-coinbase transaction without inputs raises ValueError."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")

        with pytest.raises(
            ValueError, match="Non-coinbase transactions must have inputs"
        ):
            Transaction(
                inputs=[], outputs=[output], transaction_type=TransactionType.REGULAR
            )

    def test_transaction_coinbase_with_inputs(self):
        """Test that coinbase transaction with inputs raises ValueError."""
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev"), output_index=0
        )
        output = TransactionOutput(amount=1000, recipient_address="recipient")

        with pytest.raises(
            ValueError, match="Coinbase transactions cannot have inputs"
        ):
            Transaction(
                inputs=[input_tx],
                outputs=[output],
                transaction_type=TransactionType.COINBASE,
            )

    def test_transaction_negative_gas(self):
        """Test that negative gas values raise ValueError."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")

        with pytest.raises(
            ValueError, match="Gas limit and price must be non-negative"
        ):
            Transaction(
                inputs=[],
                outputs=[output],
                transaction_type=TransactionType.COINBASE,
                gas_limit=-1,
            )

        with pytest.raises(
            ValueError, match="Gas limit and price must be non-negative"
        ):
            Transaction(
                inputs=[],
                outputs=[output],
                transaction_type=TransactionType.COINBASE,
                gas_price=-1,
            )

    def test_transaction_get_hash(self):
        """Test getting transaction hash."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(
            inputs=[], outputs=[output], transaction_type=TransactionType.COINBASE
        )

        tx_hash = transaction.get_hash()
        assert isinstance(tx_hash, Hash)

    def test_transaction_to_bytes(self):
        """Test serializing transaction to bytes."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(
            inputs=[], outputs=[output], transaction_type=TransactionType.COINBASE
        )

        data = transaction.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_transaction_copy_without_signatures(self):
        """Test copying transaction without signatures."""
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev"), output_index=0
        )
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )

        copy_tx = transaction.copy_without_signatures()
        assert len(copy_tx.inputs) == 1
        assert copy_tx.inputs[0].signature is None
        assert copy_tx.inputs[0].public_key is None

    def test_transaction_sign_input(self):
        """Test signing a transaction input."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create UTXO
        utxo = UTXO(
            tx_hash=SHA256Hasher.hash("prev_tx"),
            output_index=0,
            amount=1000,
            recipient_address=public_key.to_address(),
        )

        # Create input
        input_tx = TransactionInput(
            previous_tx_hash=utxo.tx_hash, output_index=utxo.output_index
        )

        # Create output
        output = TransactionOutput(amount=900, recipient_address="recipient")

        # Create transaction
        transaction = Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )

        # Sign the input
        signed_tx = transaction.sign_input(0, private_key)

        assert signed_tx.inputs[0].signature is not None
        assert signed_tx.inputs[0].public_key is not None

    def test_transaction_verify_signature(self):
        """Test verifying transaction signature."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create UTXO
        utxo = UTXO(
            tx_hash=SHA256Hasher.hash("prev_tx"),
            output_index=0,
            amount=1000,
            recipient_address=public_key.to_address(),
        )

        # Create input
        input_tx = TransactionInput(
            previous_tx_hash=utxo.tx_hash, output_index=utxo.output_index
        )

        # Create output
        output = TransactionOutput(amount=900, recipient_address="recipient")

        # Create and sign transaction
        transaction = Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )
        signed_tx = transaction.sign_input(0, private_key)

        # Verify signature
        assert signed_tx.verify_signature(0, utxo)

    def test_transaction_get_total_amounts(self):
        """Test getting total input and output amounts."""
        # Create UTXOs with correct keys
        hash1 = SHA256Hasher.hash("tx1")
        hash2 = SHA256Hasher.hash("tx2")
        utxos = {
            f"{hash1.to_hex()}:0": UTXO(
                tx_hash=hash1, output_index=0, amount=1000, recipient_address="sender"
            ),
            f"{hash2.to_hex()}:0": UTXO(
                tx_hash=hash2, output_index=0, amount=500, recipient_address="sender"
            ),
        }

        # Create inputs
        inputs = [
            TransactionInput(previous_tx_hash=SHA256Hasher.hash("tx1"), output_index=0),
            TransactionInput(previous_tx_hash=SHA256Hasher.hash("tx2"), output_index=0),
        ]

        # Create outputs
        outputs = [
            TransactionOutput(amount=1200, recipient_address="recipient"),
            TransactionOutput(amount=250, recipient_address="sender"),
        ]

        transaction = Transaction(
            inputs=inputs, outputs=outputs, transaction_type=TransactionType.REGULAR
        )

        assert transaction.get_total_input_amount(utxos) == 1500
        assert transaction.get_total_output_amount() == 1450
        assert transaction.get_fee(utxos) == 50

    def test_transaction_is_valid(self):
        """Test transaction validation."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create UTXO
        utxo = UTXO(
            tx_hash=SHA256Hasher.hash("prev_tx"),
            output_index=0,
            amount=1000,
            recipient_address=public_key.to_address(),
        )

        # Create valid transaction
        input_tx = TransactionInput(
            previous_tx_hash=utxo.tx_hash, output_index=utxo.output_index
        )
        output = TransactionOutput(amount=900, recipient_address="recipient")

        transaction = Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )
        signed_tx = transaction.sign_input(0, private_key)

        utxos = {utxo.get_key(): utxo}
        assert signed_tx.is_valid(utxos)

    def test_transaction_create_coinbase(self):
        """Test creating coinbase transaction."""
        coinbase_tx = Transaction.create_coinbase(
            recipient_address="miner", amount=50000000, block_height=100
        )

        assert coinbase_tx.transaction_type == TransactionType.COINBASE
        assert len(coinbase_tx.inputs) == 0
        assert len(coinbase_tx.outputs) == 1
        assert coinbase_tx.outputs[0].amount == 50000000
        assert coinbase_tx.outputs[0].recipient_address == "miner"

    def test_transaction_create_transfer(self):
        """Test creating transfer transaction."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create UTXOs
        utxos = [
            UTXO(
                tx_hash=SHA256Hasher.hash("tx1"),
                output_index=0,
                amount=1000,
                recipient_address=public_key.to_address(),
            ),
            UTXO(
                tx_hash=SHA256Hasher.hash("tx2"),
                output_index=0,
                amount=500,
                recipient_address=public_key.to_address(),
            ),
        ]

        # Create transfer transaction
        transfer_tx = Transaction.create_transfer(
            sender_private_key=private_key,
            recipient_address="recipient",
            amount=1200,
            utxos=utxos,
            fee=100,
        )

        assert transfer_tx.transaction_type == TransactionType.REGULAR
        assert len(transfer_tx.inputs) == 2
        assert len(transfer_tx.outputs) == 2  # recipient + change
        assert transfer_tx.outputs[0].amount == 1200
        assert transfer_tx.outputs[0].recipient_address == "recipient"

    def test_transaction_insufficient_funds(self):
        """Test creating transfer with insufficient funds."""
        private_key = PrivateKey.generate()
        public_key = private_key.get_public_key()

        # Create insufficient UTXOs
        utxos = [
            UTXO(
                tx_hash=SHA256Hasher.hash("tx1"),
                output_index=0,
                amount=1000,
                recipient_address=public_key.to_address(),
            )
        ]

        with pytest.raises(ValueError, match="Insufficient funds"):
            Transaction.create_transfer(
                sender_private_key=private_key,
                recipient_address="recipient",
                amount=2000,  # More than available
                utxos=utxos,
                fee=100,
            )

    def test_transaction_get_utxos_created(self):
        """Test getting UTXOs created by transaction."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(
            inputs=[], outputs=[output], transaction_type=TransactionType.COINBASE
        )

        utxos = transaction.get_utxos_created()
        assert len(utxos) == 1
        assert utxos[0].amount == 1000
        assert utxos[0].recipient_address == "recipient"

    def test_transaction_get_utxos_consumed(self):
        """Test getting UTXOs consumed by transaction."""
        input_tx = TransactionInput(
            previous_tx_hash=SHA256Hasher.hash("prev_tx"), output_index=1
        )
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(
            inputs=[input_tx],
            outputs=[output],
            transaction_type=TransactionType.REGULAR,
        )

        consumed_keys = transaction.get_utxos_consumed()
        assert len(consumed_keys) == 1
        expected_key = f"{SHA256Hasher.hash('prev_tx').to_hex()}:1"
        assert consumed_keys[0] == expected_key

    def test_transaction_string_representation(self):
        """Test string representation of transaction."""
        output = TransactionOutput(amount=1000, recipient_address="recipient")
        transaction = Transaction(
            inputs=[], outputs=[output], transaction_type=TransactionType.COINBASE
        )

        str_repr = str(transaction)
        assert "Transaction" in str_repr

        repr_str = repr(transaction)
        assert "Transaction" in repr_str
        assert "inputs=0" in repr_str
        assert "outputs=1" in repr_str
