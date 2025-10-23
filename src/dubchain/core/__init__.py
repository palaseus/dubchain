"""
Core blockchain components for GodChain.

This module provides the fundamental blockchain data structures and logic:
- Block and BlockHeader
- Transaction and UTXO
- Blockchain state management
- Consensus mechanisms
"""

import logging

logger = logging.getLogger(__name__)
from .block import Block, BlockHeader
from .blockchain import Blockchain
from .consensus import ConsensusEngine, ProofOfWork
from .transaction import UTXO, Transaction, TransactionInput, TransactionOutput

__all__ = [
    "Block",
    "BlockHeader",
    "Transaction",
    "UTXO",
    "TransactionInput",
    "TransactionOutput",
    "Blockchain",
    "ProofOfWork",
    "ConsensusEngine",
]
