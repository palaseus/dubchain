"""
Universal Data Types for ML

This module provides universal data types for blockchain ML operations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import time

class ChainType(Enum):
    """Blockchain chain types."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    POLYGON = "polygon"
    BSC = "bsc"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"

class TokenType(Enum):
    """Token types."""
    NATIVE = "native"
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    BEP20 = "bep20"
    SPL = "spl"

@dataclass
class UniversalTransaction:
    """Universal transaction representation."""
    tx_hash: str
    chain_type: ChainType
    from_address: str
    to_address: str
    value: float
    token_type: TokenType
    gas_price: Optional[float] = None
    gas_used: Optional[int] = None
    block_number: Optional[int] = None
    timestamp: float = None
    status: str = "success"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class UniversalBlock:
    """Universal block representation."""
    block_hash: str
    chain_type: ChainType
    block_number: int
    timestamp: float
    transactions: List[UniversalTransaction]
    gas_limit: Optional[int] = None
    gas_used: Optional[int] = None
    difficulty: Optional[float] = None
    size: Optional[int] = None

@dataclass
class UniversalAddress:
    """Universal address representation."""
    address: str
    chain_type: ChainType
    balance: float
    token_type: TokenType
    is_contract: bool = False
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = time.time()
        if self.last_seen is None:
            self.last_seen = time.time()

__all__ = [
    "ChainType",
    "TokenType", 
    "UniversalTransaction",
    "UniversalBlock",
    "UniversalAddress",
]
