"""
Universal asset management for DubChain.

This module provides universal asset management across chains.
"""

import logging

logger = logging.getLogger(__name__)
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class UniversalAsset:
    """Universal asset representation."""

    asset_id: str
    name: str
    symbol: str
    decimals: int
    total_supply: int
    supported_chains: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "symbol": self.symbol,
            "decimals": self.decimals,
            "total_supply": self.total_supply,
            "supported_chains": self.supported_chains,
            "created_at": self.created_at,
        }


@dataclass
class AssetRegistry:
    """Registry for universal assets."""

    assets: Dict[str, UniversalAsset] = field(default_factory=dict)

    def register_asset(self, asset: UniversalAsset) -> bool:
        """Register universal asset."""
        if asset.asset_id in self.assets:
            return False
        self.assets[asset.asset_id] = asset
        return True

    def get_asset(self, asset_id: str) -> Optional[UniversalAsset]:
        """Get universal asset."""
        return self.assets.get(asset_id)


@dataclass
class AssetConverter:
    """Converts assets between chains."""

    conversion_rates: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def set_conversion_rate(self, asset_id: str, chain_id: str, rate: float) -> None:
        """Set conversion rate for asset on chain."""
        if asset_id not in self.conversion_rates:
            self.conversion_rates[asset_id] = {}
        self.conversion_rates[asset_id][chain_id] = rate

    def convert_amount(
        self, asset_id: str, amount: int, from_chain: str, to_chain: str
    ) -> int:
        """Convert amount between chains."""
        if asset_id not in self.conversion_rates:
            return amount

        from_rate = self.conversion_rates[asset_id].get(from_chain, 1.0)
        to_rate = self.conversion_rates[asset_id].get(to_chain, 1.0)

        return int(amount * from_rate / to_rate)


@dataclass
class AssetValidator:
    """Validates universal assets."""

    validation_rules: Dict[str, Any] = field(default_factory=dict)

    def validate_asset(self, asset: UniversalAsset) -> bool:
        """Validate universal asset."""
        if not asset.asset_id or not asset.name or not asset.symbol:
            return False
        if asset.decimals < 0 or asset.total_supply < 0:
            return False
        return True
