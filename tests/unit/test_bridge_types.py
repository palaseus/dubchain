"""
Unit tests for bridge types module.
"""

import pytest

from dubchain.bridge.bridge_types import AssetType, BridgeStatus, BridgeType, ChainType


class TestBridgeType:
    """Test BridgeType enum."""

    def test_bridge_type_values(self):
        """Test that BridgeType has expected values."""
        assert BridgeType.LOCK_AND_MINT.value == "lock_and_mint"
        assert BridgeType.BURN_AND_MINT.value == "burn_and_mint"
        assert BridgeType.ATOMIC_SWAP.value == "atomic_swap"
        assert BridgeType.RELAY_CHAIN.value == "relay_chain"
        assert BridgeType.SIDECHAIN.value == "sidechain"


class TestBridgeStatus:
    """Test BridgeStatus enum."""

    def test_bridge_status_values(self):
        """Test that BridgeStatus has expected values."""
        assert BridgeStatus.ACTIVE.value == "active"
        assert BridgeStatus.INACTIVE.value == "inactive"
        assert BridgeStatus.MAINTENANCE.value == "maintenance"
        assert BridgeStatus.SUSPENDED.value == "suspended"
        assert BridgeStatus.ERROR.value == "error"


class TestChainType:
    """Test ChainType enum."""

    def test_chain_type_values(self):
        """Test that ChainType has expected values."""
        assert ChainType.MAINNET.value == "mainnet"
        assert ChainType.TESTNET.value == "testnet"
        assert ChainType.SIDECHAIN.value == "sidechain"
        assert ChainType.LAYER2.value == "layer2"
        assert ChainType.PRIVATE.value == "private"


class TestAssetType:
    """Test AssetType enum."""

    def test_asset_type_values(self):
        """Test that AssetType has expected values."""
        assert AssetType.NATIVE.value == "native"
        assert AssetType.ERC20.value == "erc20"
        assert AssetType.ERC721.value == "erc721"
        assert AssetType.ERC1155.value == "erc1155"
        assert AssetType.CUSTOM.value == "custom"
