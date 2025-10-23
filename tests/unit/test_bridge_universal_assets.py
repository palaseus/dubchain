"""
Unit tests for universal assets module.
"""

import logging

logger = logging.getLogger(__name__)
from unittest.mock import Mock, patch

import pytest

from dubchain.bridge.universal_assets import (
    AssetConverter,
    AssetRegistry,
    UniversalAsset,
)


class TestUniversalAsset:
    """Test UniversalAsset class."""

    def test_universal_asset_creation(self):
        """Test UniversalAsset creation."""
        asset = UniversalAsset(
            asset_id="asset_1",
            name="Test Asset",
            symbol="TEST",
            decimals=18,
            total_supply=1000000,
        )

        assert asset.asset_id == "asset_1"
        assert asset.name == "Test Asset"
        assert asset.symbol == "TEST"
        assert asset.decimals == 18
        assert asset.total_supply == 1000000

    def test_universal_asset_serialization(self):
        """Test UniversalAsset serialization."""
        asset = UniversalAsset(
            asset_id="asset_1",
            name="Test Asset",
            symbol="TEST",
            decimals=18,
            total_supply=1000000,
        )

        data = asset.to_dict()
        assert isinstance(data, dict)
        assert data["asset_id"] == "asset_1"
        assert data["name"] == "Test Asset"

    def test_universal_asset_defaults(self):
        """Test UniversalAsset default values."""
        asset = UniversalAsset(
            asset_id="asset_1",
            name="Test Asset",
            symbol="TEST",
            decimals=18,
            total_supply=1000000,
        )

        assert asset.supported_chains == []
        assert asset.created_at > 0


class TestAssetRegistry:
    """Test AssetRegistry class."""

    def test_asset_registry_creation(self):
        """Test AssetRegistry creation."""
        registry = AssetRegistry()
        assert registry is not None
        assert hasattr(registry, "assets")

    def test_register_asset(self):
        """Test registering an asset."""
        registry = AssetRegistry()
        asset = UniversalAsset(
            asset_id="asset_1",
            name="Test Asset",
            symbol="TEST",
            decimals=18,
            total_supply=1000000,
        )

        registry.register_asset(asset)
        assert "asset_1" in registry.assets

    def test_get_asset(self):
        """Test getting an asset."""
        registry = AssetRegistry()
        asset = UniversalAsset(
            asset_id="asset_1",
            name="Test Asset",
            symbol="TEST",
            decimals=18,
            total_supply=1000000,
        )

        registry.register_asset(asset)
        retrieved_asset = registry.get_asset("asset_1")
        assert retrieved_asset == asset

    def test_get_asset_not_found(self):
        """Test getting non-existent asset."""
        registry = AssetRegistry()
        asset = registry.get_asset("non_existent")
        assert asset is None

    def test_list_assets(self):
        """Test listing assets."""
        registry = AssetRegistry()
        asset1 = UniversalAsset(
            asset_id="asset_1",
            name="Test Asset 1",
            symbol="TEST1",
            decimals=18,
            total_supply=1000000,
        )
        asset2 = UniversalAsset(
            asset_id="asset_2",
            name="Test Asset 2",
            symbol="TEST2",
            decimals=18,
            total_supply=2000000,
        )

        registry.register_asset(asset1)
        registry.register_asset(asset2)

        # Access assets directly through the assets dict
        assert len(registry.assets) == 2
        assert "asset_1" in registry.assets
        assert "asset_2" in registry.assets
        assert registry.assets["asset_1"] == asset1
        assert registry.assets["asset_2"] == asset2

    def test_register_duplicate_asset(self):
        """Test registering duplicate asset."""
        registry = AssetRegistry()
        asset1 = UniversalAsset(
            asset_id="asset_1",
            name="Test Asset 1",
            symbol="TEST1",
            decimals=18,
            total_supply=1000000,
        )
        asset2 = UniversalAsset(
            asset_id="asset_1",  # Same ID
            name="Test Asset 2",
            symbol="TEST2",
            decimals=18,
            total_supply=2000000,
        )

        result1 = registry.register_asset(asset1)
        result2 = registry.register_asset(asset2)

        assert result1 is True
        assert result2 is False  # Should fail for duplicate
        assert len(registry.assets) == 1
        assert registry.assets["asset_1"] == asset1


class TestAssetConverter:
    """Test AssetConverter class."""

    def test_asset_converter_creation(self):
        """Test AssetConverter creation."""
        converter = AssetConverter()
        assert converter is not None
        assert hasattr(converter, "conversion_rates")

    def test_convert_amount(self):
        """Test asset amount conversion."""
        converter = AssetConverter()
        converter.set_conversion_rate("asset_1", "chain_1", 1.0)
        converter.set_conversion_rate("asset_1", "chain_2", 2.0)

        result = converter.convert_amount("asset_1", 100, "chain_1", "chain_2")
        assert result == 50  # 100 * 1.0 / 2.0

    def test_set_conversion_rate(self):
        """Test setting conversion rate."""
        converter = AssetConverter()
        converter.set_conversion_rate("asset_1", "chain_1", 2.0)

        assert "asset_1" in converter.conversion_rates
        assert converter.conversion_rates["asset_1"]["chain_1"] == 2.0

    def test_convert_amount_same_chain(self):
        """Test converting amount on same chain."""
        converter = AssetConverter()
        converter.set_conversion_rate("asset_1", "chain_1", 1.0)

        result = converter.convert_amount("asset_1", 100, "chain_1", "chain_1")
        assert result == 100

    def test_convert_amount_no_rates(self):
        """Test converting amount with no conversion rates."""
        converter = AssetConverter()

        result = converter.convert_amount("asset_1", 100, "chain_1", "chain_2")
        assert result == 100  # Returns original amount when no rates set
