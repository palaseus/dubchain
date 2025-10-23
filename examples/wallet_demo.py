#!/usr/bin/env python3
logger = logging.getLogger(__name__)
"""
Advanced Wallet System Demo for GodChain.

This demo showcases the sophisticated wallet system including HD wallets,
multi-signature wallets, encryption, and wallet management.
"""

import logging
import os
import tempfile
from dubchain.wallet import (
    WalletManager, WalletManagerConfig, HDWallet, MultisigWallet, MultisigConfig, 
    MultisigType, EncryptionConfig, Language
)


def main():
    """Run the wallet system demo."""
    logger.info("🚀 GodChain Advanced Wallet System Demo")
    logger.info("=" * 50)
    
    # Create temporary directory for wallet storage
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"📁 Using temporary storage: {temp_dir}")
        
        # Configure wallet manager
        config = WalletManagerConfig(
            storage_path=temp_dir,
            default_network="mainnet",
            encryption_enabled=True,
            mnemonic_language=Language.ENGLISH,
            auto_backup=True
        )
        
        logger.info("\n🔧 Initializing Wallet Manager...")
        manager = WalletManager(config)
        logger.info(f"✅ Wallet Manager initialized")
        logger.info(f"   - Storage path: {manager.config.storage_path}")
        logger.info(f"   - Encryption enabled: {manager.config.encryption_enabled}")
        logger.info(f"   - Max wallets: {manager.config.max_wallets}")
        
        # Create HD wallet
        logger.info("\n🔑 Creating HD Wallet...")
        hd_wallet_id = manager.create_hd_wallet(
            name="My HD Wallet",
            password="secure_password_123!",
            network="mainnet"
        )
        logger.info(f"✅ HD Wallet created: {hd_wallet_id}")
        
        # Load and interact with HD wallet
        logger.info("\n📱 Loading HD Wallet...")
        hd_wallet = manager.load_wallet(hd_wallet_id, "secure_password_123!")
        logger.info(f"✅ HD Wallet loaded: {hd_wallet}")
        logger.info(f"   - Name: {hd_wallet.metadata.name}")
        logger.info(f"   - Network: {hd_wallet.metadata.network}")
        logger.info(f"   - Accounts: {len(hd_wallet.accounts)}")
        logger.info(f"   - Current account: {hd_wallet.current_account_index}")
        
        # Create additional accounts
        logger.info("\n👥 Creating additional accounts...")
        account1 = hd_wallet.create_account(label="Savings Account")
        account2 = hd_wallet.create_account(label="Trading Account")
        logger.info(f"✅ Created accounts:")
        logger.info(f"   - Account 1: {account1.label}")
        logger.info(f"   - Account 2: {account2.label}")
        
        # Generate addresses
        logger.info("\n📍 Generating addresses...")
        addresses = hd_wallet.get_account_addresses(0, 3)
        logger.info(f"✅ Generated {len(addresses)} addresses for account 0:")
        for i, address in enumerate(addresses):
            logger.info(f"   - Address {i+1}: {address}")
        
        # Update balances
        logger.info("\n💰 Updating balances...")
        hd_wallet.update_balance(0, 1000000)  # 1M satoshis
        hd_wallet.update_balance(1, 500000)   # 500K satoshis
        hd_wallet.update_balance(2, 250000)   # 250K satoshis
        
        total_balance = hd_wallet.get_total_balance()
        logger.info(f"✅ Balances updated:")
        logger.info(f"   - Account 0: {hd_wallet.get_balance(0):,} satoshis")
        logger.info(f"   - Account 1: {hd_wallet.get_balance(1):,} satoshis")
        logger.info(f"   - Account 2: {hd_wallet.get_balance(2):,} satoshis")
        logger.info(f"   - Total balance: {total_balance:,} satoshis")
        
        # Create multisig wallet
        logger.info("\n🔐 Creating Multi-Signature Wallet...")
        multisig_config = MultisigConfig(
            multisig_type=MultisigType.M_OF_N,
            required_signatures=2,
            total_participants=3,
            timeout_seconds=3600
        )
        
        multisig_wallet_id = manager.create_multisig_wallet(
            name="Corporate Multisig",
            config=multisig_config,
            password="corporate_password_456!"
        )
        logger.info(f"✅ Multisig Wallet created: {multisig_wallet_id}")
        
        # Load and configure multisig wallet
        logger.info("\n👥 Configuring Multi-Signature Wallet...")
        multisig_wallet = manager.load_wallet(multisig_wallet_id, "corporate_password_456!")
        
        # Add participants (simulate with generated keys)
        from dubchain.crypto.signatures import PrivateKey
        
        participants = []
        for i in range(3):
            private_key = PrivateKey.generate()
            public_key = private_key.get_public_key()
            participant = multisig_wallet.add_participant(
                f"participant_{i+1}",
                public_key,
                weight=1
            )
            participants.append((private_key, participant))
            logger.info(f"   - Added participant {i+1}: {participant.participant_id}")
        
        # Create and sign a transaction
        logger.info("\n📝 Creating Multi-Signature Transaction...")
        transaction_data = b"Transfer 100,000 satoshis to address GC123..."
        transaction = multisig_wallet.create_transaction(transaction_data, expires_in_seconds=1800)
        logger.info(f"✅ Transaction created: {transaction.transaction_id}")
        logger.info(f"   - Required signatures: {transaction.required_signatures}")
        logger.info(f"   - Total participants: {transaction.total_participants}")
        
        # Sign transaction with first two participants
        logger.info("\n✍️ Signing Transaction...")
        for i, (private_key, participant) in enumerate(participants[:2]):
            success = multisig_wallet.sign_transaction(
                transaction.transaction_id,
                participant.participant_id,
                private_key
            )
            logger.info(f"   - Participant {i+1} signature: {'✅ Success' if success else '❌ Failed'}")
        
        # Verify transaction
        logger.info("\n🔍 Verifying Transaction...")
        is_complete = multisig_wallet.verify_transaction(transaction.transaction_id)
        logger.info(f"✅ Transaction verification: {'✅ Complete' if is_complete else '❌ Incomplete'}")
        
        if is_complete:
            logger.info(f"   - Signatures collected: {transaction.get_signature_count()}")
            logger.info(f"   - Required signatures: {transaction.required_signatures}")
        
        # Test encryption
        logger.info("\n🔒 Testing Wallet Encryption...")
        encryption = manager.encryption
        test_data = {"sensitive": "wallet_data", "balance": 1000000}
        password = "encryption_test_password"
        
        encrypted_data = encryption.encrypt_dict(test_data, password)
        decrypted_data = encryption.decrypt_dict(encrypted_data, password)
        
        logger.info(f"✅ Encryption test: {'✅ Success' if test_data == decrypted_data else '❌ Failed'}")
        logger.info(f"   - Original data: {test_data}")
        logger.info(f"   - Decrypted data: {decrypted_data}")
        
        # Test password validation
        logger.info("\n🔐 Testing Password Validation...")
        strong_password = manager.generate_strong_password(16)
        is_valid, errors = manager.validate_password(strong_password)
        strength = manager.get_password_strength(strong_password)
        
        logger.info(f"✅ Generated strong password: {strong_password}")
        logger.info(f"   - Valid: {'✅ Yes' if is_valid else '❌ No'}")
        logger.info(f"   - Strength: {strength}")
        if errors:
            logger.info(f"   - Errors: {errors}")
        
        # Wallet management
        logger.info("\n📊 Wallet Management Summary...")
        wallet_list = manager.get_wallet_list()
        manager_info = manager.get_manager_info()
        
        logger.info(f"✅ Wallet Manager Info:")
        logger.info(f"   - Total wallets: {manager_info['wallet_count']}")
        logger.info(f"   - Loaded wallets: {manager_info['loaded_wallets']}")
        logger.info(f"   - Storage path: {manager_info['storage_path']}")
        logger.info(f"   - Encryption enabled: {manager_info['encryption_enabled']}")
        
        logger.info(f"\n📋 Wallet List:")
        for wallet_info in wallet_list:
            logger.info(f"   - {wallet_info.name} ({wallet_info.wallet_type.value})")
            logger.info(f"     ID: {wallet_info.wallet_id}")
            logger.info(f"     Network: {wallet_info.network}")
            logger.info(f"     Encrypted: {'✅ Yes' if wallet_info.is_encrypted else '❌ No'}")
        
        # Export wallet backup
        logger.info("\n💾 Creating Wallet Backup...")
        backup_file = manager.backup_wallet(
            hd_wallet_id,
            backup_path=temp_dir,
            password="secure_password_123!"
        )
        logger.info(f"✅ Backup created: {os.path.basename(backup_file)}")
        
        logger.info("\n🎉 Wallet System Demo Completed Successfully!")
        logger.info("=" * 50)
        logger.info("✨ Features demonstrated:")
        logger.info("   - HD wallet creation and management")
        logger.info("   - Multi-account support")
        logger.info("   - Address generation")
        logger.info("   - Balance management")
        logger.info("   - Multi-signature wallets")
        logger.info("   - Transaction signing and verification")
        logger.info("   - Advanced encryption")
        logger.info("   - Password management")
        logger.info("   - Wallet backup and restore")
        logger.info("   - Comprehensive wallet management")


if __name__ == "__main__":
    main()
