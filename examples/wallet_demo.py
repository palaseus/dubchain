#!/usr/bin/env python3
"""
Advanced Wallet System Demo for GodChain.

This demo showcases the sophisticated wallet system including HD wallets,
multi-signature wallets, encryption, and wallet management.
"""

import os
import tempfile
from dubchain.wallet import (
    WalletManager, WalletManagerConfig, HDWallet, MultisigWallet, MultisigConfig, 
    MultisigType, EncryptionConfig, Language
)


def main():
    """Run the wallet system demo."""
    print("🚀 GodChain Advanced Wallet System Demo")
    print("=" * 50)
    
    # Create temporary directory for wallet storage
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary storage: {temp_dir}")
        
        # Configure wallet manager
        config = WalletManagerConfig(
            storage_path=temp_dir,
            default_network="mainnet",
            encryption_enabled=True,
            mnemonic_language=Language.ENGLISH,
            auto_backup=True
        )
        
        print("\n🔧 Initializing Wallet Manager...")
        manager = WalletManager(config)
        print(f"✅ Wallet Manager initialized")
        print(f"   - Storage path: {manager.config.storage_path}")
        print(f"   - Encryption enabled: {manager.config.encryption_enabled}")
        print(f"   - Max wallets: {manager.config.max_wallets}")
        
        # Create HD wallet
        print("\n🔑 Creating HD Wallet...")
        hd_wallet_id = manager.create_hd_wallet(
            name="My HD Wallet",
            password="secure_password_123!",
            network="mainnet"
        )
        print(f"✅ HD Wallet created: {hd_wallet_id}")
        
        # Load and interact with HD wallet
        print("\n📱 Loading HD Wallet...")
        hd_wallet = manager.load_wallet(hd_wallet_id, "secure_password_123!")
        print(f"✅ HD Wallet loaded: {hd_wallet}")
        print(f"   - Name: {hd_wallet.metadata.name}")
        print(f"   - Network: {hd_wallet.metadata.network}")
        print(f"   - Accounts: {len(hd_wallet.accounts)}")
        print(f"   - Current account: {hd_wallet.current_account_index}")
        
        # Create additional accounts
        print("\n👥 Creating additional accounts...")
        account1 = hd_wallet.create_account(label="Savings Account")
        account2 = hd_wallet.create_account(label="Trading Account")
        print(f"✅ Created accounts:")
        print(f"   - Account 1: {account1.label}")
        print(f"   - Account 2: {account2.label}")
        
        # Generate addresses
        print("\n📍 Generating addresses...")
        addresses = hd_wallet.get_account_addresses(0, 3)
        print(f"✅ Generated {len(addresses)} addresses for account 0:")
        for i, address in enumerate(addresses):
            print(f"   - Address {i+1}: {address}")
        
        # Update balances
        print("\n💰 Updating balances...")
        hd_wallet.update_balance(0, 1000000)  # 1M satoshis
        hd_wallet.update_balance(1, 500000)   # 500K satoshis
        hd_wallet.update_balance(2, 250000)   # 250K satoshis
        
        total_balance = hd_wallet.get_total_balance()
        print(f"✅ Balances updated:")
        print(f"   - Account 0: {hd_wallet.get_balance(0):,} satoshis")
        print(f"   - Account 1: {hd_wallet.get_balance(1):,} satoshis")
        print(f"   - Account 2: {hd_wallet.get_balance(2):,} satoshis")
        print(f"   - Total balance: {total_balance:,} satoshis")
        
        # Create multisig wallet
        print("\n🔐 Creating Multi-Signature Wallet...")
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
        print(f"✅ Multisig Wallet created: {multisig_wallet_id}")
        
        # Load and configure multisig wallet
        print("\n👥 Configuring Multi-Signature Wallet...")
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
            print(f"   - Added participant {i+1}: {participant.participant_id}")
        
        # Create and sign a transaction
        print("\n📝 Creating Multi-Signature Transaction...")
        transaction_data = b"Transfer 100,000 satoshis to address GC123..."
        transaction = multisig_wallet.create_transaction(transaction_data, expires_in_seconds=1800)
        print(f"✅ Transaction created: {transaction.transaction_id}")
        print(f"   - Required signatures: {transaction.required_signatures}")
        print(f"   - Total participants: {transaction.total_participants}")
        
        # Sign transaction with first two participants
        print("\n✍️ Signing Transaction...")
        for i, (private_key, participant) in enumerate(participants[:2]):
            success = multisig_wallet.sign_transaction(
                transaction.transaction_id,
                participant.participant_id,
                private_key
            )
            print(f"   - Participant {i+1} signature: {'✅ Success' if success else '❌ Failed'}")
        
        # Verify transaction
        print("\n🔍 Verifying Transaction...")
        is_complete = multisig_wallet.verify_transaction(transaction.transaction_id)
        print(f"✅ Transaction verification: {'✅ Complete' if is_complete else '❌ Incomplete'}")
        
        if is_complete:
            print(f"   - Signatures collected: {transaction.get_signature_count()}")
            print(f"   - Required signatures: {transaction.required_signatures}")
        
        # Test encryption
        print("\n🔒 Testing Wallet Encryption...")
        encryption = manager.encryption
        test_data = {"sensitive": "wallet_data", "balance": 1000000}
        password = "encryption_test_password"
        
        encrypted_data = encryption.encrypt_dict(test_data, password)
        decrypted_data = encryption.decrypt_dict(encrypted_data, password)
        
        print(f"✅ Encryption test: {'✅ Success' if test_data == decrypted_data else '❌ Failed'}")
        print(f"   - Original data: {test_data}")
        print(f"   - Decrypted data: {decrypted_data}")
        
        # Test password validation
        print("\n🔐 Testing Password Validation...")
        strong_password = manager.generate_strong_password(16)
        is_valid, errors = manager.validate_password(strong_password)
        strength = manager.get_password_strength(strong_password)
        
        print(f"✅ Generated strong password: {strong_password}")
        print(f"   - Valid: {'✅ Yes' if is_valid else '❌ No'}")
        print(f"   - Strength: {strength}")
        if errors:
            print(f"   - Errors: {errors}")
        
        # Wallet management
        print("\n📊 Wallet Management Summary...")
        wallet_list = manager.get_wallet_list()
        manager_info = manager.get_manager_info()
        
        print(f"✅ Wallet Manager Info:")
        print(f"   - Total wallets: {manager_info['wallet_count']}")
        print(f"   - Loaded wallets: {manager_info['loaded_wallets']}")
        print(f"   - Storage path: {manager_info['storage_path']}")
        print(f"   - Encryption enabled: {manager_info['encryption_enabled']}")
        
        print(f"\n📋 Wallet List:")
        for wallet_info in wallet_list:
            print(f"   - {wallet_info.name} ({wallet_info.wallet_type.value})")
            print(f"     ID: {wallet_info.wallet_id}")
            print(f"     Network: {wallet_info.network}")
            print(f"     Encrypted: {'✅ Yes' if wallet_info.is_encrypted else '❌ No'}")
        
        # Export wallet backup
        print("\n💾 Creating Wallet Backup...")
        backup_file = manager.backup_wallet(
            hd_wallet_id,
            backup_path=temp_dir,
            password="secure_password_123!"
        )
        print(f"✅ Backup created: {os.path.basename(backup_file)}")
        
        print("\n🎉 Wallet System Demo Completed Successfully!")
        print("=" * 50)
        print("✨ Features demonstrated:")
        print("   - HD wallet creation and management")
        print("   - Multi-account support")
        print("   - Address generation")
        print("   - Balance management")
        print("   - Multi-signature wallets")
        print("   - Transaction signing and verification")
        print("   - Advanced encryption")
        print("   - Password management")
        print("   - Wallet backup and restore")
        print("   - Comprehensive wallet management")


if __name__ == "__main__":
    main()
