# Cryptography

This document explains the cryptographic primitives and security mechanisms used in DubChain.

## Cryptographic Primitives

### Hash Functions

Hash functions are one-way functions that map data of arbitrary size to fixed-size output.

#### SHA-256
- **Purpose**: Block hashing, transaction hashing
- **Output**: 256 bits (32 bytes)
- **Security**: 128-bit security level
- **Usage**: Bitcoin-compatible hashing

```python
import hashlib

def sha256(data):
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).digest()

def double_sha256(data):
    """Compute double SHA-256 hash (Bitcoin style)."""
    return sha256(sha256(data))
```

#### Keccak-256
- **Purpose**: Ethereum-compatible hashing
- **Output**: 256 bits (32 bytes)
- **Security**: 128-bit security level
- **Usage**: Smart contract addresses, message hashing

```python
from Crypto.Hash import keccak

def keccak256(data):
    """Compute Keccak-256 hash of data."""
    return keccak.new(digest_bits=256).update(data).digest()
```

### Digital Signatures

Digital signatures provide authentication and non-repudiation.

#### ECDSA (secp256k1)
- **Curve**: secp256k1 (Bitcoin/Ethereum compatible)
- **Key Size**: 256 bits
- **Signature Size**: 64 bytes (r, s values)
- **Security**: 128-bit security level

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

def generate_key_pair():
    """Generate ECDSA key pair."""
    private_key = ec.generate_private_key(ec.SECP256K1())
    public_key = private_key.public_key()
    return private_key, public_key

def sign_message(private_key, message):
    """Sign message with ECDSA."""
    signature = private_key.sign(
        message,
        ec.ECDSA(hashes.SHA256())
    )
    return signature

def verify_signature(public_key, message, signature):
    """Verify ECDSA signature."""
    try:
        public_key.verify(
            signature,
            message,
            ec.ECDSA(hashes.SHA256())
        )
        return True
    except:
        return False
```

#### Ed25519
- **Curve**: Ed25519
- **Key Size**: 256 bits
- **Signature Size**: 64 bytes
- **Security**: 128-bit security level
- **Advantages**: Fast, deterministic, secure

```python
from cryptography.hazmat.primitives.asymmetric import ed25519

def generate_ed25519_key_pair():
    """Generate Ed25519 key pair."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key

def sign_ed25519(private_key, message):
    """Sign message with Ed25519."""
    return private_key.sign(message)

def verify_ed25519(public_key, message, signature):
    """Verify Ed25519 signature."""
    try:
        public_key.verify(signature, message)
        return True
    except:
        return False
```

### Merkle Trees

Merkle trees provide efficient data integrity verification.

```python
class MerkleTree:
    def __init__(self, data):
        self.data = data
        self.tree = self.build_tree()
    
    def build_tree(self):
        """Build Merkle tree from data."""
        if len(self.data) == 0:
            return []
        
        # Pad data to power of 2
        padded_data = self.data[:]
        while len(padded_data) & (len(padded_data) - 1):
            padded_data.append(padded_data[-1])
        
        # Build tree bottom-up
        tree = [sha256(item) for item in padded_data]
        level = tree
        
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else level[i]
                combined = left + right
                next_level.append(sha256(combined))
            tree.extend(next_level)
            level = next_level
        
        return tree
    
    def get_root(self):
        """Get Merkle root."""
        return self.tree[-1]
    
    def get_proof(self, index):
        """Get Merkle proof for element at index."""
        proof = []
        current_index = index
        
        for level_size in [len(self.data)]:
            if current_index ^ 1 < level_size:
                sibling_index = current_index ^ 1
                proof.append(self.tree[sibling_index])
            current_index //= 2
        
        return proof
    
    def verify_proof(self, element, proof, root):
        """Verify Merkle proof."""
        current_hash = sha256(element)
        
        for sibling_hash in proof:
            if current_hash < sibling_hash:
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = sha256(combined)
        
        return current_hash == root
```

## Key Management

### Key Generation
```python
def generate_mnemonic(strength=256):
    """Generate BIP39 mnemonic phrase."""
    import secrets
    import hashlib
    
    # Generate random entropy
    entropy = secrets.token_bytes(strength // 8)
    
    # Generate checksum
    checksum = hashlib.sha256(entropy).digest()
    checksum_bits = strength // 32
    
    # Combine entropy and checksum
    combined = entropy + checksum[:checksum_bits // 8]
    
    # Convert to mnemonic words
    wordlist = load_wordlist()  # BIP39 wordlist
    mnemonic = []
    
    for i in range(0, len(combined), 4):
        word_index = int.from_bytes(combined[i:i+4], 'big') % len(wordlist)
        mnemonic.append(wordlist[word_index])
    
    return ' '.join(mnemonic)

def mnemonic_to_seed(mnemonic, passphrase=""):
    """Convert mnemonic to seed using PBKDF2."""
    import hashlib
    import hmac
    
    mnemonic_bytes = mnemonic.encode('utf-8')
    salt = ("mnemonic" + passphrase).encode('utf-8')
    
    seed = hashlib.pbkdf2_hmac('sha512', mnemonic_bytes, salt, 2048)
    return seed
```

### Hierarchical Deterministic (HD) Wallets
```python
class HDWallet:
    def __init__(self, seed):
        self.seed = seed
        self.master_key = self.derive_master_key()
    
    def derive_master_key(self):
        """Derive master key from seed."""
        import hmac
        import hashlib
        
        # HMAC-SHA512
        hmac_result = hmac.new(
            b"Bitcoin seed",
            self.seed,
            hashlib.sha512
        ).digest()
        
        master_private_key = hmac_result[:32]
        master_chain_code = hmac_result[32:]
        
        return {
            'private_key': master_private_key,
            'chain_code': master_chain_code
        }
    
    def derive_child_key(self, index, hardened=False):
        """Derive child key from parent."""
        import hmac
        import hashlib
        
        if hardened:
            # Hardened derivation
            data = b'\x00' + self.master_key['private_key'] + index.to_bytes(4, 'big')
        else:
            # Normal derivation
            public_key = self.get_public_key()
            data = public_key + index.to_bytes(4, 'big')
        
        hmac_result = hmac.new(
            self.master_key['chain_code'],
            data,
            hashlib.sha512
        ).digest()
        
        child_private_key = hmac_result[:32]
        child_chain_code = hmac_result[32:]
        
        return {
            'private_key': child_private_key,
            'chain_code': child_chain_code
        }
```

## Security Considerations

### Random Number Generation
```python
import secrets

def generate_secure_random(length):
    """Generate cryptographically secure random bytes."""
    return secrets.token_bytes(length)

def generate_secure_random_int(min_val, max_val):
    """Generate cryptographically secure random integer."""
    return secrets.randbelow(max_val - min_val + 1) + min_val
```

### Key Storage
```python
class SecureKeyStorage:
    def __init__(self, password):
        self.password = password
        self.encrypted_keys = {}
    
    def store_key(self, key_id, private_key):
        """Store encrypted private key."""
        # Derive encryption key from password
        encryption_key = self.derive_key(self.password, key_id)
        
        # Encrypt private key
        encrypted_key = self.encrypt(private_key, encryption_key)
        
        # Store encrypted key
        self.encrypted_keys[key_id] = encrypted_key
    
    def retrieve_key(self, key_id):
        """Retrieve and decrypt private key."""
        if key_id not in self.encrypted_keys:
            raise KeyError(f"Key {key_id} not found")
        
        # Derive decryption key
        decryption_key = self.derive_key(self.password, key_id)
        
        # Decrypt private key
        private_key = self.decrypt(self.encrypted_keys[key_id], decryption_key)
        
        return private_key
```

### Signature Security
```python
def secure_sign(private_key, message, nonce=None):
    """Create secure signature with proper nonce handling."""
    if nonce is None:
        # Generate deterministic nonce (RFC 6979)
        nonce = generate_deterministic_nonce(private_key, message)
    
    # Sign with nonce
    signature = sign_with_nonce(private_key, message, nonce)
    
    return signature

def verify_signature_secure(public_key, message, signature):
    """Verify signature with security checks."""
    # Check signature format
    if len(signature) != 64:
        return False
    
    # Check signature values
    r, s = signature[:32], signature[32:]
    if r == b'\x00' * 32 or s == b'\x00' * 32:
        return False
    
    # Verify signature
    return verify_signature(public_key, message, signature)
```

## Performance Optimization

### Parallel Signature Verification
```python
import concurrent.futures

def verify_signatures_parallel(public_keys, messages, signatures):
    """Verify multiple signatures in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(public_keys)):
            future = executor.submit(
                verify_signature,
                public_keys[i],
                messages[i],
                signatures[i]
            )
            futures.append(future)
        
        results = [future.result() for future in futures]
        return results
```

### Caching
```python
class SignatureCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
    
    def get_cached_verification(self, message_hash, signature_hash):
        """Get cached verification result."""
        key = (message_hash, signature_hash)
        return self.cache.get(key)
    
    def cache_verification(self, message_hash, signature_hash, result):
        """Cache verification result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = (message_hash, signature_hash)
        self.cache[key] = result
```

## Best Practices

### Key Management
1. Use hardware security modules (HSMs) for production
2. Implement proper key rotation
3. Use secure random number generation
4. Store keys encrypted at rest
5. Implement proper access controls

### Signature Security
1. Use deterministic nonces (RFC 6979)
2. Validate all signature components
3. Implement signature caching
4. Use appropriate signature schemes
5. Regular security audits

### Performance
1. Use parallel processing for batch operations
2. Implement caching for repeated operations
3. Optimize cryptographic operations
4. Use hardware acceleration where available
5. Monitor performance metrics

## Further Reading

- [Blockchain Fundamentals](blockchain.md)
- [Consensus Mechanisms](consensus.md)
- [Smart Contracts](smart-contracts.md)
- [Performance Optimization](../../performance/README.md)
