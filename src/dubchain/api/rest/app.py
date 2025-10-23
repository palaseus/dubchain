"""
REST API for DubChain.

This module provides a comprehensive REST API with FastAPI including
all blockchain operations, smart contracts, wallets, bridges, sharding,
consensus, and governance endpoints.
"""

import logging

logger = logging.getLogger(__name__)
import asyncio
import json
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from ..core.blockchain import Blockchain
from ..core.transaction import Transaction as CoreTransaction
from ..vm.contract import SmartContract
from ..wallet.wallet_manager import WalletManager
from ..bridge.bridge_manager import BridgeManager
from ..sharding.shard_manager import ShardManager
from ..consensus.consensus_engine import ConsensusEngine
from ..governance.core import GovernanceEngine
from ..network.peer import PeerManager
from ..performance.monitoring import PerformanceMonitor
from ..common.auth import AuthManager
from ..common.rate_limit import RateLimiter
from ..common.cache import CacheManager
from ..common.monitoring import MetricsCollector

# Security
security = HTTPBearer()

# Global instances
blockchain = Blockchain()
wallet_manager = WalletManager()
bridge_manager = BridgeManager()
shard_manager = ShardManager()
consensus_engine = ConsensusEngine()
governance_engine = GovernanceEngine()
peer_manager = PeerManager()
performance_monitor = PerformanceMonitor()
auth_manager = AuthManager()
rate_limiter = RateLimiter()
cache_manager = CacheManager()
metrics_collector = MetricsCollector()

# Pydantic models for request/response validation
class TransactionRequest(BaseModel):
    """Transaction creation request."""
    from_address: str = Field(..., description="Sender address")
    to_address: str = Field(..., description="Recipient address")
    value: Decimal = Field(..., description="Amount to transfer")
    gas_price: int = Field(..., description="Gas price")
    gas_limit: int = Field(..., description="Gas limit")
    data: Optional[str] = Field(None, description="Transaction data (hex)")
    
    @validator('value')
    def validate_value(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v
    
    @validator('gas_price')
    def validate_gas_price(cls, v):
        if v <= 0:
            raise ValueError('Gas price must be positive')
        return v
    
    @validator('gas_limit')
    def validate_gas_limit(cls, v):
        if v <= 0:
            raise ValueError('Gas limit must be positive')
        return v

class ContractDeployRequest(BaseModel):
    """Contract deployment request."""
    bytecode: str = Field(..., description="Contract bytecode (hex)")
    abi: str = Field(..., description="Contract ABI (JSON string)")
    constructor_args: Optional[List[str]] = Field(None, description="Constructor arguments")

class ContractCallRequest(BaseModel):
    """Contract function call request."""
    contract_address: str = Field(..., description="Contract address")
    function_name: str = Field(..., description="Function name")
    args: Optional[List[str]] = Field(None, description="Function arguments")

class WalletCreateRequest(BaseModel):
    """Wallet creation request."""
    name: str = Field(..., description="Wallet name")
    password: Optional[str] = Field(None, description="Wallet password")

class BridgeTransferRequest(BaseModel):
    """Bridge transfer request."""
    source_chain: str = Field(..., description="Source chain")
    target_chain: str = Field(..., description="Target chain")
    source_asset: str = Field(..., description="Source asset")
    target_asset: str = Field(..., description="Target asset")
    amount: Decimal = Field(..., description="Amount to transfer")
    receiver: str = Field(..., description="Receiver address")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class ProposalCreateRequest(BaseModel):
    """Proposal creation request."""
    title: str = Field(..., description="Proposal title")
    description: str = Field(..., description="Proposal description")
    proposal_type: str = Field(..., description="Proposal type")

class VoteRequest(BaseModel):
    """Vote request."""
    proposal_id: str = Field(..., description="Proposal ID")
    vote_option: str = Field(..., description="Vote option (yes/no/abstain)")
    reason: Optional[str] = Field(None, description="Vote reason")

# Response models
class TransactionResponse(BaseModel):
    """Transaction response."""
    hash: str
    from_address: str
    to_address: str
    value: Decimal
    gas_price: int
    gas_limit: int
    gas_used: Optional[int]
    nonce: int
    data: Optional[str]
    signature: str
    status: str
    block_height: Optional[int]
    timestamp: datetime

class BlockResponse(BaseModel):
    """Block response."""
    hash: str
    previous_hash: str
    timestamp: datetime
    height: int
    transactions: List[TransactionResponse]
    merkle_root: str
    nonce: int
    difficulty: int
    validator: str
    gas_used: int
    gas_limit: int

class AccountResponse(BaseModel):
    """Account response."""
    address: str
    balance: Decimal
    nonce: int
    code_hash: Optional[str]
    storage_root: Optional[str]
    transaction_count: int
    is_contract: bool

class ContractResponse(BaseModel):
    """Contract response."""
    address: str
    name: str
    abi: str
    bytecode: str
    source_code: Optional[str]
    creator: str
    created_at: datetime
    gas_used: int
    storage_size: int

class WalletResponse(BaseModel):
    """Wallet response."""
    wallet_id: str
    name: str
    wallet_type: str
    accounts: List[AccountResponse]
    created_at: datetime
    last_accessed: datetime
    is_encrypted: bool

class BridgeTransferResponse(BaseModel):
    """Bridge transfer response."""
    transfer_id: str
    source_chain: str
    target_chain: str
    source_asset: str
    target_asset: str
    amount: Decimal
    sender: str
    receiver: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    transaction_hash: Optional[str]

class ShardResponse(BaseModel):
    """Shard response."""
    shard_id: int
    validator_count: int
    transaction_count: int
    block_height: int
    status: str
    validators: List[str]
    cross_shard_transactions: int

class ValidatorResponse(BaseModel):
    """Validator response."""
    address: str
    public_key: str
    stake: Decimal
    commission_rate: float
    status: str
    uptime: float
    blocks_proposed: int
    blocks_validated: int
    slashing_events: int
    last_active: datetime

class ProposalResponse(BaseModel):
    """Proposal response."""
    proposal_id: str
    proposer: str
    title: str
    description: str
    proposal_type: str
    status: str
    voting_start: datetime
    voting_end: datetime
    execution_time: Optional[datetime]
    total_votes: int
    yes_votes: int
    no_votes: int
    abstain_votes: int
    quorum: float
    threshold: float

class NetworkStatsResponse(BaseModel):
    """Network statistics response."""
    total_peers: int
    active_peers: int
    total_connections: int
    bytes_sent: int
    bytes_received: int
    messages_sent: int
    messages_received: int
    average_latency: float

class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response."""
    tps: float
    latency: float
    gas_efficiency: float
    memory_usage: int
    cpu_usage: float
    disk_usage: int
    network_throughput: float
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None

# FastAPI application
app = FastAPI(
    title="DubChain REST API",
    description="Advanced blockchain REST API with comprehensive endpoints",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.info(f"Error broadcasting message: {e}")

manager = ConnectionManager()

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Rate limiting dependency
async def check_rate_limit(request: Request):
    """Check rate limit for request."""
    client_ip = request.client.host if request.client else "unknown"
    if not await rate_limiter.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# Blockchain endpoints
@app.get("/v1/blockchain/blocks", response_model=List[BlockResponse])
async def get_blocks(limit: int = 10, offset: int = 0):
    """Get recent blocks."""
    try:
        blocks = []
        current_height = blockchain.get_latest_height()
        
        for i in range(offset, min(offset + limit, current_height + 1)):
            core_block = blockchain.get_block_by_height(current_height - i)
            if core_block:
                transactions = []
                for tx in core_block.transactions:
                    transactions.append(TransactionResponse(
                        hash=tx.hash,
                        from_address=tx.sender,
                        to_address=tx.recipient,
                        value=Decimal(str(tx.amount)),
                        gas_price=tx.gas_price,
                        gas_limit=tx.gas_limit,
                        gas_used=tx.gas_used,
                        nonce=tx.nonce,
                        data=tx.data.hex() if tx.data else None,
                        signature=tx.signature.hex(),
                        status="confirmed",
                        block_height=core_block.height,
                        timestamp=datetime.fromtimestamp(core_block.timestamp)
                    ))
                
                blocks.append(BlockResponse(
                    hash=core_block.hash,
                    previous_hash=core_block.previous_hash,
                    timestamp=datetime.fromtimestamp(core_block.timestamp),
                    height=core_block.height,
                    transactions=transactions,
                    merkle_root=core_block.merkle_root,
                    nonce=core_block.nonce,
                    difficulty=core_block.difficulty,
                    validator=core_block.validator,
                    gas_used=core_block.gas_used,
                    gas_limit=core_block.gas_limit
                ))
        
        return blocks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/blockchain/blocks/{block_hash}", response_model=BlockResponse)
async def get_block_by_hash(block_hash: str):
    """Get block by hash."""
    try:
        core_block = blockchain.get_block_by_hash(block_hash)
        if not core_block:
            raise HTTPException(status_code=404, detail="Block not found")
        
        transactions = []
        for tx in core_block.transactions:
            transactions.append(TransactionResponse(
                hash=tx.hash,
                from_address=tx.sender,
                to_address=tx.recipient,
                value=Decimal(str(tx.amount)),
                gas_price=tx.gas_price,
                gas_limit=tx.gas_limit,
                gas_used=tx.gas_used,
                nonce=tx.nonce,
                data=tx.data.hex() if tx.data else None,
                signature=tx.signature.hex(),
                status="confirmed",
                block_height=core_block.height,
                timestamp=datetime.fromtimestamp(core_block.timestamp)
            ))
        
        return BlockResponse(
            hash=core_block.hash,
            previous_hash=core_block.previous_hash,
            timestamp=datetime.fromtimestamp(core_block.timestamp),
            height=core_block.height,
            transactions=transactions,
            merkle_root=core_block.merkle_root,
            nonce=core_block.nonce,
            difficulty=core_block.difficulty,
            validator=core_block.validator,
            gas_used=core_block.gas_used,
            gas_limit=core_block.gas_limit
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/blockchain/blocks/height/{height}", response_model=BlockResponse)
async def get_block_by_height(height: int):
    """Get block by height."""
    try:
        core_block = blockchain.get_block_by_height(height)
        if not core_block:
            raise HTTPException(status_code=404, detail="Block not found")
        
        transactions = []
        for tx in core_block.transactions:
            transactions.append(TransactionResponse(
                hash=tx.hash,
                from_address=tx.sender,
                to_address=tx.recipient,
                value=Decimal(str(tx.amount)),
                gas_price=tx.gas_price,
                gas_limit=tx.gas_limit,
                gas_used=tx.gas_used,
                nonce=tx.nonce,
                data=tx.data.hex() if tx.data else None,
                signature=tx.signature.hex(),
                status="confirmed",
                block_height=core_block.height,
                timestamp=datetime.fromtimestamp(core_block.timestamp)
            ))
        
        return BlockResponse(
            hash=core_block.hash,
            previous_hash=core_block.previous_hash,
            timestamp=datetime.fromtimestamp(core_block.timestamp),
            height=core_block.height,
            transactions=transactions,
            merkle_root=core_block.merkle_root,
            nonce=core_block.nonce,
            difficulty=core_block.difficulty,
            validator=core_block.validator,
            gas_used=core_block.gas_used,
            gas_limit=core_block.gas_limit
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Transaction endpoints
@app.post("/v1/transactions", response_model=TransactionResponse)
async def create_transaction(
    transaction: TransactionRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Create a new transaction."""
    try:
        await check_rate_limit(request)
        
        tx_data = bytes.fromhex(transaction.data) if transaction.data else b''
        tx = CoreTransaction(
            sender=transaction.from_address,
            recipient=transaction.to_address,
            amount=int(transaction.value),
            gas_price=transaction.gas_price,
            gas_limit=transaction.gas_limit,
            data=tx_data
        )
        tx_hash = blockchain.add_transaction(tx)
        
        return TransactionResponse(
            hash=tx_hash,
            from_address=tx.sender,
            to_address=tx.recipient,
            value=transaction.value,
            gas_price=tx.gas_price,
            gas_limit=tx.gas_limit,
            gas_used=None,
            nonce=tx.nonce,
            data=transaction.data,
            signature=tx.signature.hex(),
            status="pending",
            block_height=None,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/transactions/{tx_hash}", response_model=TransactionResponse)
async def get_transaction(tx_hash: str):
    """Get transaction by hash."""
    try:
        core_tx = blockchain.get_transaction(tx_hash)
        if not core_tx:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return TransactionResponse(
            hash=core_tx.hash,
            from_address=core_tx.sender,
            to_address=core_tx.recipient,
            value=Decimal(str(core_tx.amount)),
            gas_price=core_tx.gas_price,
            gas_limit=core_tx.gas_limit,
            gas_used=core_tx.gas_used,
            nonce=core_tx.nonce,
            data=core_tx.data.hex() if core_tx.data else None,
            signature=core_tx.signature.hex(),
            status="confirmed" if core_tx.block_height else "pending",
            block_height=core_tx.block_height,
            timestamp=datetime.fromtimestamp(core_tx.timestamp)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Account endpoints
@app.get("/v1/accounts/{address}", response_model=AccountResponse)
async def get_account(address: str):
    """Get account by address."""
    try:
        account_data = blockchain.get_account(address)
        if not account_data:
            raise HTTPException(status_code=404, detail="Account not found")
        
        return AccountResponse(
            address=address,
            balance=Decimal(str(account_data.get('balance', 0))),
            nonce=account_data.get('nonce', 0),
            code_hash=account_data.get('code_hash'),
            storage_root=account_data.get('storage_root'),
            transaction_count=account_data.get('transaction_count', 0),
            is_contract=account_data.get('is_contract', False)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Contract endpoints
@app.post("/v1/contracts/deploy", response_model=ContractResponse)
async def deploy_contract(
    contract: ContractDeployRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Deploy a smart contract."""
    try:
        await check_rate_limit(request)
        
        contract_obj = SmartContract(
            bytecode=bytes.fromhex(contract.bytecode),
            abi=json.loads(contract.abi)
        )
        contract_address = blockchain.deploy_contract(contract_obj, contract.constructor_args or [])
        
        return ContractResponse(
            address=contract_address,
            name="Deployed Contract",
            abi=contract.abi,
            bytecode=contract.bytecode,
            source_code=None,
            creator=user.get('address', ''),
            created_at=datetime.now(),
            gas_used=0,
            storage_size=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/contracts/call", response_model=dict)
async def call_contract(
    call: ContractCallRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Call a smart contract function."""
    try:
        await check_rate_limit(request)
        
        result = blockchain.call_contract(call.contract_address, call.function_name, call.args or [])
        
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/contracts/{address}", response_model=ContractResponse)
async def get_contract(address: str):
    """Get contract by address."""
    try:
        contract_data = blockchain.get_contract(address)
        if not contract_data:
            raise HTTPException(status_code=404, detail="Contract not found")
        
        return ContractResponse(
            address=address,
            name=contract_data.get('name', 'Unknown'),
            abi=json.dumps(contract_data.get('abi', {})),
            bytecode=contract_data.get('bytecode', ''),
            source_code=contract_data.get('source_code'),
            creator=contract_data.get('creator', ''),
            created_at=datetime.fromtimestamp(contract_data.get('created_at', 0)),
            gas_used=contract_data.get('gas_used', 0),
            storage_size=contract_data.get('storage_size', 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Wallet endpoints
@app.post("/v1/wallets", response_model=WalletResponse)
async def create_wallet(
    wallet: WalletCreateRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Create a new wallet."""
    try:
        await check_rate_limit(request)
        
        wallet_id = wallet_manager.create_wallet(wallet.name, wallet.password)
        wallet_data = wallet_manager.get_wallet_info(wallet_id)
        
        accounts = []
        for account_data in wallet_data.get('accounts', []):
            accounts.append(AccountResponse(
                address=account_data.get('address', ''),
                balance=Decimal(str(account_data.get('balance', 0))),
                nonce=account_data.get('nonce', 0),
                code_hash=None,
                storage_root=None,
                transaction_count=account_data.get('transaction_count', 0),
                is_contract=False
            ))
        
        return WalletResponse(
            wallet_id=wallet_id,
            name=wallet_data.get('name', ''),
            wallet_type=wallet_data.get('wallet_type', ''),
            accounts=accounts,
            created_at=datetime.fromtimestamp(wallet_data.get('created_at', 0)),
            last_accessed=datetime.fromtimestamp(wallet_data.get('last_accessed', 0)),
            is_encrypted=wallet_data.get('is_encrypted', False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/wallets/{wallet_id}", response_model=WalletResponse)
async def get_wallet(wallet_id: str, user: dict = Depends(get_current_user)):
    """Get wallet by ID."""
    try:
        wallet_data = wallet_manager.get_wallet_info(wallet_id)
        if not wallet_data:
            raise HTTPException(status_code=404, detail="Wallet not found")
        
        accounts = []
        for account_data in wallet_data.get('accounts', []):
            accounts.append(AccountResponse(
                address=account_data.get('address', ''),
                balance=Decimal(str(account_data.get('balance', 0))),
                nonce=account_data.get('nonce', 0),
                code_hash=None,
                storage_root=None,
                transaction_count=account_data.get('transaction_count', 0),
                is_contract=False
            ))
        
        return WalletResponse(
            wallet_id=wallet_id,
            name=wallet_data.get('name', ''),
            wallet_type=wallet_data.get('wallet_type', ''),
            accounts=accounts,
            created_at=datetime.fromtimestamp(wallet_data.get('created_at', 0)),
            last_accessed=datetime.fromtimestamp(wallet_data.get('last_accessed', 0)),
            is_encrypted=wallet_data.get('is_encrypted', False)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Bridge endpoints
@app.post("/v1/bridge/transfers", response_model=BridgeTransferResponse)
async def create_bridge_transfer(
    transfer: BridgeTransferRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Create a bridge transfer."""
    try:
        await check_rate_limit(request)
        
        transfer_id = bridge_manager.create_transfer(
            source_chain=transfer.source_chain,
            target_chain=transfer.target_chain,
            source_asset=transfer.source_asset,
            target_asset=transfer.target_asset,
            amount=int(transfer.amount),
            receiver=transfer.receiver
        )
        
        return BridgeTransferResponse(
            transfer_id=transfer_id,
            source_chain=transfer.source_chain,
            target_chain=transfer.target_chain,
            source_asset=transfer.source_asset,
            target_asset=transfer.target_asset,
            amount=transfer.amount,
            sender=user.get('address', ''),
            receiver=transfer.receiver,
            status="pending",
            created_at=datetime.now(),
            completed_at=None,
            transaction_hash=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/bridge/transfers/{transfer_id}", response_model=BridgeTransferResponse)
async def get_bridge_transfer(transfer_id: str):
    """Get bridge transfer by ID."""
    try:
        transfer_data = bridge_manager.get_transfer(transfer_id)
        if not transfer_data:
            raise HTTPException(status_code=404, detail="Bridge transfer not found")
        
        return BridgeTransferResponse(
            transfer_id=transfer_id,
            source_chain=transfer_data.get('source_chain', ''),
            target_chain=transfer_data.get('target_chain', ''),
            source_asset=transfer_data.get('source_asset', ''),
            target_asset=transfer_data.get('target_asset', ''),
            amount=Decimal(str(transfer_data.get('amount', 0))),
            sender=transfer_data.get('sender', ''),
            receiver=transfer_data.get('receiver', ''),
            status=transfer_data.get('status', 'pending'),
            created_at=datetime.fromtimestamp(transfer_data.get('created_at', 0)),
            completed_at=datetime.fromtimestamp(transfer_data.get('completed_at', 0)) if transfer_data.get('completed_at') else None,
            transaction_hash=transfer_data.get('transaction_hash')
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sharding endpoints
@app.get("/v1/sharding/shard/{shard_id}", response_model=ShardResponse)
async def get_shard(shard_id: int):
    """Get shard by ID."""
    try:
        shard_data = shard_manager.get_shard(shard_id)
        if not shard_data:
            raise HTTPException(status_code=404, detail="Shard not found")
        
        return ShardResponse(
            shard_id=shard_id,
            validator_count=shard_data.get('validator_count', 0),
            transaction_count=shard_data.get('transaction_count', 0),
            block_height=shard_data.get('block_height', 0),
            status=shard_data.get('status', 'active'),
            validators=shard_data.get('validators', []),
            cross_shard_transactions=shard_data.get('cross_shard_transactions', 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Consensus endpoints
@app.get("/v1/consensus/validators/{address}", response_model=ValidatorResponse)
async def get_validator(address: str):
    """Get validator by address."""
    try:
        validator_data = consensus_engine.get_validator(address)
        if not validator_data:
            raise HTTPException(status_code=404, detail="Validator not found")
        
        return ValidatorResponse(
            address=address,
            public_key=validator_data.get('public_key', ''),
            stake=Decimal(str(validator_data.get('stake', 0))),
            commission_rate=validator_data.get('commission_rate', 0.0),
            status=validator_data.get('status', 'active'),
            uptime=validator_data.get('uptime', 0.0),
            blocks_proposed=validator_data.get('blocks_proposed', 0),
            blocks_validated=validator_data.get('blocks_validated', 0),
            slashing_events=validator_data.get('slashing_events', 0),
            last_active=datetime.fromtimestamp(validator_data.get('last_active', 0))
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Governance endpoints
@app.post("/v1/governance/proposals", response_model=ProposalResponse)
async def create_proposal(
    proposal: ProposalCreateRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Create a governance proposal."""
    try:
        await check_rate_limit(request)
        
        proposal_id = governance_engine.create_proposal(
            title=proposal.title,
            description=proposal.description,
            proposal_type=proposal.proposal_type
        )
        
        return ProposalResponse(
            proposal_id=proposal_id,
            proposer=user.get('address', ''),
            title=proposal.title,
            description=proposal.description,
            proposal_type=proposal.proposal_type,
            status="pending",
            voting_start=datetime.now(),
            voting_end=datetime.now(),
            execution_time=None,
            total_votes=0,
            yes_votes=0,
            no_votes=0,
            abstain_votes=0,
            quorum=0.0,
            threshold=0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/governance/vote", response_model=dict)
async def vote_on_proposal(
    vote: VoteRequest,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Vote on a proposal."""
    try:
        await check_rate_limit(request)
        
        success = governance_engine.vote(vote.proposal_id, vote.vote_option, vote.reason)
        
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/governance/proposals/{proposal_id}", response_model=ProposalResponse)
async def get_proposal(proposal_id: str):
    """Get proposal by ID."""
    try:
        proposal_data = governance_engine.get_proposal(proposal_id)
        if not proposal_data:
            raise HTTPException(status_code=404, detail="Proposal not found")
        
        return ProposalResponse(
            proposal_id=proposal_id,
            proposer=proposal_data.get('proposer', ''),
            title=proposal_data.get('title', ''),
            description=proposal_data.get('description', ''),
            proposal_type=proposal_data.get('proposal_type', ''),
            status=proposal_data.get('status', 'pending'),
            voting_start=datetime.fromtimestamp(proposal_data.get('voting_start', 0)),
            voting_end=datetime.fromtimestamp(proposal_data.get('voting_end', 0)),
            execution_time=datetime.fromtimestamp(proposal_data.get('execution_time', 0)) if proposal_data.get('execution_time') else None,
            total_votes=proposal_data.get('total_votes', 0),
            yes_votes=proposal_data.get('yes_votes', 0),
            no_votes=proposal_data.get('no_votes', 0),
            abstain_votes=proposal_data.get('abstain_votes', 0),
            quorum=proposal_data.get('quorum', 0.0),
            threshold=proposal_data.get('threshold', 0.0)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Network endpoints
@app.get("/v1/network/stats", response_model=NetworkStatsResponse)
async def get_network_stats():
    """Get network statistics."""
    try:
        stats = peer_manager.get_network_stats()
        return NetworkStatsResponse(
            total_peers=stats.get('total_peers', 0),
            active_peers=stats.get('active_peers', 0),
            total_connections=stats.get('total_connections', 0),
            bytes_sent=stats.get('bytes_sent', 0),
            bytes_received=stats.get('bytes_received', 0),
            messages_sent=stats.get('messages_sent', 0),
            messages_received=stats.get('messages_received', 0),
            average_latency=stats.get('average_latency', 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Performance endpoints
@app.get("/v1/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """Get performance metrics."""
    try:
        metrics = performance_monitor.get_current_metrics()
        return PerformanceMetricsResponse(
            tps=metrics.get('tps', 0.0),
            latency=metrics.get('latency', 0.0),
            gas_efficiency=metrics.get('gas_efficiency', 0.0),
            memory_usage=metrics.get('memory_usage', 0),
            cpu_usage=metrics.get('cpu_usage', 0.0),
            disk_usage=metrics.get('disk_usage', 0),
            network_throughput=metrics.get('network_throughput', 0.0),
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": {
            "blockchain": "running",
            "wallet": "running",
            "bridge": "running",
            "sharding": "running",
            "consensus": "running",
            "governance": "running"
        }
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back the received message
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            message=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )

# CLI entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
