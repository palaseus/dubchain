"""DubChain Storage Module.

This module provides persistent storage capabilities for the DubChain blockchain,
including database backends, indexing, migrations, and backup/recovery systems.
"""

from .backup import BackupConfig, BackupError, BackupManager
from .database import DatabaseBackend, DatabaseConfig, DatabaseError
from .indexing import IndexConfig, IndexManager, IndexType
from .isolation import IsolationLevel, TransactionIsolation
from .migrations import Migration, MigrationError, MigrationManager

# CUDA-accelerated storage imports
from .cuda_storage import (
    CUDAStorageAccelerator,
    CUDAStorageConfig,
    get_global_cuda_storage_accelerator,
)

__all__ = [
    "DatabaseBackend",
    "DatabaseConfig",
    "DatabaseError",
    "IndexManager",
    "IndexType",
    "IndexConfig",
    "MigrationManager",
    "Migration",
    "MigrationError",
    "BackupManager",
    "BackupConfig",
    "BackupError",
    "TransactionIsolation",
    "IsolationLevel",
    # CUDA Storage
    "CUDAStorageAccelerator",
    "CUDAStorageConfig",
    "get_global_cuda_storage_accelerator",
]
