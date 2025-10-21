"""
Advanced Backup System for DubChain

This module provides comprehensive backup capabilities including:
- Incremental backups
- Full backups
- Compressed backups
- Backup verification
- Backup restoration
- Backup scheduling
"""

import gzip
import json
import os
import shutil
import tarfile
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
import pickle

from ..errors import ClientError
from ..logging import get_logger

logger = get_logger(__name__)

class BackupError(Exception):
    """Base exception for backup operations."""
    pass

class BackupType(Enum):
    """Backup types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"

class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    GZIP = "gzip"
    TAR_GZ = "tar.gz"
    ZIP = "zip"

@dataclass
class BackupConfig:
    """Backup configuration."""
    backup_dir: str = "backups"
    max_backups: int = 10
    compression: CompressionType = CompressionType.GZIP
    verify_backups: bool = True
    auto_cleanup: bool = True
    retention_days: int = 30
    schedule_enabled: bool = False
    schedule_interval: int = 86400  # 24 hours in seconds

@dataclass
class BackupInfo:
    """Backup information."""
    backup_id: str
    backup_type: BackupType
    source_path: str
    backup_path: str
    size_bytes: int
    created_at: float
    status: BackupStatus
    compression: CompressionType
    checksum: Optional[str] = None
    parent_backup_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupSchedule:
    """Backup schedule."""
    schedule_id: str
    name: str
    backup_type: BackupType
    source_paths: List[str]
    enabled: bool
    interval_seconds: int
    last_run: Optional[float] = None
    next_run: Optional[float] = None

class BackupManager:
    """Main backup manager."""
    
    def __init__(self, config: BackupConfig):
        """Initialize backup manager."""
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backups: Dict[str, BackupInfo] = {}
        self.schedules: Dict[str, BackupSchedule] = {}
        self.scheduler_thread = None
        self.running = False
        
        # Load existing backups
        self._load_backup_metadata()
        
        logger.info("Initialized backup manager")
    
    def start(self) -> None:
        """Start backup manager."""
        self.running = True
        
        if self.config.schedule_enabled:
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
        
        logger.info("Backup manager started")
    
    def stop(self) -> None:
        """Stop backup manager."""
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Backup manager stopped")
    
    def create_backup(self, source_path: str, backup_type: BackupType = BackupType.FULL, 
                     name: Optional[str] = None) -> Optional[str]:
        """Create a backup."""
        try:
            backup_id = self._generate_backup_id()
            backup_name = name or f"backup_{backup_id}"
            
            # Determine backup path
            backup_path = self.backup_dir / f"{backup_name}_{backup_id}"
            
            # Create backup info
            backup_info = BackupInfo(
                backup_id=backup_id,
                backup_type=backup_type,
                source_path=source_path,
                backup_path=str(backup_path),
                size_bytes=0,
                created_at=time.time(),
                status=BackupStatus.PENDING,
                compression=self.config.compression
            )
            
            # Update status to running
            backup_info.status = BackupStatus.RUNNING
            self.backups[backup_id] = backup_info
            
            logger.info(f"Starting backup {backup_id}: {source_path}")
            
            # Perform backup
            success = self._perform_backup(backup_info)
            
            if success:
                backup_info.status = BackupStatus.COMPLETED
                
                # Verify backup if enabled
                if self.config.verify_backups:
                    if self._verify_backup(backup_info):
                        backup_info.status = BackupStatus.VERIFIED
                    else:
                        backup_info.status = BackupStatus.CORRUPTED
                        logger.error(f"Backup verification failed: {backup_id}")
                
                # Save metadata
                self._save_backup_metadata(backup_info)
                
                # Cleanup old backups
                if self.config.auto_cleanup:
                    self._cleanup_old_backups()
                
                logger.info(f"Backup completed successfully: {backup_id}")
                return backup_id
            else:
                backup_info.status = BackupStatus.FAILED
                logger.error(f"Backup failed: {backup_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return None
    
    def restore_backup(self, backup_id: str, target_path: str) -> bool:
        """Restore a backup."""
        try:
            if backup_id not in self.backups:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            backup_info = self.backups[backup_id]
            
            if backup_info.status not in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]:
                logger.error(f"Cannot restore backup {backup_id}: status is {backup_info.status}")
                return False
            
            logger.info(f"Restoring backup {backup_id} to {target_path}")
            
            # Perform restore
            success = self._perform_restore(backup_info, target_path)
            
            if success:
                logger.info(f"Backup restored successfully: {backup_id}")
                return True
            else:
                logger.error(f"Backup restore failed: {backup_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self) -> List[BackupInfo]:
        """List all backups."""
        return list(self.backups.values())
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupInfo]:
        """Get backup information."""
        return self.backups.get(backup_id)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        try:
            if backup_id not in self.backups:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            backup_info = self.backups[backup_id]
            
            # Delete backup file
            backup_path = Path(backup_info.backup_path)
            if backup_path.exists():
                if backup_path.is_file():
                    backup_path.unlink()
                elif backup_path.is_dir():
                    shutil.rmtree(backup_path)
            
            # Remove from registry
            del self.backups[backup_id]
            
            # Save updated metadata
            self._save_all_backup_metadata()
            
            logger.info(f"Deleted backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
            return False
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify a backup."""
        try:
            if backup_id not in self.backups:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            backup_info = self.backups[backup_id]
            
            logger.info(f"Verifying backup: {backup_id}")
            
            success = self._verify_backup(backup_info)
            
            if success:
                backup_info.status = BackupStatus.VERIFIED
                logger.info(f"Backup verification successful: {backup_id}")
            else:
                backup_info.status = BackupStatus.CORRUPTED
                logger.error(f"Backup verification failed: {backup_id}")
            
            self._save_backup_metadata(backup_info)
            return success
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False
    
    def create_schedule(self, schedule: BackupSchedule) -> bool:
        """Create a backup schedule."""
        try:
            self.schedules[schedule.schedule_id] = schedule
            self._save_schedule_metadata()
            
            logger.info(f"Created backup schedule: {schedule.schedule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating schedule: {e}")
            return False
    
    def _perform_backup(self, backup_info: BackupInfo) -> bool:
        """Perform the actual backup."""
        try:
            source_path = Path(backup_info.source_path)
            backup_path = Path(backup_info.backup_path)
            
            if not source_path.exists():
                logger.error(f"Source path does not exist: {source_path}")
                return False
            
            # Create backup based on type
            if backup_info.backup_type == BackupType.FULL:
                success = self._create_full_backup(source_path, backup_path, backup_info)
            elif backup_info.backup_type == BackupType.INCREMENTAL:
                success = self._create_incremental_backup(source_path, backup_path, backup_info)
            elif backup_info.backup_type == BackupType.DIFFERENTIAL:
                success = self._create_differential_backup(source_path, backup_path, backup_info)
            elif backup_info.backup_type == BackupType.SNAPSHOT:
                success = self._create_snapshot_backup(source_path, backup_path, backup_info)
            else:
                logger.error(f"Unknown backup type: {backup_info.backup_type}")
                return False
            
            if success:
                # Update backup info
                backup_info.size_bytes = self._get_path_size(backup_path)
                backup_info.checksum = self._calculate_checksum(backup_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Error performing backup: {e}")
            return False
    
    def _create_full_backup(self, source_path: Path, backup_path: Path, backup_info: BackupInfo) -> bool:
        """Create full backup."""
        try:
            if source_path.is_file():
                # File backup
                if self.config.compression == CompressionType.GZIP:
                    with open(source_path, 'rb') as f_in:
                        with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    backup_info.backup_path = f"{backup_path}.gz"
                else:
                    shutil.copy2(source_path, backup_path)
            else:
                # Directory backup
                if self.config.compression == CompressionType.TAR_GZ:
                    with tarfile.open(f"{backup_path}.tar.gz", "w:gz") as tar:
                        tar.add(source_path, arcname=source_path.name)
                    backup_info.backup_path = f"{backup_path}.tar.gz"
                else:
                    shutil.copytree(source_path, backup_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating full backup: {e}")
            return False
    
    def _create_incremental_backup(self, source_path: Path, backup_path: Path, backup_info: BackupInfo) -> bool:
        """Create incremental backup."""
        try:
            # Find the most recent full backup
            parent_backup = self._find_parent_backup(backup_info)
            
            if not parent_backup:
                logger.warning("No parent backup found, creating full backup instead")
                return self._create_full_backup(source_path, backup_path, backup_info)
            
            # Create incremental backup (simplified implementation)
            # In real implementation, would track file modifications
            return self._create_full_backup(source_path, backup_path, backup_info)
            
        except Exception as e:
            logger.error(f"Error creating incremental backup: {e}")
            return False
    
    def _create_differential_backup(self, source_path: Path, backup_path: Path, backup_info: BackupInfo) -> bool:
        """Create differential backup."""
        try:
            # Find the most recent full backup
            parent_backup = self._find_parent_backup(backup_info)
            
            if not parent_backup:
                logger.warning("No parent backup found, creating full backup instead")
                return self._create_full_backup(source_path, backup_path, backup_info)
            
            # Create differential backup (simplified implementation)
            return self._create_full_backup(source_path, backup_path, backup_info)
            
        except Exception as e:
            logger.error(f"Error creating differential backup: {e}")
            return False
    
    def _create_snapshot_backup(self, source_path: Path, backup_path: Path, backup_info: BackupInfo) -> bool:
        """Create snapshot backup."""
        try:
            # Snapshot backup (simplified implementation)
            # In real implementation, would use filesystem snapshots
            return self._create_full_backup(source_path, backup_path, backup_info)
            
        except Exception as e:
            logger.error(f"Error creating snapshot backup: {e}")
            return False
    
    def _perform_restore(self, backup_info: BackupInfo, target_path: str) -> bool:
        """Perform the actual restore."""
        try:
            backup_path = Path(backup_info.backup_path)
            target_path = Path(target_path)
            
            if not backup_path.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore based on compression type
            if backup_info.compression == CompressionType.GZIP and str(backup_path).endswith('.gz'):
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(target_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            elif backup_info.compression == CompressionType.TAR_GZ and str(backup_path).endswith('.tar.gz'):
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall(target_path.parent)
            else:
                if backup_path.is_file():
                    shutil.copy2(backup_path, target_path)
                else:
                    shutil.copytree(backup_path, target_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing restore: {e}")
            return False
    
    def _verify_backup(self, backup_info: BackupInfo) -> bool:
        """Verify backup integrity."""
        try:
            backup_path = Path(backup_info.backup_path)
            
            if not backup_path.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            # Verify checksum
            if backup_info.checksum:
                current_checksum = self._calculate_checksum(backup_path)
                if current_checksum != backup_info.checksum:
                    logger.error(f"Checksum mismatch for backup {backup_info.backup_id}")
                    return False
            
            # Verify file structure
            if backup_info.compression == CompressionType.TAR_GZ:
                try:
                    with tarfile.open(backup_path, "r:gz") as tar:
                        tar.getmembers()  # This will raise an exception if corrupted
                except Exception as e:
                    logger.error(f"Tar file verification failed: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False
    
    def _find_parent_backup(self, backup_info: BackupInfo) -> Optional[BackupInfo]:
        """Find parent backup for incremental/differential backups."""
        try:
            # Find the most recent full backup
            full_backups = [
                b for b in self.backups.values() 
                if b.backup_type == BackupType.FULL and b.status in [BackupStatus.COMPLETED, BackupStatus.VERIFIED]
            ]
            
            if not full_backups:
                return None
            
            # Return the most recent full backup
            return max(full_backups, key=lambda b: b.created_at)
            
        except Exception as e:
            logger.error(f"Error finding parent backup: {e}")
            return None
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        return f"backup_{int(time.time())}_{hash(str(time.time())) % 10000:04d}"
    
    def _get_path_size(self, path: Path) -> int:
        """Get total size of path."""
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                total_size = 0
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                return total_size
            else:
                return 0
        except Exception:
            return 0
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate checksum for path."""
        try:
            hasher = hashlib.sha256()
            
            if path.is_file():
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            elif path.is_dir():
                # Calculate checksum for directory by hashing all files
                for file_path in sorted(path.rglob('*')):
                    if file_path.is_file():
                        with open(file_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def _cleanup_old_backups(self) -> None:
        """Cleanup old backups based on retention policy."""
        try:
            cutoff_time = time.time() - (self.config.retention_days * 86400)
            
            # Find old backups
            old_backups = [
                backup_id for backup_id, backup_info in self.backups.items()
                if backup_info.created_at < cutoff_time
            ]
            
            # Delete old backups
            for backup_id in old_backups:
                self.delete_backup(backup_id)
            
            # Limit number of backups
            if len(self.backups) > self.config.max_backups:
                # Sort by creation time and delete oldest
                sorted_backups = sorted(
                    self.backups.items(),
                    key=lambda x: x[1].created_at
                )
                
                excess_count = len(self.backups) - self.config.max_backups
                for backup_id, _ in sorted_backups[:excess_count]:
                    self.delete_backup(backup_id)
            
            logger.info(f"Cleaned up {len(old_backups)} old backups")
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def _load_backup_metadata(self) -> None:
        """Load backup metadata from disk."""
        try:
            metadata_file = self.backup_dir / "backup_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                for backup_data in data.get('backups', []):
                    backup_info = BackupInfo(**backup_data)
                    self.backups[backup_info.backup_id] = backup_info
                
                logger.info(f"Loaded {len(self.backups)} backup metadata entries")
            
        except Exception as e:
            logger.error(f"Error loading backup metadata: {e}")
    
    def _save_backup_metadata(self, backup_info: BackupInfo) -> None:
        """Save backup metadata to disk."""
        try:
            metadata_file = self.backup_dir / "backup_metadata.json"
            
            # Load existing data
            data = {"backups": []}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
            
            # Update backup info
            backup_data = {
                "backup_id": backup_info.backup_id,
                "backup_type": backup_info.backup_type.value,
                "source_path": backup_info.source_path,
                "backup_path": backup_info.backup_path,
                "size_bytes": backup_info.size_bytes,
                "created_at": backup_info.created_at,
                "status": backup_info.status.value,
                "compression": backup_info.compression.value,
                "checksum": backup_info.checksum,
                "parent_backup_id": backup_info.parent_backup_id,
                "metadata": backup_info.metadata
            }
            
            # Update or add backup data
            backups = data["backups"]
            for i, existing_backup in enumerate(backups):
                if existing_backup["backup_id"] == backup_info.backup_id:
                    backups[i] = backup_data
                    break
            else:
                backups.append(backup_data)
            
            # Save updated data
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving backup metadata: {e}")
    
    def _save_all_backup_metadata(self) -> None:
        """Save all backup metadata to disk."""
        try:
            metadata_file = self.backup_dir / "backup_metadata.json"
            
            data = {
                "backups": [
                    {
                        "backup_id": backup_info.backup_id,
                        "backup_type": backup_info.backup_type.value,
                        "source_path": backup_info.source_path,
                        "backup_path": backup_info.backup_path,
                        "size_bytes": backup_info.size_bytes,
                        "created_at": backup_info.created_at,
                        "status": backup_info.status.value,
                        "compression": backup_info.compression.value,
                        "checksum": backup_info.checksum,
                        "parent_backup_id": backup_info.parent_backup_id,
                        "metadata": backup_info.metadata
                    }
                    for backup_info in self.backups.values()
                ]
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving all backup metadata: {e}")
    
    def _save_schedule_metadata(self) -> None:
        """Save schedule metadata to disk."""
        try:
            metadata_file = self.backup_dir / "schedule_metadata.json"
            
            data = {
                "schedules": [
                    {
                        "schedule_id": schedule.schedule_id,
                        "name": schedule.name,
                        "backup_type": schedule.backup_type.value,
                        "source_paths": schedule.source_paths,
                        "enabled": schedule.enabled,
                        "interval_seconds": schedule.interval_seconds,
                        "last_run": schedule.last_run,
                        "next_run": schedule.next_run
                    }
                    for schedule in self.schedules.values()
                ]
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving schedule metadata: {e}")
    
    def _scheduler_loop(self) -> None:
        """Scheduler loop for automatic backups."""
        while self.running:
            try:
                current_time = time.time()
                
                for schedule in self.schedules.values():
                    if (schedule.enabled and 
                        schedule.next_run and 
                        current_time >= schedule.next_run):
                        
                        logger.info(f"Running scheduled backup: {schedule.schedule_id}")
                        
                        # Run backup for each source path
                        for source_path in schedule.source_paths:
                            self.create_backup(source_path, schedule.backup_type)
                        
                        # Update schedule
                        schedule.last_run = current_time
                        schedule.next_run = current_time + schedule.interval_seconds
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")

__all__ = [
    "BackupManager",
    "BackupInfo",
    "BackupConfig",
    "BackupSchedule",
    "BackupType",
    "BackupStatus",
    "CompressionType",
]