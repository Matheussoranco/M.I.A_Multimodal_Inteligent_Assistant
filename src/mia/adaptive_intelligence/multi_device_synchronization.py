import asyncio
import base64
import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..providers import provider_registry

logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    """Directions for data synchronization."""

    UPLOAD = "upload"  # Device to cloud
    DOWNLOAD = "download"  # Cloud to device
    BIDIRECTIONAL = "bidirectional"  # Both ways


class SyncPriority(Enum):
    """Priority levels for synchronization."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class DeviceType(Enum):
    """Types of devices that can be synchronized."""

    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"
    WEARABLE = "wearable"
    BROWSER = "browser"
    SERVER = "server"


class SyncStatus(Enum):
    """Status of synchronization operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class Device:
    """Represents a synchronized device."""

    id: str
    user_id: str
    device_type: DeviceType
    device_name: str
    platform: str  # OS/platform info
    app_version: str
    capabilities: Set[str] = field(default_factory=set)  # Device capabilities
    last_seen: datetime = field(default_factory=datetime.now)
    is_online: bool = True
    sync_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncData:
    """Represents data to be synchronized."""

    id: str
    user_id: str
    data_type: str  # Type of data (profile, memory, settings, etc.)
    content: Any  # The actual data
    version: int = 1
    checksum: str = ""  # Data integrity check
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    source_device_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    priority: SyncPriority = SyncPriority.NORMAL
    ttl: Optional[timedelta] = None  # Time to live


@dataclass
class SyncOperation:
    """Represents a synchronization operation."""

    id: str
    user_id: str
    source_device_id: str
    target_device_ids: List[str]
    data_items: List[SyncData]
    direction: SyncDirection
    priority: SyncPriority
    status: SyncStatus = SyncStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class SyncConflict:
    """Represents a synchronization conflict."""

    id: str
    user_id: str
    data_id: str
    conflicting_versions: List[SyncData]
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    resolved_version: Optional[SyncData] = None


class DeviceManager:
    """
    Manages device registration and status tracking.

    Handles device discovery, registration, and heartbeat monitoring.
    """

    def __init__(self):
        self.devices: Dict[str, Device] = {}
        self.user_devices: Dict[str, Set[str]] = {}  # user_id -> device_ids
        self.online_devices: Set[str] = set()
        self._lock = threading.RLock()

        # Heartbeat monitoring
        self.heartbeat_timeout = timedelta(minutes=5)
        self.cleanup_task: Optional[asyncio.Task] = None

    def register_device(self, device: Device) -> str:
        """Register a new device."""
        with self._lock:
            device_id = device.id or str(uuid.uuid4())
            device.id = device_id
            device.last_seen = datetime.now()
            device.is_online = True

            self.devices[device_id] = device

            # Update user device mapping
            if device.user_id not in self.user_devices:
                self.user_devices[device.user_id] = set()
            self.user_devices[device.user_id].add(device_id)

            self.online_devices.add(device_id)

            logger.info(
                f"Registered device: {device_id} ({device.device_name}) for user {device.user_id}"
            )
            return device_id

    def update_device_status(
        self,
        device_id: str,
        is_online: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update device status."""
        with self._lock:
            if device_id in self.devices:
                device = self.devices[device_id]
                device.is_online = is_online
                device.last_seen = datetime.now()

                if metadata:
                    device.metadata.update(metadata)

                if is_online:
                    self.online_devices.add(device_id)
                else:
                    self.online_devices.discard(device_id)

                logger.debug(
                    f"Updated device status: {device_id} - online: {is_online}"
                )

    def get_device(self, device_id: str) -> Optional[Device]:
        """Get device information."""
        with self._lock:
            return self.devices.get(device_id)

    def get_user_devices(
        self, user_id: str, online_only: bool = False
    ) -> List[Device]:
        """Get all devices for a user."""
        with self._lock:
            device_ids = self.user_devices.get(user_id, set())

            devices = []
            for device_id in device_ids:
                device = self.devices.get(device_id)
                if device and (not online_only or device.is_online):
                    devices.append(device)

            return devices

    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device."""
        with self._lock:
            if device_id not in self.devices:
                return False

            device = self.devices[device_id]
            user_id = device.user_id

            # Remove from mappings
            self.user_devices[user_id].discard(device_id)
            self.online_devices.discard(device_id)

            del self.devices[device_id]

            logger.info(f"Unregistered device: {device_id}")
            return True

    def check_device_heartbeats(self):
        """Check for offline devices based on heartbeat timeout."""
        with self._lock:
            current_time = datetime.now()
            offline_devices = []

            for device_id, device in self.devices.items():
                if (
                    device.is_online
                    and current_time - device.last_seen
                    > self.heartbeat_timeout
                ):
                    device.is_online = False
                    self.online_devices.discard(device_id)
                    offline_devices.append(device_id)

            if offline_devices:
                logger.info(
                    f"Marked {len(offline_devices)} devices as offline due to heartbeat timeout"
                )

            return offline_devices

    def get_device_stats(self) -> Dict[str, Any]:
        """Get device statistics."""
        with self._lock:
            total_devices = len(self.devices)
            online_devices = len(self.online_devices)

            device_types = {}
            platforms = {}

            for device in self.devices.values():
                device_types[device.device_type.value] = (
                    device_types.get(device.device_type.value, 0) + 1
                )
                platforms[device.platform] = (
                    platforms.get(device.platform, 0) + 1
                )

            return {
                "total_devices": total_devices,
                "online_devices": online_devices,
                "offline_devices": total_devices - online_devices,
                "device_types": device_types,
                "platforms": platforms,
            }


class SyncDataStore:
    """
    Storage system for synchronized data.

    Manages data versioning, integrity, and conflict resolution.
    """

    def __init__(self):
        self.data_store: Dict[str, Dict[str, SyncData]] = (
            {}
        )  # user_id -> data_id -> data
        self.data_versions: Dict[str, Dict[str, List[SyncData]]] = (
            {}
        )  # user_id -> data_id -> versions
        self.data_checksums: Dict[str, str] = {}  # data_id -> checksum
        self._lock = threading.RLock()

        # Cleanup settings
        self.max_versions_per_data = 10
        self.cleanup_task: Optional[asyncio.Task] = None

    def store_data(self, data: SyncData) -> str:
        """Store synchronized data."""
        with self._lock:
            data_id = data.id or str(uuid.uuid4())
            data.id = data_id

            # Generate checksum
            data.checksum = self._generate_checksum(data.content)

            # Initialize user data store
            if data.user_id not in self.data_store:
                self.data_store[data.user_id] = {}
                self.data_versions[data.user_id] = {}

            # Store current version
            self.data_store[data.user_id][data_id] = data

            # Store version history
            if data_id not in self.data_versions[data.user_id]:
                self.data_versions[data.user_id][data_id] = []

            self.data_versions[data.user_id][data_id].append(data)

            # Keep only recent versions
            if (
                len(self.data_versions[data.user_id][data_id])
                > self.max_versions_per_data
            ):
                self.data_versions[data.user_id][data_id] = self.data_versions[
                    data.user_id
                ][data_id][-self.max_versions_per_data :]

            # Update checksum cache
            self.data_checksums[data_id] = data.checksum

            logger.debug(
                f"Stored sync data: {data_id} for user {data.user_id}"
            )
            return data_id

    def get_data(self, user_id: str, data_id: str) -> Optional[SyncData]:
        """Get synchronized data."""
        with self._lock:
            user_data = self.data_store.get(user_id, {})
            return user_data.get(data_id)

    def get_user_data(
        self, user_id: str, data_type: Optional[str] = None
    ) -> List[SyncData]:
        """Get all data for a user."""
        with self._lock:
            user_data = self.data_store.get(user_id, {})

            data_list = list(user_data.values())

            if data_type:
                data_list = [d for d in data_list if d.data_type == data_type]

            return data_list

    def update_data(
        self,
        user_id: str,
        data_id: str,
        new_content: Any,
        source_device_id: Optional[str] = None,
    ) -> Optional[SyncData]:
        """Update synchronized data."""
        with self._lock:
            existing_data = self.get_data(user_id, data_id)
            if not existing_data:
                return None

            # Create new version
            new_version = SyncData(
                id=data_id,
                user_id=user_id,
                data_type=existing_data.data_type,
                content=new_content,
                version=existing_data.version + 1,
                source_device_id=source_device_id,
                tags=existing_data.tags.copy(),
                priority=existing_data.priority,
                ttl=existing_data.ttl,
            )

            self.store_data(new_version)
            return new_version

    def delete_data(self, user_id: str, data_id: str) -> bool:
        """Delete synchronized data."""
        with self._lock:
            if (
                user_id not in self.data_store
                or data_id not in self.data_store[user_id]
            ):
                return False

            del self.data_store[user_id][data_id]

            # Also remove from versions
            if (
                user_id in self.data_versions
                and data_id in self.data_versions[user_id]
            ):
                del self.data_versions[user_id][data_id]

            # Remove checksum
            if data_id in self.data_checksums:
                del self.data_checksums[data_id]

            logger.debug(f"Deleted sync data: {data_id} for user {user_id}")
            return True

    def detect_conflicts(
        self, user_id: str, data_id: str, new_data: SyncData
    ) -> Optional[SyncConflict]:
        """Detect conflicts in data synchronization."""
        with self._lock:
            existing_data = self.get_data(user_id, data_id)
            if not existing_data:
                return None

            # Check if versions conflict
            if new_data.version <= existing_data.version:
                # Check content differences
                new_checksum = self._generate_checksum(new_data.content)
                existing_checksum = existing_data.checksum

                if new_checksum != existing_checksum:
                    # Content conflict detected
                    conflict = SyncConflict(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        data_id=data_id,
                        conflicting_versions=[existing_data, new_data],
                    )

                    logger.warning(
                        f"Sync conflict detected: {data_id} for user {user_id}"
                    )
                    return conflict

            return None

    def resolve_conflict(
        self,
        conflict: SyncConflict,
        resolution_strategy: str,
        resolved_content: Any,
    ) -> SyncData:
        """Resolve a synchronization conflict."""
        with self._lock:
            # Create resolved version
            resolved_data = SyncData(
                id=conflict.data_id,
                user_id=conflict.user_id,
                data_type=conflict.conflicting_versions[0].data_type,
                content=resolved_content,
                version=max(v.version for v in conflict.conflicting_versions)
                + 1,
                tags=conflict.conflicting_versions[0].tags.copy(),
            )

            # Store resolved data
            self.store_data(resolved_data)

            # Mark conflict as resolved
            conflict.resolved = True
            conflict.resolution_strategy = resolution_strategy
            conflict.resolved_version = resolved_data

            logger.info(
                f"Resolved sync conflict: {conflict.id} using strategy {resolution_strategy}"
            )
            return resolved_data

    def get_data_versions(self, user_id: str, data_id: str) -> List[SyncData]:
        """Get version history for data."""
        with self._lock:
            if (
                user_id not in self.data_versions
                or data_id not in self.data_versions[user_id]
            ):
                return []

            return self.data_versions[user_id][data_id].copy()

    def _generate_checksum(self, content: Any) -> str:
        """Generate checksum for data integrity."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def cleanup_expired_data(self):
        """Clean up expired data based on TTL."""
        with self._lock:
            current_time = datetime.now()
            expired_data = []

            for user_id, user_data in self.data_store.items():
                for data_id, data in user_data.items():
                    if data.ttl and current_time - data.created_at > data.ttl:
                        expired_data.append((user_id, data_id))

            for user_id, data_id in expired_data:
                self.delete_data(user_id, data_id)

            if expired_data:
                logger.info(
                    f"Cleaned up {len(expired_data)} expired data items"
                )

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            total_users = len(self.data_store)
            total_data_items = sum(
                len(user_data) for user_data in self.data_store.values()
            )
            total_versions = sum(
                len(versions)
                for user_versions in self.data_versions.values()
                for versions in user_versions.values()
            )

            data_types = {}
            for user_data in self.data_store.values():
                for data in user_data.values():
                    data_types[data.data_type] = (
                        data_types.get(data.data_type, 0) + 1
                    )

            return {
                "total_users": total_users,
                "total_data_items": total_data_items,
                "total_versions": total_versions,
                "data_types": data_types,
                "avg_versions_per_item": total_versions
                / max(total_data_items, 1),
            }


class SyncCoordinator:
    """
    Coordinates synchronization operations across devices.

    Manages sync queues, conflict resolution, and progress tracking.
    """

    def __init__(
        self, device_manager: DeviceManager, data_store: SyncDataStore
    ):
        self.device_manager = device_manager
        self.data_store = data_store

        self.sync_operations: Dict[str, SyncOperation] = {}
        self.pending_operations: asyncio.Queue = asyncio.Queue()
        self.active_operations: Set[str] = set()

        self.conflicts: Dict[str, SyncConflict] = {}  # conflict_id -> conflict

        self._lock = threading.RLock()
        self.max_concurrent_operations = 5

        # Background processing
        self.sync_task: Optional[asyncio.Task] = None
        self.conflict_resolution_task: Optional[asyncio.Task] = None

    def queue_sync_operation(self, operation: SyncOperation):
        """Queue a synchronization operation."""
        with self._lock:
            operation_id = operation.id or str(uuid.uuid4())
            operation.id = operation_id

            self.sync_operations[operation_id] = operation

            # Add to async queue
            asyncio.create_task(self.pending_operations.put(operation))

            logger.info(
                f"Queued sync operation: {operation_id} for user {operation.user_id}"
            )
            return operation_id

    async def process_sync_operations(self):
        """Process pending synchronization operations."""
        while True:
            try:
                # Get next operation
                operation = await self.pending_operations.get()

                # Check concurrency limit
                if (
                    len(self.active_operations)
                    >= self.max_concurrent_operations
                ):
                    # Put back in queue and wait
                    await self.pending_operations.put(operation)
                    await asyncio.sleep(1)
                    continue

                # Start processing
                self.active_operations.add(operation.id)
                operation.status = SyncStatus.IN_PROGRESS
                operation.started_at = datetime.now()

                try:
                    await self._execute_sync_operation(operation)
                    operation.status = SyncStatus.COMPLETED
                    operation.completed_at = datetime.now()
                    operation.progress = 1.0

                except Exception as exc:
                    operation.status = SyncStatus.FAILED
                    operation.error_message = str(exc)
                    operation.completed_at = datetime.now()

                    # Retry logic
                    if operation.retry_count < operation.max_retries:
                        operation.retry_count += 1
                        operation.status = SyncStatus.PENDING
                        await self.pending_operations.put(operation)
                        logger.warning(
                            f"Retrying sync operation {operation.id} (attempt {operation.retry_count})"
                        )
                    else:
                        logger.error(
                            f"Sync operation failed permanently: {operation.id} - {exc}"
                        )

                finally:
                    self.active_operations.discard(operation.id)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Sync processing error: {exc}")
                await asyncio.sleep(5)

    async def _execute_sync_operation(self, operation: SyncOperation):
        """Execute a synchronization operation."""
        logger.info(
            f"Executing sync operation: {operation.id} ({operation.direction.value})"
        )

        if operation.direction == SyncDirection.UPLOAD:
            await self._execute_upload_sync(operation)
        elif operation.direction == SyncDirection.DOWNLOAD:
            await self._execute_download_sync(operation)
        elif operation.direction == SyncDirection.BIDIRECTIONAL:
            await self._execute_bidirectional_sync(operation)

    async def _execute_upload_sync(self, operation: SyncOperation):
        """Execute upload synchronization (device to cloud)."""
        for data_item in operation.data_items:
            # Check for conflicts
            conflict = self.data_store.detect_conflicts(
                operation.user_id, data_item.id, data_item
            )

            if conflict:
                # Handle conflict
                await self._handle_sync_conflict(conflict, operation)
            else:
                # Store data
                self.data_store.store_data(data_item)

            # Update progress
            operation.progress += 1.0 / len(operation.data_items)

    async def _execute_download_sync(self, operation: SyncOperation):
        """Execute download synchronization (cloud to device)."""
        # In a real implementation, this would send data to target devices
        # For now, just mark as completed
        for data_item in operation.data_items:
            # Simulate sending to devices
            await asyncio.sleep(0.01)  # Simulate network delay
            operation.progress += 1.0 / len(operation.data_items)

    async def _execute_bidirectional_sync(self, operation: SyncOperation):
        """Execute bidirectional synchronization."""
        # Combine upload and download
        await self._execute_upload_sync(operation)
        await self._execute_download_sync(operation)

    async def _handle_sync_conflict(
        self, conflict: SyncConflict, operation: SyncOperation
    ):
        """Handle a synchronization conflict."""
        # Store conflict for resolution
        self.conflicts[conflict.id] = conflict

        # For now, use automatic resolution strategy (latest wins)
        # In a real implementation, this might involve user interaction
        latest_version = max(
            conflict.conflicting_versions, key=lambda v: v.modified_at
        )

        resolved_data = self.data_store.resolve_conflict(
            conflict, "latest_wins", latest_version.content
        )

        logger.info(
            f"Auto-resolved conflict {conflict.id} using latest version"
        )

    def get_operation_status(
        self, operation_id: str
    ) -> Optional[SyncOperation]:
        """Get the status of a sync operation."""
        with self._lock:
            return self.sync_operations.get(operation_id)

    def get_pending_conflicts(self, user_id: str) -> List[SyncConflict]:
        """Get pending conflicts for a user."""
        with self._lock:
            user_conflicts = [
                conflict
                for conflict in self.conflicts.values()
                if conflict.user_id == user_id and not conflict.resolved
            ]
            return user_conflicts

    def resolve_conflict_manual(
        self, conflict_id: str, resolution_strategy: str, resolved_content: Any
    ) -> bool:
        """Manually resolve a conflict."""
        with self._lock:
            if conflict_id not in self.conflicts:
                return False

            conflict = self.conflicts[conflict_id]

            self.data_store.resolve_conflict(
                conflict, resolution_strategy, resolved_content
            )

            return True

    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        with self._lock:
            total_operations = len(self.sync_operations)
            completed_operations = len(
                [
                    op
                    for op in self.sync_operations.values()
                    if op.status == SyncStatus.COMPLETED
                ]
            )
            failed_operations = len(
                [
                    op
                    for op in self.sync_operations.values()
                    if op.status == SyncStatus.FAILED
                ]
            )
            pending_operations = self.pending_operations.qsize()
            active_operations = len(self.active_operations)

            return {
                "total_operations": total_operations,
                "completed_operations": completed_operations,
                "failed_operations": failed_operations,
                "pending_operations": pending_operations,
                "active_operations": active_operations,
                "pending_conflicts": len(
                    [c for c in self.conflicts.values() if not c.resolved]
                ),
            }


class SyncConnectors:
    """
    Connectors for different synchronization protocols and platforms.

    Handles device-specific synchronization logic.
    """

    def __init__(self):
        self.connectors: Dict[str, Callable] = {}
        self.device_connectors: Dict[str, Any] = (
            {}
        )  # device_id -> connector instance

    def register_connector(self, platform: str, connector_class: Callable):
        """Register a connector for a specific platform."""
        self.connectors[platform] = connector_class
        logger.info(f"Registered sync connector for platform: {platform}")

    def get_connector(self, device: Device):
        """Get or create a connector for a device."""
        if device.id in self.device_connectors:
            return self.device_connectors[device.id]

        platform = device.platform.lower()
        if platform in self.connectors:
            connector_class = self.connectors[platform]
            connector = connector_class(device)
            self.device_connectors[device.id] = connector
            return connector

        # Return generic connector
        return GenericSyncConnector(device)

    async def sync_device(
        self, device: Device, operation: SyncOperation
    ) -> Dict[str, Any]:
        """Synchronize data with a specific device."""
        connector = self.get_connector(device)

        try:
            result = await connector.sync(operation)
            return result
        except Exception as exc:
            logger.error(f"Device sync failed for {device.id}: {exc}")
            raise exc


class GenericSyncConnector:
    """Generic synchronization connector."""

    def __init__(self, device: Device):
        self.device = device

    async def sync(self, operation: SyncOperation) -> Dict[str, Any]:
        """Perform synchronization with the device."""
        # Generic sync logic - in a real implementation, this would use
        # platform-specific APIs (WebSocket, REST API, etc.)

        # Simulate sync operation
        await asyncio.sleep(0.1)

        return {
            "device_id": self.device.id,
            "operation_id": operation.id,
            "status": "completed",
            "bytes_transferred": 1024,  # Simulated
            "items_synced": len(operation.data_items),
        }


class SyncDashboard:
    """
    Dashboard for monitoring and managing synchronization.

    Provides insights and controls for sync operations.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        coordinator: SyncCoordinator,
        data_store: SyncDataStore,
    ):
        self.device_manager = device_manager
        self.coordinator = coordinator
        self.data_store = data_store

    def get_dashboard_data(
        self, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get dashboard data for synchronization status."""
        device_stats = self.device_manager.get_device_stats()
        sync_stats = self.coordinator.get_sync_stats()
        storage_stats = self.data_store.get_storage_stats()

        # Get user-specific data if provided
        user_data = {}
        if user_id:
            user_devices = self.device_manager.get_user_devices(user_id)
            user_sync_data = self.data_store.get_user_data(user_id)
            pending_conflicts = self.coordinator.get_pending_conflicts(user_id)

            user_data = {
                "devices": [
                    {
                        "id": d.id,
                        "name": d.device_name,
                        "type": d.device_type.value,
                        "online": d.is_online,
                        "last_seen": d.last_seen.isoformat(),
                    }
                    for d in user_devices
                ],
                "data_items": len(user_sync_data),
                "pending_conflicts": len(pending_conflicts),
                "storage_used": len(user_sync_data) * 1024,  # Rough estimate
            }

        return {
            "global_stats": {**device_stats, **sync_stats, **storage_stats},
            "user_data": user_data,
            "recent_activity": self._get_recent_activity(),
            "health_status": self._get_health_status(),
        }

    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent synchronization activity."""
        # Get recent operations
        recent_ops = [
            op
            for op in self.coordinator.sync_operations.values()
            if op.completed_at
            and datetime.now() - op.completed_at < timedelta(hours=24)
        ]

        recent_ops.sort(
            key=lambda op: op.completed_at or datetime.min, reverse=True
        )

        return [
            {
                "operation_id": op.id,
                "user_id": op.user_id,
                "direction": op.direction.value,
                "status": op.status.value,
                "completed_at": (
                    op.completed_at.isoformat() if op.completed_at else None
                ),
                "items_synced": len(op.data_items),
            }
            for op in recent_ops[:10]
        ]

    def _get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the sync system."""
        sync_stats = self.coordinator.get_sync_stats()

        # Calculate health score
        success_rate = sync_stats["completed_operations"] / max(
            sync_stats["total_operations"], 1
        )
        failure_rate = sync_stats["failed_operations"] / max(
            sync_stats["total_operations"], 1
        )

        health_score = (success_rate * 0.7 + (1 - failure_rate) * 0.3) * 100

        status = "healthy"
        if health_score < 70:
            status = "warning"
        if health_score < 50:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "pending_operations": sync_stats["pending_operations"],
            "active_operations": sync_stats["active_operations"],
        }


class MultiDeviceSynchronizationSystem:
    """
    Multi-device synchronization system.
    """

    def __init__(self, config_manager=None):
        self.config_manager = config_manager

        # Core components
        self.device_manager = DeviceManager()
        self.data_store = SyncDataStore()
        self.coordinator = SyncCoordinator(
            self.device_manager, self.data_store
        )
        self.connectors = SyncConnectors()
        self.dashboard = SyncDashboard(
            self.device_manager, self.coordinator, self.data_store
        )

        # Background tasks
        self.sync_processing_task: Optional[asyncio.Task] = None
        self.device_cleanup_task: Optional[asyncio.Task] = None
        self.data_cleanup_task: Optional[asyncio.Task] = None

        logger.info("Multi-Device Synchronization System initialized")

    async def start(self):
        """Start the synchronization system."""
        # Start background tasks
        self.sync_processing_task = asyncio.create_task(
            self.coordinator.process_sync_operations()
        )
        self.device_cleanup_task = asyncio.create_task(
            self._device_cleanup_loop()
        )
        self.data_cleanup_task = asyncio.create_task(self._data_cleanup_loop())

        logger.info("Multi-Device Synchronization System started")

    async def stop(self):
        """Stop the synchronization system."""
        tasks_to_cancel = []

        if self.sync_processing_task:
            self.sync_processing_task.cancel()
            tasks_to_cancel.append(self.sync_processing_task)

        if self.device_cleanup_task:
            self.device_cleanup_task.cancel()
            tasks_to_cancel.append(self.device_cleanup_task)

        if self.data_cleanup_task:
            self.data_cleanup_task.cancel()
            tasks_to_cancel.append(self.data_cleanup_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        logger.info("Multi-Device Synchronization System stopped")

    async def _device_cleanup_loop(self):
        """Periodic device cleanup."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                self.device_manager.check_device_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Device cleanup error: {exc}")

    async def _data_cleanup_loop(self):
        """Periodic data cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                self.data_store.cleanup_expired_data()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Data cleanup error: {exc}")

    def register_device(
        self, user_id: str, device_info: Dict[str, Any]
    ) -> str:
        """Register a new device for synchronization."""
        device = Device(
            id="",
            user_id=user_id,
            device_type=DeviceType(device_info.get("device_type", "desktop")),
            device_name=device_info.get("device_name", "Unknown Device"),
            platform=device_info.get("platform", "unknown"),
            app_version=device_info.get("app_version", "1.0.0"),
            capabilities=set(device_info.get("capabilities", [])),
        )

        return self.device_manager.register_device(device)

    def sync_data(
        self,
        user_id: str,
        data: Dict[str, Any],
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
    ) -> str:
        """Synchronize data across devices."""
        # Create sync data
        sync_data = SyncData(
            id=data.get("id") or "",
            user_id=user_id,
            data_type=data.get("type", "generic"),
            content=data.get("content"),
            source_device_id=data.get("source_device_id"),
            tags=set(data.get("tags", [])),
            priority=SyncPriority(data.get("priority", "normal")),
            ttl=data.get("ttl"),  # Should be timedelta if provided
        )

        # Get user's devices
        user_devices = self.device_manager.get_user_devices(
            user_id, online_only=True
        )
        target_device_ids = [d.id for d in user_devices]

        # Create sync operation
        operation = SyncOperation(
            id="",
            user_id=user_id,
            source_device_id=sync_data.source_device_id or "system",
            target_device_ids=target_device_ids,
            data_items=[sync_data],
            direction=direction,
            priority=sync_data.priority,
        )

        return self.coordinator.queue_sync_operation(operation)

    def get_sync_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a synchronization operation."""
        operation = self.coordinator.get_operation_status(operation_id)
        if not operation:
            return None

        return {
            "id": operation.id,
            "status": operation.status.value,
            "progress": operation.progress,
            "started_at": (
                operation.started_at.isoformat()
                if operation.started_at
                else None
            ),
            "completed_at": (
                operation.completed_at.isoformat()
                if operation.completed_at
                else None
            ),
            "error_message": operation.error_message,
        }

    def get_user_devices(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all devices for a user."""
        devices = self.device_manager.get_user_devices(user_id)

        return [
            {
                "id": d.id,
                "name": d.device_name,
                "type": d.device_type.value,
                "platform": d.platform,
                "online": d.is_online,
                "last_seen": d.last_seen.isoformat(),
                "capabilities": list(d.capabilities),
            }
            for d in devices
        ]

    def get_user_data(
        self, user_id: str, data_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get synchronized data for a user."""
        data_items = self.data_store.get_user_data(user_id, data_type)

        return [
            {
                "id": d.id,
                "type": d.data_type,
                "content": d.content,
                "version": d.version,
                "modified_at": d.modified_at.isoformat(),
                "source_device": d.source_device_id,
                "tags": list(d.tags),
            }
            for d in data_items
        ]

    def resolve_conflict(
        self, conflict_id: str, resolution: Dict[str, Any]
    ) -> bool:
        """Resolve a synchronization conflict."""
        return self.coordinator.resolve_conflict_manual(
            conflict_id,
            resolution.get("strategy", "manual"),
            resolution.get("content"),
        )

    def get_dashboard(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get synchronization dashboard data."""
        return self.dashboard.get_dashboard_data(user_id)

    def update_device_status(self, device_id: str, status: Dict[str, Any]):
        """Update device status."""
        self.device_manager.update_device_status(
            device_id, status.get("online", True), status.get("metadata", {})
        )

    def unregister_device(self, device_id: str) -> bool:
        """Unregister a device."""
        return self.device_manager.unregister_device(device_id)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        device_stats = self.device_manager.get_device_stats()
        sync_stats = self.coordinator.get_sync_stats()
        storage_stats = self.data_store.get_storage_stats()

        return {
            "devices": device_stats,
            "sync_operations": sync_stats,
            "storage": storage_stats,
            "system_health": self.dashboard._get_health_status(),
        }


# Register with provider registry
def create_multi_device_sync_system(config_manager=None, **kwargs):
    """Factory function for MultiDeviceSynchronizationSystem."""
    return MultiDeviceSynchronizationSystem(config_manager=config_manager)


provider_registry.register_lazy(
    "adaptive_intelligence",
    "multi_device_sync",
    "mia.adaptive_intelligence.multi_device_synchronization",
    "create_multi_device_sync_system",
    default=True,
)
