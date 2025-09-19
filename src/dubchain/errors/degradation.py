"""Graceful degradation system for DubChain.

This module provides graceful degradation capabilities to maintain system
stability and performance under adverse conditions.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .exceptions import DubChainError, ErrorSeverity


class DegradationLevel(Enum):
    """Degradation levels."""

    NONE = "0_none"  # No degradation
    MINIMAL = "1_minimal"  # Minimal impact
    MODERATE = "2_moderate"  # Moderate impact
    SEVERE = "3_severe"  # Severe impact
    CRITICAL = "4_critical"  # Critical impact


class ServiceLevel(Enum):
    """Service levels."""

    FULL = "full"  # Full service
    REDUCED = "reduced"  # Reduced service
    LIMITED = "limited"  # Limited service
    EMERGENCY = "emergency"  # Emergency service only
    OFFLINE = "offline"  # Service offline


class DegradationStrategy(Enum):
    """Degradation strategies."""

    THROTTLE = "throttle"  # Throttle requests
    QUEUE = "queue"  # Queue requests
    CACHE_ONLY = "cache_only"  # Serve from cache only
    READ_ONLY = "read_only"  # Read-only mode
    MAINTENANCE = "maintenance"  # Maintenance mode
    SHUTDOWN = "shutdown"  # Graceful shutdown


@dataclass
class DegradationRule:
    """Degradation rule configuration."""

    name: str
    condition: Callable[[], bool]
    level: DegradationLevel
    strategy: DegradationStrategy
    threshold: float = 0.8
    cooldown: float = 300.0  # 5 minutes
    enabled: bool = True

    def __post_init__(self):
        self.last_triggered = 0.0
        self.trigger_count = 0


@dataclass
class ServiceHealth:
    """Service health information."""

    service_name: str
    level: ServiceLevel
    degradation_level: DegradationLevel
    last_check: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class HealthChecker(ABC):
    """Abstract health checker."""

    @abstractmethod
    def check_health(self) -> ServiceHealth:
        """Check service health."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        pass


class SystemHealthChecker(HealthChecker):
    """System-level health checker."""

    def __init__(self, service_name: str = "system"):
        self.service_name = service_name
        self._logger = logging.getLogger(f"{__name__}.{service_name}")

    def check_health(self) -> ServiceHealth:
        """Check system health."""
        import psutil

        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Check disk usage
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            # Determine degradation level
            degradation_level = DegradationLevel.NONE
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                degradation_level = DegradationLevel.CRITICAL
            elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 90:
                degradation_level = DegradationLevel.SEVERE
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 80:
                degradation_level = DegradationLevel.MODERATE
            elif cpu_percent > 60 or memory_percent > 60 or disk_percent > 70:
                degradation_level = DegradationLevel.MINIMAL

            # Determine service level
            service_level = ServiceLevel.FULL
            if degradation_level == DegradationLevel.CRITICAL:
                service_level = ServiceLevel.OFFLINE
            elif degradation_level == DegradationLevel.SEVERE:
                service_level = ServiceLevel.EMERGENCY
            elif degradation_level == DegradationLevel.MODERATE:
                service_level = ServiceLevel.LIMITED
            elif degradation_level == DegradationLevel.MINIMAL:
                service_level = ServiceLevel.REDUCED

            # Collect metrics
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "memory_available": memory.available,
                "disk_free": disk.free,
            }

            # Collect errors and warnings
            errors = []
            warnings = []

            if cpu_percent > 90:
                errors.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                warnings.append(f"Elevated CPU usage: {cpu_percent:.1f}%")

            if memory_percent > 90:
                errors.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                warnings.append(f"Elevated memory usage: {memory_percent:.1f}%")

            if disk_percent > 95:
                errors.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > 90:
                warnings.append(f"Elevated disk usage: {disk_percent:.1f}%")

            return ServiceHealth(
                service_name=self.service_name,
                level=service_level,
                degradation_level=degradation_level,
                last_check=time.time(),
                metrics=metrics,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            self._logger.error(f"Error checking system health: {e}")
            return ServiceHealth(
                service_name=self.service_name,
                level=ServiceLevel.OFFLINE,
                degradation_level=DegradationLevel.CRITICAL,
                last_check=time.time(),
                errors=[f"Health check failed: {e}"],
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (
                    psutil.disk_usage("/").used / psutil.disk_usage("/").total
                )
                * 100,
                "process_count": len(psutil.pids()),
                "boot_time": psutil.boot_time(),
            }
        except Exception as e:
            self._logger.error(f"Error getting system metrics: {e}")
            return {}


class NetworkHealthChecker(HealthChecker):
    """Network health checker."""

    def __init__(self, service_name: str = "network", endpoints: List[str] = None):
        self.service_name = service_name
        self.endpoints = endpoints or []
        self._logger = logging.getLogger(f"{__name__}.{service_name}")

    def check_health(self) -> ServiceHealth:
        """Check network health."""
        try:
            import socket
            import urllib.request

            # Check network connectivity
            connectivity_issues = 0
            total_endpoints = len(self.endpoints)

            for endpoint in self.endpoints:
                try:
                    if endpoint.startswith("http"):
                        urllib.request.urlopen(endpoint, timeout=5)
                    else:
                        # Assume it's a host:port
                        host, port = endpoint.split(":")
                        socket.create_connection((host, int(port)), timeout=5)
                except Exception:
                    connectivity_issues += 1

            # Determine degradation level
            if total_endpoints == 0:
                degradation_level = DegradationLevel.NONE
            elif connectivity_issues == total_endpoints:
                degradation_level = DegradationLevel.CRITICAL
            elif connectivity_issues > total_endpoints * 0.5:
                degradation_level = DegradationLevel.SEVERE
            elif connectivity_issues > total_endpoints * 0.25:
                degradation_level = DegradationLevel.MODERATE
            else:
                degradation_level = DegradationLevel.MINIMAL

            # Determine service level
            service_level = ServiceLevel.FULL
            if degradation_level == DegradationLevel.CRITICAL:
                service_level = ServiceLevel.OFFLINE
            elif degradation_level == DegradationLevel.SEVERE:
                service_level = ServiceLevel.EMERGENCY
            elif degradation_level == DegradationLevel.MODERATE:
                service_level = ServiceLevel.LIMITED
            elif degradation_level == DegradationLevel.MINIMAL:
                service_level = ServiceLevel.REDUCED

            # Collect metrics
            metrics = {
                "total_endpoints": total_endpoints,
                "connectivity_issues": connectivity_issues,
                "connectivity_rate": (total_endpoints - connectivity_issues)
                / total_endpoints
                if total_endpoints > 0
                else 1.0,
            }

            # Collect errors and warnings
            errors = []
            warnings = []

            if connectivity_issues == total_endpoints and total_endpoints > 0:
                errors.append("All network endpoints unreachable")
            elif connectivity_issues > total_endpoints * 0.5:
                errors.append(
                    f"Multiple network endpoints unreachable: {connectivity_issues}/{total_endpoints}"
                )
            elif connectivity_issues > 0:
                warnings.append(
                    f"Some network endpoints unreachable: {connectivity_issues}/{total_endpoints}"
                )

            return ServiceHealth(
                service_name=self.service_name,
                level=service_level,
                degradation_level=degradation_level,
                last_check=time.time(),
                metrics=metrics,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            self._logger.error(f"Error checking network health: {e}")
            return ServiceHealth(
                service_name=self.service_name,
                level=ServiceLevel.OFFLINE,
                degradation_level=DegradationLevel.CRITICAL,
                last_check=time.time(),
                errors=[f"Network health check failed: {e}"],
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get network metrics."""
        return {
            "total_endpoints": len(self.endpoints),
            "connectivity_issues": 0,  # Would be calculated in real implementation
            "connectivity_rate": 1.0,
        }


class GracefulDegradationManager:
    """Graceful degradation manager."""

    def __init__(self):
        self._degradation_rules: List[DegradationRule] = []
        self._health_checkers: Dict[str, HealthChecker] = {}
        self._current_degradation_level = DegradationLevel.NONE
        self._current_service_level = ServiceLevel.FULL
        self._degradation_actions: Dict[DegradationLevel, Callable] = {}
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)

        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False
        self._monitoring_interval = 30.0  # 30 seconds

        # Setup default actions
        self._setup_default_actions()

    def _setup_default_actions(self) -> None:
        """Setup default degradation actions."""
        self._degradation_actions[DegradationLevel.NONE] = self._no_degradation_action
        self._degradation_actions[
            DegradationLevel.MINIMAL
        ] = self._minimal_degradation_action
        self._degradation_actions[
            DegradationLevel.MODERATE
        ] = self._moderate_degradation_action
        self._degradation_actions[
            DegradationLevel.SEVERE
        ] = self._severe_degradation_action
        self._degradation_actions[
            DegradationLevel.CRITICAL
        ] = self._critical_degradation_action

    def add_degradation_rule(self, rule: DegradationRule) -> None:
        """Add a degradation rule."""
        with self._lock:
            self._degradation_rules.append(rule)
            self._logger.info(f"Added degradation rule: {rule.name}")

    def remove_degradation_rule(self, name: str) -> None:
        """Remove a degradation rule."""
        with self._lock:
            self._degradation_rules = [
                rule for rule in self._degradation_rules if rule.name != name
            ]
            self._logger.info(f"Removed degradation rule: {name}")

    def add_health_checker(self, name: str, checker: HealthChecker) -> None:
        """Add a health checker."""
        with self._lock:
            self._health_checkers[name] = checker
            self._logger.info(f"Added health checker: {name}")

    def remove_health_checker(self, name: str) -> None:
        """Remove a health checker."""
        with self._lock:
            if name in self._health_checkers:
                del self._health_checkers[name]
                self._logger.info(f"Removed health checker: {name}")

    def set_degradation_action(self, level: DegradationLevel, action: Callable) -> None:
        """Set degradation action for a level."""
        with self._lock:
            self._degradation_actions[level] = action
            self._logger.info(f"Set degradation action for level: {level.value}")

    def start_monitoring(self) -> None:
        """Start degradation monitoring."""
        with self._lock:
            if self._monitoring_thread is not None:
                return

            self._running = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_worker, daemon=True
            )
            self._monitoring_thread.start()

            self._logger.info("Started degradation monitoring")

    def stop_monitoring(self) -> None:
        """Stop degradation monitoring."""
        with self._lock:
            self._running = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5.0)

            self._logger.info("Stopped degradation monitoring")

    def _monitoring_worker(self) -> None:
        """Background worker for monitoring degradation."""
        while self._running:
            try:
                time.sleep(self._monitoring_interval)

                if not self._running:
                    break

                # Check degradation rules
                self._check_degradation_rules()

                # Check health checkers
                self._check_health_checkers()

            except Exception as e:
                self._logger.error(f"Degradation monitoring error: {e}")

    def _check_degradation_rules(self) -> None:
        """Check degradation rules."""
        with self._lock:
            current_time = time.time()
            max_degradation_level = DegradationLevel.NONE

            for rule in self._degradation_rules:
                if not rule.enabled:
                    continue

                # Check cooldown
                if current_time - rule.last_triggered < rule.cooldown:
                    continue

                # Check condition
                try:
                    if rule.condition():
                        rule.last_triggered = current_time
                        rule.trigger_count += 1

                        # Update max degradation level
                        if rule.level.value > max_degradation_level.value:
                            max_degradation_level = rule.level

                        self._logger.warning(
                            f"Degradation rule '{rule.name}' triggered "
                            f"(level: {rule.level.value}, count: {rule.trigger_count})"
                        )
                except Exception as e:
                    self._logger.error(
                        f"Error checking degradation rule '{rule.name}': {e}"
                    )

            # Update degradation level if changed
            if max_degradation_level != self._current_degradation_level:
                self._update_degradation_level(max_degradation_level)

    def _check_health_checkers(self) -> None:
        """Check health checkers."""
        with self._lock:
            max_degradation_level = DegradationLevel.NONE

            for name, checker in self._health_checkers.items():
                try:
                    health = checker.check_health()

                    # Update max degradation level
                    if health.degradation_level.value > max_degradation_level.value:
                        max_degradation_level = health.degradation_level

                    # Log health status
                    if health.degradation_level != DegradationLevel.NONE:
                        self._logger.warning(
                            f"Health checker '{name}' reports degradation level: "
                            f"{health.degradation_level.value}"
                        )

                        if health.errors:
                            for error in health.errors:
                                self._logger.error(
                                    f"Health checker '{name}' error: {error}"
                                )

                        if health.warnings:
                            for warning in health.warnings:
                                self._logger.warning(
                                    f"Health checker '{name}' warning: {warning}"
                                )

                except Exception as e:
                    self._logger.error(f"Error checking health checker '{name}': {e}")

            # Update degradation level if changed
            if max_degradation_level != self._current_degradation_level:
                self._update_degradation_level(max_degradation_level)

    def _update_degradation_level(self, new_level: DegradationLevel) -> None:
        """Update degradation level and execute actions."""
        old_level = self._current_degradation_level
        self._current_degradation_level = new_level

        # Update service level
        if new_level == DegradationLevel.CRITICAL:
            self._current_service_level = ServiceLevel.OFFLINE
        elif new_level == DegradationLevel.SEVERE:
            self._current_service_level = ServiceLevel.EMERGENCY
        elif new_level == DegradationLevel.MODERATE:
            self._current_service_level = ServiceLevel.LIMITED
        elif new_level == DegradationLevel.MINIMAL:
            self._current_service_level = ServiceLevel.REDUCED
        else:
            self._current_service_level = ServiceLevel.FULL

        # Execute degradation action
        action = self._degradation_actions.get(new_level)
        if action:
            try:
                action(old_level, new_level)
            except Exception as e:
                self._logger.error(f"Error executing degradation action: {e}")

        self._logger.info(
            f"Degradation level changed from {old_level.value} to {new_level.value}, "
            f"service level: {self._current_service_level.value}"
        )

    def _no_degradation_action(
        self, old_level: DegradationLevel, new_level: DegradationLevel
    ) -> None:
        """No degradation action."""
        pass

    def _minimal_degradation_action(
        self, old_level: DegradationLevel, new_level: DegradationLevel
    ) -> None:
        """Minimal degradation action."""
        self._logger.info("Applying minimal degradation measures")

    def _moderate_degradation_action(
        self, old_level: DegradationLevel, new_level: DegradationLevel
    ) -> None:
        """Moderate degradation action."""
        self._logger.info("Applying moderate degradation measures")

    def _severe_degradation_action(
        self, old_level: DegradationLevel, new_level: DegradationLevel
    ) -> None:
        """Severe degradation action."""
        self._logger.warning("Applying severe degradation measures")

    def _critical_degradation_action(
        self, old_level: DegradationLevel, new_level: DegradationLevel
    ) -> None:
        """Critical degradation action."""
        self._logger.critical("Applying critical degradation measures")

    def get_current_degradation_level(self) -> DegradationLevel:
        """Get current degradation level."""
        with self._lock:
            return self._current_degradation_level

    def get_current_service_level(self) -> ServiceLevel:
        """Get current service level."""
        with self._lock:
            return self._current_service_level

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        with self._lock:
            health_status = {
                "degradation_level": self._current_degradation_level.value,
                "service_level": self._current_service_level.value,
                "health_checkers": {},
                "degradation_rules": [],
                "timestamp": time.time(),
            }

            # Get health checker status
            for name, checker in self._health_checkers.items():
                try:
                    health = checker.check_health()
                    health_status["health_checkers"][name] = {
                        "level": health.level.value,
                        "degradation_level": health.degradation_level.value,
                        "last_check": health.last_check,
                        "errors": health.errors,
                        "warnings": health.warnings,
                    }
                except Exception as e:
                    health_status["health_checkers"][name] = {"error": str(e)}

            # Get degradation rule status
            for rule in self._degradation_rules:
                health_status["degradation_rules"].append(
                    {
                        "name": rule.name,
                        "level": rule.level.value,
                        "strategy": rule.strategy.value,
                        "enabled": rule.enabled,
                        "trigger_count": rule.trigger_count,
                        "last_triggered": rule.last_triggered,
                    }
                )

            return health_status

    def force_degradation_level(self, level: DegradationLevel) -> None:
        """Force degradation level (for testing)."""
        with self._lock:
            self._update_degradation_level(level)

    def reset_degradation(self) -> None:
        """Reset degradation to normal level."""
        with self._lock:
            self._update_degradation_level(DegradationLevel.NONE)

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
