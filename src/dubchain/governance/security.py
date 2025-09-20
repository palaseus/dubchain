"""
Security mechanisms for governance system.

This module implements various security measures to defend against attacks
such as Sybil attacks, vote buying, flash loan attacks, and governance front-running.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..crypto.hashing import Hash, SHA256Hasher
from ..errors.exceptions import ValidationError, GovernanceError
from .core import Vote, Proposal, VotingPower


@dataclass
class SecurityAlert:
    """A security alert from the governance system."""
    
    alert_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    description: str
    proposal_id: Optional[str] = None
    voter_address: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "description": self.description,
            "proposal_id": self.proposal_id,
            "voter_address": self.voter_address,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AttackDetector(ABC):
    """Abstract base class for attack detectors."""
    
    @abstractmethod
    def detect_attack(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> Optional[SecurityAlert]:
        """Detect if a vote represents an attack."""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Get the name of this detector."""
        pass


class SybilDetector(AttackDetector):
    """Detects Sybil attacks in governance voting."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Sybil detector."""
        self.config = config or {}
        self.suspicious_patterns: Dict[str, List[str]] = {}  # pattern -> addresses
        self.vote_history: Dict[str, List[Vote]] = {}  # address -> votes
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.min_votes_for_analysis = self.config.get("min_votes_for_analysis", 10)
    
    def detect_attack(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> Optional[SecurityAlert]:
        """Detect Sybil attack patterns."""
        voter_address = vote.voter_address
        
        # Track vote history
        if voter_address not in self.vote_history:
            self.vote_history[voter_address] = []
        self.vote_history[voter_address].append(vote)
        
        # Check for suspicious voting patterns
        if self._is_suspicious_voting_pattern(voter_address, vote, proposal):
            return SecurityAlert(
                alert_id=f"sybil_{voter_address}_{int(time.time())}",
                alert_type="sybil_attack",
                severity="high",
                description=f"Suspicious voting pattern detected for address {voter_address}",
                proposal_id=proposal.proposal_id,
                voter_address=voter_address,
                metadata={
                    "voting_pattern": "suspicious",
                    "vote_count": len(self.vote_history[voter_address]),
                }
            )
        
        # Check for coordinated voting
        if self._is_coordinated_voting(vote, proposal, context):
            return SecurityAlert(
                alert_id=f"coordinated_{voter_address}_{int(time.time())}",
                alert_type="coordinated_voting",
                severity="medium",
                description=f"Coordinated voting detected for address {voter_address}",
                proposal_id=proposal.proposal_id,
                voter_address=voter_address,
                metadata={
                    "coordination_type": "suspicious_timing",
                }
            )
        
        return None
    
    def _is_suspicious_voting_pattern(
        self,
        voter_address: str,
        vote: Vote,
        proposal: Proposal
    ) -> bool:
        """Check for suspicious voting patterns."""
        vote_history = self.vote_history.get(voter_address, [])
        
        if len(vote_history) < self.min_votes_for_analysis:
            return False
        
        # Check for identical voting patterns
        recent_votes = vote_history[-10:]  # Last 10 votes
        identical_votes = sum(1 for v in recent_votes if v.choice == vote.choice)
        
        if identical_votes / len(recent_votes) > self.similarity_threshold:
            return True
        
        # Check for voting power patterns
        voting_powers = [v.voting_power.power for v in recent_votes]
        if len(set(voting_powers)) == 1:  # All votes have same power
            return True
        
        return False
    
    def _is_coordinated_voting(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> bool:
        """Check for coordinated voting patterns."""
        # Check for votes cast within a short time window
        recent_votes = context.get("recent_votes", [])
        if not recent_votes:
            return False
        
        vote_time = vote.timestamp
        recent_vote_times = [v.timestamp for v in recent_votes[-10:]]
        
        # Check if vote was cast within 1 minute of other votes
        for other_time in recent_vote_times:
            if abs(vote_time - other_time) < 60:  # 1 minute
                return True
        
        return False
    
    def get_detector_name(self) -> str:
        """Get detector name."""
        return "sybil_detector"


class VoteBuyingDetector(AttackDetector):
    """Detects vote buying and bribery attacks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize vote buying detector."""
        self.config = config or {}
        self.suspicious_transactions: Dict[str, List[Dict[str, Any]]] = {}  # address -> transactions
        self.vote_buying_threshold = self.config.get("vote_buying_threshold", 1000)
        self.analysis_window = self.config.get("analysis_window", 3600)  # 1 hour
    
    def detect_attack(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> Optional[SecurityAlert]:
        """Detect vote buying patterns."""
        voter_address = vote.voter_address
        
        # Check for suspicious transactions before voting
        if self._has_suspicious_transactions(voter_address, vote.timestamp):
            return SecurityAlert(
                alert_id=f"vote_buying_{voter_address}_{int(time.time())}",
                alert_type="vote_buying",
                severity="critical",
                description=f"Vote buying detected for address {voter_address}",
                proposal_id=proposal.proposal_id,
                voter_address=voter_address,
                metadata={
                    "suspicious_transactions": len(self.suspicious_transactions.get(voter_address, [])),
                    "analysis_window": self.analysis_window,
                }
            )
        
        # Check for unusual voting power changes
        if self._has_unusual_power_change(vote, context):
            return SecurityAlert(
                alert_id=f"power_manipulation_{voter_address}_{int(time.time())}",
                alert_type="power_manipulation",
                severity="high",
                description=f"Unusual voting power change detected for address {voter_address}",
                proposal_id=proposal.proposal_id,
                voter_address=voter_address,
                metadata={
                    "power_change": "suspicious",
                }
            )
        
        return None
    
    def _has_suspicious_transactions(self, voter_address: str, vote_timestamp: float) -> bool:
        """Check for suspicious transactions before voting."""
        if voter_address not in self.suspicious_transactions:
            return False
        
        transactions = self.suspicious_transactions[voter_address]
        
        # Check for transactions within analysis window
        suspicious_count = 0
        for tx in transactions:
            # Transaction should be before the vote and within analysis window
            time_diff = vote_timestamp - tx["timestamp"]
            if 0 <= time_diff <= self.analysis_window:
                if tx["amount"] >= self.vote_buying_threshold:
                    suspicious_count += 1
        
        return suspicious_count > 0
    
    def _has_unusual_power_change(self, vote: Vote, context: Dict[str, Any]) -> bool:
        """Check for unusual voting power changes."""
        voter_address = vote.voter_address
        current_power = vote.voting_power.power
        
        # Get historical voting power
        historical_power = context.get("historical_voting_power", {}).get(voter_address)
        if not historical_power:
            return False
        
        # Check for sudden power increase
        power_increase = current_power - historical_power
        if power_increase > historical_power * 0.5:  # 50% increase
            return True
        
        return False
    
    def add_suspicious_transaction(
        self,
        voter_address: str,
        transaction_data: Dict[str, Any]
    ) -> None:
        """Add a suspicious transaction for analysis."""
        if voter_address not in self.suspicious_transactions:
            self.suspicious_transactions[voter_address] = []
        
        self.suspicious_transactions[voter_address].append(transaction_data)
    
    def get_detector_name(self) -> str:
        """Get detector name."""
        return "vote_buying_detector"


class FlashLoanDetector(AttackDetector):
    """Detects flash loan attacks in governance voting."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize flash loan detector."""
        self.config = config or {}
        self.flash_loan_threshold = self.config.get("flash_loan_threshold", 1000000)
        self.analysis_window = self.config.get("analysis_window", 300)  # 5 minutes
        self.voting_power_snapshots: Dict[str, List[Tuple[float, int]]] = {}  # address -> [(timestamp, power)]
    
    def detect_attack(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> Optional[SecurityAlert]:
        """Detect flash loan attacks."""
        voter_address = vote.voter_address
        current_power = vote.voting_power.power
        
        # Track voting power snapshots
        if voter_address not in self.voting_power_snapshots:
            self.voting_power_snapshots[voter_address] = []
        
        self.voting_power_snapshots[voter_address].append((vote.timestamp, current_power))
        
        # Check for flash loan pattern
        if self._is_flash_loan_pattern(voter_address, vote.timestamp):
            return SecurityAlert(
                alert_id=f"flash_loan_{voter_address}_{int(time.time())}",
                alert_type="flash_loan_attack",
                severity="critical",
                description=f"Flash loan attack detected for address {voter_address}",
                proposal_id=proposal.proposal_id,
                voter_address=voter_address,
                metadata={
                    "flash_loan_amount": current_power,
                    "analysis_window": self.analysis_window,
                }
            )
        
        return None
    
    def _is_flash_loan_pattern(self, voter_address: str, vote_timestamp: float) -> bool:
        """Check for flash loan attack pattern."""
        if voter_address not in self.voting_power_snapshots:
            return False
        
        snapshots = self.voting_power_snapshots[voter_address]
        
        # Look for sudden power increase followed by decrease
        for i, (timestamp, power) in enumerate(snapshots):
            if abs(timestamp - vote_timestamp) <= self.analysis_window:
                # Check if power is significantly higher than recent average
                recent_powers = [p for t, p in snapshots[max(0, i-5):i] if abs(t - timestamp) <= self.analysis_window]
                if recent_powers:
                    avg_power = sum(recent_powers) / len(recent_powers)
                    if power > avg_power * 2:  # 2x increase
                        return True
        
        return False
    
    def get_detector_name(self) -> str:
        """Get detector name."""
        return "flash_loan_detector"


class GovernanceFrontRunningDetector(AttackDetector):
    """Detects governance front-running attacks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize front-running detector."""
        self.config = config or {}
        self.front_running_threshold = self.config.get("front_running_threshold", 0.1)  # 10%
        self.analysis_window = self.config.get("analysis_window", 60)  # 1 minute
    
    def detect_attack(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> Optional[SecurityAlert]:
        """Detect governance front-running."""
        voter_address = vote.voter_address
        
        # Check for front-running pattern
        if self._is_front_running_pattern(vote, proposal, context):
            return SecurityAlert(
                alert_id=f"front_running_{voter_address}_{int(time.time())}",
                alert_type="governance_front_running",
                severity="high",
                description=f"Governance front-running detected for address {voter_address}",
                proposal_id=proposal.proposal_id,
                voter_address=voter_address,
                metadata={
                    "front_running_type": "governance_manipulation",
                }
            )
        
        return None
    
    def _is_front_running_pattern(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> bool:
        """Check for front-running pattern."""
        # Check if vote was cast very early in the voting period
        if proposal.start_block and proposal.end_block:
            voting_duration = proposal.end_block - proposal.start_block
            current_block = context.get("current_block", 0)
            
            if current_block > proposal.start_block:
                voting_progress = (current_block - proposal.start_block) / voting_duration
                if voting_progress < self.front_running_threshold:
                    return True
        
        return False
    
    def get_detector_name(self) -> str:
        """Get detector name."""
        return "governance_front_running_detector"


class SecurityManager:
    """Main security manager for the governance system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize security manager."""
        self.config = config or {}
        self.detectors: List[AttackDetector] = []
        self.alerts: List[SecurityAlert] = []
        self.blocked_addresses: Set[str] = set()
        self.suspicious_addresses: Set[str] = set()
        
        # Initialize detectors
        self._initialize_detectors()
    
    def _initialize_detectors(self) -> None:
        """Initialize attack detectors."""
        self.detectors = [
            SybilDetector(self.config.get("sybil_detector", {})),
            VoteBuyingDetector(self.config.get("vote_buying_detector", {})),
            FlashLoanDetector(self.config.get("flash_loan_detector", {})),
            GovernanceFrontRunningDetector(self.config.get("front_running_detector", {})),
        ]
    
    def analyze_vote(
        self,
        vote: Vote,
        proposal: Proposal,
        context: Dict[str, Any]
    ) -> List[SecurityAlert]:
        """Analyze a vote for security threats."""
        alerts = []
        
        # Check if address is blocked
        if vote.voter_address in self.blocked_addresses:
            alert = SecurityAlert(
                alert_id=f"blocked_address_{vote.voter_address}_{int(time.time())}",
                alert_type="blocked_address",
                severity="critical",
                description=f"Vote from blocked address {vote.voter_address}",
                proposal_id=proposal.proposal_id,
                voter_address=vote.voter_address,
            )
            alerts.append(alert)
            return alerts
        
        # Run all detectors
        for detector in self.detectors:
            try:
                alert = detector.detect_attack(vote, proposal, context)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                # Log detector error but don't fail the analysis
                print(f"Error in detector {detector.get_detector_name()}: {e}")
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        return alerts
    
    def block_address(self, address: str, reason: str) -> None:
        """Block an address from participating in governance."""
        self.blocked_addresses.add(address)
        
        # Create alert
        alert = SecurityAlert(
            alert_id=f"address_blocked_{address}_{int(time.time())}",
            alert_type="address_blocked",
            severity="high",
            description=f"Address {address} blocked: {reason}",
            voter_address=address,
            metadata={"reason": reason},
        )
        self.alerts.append(alert)
    
    def unblock_address(self, address: str) -> None:
        """Unblock an address."""
        self.blocked_addresses.discard(address)
    
    def mark_suspicious(self, address: str, reason: str) -> None:
        """Mark an address as suspicious."""
        self.suspicious_addresses.add(address)
        
        # Create alert
        alert = SecurityAlert(
            alert_id=f"address_suspicious_{address}_{int(time.time())}",
            alert_type="address_suspicious",
            severity="medium",
            description=f"Address {address} marked as suspicious: {reason}",
            voter_address=address,
            metadata={"reason": reason},
        )
        self.alerts.append(alert)
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security statistics."""
        alert_counts = {}
        for alert in self.alerts:
            alert_type = alert.alert_type
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        return {
            "total_alerts": len(self.alerts),
            "alert_counts": alert_counts,
            "blocked_addresses": len(self.blocked_addresses),
            "suspicious_addresses": len(self.suspicious_addresses),
            "active_detectors": len(self.detectors),
        }
    
    def get_recent_alerts(self, limit: int = 100) -> List[SecurityAlert]:
        """Get recent security alerts."""
        return sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
    
    def add_suspicious_transaction(
        self,
        voter_address: str,
        transaction_data: Dict[str, Any]
    ) -> None:
        """Add a suspicious transaction for analysis."""
        # Find vote buying detector and add transaction
        for detector in self.detectors:
            if isinstance(detector, VoteBuyingDetector):
                detector.add_suspicious_transaction(voter_address, transaction_data)
                break
