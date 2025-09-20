"""
Unit tests for governance security system.

This module tests security mechanisms including attack detection,
Sybil detection, vote buying detection, flash loan detection,
and governance front-running detection.
"""

import pytest
import time
from unittest.mock import Mock, patch

from dubchain.governance.security import (
    SecurityAlert,
    AttackDetector,
    SybilDetector,
    VoteBuyingDetector,
    FlashLoanDetector,
    GovernanceFrontRunningDetector,
    SecurityManager,
)
from dubchain.governance.core import Vote, VoteChoice, VotingPower, Proposal, ProposalType


class TestSecurityAlert:
    """Test SecurityAlert class."""
    
    def test_security_alert_creation(self):
        """Test creating a security alert."""
        alert = SecurityAlert(
            alert_id="alert_123",
            alert_type="sybil_attack",
            severity="high",
            description="Suspicious voting pattern detected",
            proposal_id="prop_123",
            voter_address="0x123"
        )
        
        assert alert.alert_id == "alert_123"
        assert alert.alert_type == "sybil_attack"
        assert alert.severity == "high"
        assert alert.description == "Suspicious voting pattern detected"
        assert alert.proposal_id == "prop_123"
        assert alert.voter_address == "0x123"
        assert alert.timestamp > 0
        assert len(alert.metadata) == 0
    
    def test_security_alert_with_metadata(self):
        """Test creating a security alert with metadata."""
        metadata = {"pattern": "suspicious", "vote_count": 5}
        alert = SecurityAlert(
            alert_id="alert_123",
            alert_type="vote_buying",
            severity="critical",
            description="Vote buying detected",
            metadata=metadata
        )
        
        assert alert.metadata == metadata
    
    def test_security_alert_serialization(self):
        """Test security alert serialization."""
        alert = SecurityAlert(
            alert_id="alert_123",
            alert_type="flash_loan_attack",
            severity="critical",
            description="Flash loan attack detected",
            proposal_id="prop_123",
            voter_address="0x123",
            metadata={"amount": 1000000}
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["alert_id"] == "alert_123"
        assert alert_dict["alert_type"] == "flash_loan_attack"
        assert alert_dict["severity"] == "critical"
        assert alert_dict["description"] == "Flash loan attack detected"
        assert alert_dict["proposal_id"] == "prop_123"
        assert alert_dict["voter_address"] == "0x123"
        assert alert_dict["metadata"]["amount"] == 1000000
        assert alert_dict["timestamp"] > 0


class TestSybilDetector:
    """Test SybilDetector class."""
    
    def test_sybil_detector_creation(self):
        """Test creating Sybil detector."""
        detector = SybilDetector()
        
        assert detector.get_detector_name() == "sybil_detector"
        assert detector.similarity_threshold == 0.8
        assert detector.min_votes_for_analysis == 10
        assert len(detector.vote_history) == 0
    
    def test_sybil_detector_with_config(self):
        """Test creating Sybil detector with configuration."""
        config = {
            "similarity_threshold": 0.9,
            "min_votes_for_analysis": 5
        }
        detector = SybilDetector(config)
        
        assert detector.similarity_threshold == 0.9
        assert detector.min_votes_for_analysis == 5
    
    def test_detect_attack_no_suspicious_pattern(self):
        """Test detecting no suspicious pattern."""
        detector = SybilDetector()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {"recent_votes": []}
        
        alert = detector.detect_attack(vote, proposal, context)
        assert alert is None
    
    def test_detect_attack_suspicious_voting_pattern(self):
        """Test detecting suspicious voting pattern."""
        detector = SybilDetector({"min_votes_for_analysis": 3})
        
        # Create vote history with identical patterns
        for i in range(5):
            voting_power = VotingPower(
                voter_address="0x123",
                power=1000,
                token_balance=1000
            )
            
            vote = Vote(
                proposal_id=f"prop_{i}",
                voter_address="0x123",
                choice=VoteChoice.FOR,  # Always FOR
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            
            detector.vote_history["0x123"] = detector.vote_history.get("0x123", [])
            detector.vote_history["0x123"].append(vote)
        
        # Create new vote with same pattern
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_5",
            voter_address="0x123",
            choice=VoteChoice.FOR,  # Same pattern
            voting_power=voting_power,
            signature="0x5"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {"recent_votes": []}
        
        alert = detector.detect_attack(vote, proposal, context)
        
        assert alert is not None
        assert alert.alert_type == "sybil_attack"
        assert alert.severity == "high"
        assert alert.voter_address == "0x123"
    
    def test_detect_attack_coordinated_voting(self):
        """Test detecting coordinated voting."""
        detector = SybilDetector()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123",
            timestamp=time.time()
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        # Create recent votes within 1 minute
        recent_votes = []
        for i in range(3):
            recent_vote = Vote(
                proposal_id=f"prop_{i}",
                voter_address=f"0x{i}",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature=f"0x{i}",
                timestamp=time.time() - 30  # 30 seconds ago
            )
            recent_votes.append(recent_vote)
        
        context = {"recent_votes": recent_votes}
        
        alert = detector.detect_attack(vote, proposal, context)
        
        assert alert is not None
        assert alert.alert_type == "coordinated_voting"
        assert alert.severity == "medium"
        assert alert.voter_address == "0x123"
    
    def test_is_suspicious_voting_pattern_insufficient_votes(self):
        """Test suspicious pattern detection with insufficient votes."""
        detector = SybilDetector({"min_votes_for_analysis": 10})
        
        # Create only 5 votes (less than min_votes_for_analysis)
        for i in range(5):
            voting_power = VotingPower(
                voter_address="0x123",
                power=1000,
                token_balance=1000
            )
            
            vote = Vote(
                proposal_id=f"prop_{i}",
                voter_address="0x123",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            
            detector.vote_history["0x123"] = detector.vote_history.get("0x123", [])
            detector.vote_history["0x123"].append(vote)
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_5",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0x5"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        is_suspicious = detector._is_suspicious_voting_pattern("0x123", vote, proposal)
        assert is_suspicious is False
    
    def test_is_suspicious_voting_pattern_identical_powers(self):
        """Test suspicious pattern detection with identical voting powers."""
        detector = SybilDetector({"min_votes_for_analysis": 3})
        
        # Create votes with identical voting powers
        for i in range(5):
            voting_power = VotingPower(
                voter_address="0x123",
                power=1000,  # Always same power
                token_balance=1000
            )
            
            vote = Vote(
                proposal_id=f"prop_{i}",
                voter_address="0x123",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            
            detector.vote_history["0x123"] = detector.vote_history.get("0x123", [])
            detector.vote_history["0x123"].append(vote)
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,  # Same power again
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_5",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0x5"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        is_suspicious = detector._is_suspicious_voting_pattern("0x123", vote, proposal)
        assert is_suspicious is True


class TestVoteBuyingDetector:
    """Test VoteBuyingDetector class."""
    
    def test_vote_buying_detector_creation(self):
        """Test creating vote buying detector."""
        detector = VoteBuyingDetector()
        
        assert detector.get_detector_name() == "vote_buying_detector"
        assert detector.vote_buying_threshold == 1000
        assert detector.analysis_window == 3600
        assert len(detector.suspicious_transactions) == 0
    
    def test_vote_buying_detector_with_config(self):
        """Test creating vote buying detector with configuration."""
        config = {
            "vote_buying_threshold": 5000,
            "analysis_window": 1800
        }
        detector = VoteBuyingDetector(config)
        
        assert detector.vote_buying_threshold == 5000
        assert detector.analysis_window == 1800
    
    def test_detect_attack_no_suspicious_transactions(self):
        """Test detecting no suspicious transactions."""
        detector = VoteBuyingDetector()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {}
        
        alert = detector.detect_attack(vote, proposal, context)
        assert alert is None
    
    def test_detect_attack_suspicious_transactions(self):
        """Test detecting suspicious transactions."""
        detector = VoteBuyingDetector({"vote_buying_threshold": 1000})
        
        # Add suspicious transaction
        detector.add_suspicious_transaction(
            "0x123",
            {
                "timestamp": time.time() - 1800,  # 30 minutes ago
                "amount": 1500,  # Above threshold
                "from": "0x999",
                "to": "0x123"
            }
        )
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123",
            timestamp=time.time()
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {}
        
        alert = detector.detect_attack(vote, proposal, context)
        
        assert alert is not None
        assert alert.alert_type == "vote_buying"
        assert alert.severity == "critical"
        assert alert.voter_address == "0x123"
        assert alert.metadata["suspicious_transactions"] == 1
    
    def test_detect_attack_unusual_power_change(self):
        """Test detecting unusual power change."""
        detector = VoteBuyingDetector()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=2000,  # High power
            token_balance=2000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        # Historical voting power was much lower
        context = {
            "historical_voting_power": {
                "0x123": 1000  # 50% increase (2000 vs 1000)
            }
        }
        
        alert = detector.detect_attack(vote, proposal, context)
        
        assert alert is not None
        assert alert.alert_type == "power_manipulation"
        assert alert.severity == "high"
        assert alert.voter_address == "0x123"
    
    def test_has_suspicious_transactions(self):
        """Test checking for suspicious transactions."""
        detector = VoteBuyingDetector({"vote_buying_threshold": 1000})
        
        # Add transaction within analysis window
        detector.add_suspicious_transaction(
            "0x123",
            {
                "timestamp": time.time() - 1800,  # 30 minutes ago
                "amount": 1500,
                "from": "0x999",
                "to": "0x123"
            }
        )
        
        # Check within analysis window
        has_suspicious = detector._has_suspicious_transactions("0x123", time.time())
        assert has_suspicious is True
        
        # Check outside analysis window (check at a time that's 2 hours ago, so the 30-minute-ago transaction is outside the window)
        has_suspicious = detector._has_suspicious_transactions("0x123", time.time() - 7200)  # 2 hours ago
        assert has_suspicious is False
    
    def test_has_unusual_power_change(self):
        """Test checking for unusual power change."""
        detector = VoteBuyingDetector()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=2000,
            token_balance=2000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        # 50% increase (should be detected)
        context = {
            "historical_voting_power": {
                "0x123": 1000
            }
        }
        
        has_unusual = detector._has_unusual_power_change(vote, context)
        assert has_unusual is True
        
        # 30% increase (should not be detected)
        voting_power.power = 1300
        has_unusual = detector._has_unusual_power_change(vote, context)
        assert has_unusual is False


class TestFlashLoanDetector:
    """Test FlashLoanDetector class."""
    
    def test_flash_loan_detector_creation(self):
        """Test creating flash loan detector."""
        detector = FlashLoanDetector()
        
        assert detector.get_detector_name() == "flash_loan_detector"
        assert detector.flash_loan_threshold == 1000000
        assert detector.analysis_window == 300
        assert len(detector.voting_power_snapshots) == 0
    
    def test_detect_attack_no_flash_loan_pattern(self):
        """Test detecting no flash loan pattern."""
        detector = FlashLoanDetector()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {}
        
        alert = detector.detect_attack(vote, proposal, context)
        assert alert is None
    
    def test_detect_attack_flash_loan_pattern(self):
        """Test detecting flash loan pattern."""
        detector = FlashLoanDetector({"analysis_window": 300})
        
        # Create voting power snapshots showing sudden increase
        current_time = time.time()
        
        # Add historical snapshots with low power
        detector.voting_power_snapshots["0x123"] = [
            (current_time - 600, 1000),  # 10 minutes ago
            (current_time - 300, 1000),  # 5 minutes ago
            (current_time - 60, 1000),   # 1 minute ago
        ]
        
        # Create vote with suddenly high power
        voting_power = VotingPower(
            voter_address="0x123",
            power=5000,  # 5x increase
            token_balance=5000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123",
            timestamp=current_time
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {}
        
        alert = detector.detect_attack(vote, proposal, context)
        
        assert alert is not None
        assert alert.alert_type == "flash_loan_attack"
        assert alert.severity == "critical"
        assert alert.voter_address == "0x123"
        assert alert.metadata["flash_loan_amount"] == 5000
    
    def test_is_flash_loan_pattern(self):
        """Test flash loan pattern detection."""
        detector = FlashLoanDetector({"analysis_window": 300})
        
        current_time = time.time()
        
        # Add snapshots with low power
        detector.voting_power_snapshots["0x123"] = [
            (current_time - 600, 1000),
            (current_time - 300, 1000),
            (current_time - 60, 1000),
        ]
        
        # Check for flash loan pattern
        is_flash_loan = detector._is_flash_loan_pattern("0x123", current_time)
        assert is_flash_loan is False
        
        # Add snapshot with high power
        detector.voting_power_snapshots["0x123"].append((current_time, 5000))
        
        is_flash_loan = detector._is_flash_loan_pattern("0x123", current_time)
        assert is_flash_loan is True


class TestGovernanceFrontRunningDetector:
    """Test GovernanceFrontRunningDetector class."""
    
    def test_front_running_detector_creation(self):
        """Test creating front-running detector."""
        detector = GovernanceFrontRunningDetector()
        
        assert detector.get_detector_name() == "governance_front_running_detector"
        assert detector.front_running_threshold == 0.1
        assert detector.analysis_window == 60
    
    def test_detect_attack_no_front_running(self):
        """Test detecting no front-running."""
        detector = GovernanceFrontRunningDetector()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal",
            start_block=100,
            end_block=200
        )
        
        # Vote cast in middle of voting period
        context = {"current_block": 150}
        
        alert = detector.detect_attack(vote, proposal, context)
        assert alert is None
    
    def test_detect_attack_front_running(self):
        """Test detecting front-running."""
        detector = GovernanceFrontRunningDetector({"front_running_threshold": 0.1})
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal",
            start_block=100,
            end_block=200
        )
        
        # Vote cast very early in voting period (5% progress)
        context = {"current_block": 105}
        
        alert = detector.detect_attack(vote, proposal, context)
        
        assert alert is not None
        assert alert.alert_type == "governance_front_running"
        assert alert.severity == "high"
        assert alert.voter_address == "0x123"
    
    def test_is_front_running_pattern(self):
        """Test front-running pattern detection."""
        detector = GovernanceFrontRunningDetector({"front_running_threshold": 0.1})
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=VotingPower("0x123", 1000, 1000),
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal",
            start_block=100,
            end_block=200
        )
        
        # Early voting (5% progress)
        context = {"current_block": 105}
        is_front_running = detector._is_front_running_pattern(vote, proposal, context)
        assert is_front_running is True
        
        # Late voting (50% progress)
        context = {"current_block": 150}
        is_front_running = detector._is_front_running_pattern(vote, proposal, context)
        assert is_front_running is False


class TestSecurityManager:
    """Test SecurityManager class."""
    
    def test_security_manager_creation(self):
        """Test creating security manager."""
        manager = SecurityManager()
        
        assert len(manager.detectors) == 4
        assert len(manager.alerts) == 0
        assert len(manager.blocked_addresses) == 0
        assert len(manager.suspicious_addresses) == 0
        
        # Check that all detectors are initialized
        detector_names = [detector.get_detector_name() for detector in manager.detectors]
        assert "sybil_detector" in detector_names
        assert "vote_buying_detector" in detector_names
        assert "flash_loan_detector" in detector_names
        assert "governance_front_running_detector" in detector_names
    
    def test_security_manager_with_config(self):
        """Test creating security manager with configuration."""
        config = {
            "sybil_detector": {"similarity_threshold": 0.9},
            "vote_buying_detector": {"vote_buying_threshold": 5000}
        }
        manager = SecurityManager(config)
        
        assert len(manager.detectors) == 4
    
    def test_analyze_vote_no_alerts(self):
        """Test analyzing vote with no security alerts."""
        manager = SecurityManager()
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {}
        
        alerts = manager.analyze_vote(vote, proposal, context)
        assert len(alerts) == 0
    
    def test_analyze_vote_blocked_address(self):
        """Test analyzing vote from blocked address."""
        manager = SecurityManager()
        
        # Block address
        manager.block_address("0x123", "Suspicious activity")
        
        voting_power = VotingPower(
            voter_address="0x123",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id="prop_123",
            voter_address="0x123",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature="0xabc123"
        )
        
        proposal = Proposal(
            proposer_address="0x456",
            title="Test Proposal",
            description="This is a test proposal"
        )
        
        context = {}
        
        alerts = manager.analyze_vote(vote, proposal, context)
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == "blocked_address"
        assert alerts[0].severity == "critical"
        assert alerts[0].voter_address == "0x123"
    
    def test_block_unblock_address(self):
        """Test blocking and unblocking addresses."""
        manager = SecurityManager()
        
        # Block address
        manager.block_address("0x123", "Suspicious activity")
        
        assert "0x123" in manager.blocked_addresses
        assert len(manager.alerts) == 1
        assert manager.alerts[0].alert_type == "address_blocked"
        
        # Unblock address
        manager.unblock_address("0x123")
        
        assert "0x123" not in manager.blocked_addresses
    
    def test_mark_suspicious(self):
        """Test marking address as suspicious."""
        manager = SecurityManager()
        
        manager.mark_suspicious("0x123", "Unusual voting pattern")
        
        assert "0x123" in manager.suspicious_addresses
        assert len(manager.alerts) == 1
        assert manager.alerts[0].alert_type == "address_suspicious"
        assert manager.alerts[0].severity == "medium"
    
    def test_get_security_statistics(self):
        """Test getting security statistics."""
        manager = SecurityManager()
        
        # Add some test data
        manager.block_address("0x123", "Test")
        manager.mark_suspicious("0x456", "Test")
        
        # Create a test alert
        alert = SecurityAlert(
            alert_id="test_alert",
            alert_type="test_attack",
            severity="high",
            description="Test alert"
        )
        manager.alerts.append(alert)
        
        stats = manager.get_security_statistics()
        
        assert stats["total_alerts"] == 3
        assert stats["alert_counts"]["address_blocked"] == 1
        assert stats["alert_counts"]["address_suspicious"] == 1
        assert stats["blocked_addresses"] == 1
        assert stats["suspicious_addresses"] == 1
        assert stats["active_detectors"] == 4
    
    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        manager = SecurityManager()
        
        # Add alerts with different timestamps
        alert1 = SecurityAlert(
            alert_id="alert1",
            alert_type="test1",
            severity="low",
            description="Old alert",
            timestamp=time.time() - 3600
        )
        
        alert2 = SecurityAlert(
            alert_id="alert2",
            alert_type="test2",
            severity="high",
            description="Recent alert",
            timestamp=time.time()
        )
        
        manager.alerts = [alert1, alert2]
        
        recent_alerts = manager.get_recent_alerts(limit=1)
        
        assert len(recent_alerts) == 1
        assert recent_alerts[0].alert_id == "alert2"  # Most recent
    
    def test_add_suspicious_transaction(self):
        """Test adding suspicious transaction."""
        manager = SecurityManager()
        
        transaction_data = {
            "timestamp": time.time(),
            "amount": 5000,
            "from": "0x999",
            "to": "0x123"
        }
        
        manager.add_suspicious_transaction("0x123", transaction_data)
        
        # Find vote buying detector and check if transaction was added
        vote_buying_detector = None
        for detector in manager.detectors:
            if isinstance(detector, VoteBuyingDetector):
                vote_buying_detector = detector
                break
        
        assert vote_buying_detector is not None
        assert "0x123" in vote_buying_detector.suspicious_transactions
        assert len(vote_buying_detector.suspicious_transactions["0x123"]) == 1
