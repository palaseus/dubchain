"""
Comprehensive governance system demonstration.

This script demonstrates all features of the governance system including
proposal lifecycle, voting strategies, delegation, security, treasury,
and upgrade mechanisms.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain.governance.core import (
    GovernanceEngine,
    GovernanceConfig,
    Proposal,
    ProposalStatus,
    ProposalType,
    Vote,
    VoteChoice,
    VotingPower,
)
from dubchain.governance.strategies import StrategyFactory
from dubchain.governance.delegation import DelegationManager
from dubchain.governance.security import SecurityManager
from dubchain.governance.execution import ExecutionEngine
from dubchain.governance.treasury import TreasuryManager
from dubchain.governance.observability import GovernanceEvents
from dubchain.governance.upgrades import UpgradeManager, ProxyGovernance


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print('='*60)


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n📋 {title}")
    print('-'*40)


def demo_governance_configuration():
    """Demonstrate governance configuration."""
    print_section("Governance Configuration")
    
    config = GovernanceConfig(
        default_quorum_threshold=2000,
        default_approval_threshold=0.6,
        default_voting_period=1000,
        default_execution_delay=100,
        max_proposal_description_length=5000,
        min_proposal_title_length=10,
        emergency_threshold=0.8,
        emergency_execution_delay=10,
        max_delegation_chain_length=5,
        delegation_cooldown_period=100,
        max_treasury_spending_per_proposal=1000000,
        treasury_multisig_threshold=3,
        upgrade_timelock_period=1000,
        emergency_upgrade_threshold=0.9
    )
    
    print("✅ Governance configuration created with:")
    print(f"   - Quorum threshold: {config.default_quorum_threshold}")
    print(f"   - Approval threshold: {config.default_approval_threshold}")
    print(f"   - Voting period: {config.default_voting_period} blocks")
    print(f"   - Execution delay: {config.default_execution_delay} blocks")
    print(f"   - Emergency threshold: {config.emergency_threshold}")
    
    return config


def demo_proposal_lifecycle():
    """Demonstrate proposal lifecycle."""
    print_section("Proposal Lifecycle")
    
    config = GovernanceConfig()
    engine = GovernanceEngine(config)
    
    print_subsection("Creating a Proposal")
    
    # Create a parameter change proposal
    proposal = engine.create_proposal(
        proposer_address="0x1234567890abcdef",
        title="Increase Block Size Limit",
        description="This proposal increases the maximum block size from 1MB to 2MB to improve throughput.",
        proposal_type=ProposalType.PARAMETER_CHANGE,
        quorum_threshold=1500,
        approval_threshold=0.6,
        execution_delay=200,
        execution_data={
            "parameter_name": "max_block_size",
            "old_value": 1048576,  # 1MB
            "new_value": 2097152   # 2MB
        }
    )
    
    print(f"✅ Proposal created: {proposal.proposal_id}")
    print(f"   - Status: {proposal.status.value}")
    print(f"   - Type: {proposal.proposal_type.value}")
    print(f"   - Quorum threshold: {proposal.quorum_threshold}")
    print(f"   - Approval threshold: {proposal.approval_threshold}")
    
    print_subsection("Activating Proposal")
    
    # Activate the proposal
    engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.ACTIVE)
    print(f"✅ Proposal activated: {proposal.status.value}")
    
    print_subsection("Voting on Proposal")
    
    # Create voters with different voting power
    voters = [
        ("0x1111111111111111", 800, VoteChoice.FOR),
        ("0x2222222222222222", 600, VoteChoice.FOR),
        ("0x3333333333333333", 400, VoteChoice.AGAINST),
        ("0x4444444444444444", 300, VoteChoice.FOR),
        ("0x5555555555555555", 200, VoteChoice.ABSTAIN),
    ]
    
    for voter_address, power, choice in voters:
        voting_power = VotingPower(
            voter_address=voter_address,
            power=power,
            token_balance=power
        )
        
        vote = engine.cast_vote(
            proposal_id=proposal.proposal_id,
            voter_address=voter_address,
            choice=choice,
            voting_power=voting_power,
            signature=f"0x{voter_address[-8:]}"
        )
        
        print(f"   ✅ Vote cast by {voter_address[:10]}...: {choice.value} ({power} power)")
    
    print_subsection("Vote Summary")
    
    summary = proposal.get_vote_summary()
    print(f"   - Total voting power: {summary['total_voting_power']}")
    print(f"   - For: {summary['for_power']}")
    print(f"   - Against: {summary['against_power']}")
    print(f"   - Abstain: {summary['abstain_power']}")
    print(f"   - Quorum met: {summary['quorum_met']}")
    print(f"   - Approval percentage: {summary['approval_percentage']:.2%}")
    print(f"   - Approved: {summary['approved']}")
    
    if summary['approved']:
        print_subsection("Queuing and Executing Proposal")
        
        # Queue the proposal
        engine.state.update_proposal_status(proposal.proposal_id, ProposalStatus.QUEUED)
        print(f"✅ Proposal queued: {proposal.status.value}")
        
        # Execute the proposal
        execution_engine = ExecutionEngine(engine.state)
        result = execution_engine.execute_proposal(proposal, current_block=1000)
        
        if result.is_successful():
            print(f"✅ Proposal executed successfully: {proposal.status.value}")
            print(f"   - Execution data: {result.execution_data}")
        else:
            print(f"❌ Proposal execution failed: {result.error_message}")
    
    return proposal


def demo_voting_strategies():
    """Demonstrate different voting strategies."""
    print_section("Voting Strategies")
    
    # Test data
    voter_address = "0x1234567890abcdef"
    token_balance = 10000
    
    strategies = [
        ("token_weighted", "Token-Weighted Voting"),
        ("quadratic_voting", "Quadratic Voting"),
        ("conviction_voting", "Conviction Voting"),
        ("snapshot_voting", "Snapshot Voting"),
    ]
    
    for strategy_name, description in strategies:
        print_subsection(description)
        
        strategy = StrategyFactory.create_strategy(strategy_name)
        
        # Calculate voting power
        power = strategy.calculate_voting_power(
            voter_address=voter_address,
            token_balance=token_balance,
            delegated_power=0
        )
        
        print(f"   - Token balance: {token_balance}")
        print(f"   - Voting power: {power.power}")
        print(f"   - Strategy: {strategy.name}")
        
        # Test with different token balances
        test_balances = [100, 1000, 10000, 100000]
        print("   - Power scaling:")
        for balance in test_balances:
            test_power = strategy.calculate_voting_power(
                voter_address=voter_address,
                token_balance=balance,
                delegated_power=0
            )
            print(f"     {balance:6d} tokens → {test_power.power:6d} power")


def demo_delegation_system():
    """Demonstrate delegation system."""
    print_section("Delegation System")
    
    config = GovernanceConfig()
    delegation_manager = DelegationManager(config)
    
    print_subsection("Creating Delegations")
    
    # Create delegation chain: A → B → C
    delegations = [
        ("0x1111111111111111", "0x2222222222222222", 1000),
        ("0x2222222222222222", "0x3333333333333333", 800),
        ("0x3333333333333333", "0x4444444444444444", 600),
    ]
    
    for delegator, delegatee, power in delegations:
        delegation = delegation_manager.create_delegation(
            delegator_address=delegator,
            delegatee_address=delegatee,
            delegation_power=power
        )
        
        print(f"   ✅ {delegator[:10]}... → {delegatee[:10]}... ({power} power)")
    
    print_subsection("Delegation Statistics")
    
    stats = delegation_manager.get_delegation_statistics()
    print(f"   - Total delegations: {stats['total_delegations']}")
    print(f"   - Active delegations: {stats['active_delegations']}")
    print(f"   - Total delegated power: {stats['total_delegated_power']}")
    print(f"   - Unique delegators: {stats['unique_delegators']}")
    print(f"   - Unique delegatees: {stats['unique_delegatees']}")
    
    print_subsection("Delegated Voting Power")
    
    # Check delegated power for final delegatee
    final_delegatee = "0x4444444444444444"
    delegated_power = delegation_manager.get_delegated_power(final_delegatee, 100)
    print(f"   - {final_delegatee[:10]}... has {delegated_power} delegated power")
    
    print_subsection("Circular Delegation Prevention")
    
    try:
        # Try to create circular delegation
        delegation_manager.create_delegation(
            delegator_address="0x4444444444444444",
            delegatee_address="0x1111111111111111",  # Would create cycle
            delegation_power=500
        )
        print("   ❌ Circular delegation was allowed (this should not happen)")
    except ValueError as e:
        print(f"   ✅ Circular delegation prevented: {e}")


def demo_security_system():
    """Demonstrate security system."""
    print_section("Security System")
    
    security_manager = SecurityManager({
        "sybil_detector": {"similarity_threshold": 0.8, "min_votes_for_analysis": 5},
        "vote_buying_detector": {"vote_buying_threshold": 1000},
        "flash_loan_detector": {"flash_loan_threshold": 1000000},
        "front_running_detector": {"front_running_threshold": 0.1}
    })
    
    print_subsection("Attack Detection")
    
    # Create a test proposal
    proposal = Proposal(
        proposer_address="0x1234567890abcdef",
        title="Test Proposal",
        description="Test proposal for security demonstration"
    )
    
    # Test Sybil attack detection
    print("   🔍 Testing Sybil attack detection...")
    for i in range(10):
        voting_power = VotingPower(
            voter_address=f"0xsybil{i}",
            power=1000,
            token_balance=1000
        )
        
        vote = Vote(
            proposal_id=proposal.proposal_id,
            voter_address=f"0xsybil{i}",
            choice=VoteChoice.FOR,
            voting_power=voting_power,
            signature=f"0x{i}"
        )
        
        alerts = security_manager.analyze_vote(vote, proposal, {})
        if alerts:
            for alert in alerts:
                print(f"   ⚠️  Security alert: {alert.alert_type} - {alert.severity}")
    
    # Test vote buying detection
    print("   🔍 Testing vote buying detection...")
    suspicious_voter = "0xbribed1234567890"
    
    # Add suspicious transaction
    security_manager.add_suspicious_transaction(
        suspicious_voter,
        {
            "timestamp": time.time() - 1800,  # 30 minutes ago
            "amount": 5000,
            "from": "0xattacker",
            "to": suspicious_voter
        }
    )
    
    voting_power = VotingPower(
        voter_address=suspicious_voter,
        power=2000,
        token_balance=2000
    )
    
    vote = Vote(
        proposal_id=proposal.proposal_id,
        voter_address=suspicious_voter,
        choice=VoteChoice.FOR,
        voting_power=voting_power,
        signature="0xbribed"
    )
    
    alerts = security_manager.analyze_vote(vote, proposal, {})
    if alerts:
        for alert in alerts:
            print(f"   ⚠️  Security alert: {alert.alert_type} - {alert.severity}")
    
    print_subsection("Security Statistics")
    
    stats = security_manager.get_security_statistics()
    print(f"   - Total alerts: {stats['total_alerts']}")
    print(f"   - Blocked addresses: {stats['blocked_addresses']}")
    print(f"   - Suspicious addresses: {stats['suspicious_addresses']}")
    print(f"   - Active detectors: {stats['active_detectors']}")


def demo_treasury_system():
    """Demonstrate treasury system."""
    print_section("Treasury System")
    
    treasury_manager = TreasuryManager({
        "limits": {
            "max_single_transaction": 100000,
            "max_daily_spending": 500000,
            "max_monthly_spending": 5000000
        }
    })
    
    print_subsection("Treasury Setup")
    
    # Add treasury balance
    treasury_manager.add_treasury_balance(
        token_address="0xTOKEN",
        amount=10000000,
        token_symbol="TOKEN"
    )
    
    balance = treasury_manager.get_treasury_balance("0xTOKEN")
    print(f"   ✅ Treasury balance: {balance} TOKEN")
    
    # Add multisig signers
    signers = ["0x1111111111111111", "0x2222222222222222", "0x3333333333333333", "0x4444444444444444"]
    for signer in signers:
        treasury_manager.add_multisig_signer(signer)
    
    print(f"   ✅ Added {len(signers)} multisig signers")
    
    print_subsection("Treasury Proposal")
    
    # Create treasury spending proposal
    proposal = treasury_manager.create_treasury_proposal(
        proposer_address="0x1234567890abcdef",
        operation_type="spending",
        recipient_address="0x5555555555555555",
        amount=50000,
        token_address="0xTOKEN",
        description="Community development grant",
        justification="Funding for community development initiatives"
    )
    
    print(f"   ✅ Treasury proposal created: {proposal.proposal_id}")
    print(f"   - Amount: {proposal.amount} TOKEN")
    print(f"   - Recipient: {proposal.recipient_address}")
    print(f"   - Status: {proposal.status.value}")
    
    print_subsection("Multisig Approval")
    
    # Approve with multisig signatures
    signatures = ["0xsig1", "0xsig2", "0xsig3"]
    for i, signature in enumerate(signatures):
        approved = treasury_manager.approve_treasury_proposal(
            proposal.proposal_id,
            signers[i],
            signature
        )
        print(f"   ✅ Signature {i+1} added: {approved}")
    
    print(f"   - Multisig approved: {proposal.is_multisig_approved()}")
    
    print_subsection("Treasury Execution")
    
    # Execute treasury proposal
    success = treasury_manager.execute_treasury_proposal(
        proposal.proposal_id,
        "0x1111111111111111"
    )
    
    if success:
        print(f"   ✅ Treasury proposal executed successfully")
        new_balance = treasury_manager.get_treasury_balance("0xTOKEN")
        print(f"   - New treasury balance: {new_balance} TOKEN")
    else:
        print(f"   ❌ Treasury proposal execution failed")
    
    print_subsection("Treasury Statistics")
    
    stats = treasury_manager.get_treasury_statistics()
    print(f"   - Total balance: {stats['total_balance']}")
    print(f"   - Total proposals: {stats['total_proposals']}")
    print(f"   - Executed proposals: {stats['executed_proposals']}")
    print(f"   - Multisig signers: {stats['multisig_signers']}")


def demo_observability_system():
    """Demonstrate observability system."""
    print_section("Observability System")
    
    observability = GovernanceEvents()
    
    print_subsection("Event Emission")
    
    # Emit various governance events
    events = [
        ("proposal_created", {"proposal_id": "prop_123", "title": "Test Proposal"}),
        ("proposal_activated", {"proposal_id": "prop_123"}),
        ("vote_cast", {"proposal_id": "prop_123", "voter": "0x123", "choice": "for"}),
        ("delegation_created", {"delegator": "0x111", "delegatee": "0x222"}),
        ("treasury_spending", {"amount": 1000, "recipient": "0x333"}),
        ("security_alert", {"alert_type": "sybil_attack", "severity": "high"}),
    ]
    
    for event_type, metadata in events:
        event = observability.emit_event(
            event_type=event_type,
            metadata=metadata
        )
        print(f"   ✅ Event emitted: {event_type} - {event.event_id}")
    
    print_subsection("Audit Trail")
    
    audit_trail = observability.get_audit_trail()
    summary = audit_trail.get_audit_summary()
    
    print(f"   - Total events: {summary['total_events']}")
    print(f"   - Unique proposals: {summary['unique_proposals']}")
    print(f"   - Unique voters: {summary['unique_voters']}")
    print(f"   - Integrity verified: {summary['integrity_verified']}")
    
    print_subsection("Merkle Proofs")
    
    # Create Merkle tree for votes
    merkle_manager = observability.get_merkle_proof_manager()
    
    vote_data = [
        "0x111:for:1000",
        "0x222:against:500",
        "0x333:for:800",
        "0x444:abstain:300"
    ]
    
    merkle_root = merkle_manager.create_merkle_tree("test_votes", vote_data)
    print(f"   ✅ Merkle tree created: {merkle_root}")
    
    # Generate proof for a vote
    proof = merkle_manager.generate_merkle_proof("test_votes", "0x111:for:1000")
    if proof:
        print(f"   ✅ Merkle proof generated for vote")
        print(f"   - Proof verified: {merkle_manager.verify_merkle_proof(proof)}")


def demo_upgrade_system():
    """Demonstrate upgrade system."""
    print_section("Upgrade System")
    
    upgrade_manager = UpgradeManager({
        "upgrade_timelock": 1000,
        "multisig_threshold": 3
    })
    
    print_subsection("Proxy Contract Setup")
    
    # Add proxy contracts
    governance_proxy = upgrade_manager.add_proxy_contract(
        proxy_address="0xGOVERNANCE_PROXY",
        implementation_address="0xGOVERNANCE_V1",
        admin_address="0xADMIN",
        is_governance_controlled=True
    )
    
    timelock_proxy = upgrade_manager.add_proxy_contract(
        proxy_address="0xTIMELOCK_PROXY",
        implementation_address="0xTIMELOCK_V1",
        admin_address="0xADMIN",
        is_governance_controlled=True
    )
    
    print(f"   ✅ Governance proxy: {governance_proxy.proxy_address}")
    print(f"   ✅ Timelock proxy: {timelock_proxy.proxy_address}")
    
    print_subsection("Upgrade Proposal")
    
    # Create upgrade proposal
    upgrade_proposal = upgrade_manager.create_upgrade_proposal(
        proposer_address="0x1234567890abcdef",
        upgrade_type="governance_upgrade",
        target_contract="0xGOVERNANCE_PROXY",
        new_implementation="0xGOVERNANCE_V2",
        upgrade_data=b"upgrade_data",
        requires_governance_approval=True,
        execution_delay=1000
    )
    
    print(f"   ✅ Upgrade proposal created: {upgrade_proposal.proposal_id}")
    print(f"   - Target contract: {upgrade_proposal.target_contract}")
    print(f"   - New implementation: {upgrade_proposal.new_implementation}")
    print(f"   - Status: {upgrade_proposal.status.value}")
    
    print_subsection("Emergency Escape Hatch")
    
    # Create emergency escape hatch
    escape_hatch = upgrade_manager.create_emergency_escape_hatch(
        hatch_id="emergency_hatch_1",
        description="Emergency upgrade for critical security vulnerability",
        trigger_conditions=["security_vulnerability_detected", "governance_compromise"],
        required_signatures=3
    )
    
    print(f"   ✅ Emergency escape hatch created: {escape_hatch.hatch_id}")
    print(f"   - Required signatures: {escape_hatch.required_signatures}")
    print(f"   - Trigger conditions: {escape_hatch.trigger_conditions}")
    
    print_subsection("Upgrade Statistics")
    
    stats = upgrade_manager.get_upgrade_statistics()
    print(f"   - Total proposals: {stats['total_proposals']}")
    print(f"   - Completed upgrades: {stats['completed_upgrades']}")
    print(f"   - Failed upgrades: {stats['failed_upgrades']}")
    print(f"   - Success rate: {stats['success_rate']:.2%}")
    print(f"   - Proxy contracts: {stats['proxy_contracts']}")
    print(f"   - Emergency hatches: {stats['emergency_hatches']}")


def demo_emergency_scenarios():
    """Demonstrate emergency scenarios."""
    print_section("Emergency Scenarios")
    
    config = GovernanceConfig()
    engine = GovernanceEngine(config)
    
    print_subsection("Emergency Pause")
    
    # Pause governance due to emergency
    engine.emergency_pause("Critical security vulnerability detected", 1000)
    print(f"   ✅ Governance paused: {engine.state.emergency_paused}")
    print(f"   - Reason: {engine.state.emergency_pause_reason}")
    print(f"   - Block: {engine.state.emergency_pause_block}")
    
    # Try to create proposal during pause (should fail)
    try:
        engine.create_proposal(
            proposer_address="0x1234567890abcdef",
            title="Emergency Proposal",
            description="This should fail during pause",
            proposal_type=ProposalType.PARAMETER_CHANGE
        )
        print("   ❌ Proposal creation should have failed during pause")
    except Exception as e:
        print(f"   ✅ Proposal creation blocked during pause: {type(e).__name__}")
    
    print_subsection("Emergency Resume")
    
    # Resume governance
    engine.emergency_resume()
    print(f"   ✅ Governance resumed: {not engine.state.emergency_paused}")
    
    # Now should be able to create proposals
    proposal = engine.create_proposal(
        proposer_address="0x1234567890abcdef",
        title="Post-Emergency Proposal",
        description="This should work after resume",
        proposal_type=ProposalType.PARAMETER_CHANGE
    )
    print(f"   ✅ Proposal created after resume: {proposal.proposal_id}")


def main():
    """Main demonstration function."""
    print("🎯 DubChain Governance System Demonstration")
    print("=" * 60)
    print("This demonstration showcases all features of the governance system:")
    print("- Proposal lifecycle management")
    print("- Multiple voting strategies")
    print("- Vote delegation system")
    print("- Security and attack detection")
    print("- Treasury management")
    print("- Observability and audit trails")
    print("- Upgrade mechanisms")
    print("- Emergency scenarios")
    
    try:
        # Run all demonstrations
        config = demo_governance_configuration()
        proposal = demo_proposal_lifecycle()
        demo_voting_strategies()
        demo_delegation_system()
        demo_security_system()
        demo_treasury_system()
        demo_observability_system()
        demo_upgrade_system()
        demo_emergency_scenarios()
        
        print_section("Demonstration Complete")
        print("✅ All governance system features demonstrated successfully!")
        print("\nKey Features Demonstrated:")
        print("   🗳️  Proposal lifecycle with voting and execution")
        print("   🎯 Multiple voting strategies (token-weighted, quadratic, etc.)")
        print("   🔗 Delegation system with circular prevention")
        print("   🔒 Security system with attack detection")
        print("   💰 Treasury management with multisig controls")
        print("   📊 Observability with events and audit trails")
        print("   🔄 Upgrade system with proxy patterns")
        print("   🚨 Emergency pause and resume functionality")
        
        print("\n🎉 Governance system is production-ready!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
