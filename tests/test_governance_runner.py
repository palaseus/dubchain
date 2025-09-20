"""
Comprehensive test runner for governance system.

This module provides a test runner that executes all governance tests
and generates coverage reports.
"""

import pytest
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain.governance.core import GovernanceEngine, GovernanceConfig
from dubchain.governance.strategies import StrategyFactory
from dubchain.governance.delegation import DelegationManager
from dubchain.governance.security import SecurityManager
from dubchain.governance.execution import ExecutionEngine
from dubchain.governance.treasury import TreasuryManager
from dubchain.governance.observability import GovernanceEvents


class GovernanceTestRunner:
    """Comprehensive test runner for governance system."""
    
    def __init__(self):
        """Initialize test runner."""
        self.test_results = {}
        self.coverage_results = {}
        self.performance_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all governance tests."""
        print("🚀 Starting comprehensive governance system tests...")
        
        # Test categories
        test_categories = [
            ("unit", "Unit Tests"),
            ("integration", "Integration Tests"),
            ("property", "Property-Based Tests"),
            ("adversarial", "Adversarial Tests"),
            ("fuzz", "Fuzz Tests"),
        ]
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for category, description in test_categories:
            print(f"\n📋 Running {description}...")
            
            try:
                # Run tests for this category
                result = self._run_test_category(category)
                
                self.test_results[category] = result
                total_tests += result["total"]
                total_passed += result["passed"]
                total_failed += result["failed"]
                
                print(f"✅ {description}: {result['passed']}/{result['total']} passed")
                
            except Exception as e:
                print(f"❌ {description} failed: {e}")
                self.test_results[category] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 1,
                    "error": str(e)
                }
                total_failed += 1
        
        # Generate summary
        summary = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": total_passed / max(total_tests, 1),
            "categories": self.test_results
        }
        
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Success Rate: {summary['success_rate']:.2%}")
        
        return summary
    
    def _run_test_category(self, category: str) -> Dict[str, Any]:
        """Run tests for a specific category."""
        test_dir = Path(__file__).parent / category
        
        if not test_dir.exists():
            return {"total": 0, "passed": 0, "failed": 0, "error": "Test directory not found"}
        
        # Find test files
        test_files = list(test_dir.glob("test_*.py"))
        
        if not test_files:
            return {"total": 0, "passed": 0, "failed": 0, "error": "No test files found"}
        
        # Run pytest on test files
        pytest_args = [
            str(test_dir),
            "-v",
            "--tb=short",
            "--disable-warnings",
            "--no-header",
            "-q"
        ]
        
        try:
            result = pytest.main(pytest_args)
            
            # Parse result (simplified)
            if result == 0:
                return {"total": 1, "passed": 1, "failed": 0}
            else:
                return {"total": 1, "passed": 0, "failed": 1}
                
        except Exception as e:
            return {"total": 0, "passed": 0, "failed": 1, "error": str(e)}
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis on governance code."""
        print("\n📊 Running coverage analysis...")
        
        try:
            import coverage
            
            # Initialize coverage
            cov = coverage.Coverage(source=["dubchain.governance"])
            cov.start()
            
            # Run a subset of tests for coverage
            self._run_coverage_tests()
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            # Get coverage data
            coverage_data = {
                "total_lines": cov.total_lines,
                "covered_lines": cov.covered_lines,
                "coverage_percentage": cov.covered_lines / max(cov.total_lines, 1) * 100
            }
            
            print(f"✅ Coverage: {coverage_data['coverage_percentage']:.1f}%")
            print(f"   Total Lines: {coverage_data['total_lines']}")
            print(f"   Covered Lines: {coverage_data['covered_lines']}")
            
            return coverage_data
            
        except ImportError:
            print("⚠️  Coverage module not available, skipping coverage analysis")
            return {"error": "Coverage module not available"}
        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            return {"error": str(e)}
    
    def _run_coverage_tests(self):
        """Run tests for coverage analysis."""
        # Run unit tests
        pytest.main([str(Path(__file__).parent / "unit"), "-q"])
        
        # Run integration tests
        pytest.main([str(Path(__file__).parent / "integration"), "-q"])
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("\n⚡ Running performance benchmarks...")
        
        benchmarks = {}
        
        # Benchmark proposal creation
        start_time = time.time()
        config = GovernanceConfig()
        engine = GovernanceEngine(config)
        
        for i in range(1000):
            engine.create_proposal(
                proposer_address=f"0x{i}",
                title=f"Proposal {i}",
                description=f"Description {i}",
                proposal_type="parameter_change"
            )
        
        benchmarks["proposal_creation"] = {
            "operations": 1000,
            "time_seconds": time.time() - start_time,
            "ops_per_second": 1000 / (time.time() - start_time)
        }
        
        # Benchmark vote casting
        start_time = time.time()
        proposal = engine.create_proposal(
            proposer_address="0x123",
            title="Benchmark Proposal",
            description="Benchmark description",
            proposal_type="parameter_change"
        )
        
        for i in range(1000):
            from dubchain.governance.core import Vote, VoteChoice, VotingPower
            voting_power = VotingPower(
                voter_address=f"0x{i}",
                power=1000,
                token_balance=1000
            )
            vote = Vote(
                proposal_id=proposal.proposal_id,
                voter_address=f"0x{i}",
                choice=VoteChoice.FOR,
                voting_power=voting_power,
                signature=f"0x{i}"
            )
            proposal.add_vote(vote)
        
        benchmarks["vote_casting"] = {
            "operations": 1000,
            "time_seconds": time.time() - start_time,
            "ops_per_second": 1000 / (time.time() - start_time)
        }
        
        # Benchmark delegation creation
        start_time = time.time()
        delegation_manager = DelegationManager(config)
        
        for i in range(1000):
            try:
                delegation_manager.create_delegation(
                    delegator_address=f"0x{i}",
                    delegatee_address=f"0x{i + 1000}",
                    delegation_power=1000
                )
            except ValueError:
                continue  # Skip circular delegations
        
        benchmarks["delegation_creation"] = {
            "operations": 1000,
            "time_seconds": time.time() - start_time,
            "ops_per_second": 1000 / (time.time() - start_time)
        }
        
        print("✅ Performance benchmarks completed:")
        for benchmark_name, data in benchmarks.items():
            print(f"   {benchmark_name}: {data['ops_per_second']:.0f} ops/sec")
        
        return benchmarks
    
    def run_security_audit(self) -> Dict[str, Any]:
        """Run security audit tests."""
        print("\n🔒 Running security audit...")
        
        security_results = {
            "sybil_detection": False,
            "vote_buying_detection": False,
            "flash_loan_detection": False,
            "front_running_detection": False,
            "circular_delegation_prevention": False
        }
        
        try:
            # Test Sybil detection
            config = GovernanceConfig()
            engine = GovernanceEngine(config)
            engine.security_manager = SecurityManager()
            
            # Create proposal
            proposal = engine.create_proposal(
                proposer_address="0x123",
                title="Security Test Proposal",
                description="Testing security measures",
                proposal_type="parameter_change"
            )
            
            # Test Sybil detection
            from dubchain.governance.core import Vote, VoteChoice, VotingPower
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
                
                alerts = engine.security_manager.analyze_vote(vote, proposal, {})
                if any(alert.alert_type == "sybil_attack" for alert in alerts):
                    security_results["sybil_detection"] = True
                    break
            
            # Test circular delegation prevention
            delegation_manager = DelegationManager(config)
            try:
                delegation_manager.create_delegation("0x123", "0x456", 1000)
                delegation_manager.create_delegation("0x456", "0x123", 1000)  # Should fail
            except ValueError:
                security_results["circular_delegation_prevention"] = True
            
            print("✅ Security audit completed:")
            for test_name, passed in security_results.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}")
            
        except Exception as e:
            print(f"❌ Security audit failed: {e}")
            security_results["error"] = str(e)
        
        return security_results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("# Governance System Test Report")
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Test results
        report.append("## Test Results")
        for category, result in self.test_results.items():
            report.append(f"### {category.title()} Tests")
            if "error" in result:
                report.append(f"Error: {result['error']}")
            else:
                report.append(f"- Total: {result['total']}")
                report.append(f"- Passed: {result['passed']}")
                report.append(f"- Failed: {result['failed']}")
            report.append("")
        
        # Coverage results
        if self.coverage_results:
            report.append("## Coverage Results")
            if "error" in self.coverage_results:
                report.append(f"Error: {self.coverage_results['error']}")
            else:
                report.append(f"- Coverage: {self.coverage_results['coverage_percentage']:.1f}%")
                report.append(f"- Total Lines: {self.coverage_results['total_lines']}")
                report.append(f"- Covered Lines: {self.coverage_results['covered_lines']}")
            report.append("")
        
        # Performance results
        if self.performance_results:
            report.append("## Performance Results")
            for benchmark_name, data in self.performance_results.items():
                report.append(f"### {benchmark_name.title()}")
                report.append(f"- Operations: {data['operations']}")
                report.append(f"- Time: {data['time_seconds']:.2f} seconds")
                report.append(f"- Rate: {data['ops_per_second']:.0f} ops/sec")
                report.append("")
        
        return "\n".join(report)
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("🎯 Starting comprehensive governance test suite...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        test_results = self.run_all_tests()
        
        # Run coverage analysis
        self.coverage_results = self.run_coverage_analysis()
        
        # Run performance benchmarks
        self.performance_results = self.run_performance_benchmarks()
        
        # Run security audit
        security_results = self.run_security_audit()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        report_path = Path(__file__).parent / "governance_test_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 Comprehensive test suite completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📄 Report saved to: {report_path}")
        
        return {
            "test_results": test_results,
            "coverage_results": self.coverage_results,
            "performance_results": self.performance_results,
            "security_results": security_results,
            "total_time": total_time,
            "report_path": str(report_path)
        }


def main():
    """Main entry point for test runner."""
    runner = GovernanceTestRunner()
    results = runner.run_comprehensive_test_suite()
    
    # Exit with appropriate code
    if results["test_results"]["total_failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
