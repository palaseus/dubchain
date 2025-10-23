#!/usr/bin/env python3
"""
DubChain Master Test Runner

This script orchestrates all DubChain test suites:
- Comprehensive blockchain tests
- Performance stress tests
- Security and adversarial tests
- Integration tests
- Generates unified reports
"""

import asyncio
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import argparse

logger = logging.getLogger(__name__)

import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class TestSuiteResult:
    """Test suite result data structure."""
    name: str
    success: bool
    duration: float
    tests_passed: int
    tests_failed: int
    total_tests: int
    success_rate: float
    output_dir: str
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MasterTestResult:
    """Master test result data structure."""
    start_time: float
    end_time: float
    total_duration: float
    test_suites: List[TestSuiteResult] = field(default_factory=list)
    overall_success: bool = True
    total_tests_passed: int = 0
    total_tests_failed: int = 0
    total_tests: int = 0
    overall_success_rate: float = 0.0


class MasterTestRunner:
    """Master test runner for all DubChain test suites."""
    
    def __init__(self, output_dir: str = "master_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.scripts_dir = Path(__file__).parent
        self.result = MasterTestResult(
            start_time=time.time(),
            end_time=0.0,
            total_duration=0.0
        )
        
    def run_all_tests(self, test_types: List[str] = None) -> MasterTestResult:
        """Run all test suites."""
        logger.info("🚀 Starting DubChain Master Test Suite")
        logger.info("=" * 80)
        
        if test_types is None:
            test_types = ["comprehensive", "performance", "security", "integration"]
        
        try:
            # Run each test suite
            for test_type in test_types:
                if test_type == "comprehensive":
                    self._run_comprehensive_tests()
                elif test_type == "performance":
                    self._run_performance_tests()
                elif test_type == "security":
                    self._run_security_tests()
                elif test_type == "integration":
                    self._run_integration_tests()
                elif test_type == "existing":
                    self._run_existing_tests()
                else:
                    logger.info(f"⚠️  Unknown test type: {test_type}")
                    
            # Finalize results
            self._finalize_results()
            
            # Generate master report
            self._generate_master_report()
            
        except Exception as e:
            logger.info(f"❌ Master test runner failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.result.end_time = time.time()
            self.result.total_duration = self.result.end_time - self.result.start_time
            
        logger.info(f"\n✅ Master test suite completed in {self.result.total_duration:.2f} seconds")
        return self.result
        
    def _run_comprehensive_tests(self):
        """Run comprehensive blockchain tests."""
        logger.info("\n📦 Running Comprehensive Blockchain Tests...")
        
        start_time = time.time()
        script_path = self.scripts_dir / "comprehensive_blockchain_test.py"
        output_dir = self.output_dir / "comprehensive_results"
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path),
                "--output-dir", str(output_dir)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            duration = time.time() - start_time
            
            # Parse results from output
            success = result.returncode == 0
            tests_passed = 0
            tests_failed = 0
            
            if success:
                # Try to parse JSON results
                json_file = output_dir / "comprehensive_test_report.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        tests_passed = data.get("test_summary", {}).get("passed", 0)
                        tests_failed = data.get("test_summary", {}).get("failed", 0)
            else:
                tests_failed = 1
                
            total_tests = tests_passed + tests_failed
            success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
            
            test_result = TestSuiteResult(
                name="comprehensive_blockchain_tests",
                success=success,
                duration=duration,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                total_tests=total_tests,
                success_rate=success_rate)
                output_dir=str(output_dir),
                error=result.stderr if not success else None,
                details={
                    "return_code": result.returncode,
                    "stdout_lines": len(result.stdout.split('\n')),
                    "stderr_lines": len(result.stderr.split('\n'))
                }
            )
            
            self.result.test_suites.append(test_result)
            logger.info(f"  ✅ Comprehensive tests: {tests_passed}/{total_tests} passed ({duration:.2f}s)")
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="comprehensive_blockchain_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error="Test suite timed out after 5 minutes"
            )
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Comprehensive tests: TIMEOUT ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="comprehensive_blockchain_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error=str(e)
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Comprehensive tests: ERROR - {e}")
            
    def _run_performance_tests(self):
        """Run performance stress tests."""
        logger.info("\n⚡ Running Performance Stress Tests...")
        
        start_time = time.time()
        script_path = self.scripts_dir / "performance_stress_test.py"
        output_dir = self.output_dir / "performance_results"
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path),
                "--output-dir", str(output_dir)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            duration = time.time() - start_time
            
            success = result.returncode == 0
            tests_passed = 0
            tests_failed = 0
            
            if success:
                json_file = output_dir / "performance_stress_report.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        tests_passed = data.get("performance_summary", {}).get("successful_tests", 0)
                        tests_failed = data.get("performance_summary", {}).get("failed_tests", 0)
            else:
                tests_failed = 1
                
            total_tests = tests_passed + tests_failed
            success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
            
            test_result = TestSuiteResult(
                name="performance_stress_tests",
                success=success,
                duration=duration,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                    total_tests=total_tests,
                    success_rate=success_rate)
                output_dir=str(output_dir),
                error=result.stderr if not success else None
            )
            
            self.result.test_suites.append(test_result)
            logger.info(f"  ✅ Performance tests: {tests_passed}/{total_tests} passed ({duration:.2f}s)")
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="performance_stress_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error="Performance tests timed out after 10 minutes"
            )
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Performance tests: TIMEOUT ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="performance_stress_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error=str(e)
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Performance tests: ERROR - {e}")
            
    def _run_security_tests(self):
        """Run security and adversarial tests."""
        logger.info("\n🔒 Running Security and Adversarial Tests...")
        
        start_time = time.time()
        script_path = self.scripts_dir / "security_adversarial_test.py"
        output_dir = self.output_dir / "security_results"
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path),
                "--output-dir", str(output_dir)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            duration = time.time() - start_time
            
            success = result.returncode == 0
            tests_passed = 0
            tests_failed = 0
            
            if success:
                json_file = output_dir / "security_adversarial_report.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        tests_passed = data.get("security_summary", {}).get("passed_tests", 0)
                        tests_failed = data.get("security_summary", {}).get("failed_tests", 0)
            else:
                tests_failed = 1
                
            total_tests = tests_passed + tests_failed
            success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
            
            test_result = TestSuiteResult(
                name="security_adversarial_tests",
                success=success,
                duration=duration,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                    total_tests=total_tests,
                    success_rate=success_rate)
                output_dir=str(output_dir),
                error=result.stderr if not success else None
            )
            
            self.result.test_suites.append(test_result)
            logger.info(f"  ✅ Security tests: {tests_passed}/{total_tests} passed ({duration:.2f}s)")
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="security_adversarial_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error="Security tests timed out after 10 minutes"
            )
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Security tests: TIMEOUT ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="security_adversarial_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error=str(e)
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Security tests: ERROR - {e}")
            
    def _run_integration_tests(self):
        """Run integration tests."""
        logger.info("\n🔗 Running Integration Tests...")
        
        start_time = time.time()
        script_path = self.scripts_dir / "integration_comprehensive_test.py"
        output_dir = self.output_dir / "integration_results"
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path),
                "--output-dir", str(output_dir)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            duration = time.time() - start_time
            
            success = result.returncode == 0
            tests_passed = 0
            tests_failed = 0
            
            if success:
                json_file = output_dir / "integration_comprehensive_report.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        tests_passed = data.get("integration_summary", {}).get("passed_tests", 0)
                        tests_failed = data.get("integration_summary", {}).get("failed_tests", 0)
            else:
                tests_failed = 1
                
            total_tests = tests_passed + tests_failed
            success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
            
            test_result = TestSuiteResult(
                name="integration_tests",
                success=success,
                duration=duration,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                    total_tests=total_tests,
                    success_rate=success_rate)
                output_dir=str(output_dir),
                error=result.stderr if not success else None
            )
            
            self.result.test_suites.append(test_result)
            logger.info(f"  ✅ Integration tests: {tests_passed}/{total_tests} passed ({duration:.2f}s)")
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="integration_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error="Integration tests timed out after 10 minutes"
            )
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Integration tests: TIMEOUT ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestSuiteResult(
                name="integration_tests",
                success=False,
                duration=duration,
                tests_passed=0,
                tests_failed=1,
                    total_tests=1)
                success_rate=0.0)
                output_dir=str(output_dir),
                error=str(e)
            self.result.test_suites.append(test_result)
            logger.info(f"  ❌ Integration tests: ERROR - {e}")
            
    def _run_existing_tests(self):
        """Run existing test scripts."""
        logger.info("\n📜 Running Existing Test Scripts...")
        
        existing_scripts = [
            ("chaos_mode", "chaos_mode.py"),
            ("consensus_tests", "run_consensus_tests.py"),
            ("cleanup_results", "cleanup_results.py")
        ]
        
        for script_name, script_file in existing_scripts:
            start_time = time.time()
            script_path = self.scripts_dir / script_file
            
            try:
                if script_name == "consensus_tests":
                    result = subprocess.run([
                        sys.executable, str(script_path), "--all"
                    ], capture_output=True, text=True, timeout=300)
                else:
                    result = subprocess.run([
                        sys.executable, str(script_path)
                    ], capture_output=True, text=True, timeout=300)
                
                duration = time.time() - start_time
                success = result.returncode == 0
                
                test_result = TestSuiteResult(
                    name=f"existing_{script_name}",
                    success=success,
                    duration=duration,
                    tests_passed=1 if success else 0,
                    tests_failed=0 if success else 1,
                    total_tests=1)
                    success_rate=100.0 if success else 0.0)
                    output_dir="")
                    error=result.stderr if not success else None
                )
                
                self.result.test_suites.append(test_result)
                status = "✅" if success else "❌"
                logger.info(f"  {status} {script_name}: {'PASSED' if success else 'FAILED'} ({duration:.2f}s)")
                
            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                test_result = TestSuiteResult(
                    name=f"existing_{script_name}",
                    success=False,
                    duration=duration,
                    tests_passed=0,
                    tests_failed=1,
                    total_tests=1)
                    success_rate=0.0)
                    output_dir="")
                    error="Script timed out after 5 minutes"
                )
                self.result.test_suites.append(test_result)
                logger.info(f"  ❌ {script_name}: TIMEOUT ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                test_result = TestSuiteResult(
                    name=f"existing_{script_name}",
                    success=False,
                    duration=duration,
                    tests_passed=0,
                    tests_failed=1,
                    total_tests=1)
                    success_rate=0.0)
                    output_dir="")
                    error=str(e)
                self.result.test_suites.append(test_result)
                logger.info(f"  ❌ {script_name}: ERROR - {e}")
                
    def _finalize_results(self):
        """Finalize test results."""
        self.result.total_tests_passed = sum(suite.tests_passed for suite in self.result.test_suites)
        self.result.total_tests_failed = sum(suite.tests_failed for suite in self.result.test_suites)
        self.result.total_tests = self.result.total_tests_passed + self.result.total_tests_failed
        self.result.overall_success_rate = (self.result.total_tests_passed / self.result.total_tests * 100) if self.result.total_tests > 0 else 0
        self.result.overall_success = self.result.total_tests_failed == 0
        
    def _generate_master_report(self):
        """Generate master test report."""
        logger.info("\n📋 Generating Master Test Report...")
        
        # Generate JSON report
        report_data = {
            "master_test_summary": {
                "start_time": self.result.start_time,
                "end_time": self.result.end_time,
                "total_duration": self.result.total_duration,
                "overall_success": self.result.overall_success,
                "total_tests_passed": self.result.total_tests_passed,
                "total_tests_failed": self.result.total_tests_failed,
                "total_tests": self.result.total_tests,
                "overall_success_rate": self.result.overall_success_rate
            },
            "test_suites": [
                {
                    "name": suite.name,
                    "success": suite.success,
                    "duration": suite.duration,
                    "tests_passed": suite.tests_passed,
                    "tests_failed": suite.tests_failed,
                    "total_tests": suite.total_tests,
                    "success_rate": suite.success_rate,
                    "output_dir": suite.output_dir,
                    "error": suite.error,
                    "details": suite.details
                }
                for suite in self.result.test_suites
            ]
        }
        
        # Save JSON report
        json_file = self.output_dir / "master_test_report.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Generate markdown report
        markdown_file = self.output_dir / "master_test_report.md"
        with open(markdown_file, 'w') as f:
            f.write(self._generate_markdown_report(report_data)
        logger.info(f"📁 JSON report saved to: {json_file}")
        logger.info(f"📋 Markdown report saved to: {markdown_file}")
        
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown master report."""
        lines = [
            "# DubChain Master Test Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"- **Total Duration**: {report_data['master_test_summary']['total_duration']:.2f} seconds",
            f"- **Overall Success**: {'✅ YES' if report_data['master_test_summary']['overall_success'] else '❌ NO'}",
            f"- **Total Tests**: {report_data['master_test_summary']['total_tests']}",
            f"- **Passed**: {report_data['master_test_summary']['total_tests_passed']}",
            f"- **Failed**: {report_data['master_test_summary']['total_tests_failed']}",
            f"- **Success Rate**: {report_data['master_test_summary']['overall_success_rate']:.1f}%",
            "",
            "## Test Suite Results",
            ""]
        
        for suite in report_data["test_suites"]:
            status = "✅ PASSED" if suite["success"] else "❌ FAILED"
            lines.extend([
                f"### {suite['name']} {status}",
                f"- **Duration**: {suite['duration']:.2f}s")
                f"- **Tests**: {suite['tests_passed']}/{suite['total_tests']} passed")
                f"- **Success Rate**: {suite['success_rate']:.1f}%")
                f"- **Output Directory**: {suite['output_dir']}",
                ""])
            
            if suite["error"]:
                lines.append(f"**Error**: {suite['error']}")
                lines.append("")
                
        # Add recommendations
        lines.extend([
            "## Recommendations",
            "",
            "Based on the master test results:",
            ""])
        
        if report_data['master_test_summary']['overall_success']:
            lines.append("✅ **All test suites passed successfully!**")
            lines.append("- The DubChain system is functioning correctly")
            lines.append("- All components are properly integrated")
            lines.append("- Performance and security requirements are met")
        else:
            lines.append("⚠️ **Some test suites failed**")
            lines.append("- Review failed test suites for issues")
            lines.append("- Check error messages for specific problems")
            lines.append("- Address issues before deployment")
            
        lines.extend([
            "",
            "## Next Steps",
            "",
            "1. **Review individual test reports** for detailed analysis",
            "2. **Address any failed tests** before production deployment",
            "3. **Monitor system performance** in production",
            "4. **Schedule regular testing** to maintain quality",
            "5. **Update test suites** as new features are added",
            ""])
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run DubChain master test suite")
    parser.add_argument(
        "--output-dir")
        default="master_test_results")
        help="Output directory for test results"
    )
    parser.add_argument(
        "--test-types",
        nargs="+")
        choices=["comprehensive", "performance", "security", "integration", "existing", "all"])
        default=["all"])
        help="Test types to run"
    )
    parser.add_argument(
        "--quick")
        action="store_true")
        help="Run quick tests only"
    )
    
    args = parser.parse_args()
    
    # Handle "all" test type
    if "all" in args.test_types:
        args.test_types = ["comprehensive", "performance", "security", "integration", "existing"]
    
    # Create runner
    runner = MasterTestRunner(args.output_dir)
    
    # Run tests
    try:
        result = runner.run_all_tests(args.test_types)
        
        logger.info(f"\n🎉 Master test suite completed!")
        logger.info(f"📊 Overall Results: {result.total_tests_passed}/{result.total_tests} tests passed")
        logger.info(f"⏱️  Total Duration: {result.total_duration:.2f} seconds")
        logger.info(f"📈 Overall Success Rate: {result.overall_success_rate:.1f}%")
        
        if result.overall_success:
            logger.info("✨ All test suites passed!")
            sys.exit(0)
        else:
            logger.info(f"⚠️  {result.total_tests_failed} tests failed - check the detailed report")
            sys.exit(1)
            
    except Exception as e:
        logger.info(f"❌ Master test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
