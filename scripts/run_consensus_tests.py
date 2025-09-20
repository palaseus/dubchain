#!/usr/bin/env python3
"""
Comprehensive test runner for DubChain consensus mechanisms.

This script runs all consensus-related tests including unit tests,
adversarial tests, property-based tests, and benchmarks.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Exit code: {result.returncode}")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    
    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run DubChain consensus tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--adversarial", action="store_true", help="Run adversarial tests")
    parser.add_argument("--property", action="store_true", help="Run property-based tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only")
    
    args = parser.parse_args()
    
    # If no specific test type is specified, run all
    if not any([args.unit, args.adversarial, args.property, args.benchmark]):
        args.all = True
    
    # Base pytest command
    base_cmd = ["python3", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    else:
        base_cmd.append("-q")
    
    if args.coverage:
        base_cmd.extend(["--cov=src/dubchain/consensus", "--cov-report=html", "--cov-report=term"])
    
    # Test results
    results = {}
    
    # Unit tests
    if args.unit or args.all:
        cmd = base_cmd + [
            "tests/unit/test_consensus_new_mechanisms.py"
        ]
        if args.fast:
            cmd.extend(["-k", "not slow"])
        
        results["unit"] = run_command(cmd, "Unit Tests")
    
    # Adversarial tests
    if args.adversarial or args.all:
        cmd = base_cmd + ["tests/adversarial/test_consensus_adversarial.py"]
        if args.fast:
            cmd.extend(["-k", "not slow"])
        
        results["adversarial"] = run_command(cmd, "Adversarial Tests")
    
    # Property-based tests
    if args.property or args.all:
        cmd = base_cmd + ["tests/property/test_consensus_property.py"]
        if args.fast:
            cmd.extend(["--max-examples=5"])  # Reduce examples for faster run
        
        results["property"] = run_command(cmd, "Property-Based Tests")
    
    # Benchmark tests
    if args.benchmark or args.all:
        cmd = base_cmd + ["tests/benchmark/test_consensus_benchmark.py"]
        if args.fast:
            cmd.extend(["-k", "not slow"])
        
        results["benchmark"] = run_command(cmd, "Benchmark Tests")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    for test_type, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_type.upper():<15}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
    
    if args.coverage:
        print(f"\nCoverage report generated in htmlcov/index.html")
    
    # Exit with error code if any tests failed
    if passed_tests < total_tests:
        sys.exit(1)
    else:
        print("\nAll tests passed! 🎉")
        sys.exit(0)


if __name__ == "__main__":
    main()
