"""Coverage analysis infrastructure for DubChain.

This module provides code coverage analysis capabilities for testing.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ..logging import get_logger
from .base import ExecutionResult, ExecutionStatus


@dataclass
class CoverageMetrics:
    """Coverage metrics data structure."""

    # Line coverage
    total_lines: int = 0
    covered_lines: int = 0
    line_coverage_percent: float = 0.0

    # Branch coverage
    total_branches: int = 0
    covered_branches: int = 0
    branch_coverage_percent: float = 0.0

    # Function coverage
    total_functions: int = 0
    covered_functions: int = 0
    function_coverage_percent: float = 0.0

    # File coverage
    total_files: int = 0
    covered_files: int = 0
    file_coverage_percent: float = 0.0

    # Overall coverage
    overall_coverage_percent: float = 0.0

    # Detailed coverage data
    file_coverage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    uncovered_lines: Dict[str, List[int]] = field(default_factory=dict)
    uncovered_branches: Dict[str, List[tuple]] = field(default_factory=dict)
    uncovered_functions: Dict[str, List[str]] = field(default_factory=dict)


class CoverageCollector:
    """Coverage data collector."""

    def __init__(self):
        self.coverage_data: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.logger = get_logger("coverage_collector")

    def start_collection(self) -> None:
        """Start coverage collection."""
        with self._lock:
            self.coverage_data.clear()
            self.logger.info("Started coverage collection")

    def stop_collection(self) -> None:
        """Stop coverage collection."""
        with self._lock:
            self.logger.info("Stopped coverage collection")

    def add_coverage_data(
        self, file_path: str, line_number: int, branch_id: str = None
    ) -> None:
        """Add coverage data for a line."""
        with self._lock:
            if file_path not in self.coverage_data:
                self.coverage_data[file_path] = {
                    "lines": set(),
                    "branches": set(),
                    "functions": set(),
                }

            self.coverage_data[file_path]["lines"].add(line_number)

            if branch_id:
                self.coverage_data[file_path]["branches"].add(branch_id)

    def add_function_coverage(self, file_path: str, function_name: str) -> None:
        """Add function coverage data."""
        with self._lock:
            if file_path not in self.coverage_data:
                self.coverage_data[file_path] = {
                    "lines": set(),
                    "branches": set(),
                    "functions": set(),
                }

            self.coverage_data[file_path]["functions"].add(function_name)

    def get_coverage_data(self) -> Dict[str, Any]:
        """Get collected coverage data."""
        with self._lock:
            return self.coverage_data.copy()


class CoverageAnalyzer:
    """Coverage data analyzer."""

    def __init__(self):
        self.metrics = CoverageMetrics()
        self.logger = get_logger("coverage_analyzer")

    def analyze_coverage(
        self, coverage_data: Dict[str, Any], source_files: Dict[str, List[str]] = None
    ) -> CoverageMetrics:
        """Analyze coverage data and calculate metrics."""
        self.metrics = CoverageMetrics()

        # Analyze file coverage
        for file_path, file_data in coverage_data.items():
            self._analyze_file_coverage(file_path, file_data, source_files)

        # Calculate overall metrics
        self._calculate_overall_metrics()

        return self.metrics

    def _analyze_file_coverage(
        self,
        file_path: str,
        file_data: Dict[str, Any],
        source_files: Dict[str, List[str]] = None,
    ) -> None:
        """Analyze coverage for a single file."""
        covered_lines = file_data.get("lines", set())
        covered_branches = file_data.get("branches", set())
        covered_functions = file_data.get("functions", set())

        # Get source file info if available
        if source_files and file_path in source_files:
            source_lines = source_files[file_path]
            total_lines = len(source_lines)

            # Find uncovered lines
            uncovered_lines = []
            for i, line in enumerate(source_lines):
                if (
                    i + 1 not in covered_lines and line.strip()
                ):  # +1 for 1-based line numbers
                    uncovered_lines.append(i + 1)

            self.metrics.uncovered_lines[file_path] = uncovered_lines
        else:
            total_lines = max(covered_lines) if covered_lines else 0

        # Calculate file metrics
        file_metrics = {
            "total_lines": total_lines,
            "covered_lines": len(covered_lines),
            "line_coverage_percent": (len(covered_lines) / total_lines * 100)
            if total_lines > 0
            else 0,
            "total_branches": len(covered_branches),  # Simplified
            "covered_branches": len(covered_branches),
            "branch_coverage_percent": 100.0 if covered_branches else 0,
            "total_functions": len(covered_functions),  # Simplified
            "covered_functions": len(covered_functions),
            "function_coverage_percent": 100.0 if covered_functions else 0,
        }

        self.metrics.file_coverage[file_path] = file_metrics

        # Update overall metrics
        self.metrics.total_lines += total_lines
        self.metrics.covered_lines += len(covered_lines)
        self.metrics.total_branches += file_metrics["total_branches"]
        self.metrics.covered_branches += file_metrics["covered_branches"]
        self.metrics.total_functions += file_metrics["total_functions"]
        self.metrics.covered_functions += file_metrics["covered_functions"]
        self.metrics.total_files += 1

        if file_metrics["line_coverage_percent"] > 0:
            self.metrics.covered_files += 1

    def _calculate_overall_metrics(self) -> None:
        """Calculate overall coverage metrics."""
        # Line coverage
        self.metrics.line_coverage_percent = (
            (self.metrics.covered_lines / self.metrics.total_lines * 100)
            if self.metrics.total_lines > 0
            else 0
        )

        # Branch coverage
        self.metrics.branch_coverage_percent = (
            (self.metrics.covered_branches / self.metrics.total_branches * 100)
            if self.metrics.total_branches > 0
            else 0
        )

        # Function coverage
        self.metrics.function_coverage_percent = (
            (self.metrics.covered_functions / self.metrics.total_functions * 100)
            if self.metrics.total_functions > 0
            else 0
        )

        # File coverage
        self.metrics.file_coverage_percent = (
            (self.metrics.covered_files / self.metrics.total_files * 100)
            if self.metrics.total_files > 0
            else 0
        )

        # Overall coverage (weighted average)
        self.metrics.overall_coverage_percent = (
            self.metrics.line_coverage_percent * 0.4
            + self.metrics.branch_coverage_percent * 0.3
            + self.metrics.function_coverage_percent * 0.2
            + self.metrics.file_coverage_percent * 0.1
        )

    def get_coverage_report(self) -> Dict[str, Any]:
        """Get detailed coverage report."""
        return {
            "overall_metrics": {
                "total_lines": self.metrics.total_lines,
                "covered_lines": self.metrics.covered_lines,
                "line_coverage_percent": self.metrics.line_coverage_percent,
                "total_branches": self.metrics.total_branches,
                "covered_branches": self.metrics.covered_branches,
                "branch_coverage_percent": self.metrics.branch_coverage_percent,
                "total_functions": self.metrics.total_functions,
                "covered_functions": self.metrics.covered_functions,
                "function_coverage_percent": self.metrics.function_coverage_percent,
                "total_files": self.metrics.total_files,
                "covered_files": self.metrics.covered_files,
                "file_coverage_percent": self.metrics.file_coverage_percent,
                "overall_coverage_percent": self.metrics.overall_coverage_percent,
            },
            "file_coverage": self.metrics.file_coverage,
            "uncovered_lines": self.metrics.uncovered_lines,
            "uncovered_branches": self.metrics.uncovered_branches,
            "uncovered_functions": self.metrics.uncovered_functions,
        }

    def check_coverage_threshold(self, threshold: float = 80.0) -> bool:
        """Check if coverage meets threshold."""
        return self.metrics.overall_coverage_percent >= threshold

    def get_files_below_threshold(self, threshold: float = 80.0) -> List[str]:
        """Get files with coverage below threshold."""
        below_threshold = []

        for file_path, file_metrics in self.metrics.file_coverage.items():
            if file_metrics["line_coverage_percent"] < threshold:
                below_threshold.append(file_path)

        return below_threshold


class CoverageReporter:
    """Coverage report generator."""

    def __init__(self):
        self.logger = get_logger("coverage_reporter")

    def generate_text_report(self, metrics: CoverageMetrics) -> str:
        """Generate text coverage report."""
        report_lines = [
            "Coverage Report",
            "=" * 50,
            f"Overall Coverage: {metrics.overall_coverage_percent:.2f}%",
            "",
            "Summary:",
            f"  Lines: {metrics.covered_lines}/{metrics.total_lines} ({metrics.line_coverage_percent:.2f}%)",
            f"  Branches: {metrics.covered_branches}/{metrics.total_branches} ({metrics.branch_coverage_percent:.2f}%)",
            f"  Functions: {metrics.covered_functions}/{metrics.total_functions} ({metrics.function_coverage_percent:.2f}%)",
            f"  Files: {metrics.covered_files}/{metrics.total_files} ({metrics.file_coverage_percent:.2f}%)",
            "",
            "File Coverage:",
        ]

        for file_path, file_metrics in metrics.file_coverage.items():
            report_lines.append(
                f"  {file_path}: {file_metrics['line_coverage_percent']:.2f}% "
                f"({file_metrics['covered_lines']}/{file_metrics['total_lines']} lines)"
            )

        return "\n".join(report_lines)

    def generate_html_report(self, metrics: CoverageMetrics) -> str:
        """Generate HTML coverage report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coverage Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .file-list {{ margin: 20px 0; }}
                .file-item {{ margin: 5px 0; padding: 5px; border-left: 3px solid #ccc; }}
                .coverage-high {{ border-left-color: #4CAF50; }}
                .coverage-medium {{ border-left-color: #FF9800; }}
                .coverage-low {{ border-left-color: #F44336; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Coverage Report</h1>
                <p>Overall Coverage: <strong>{metrics.overall_coverage_percent:.2f}%</strong></p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <ul>
                    <li>Lines: {metrics.covered_lines}/{metrics.total_lines} ({metrics.line_coverage_percent:.2f}%)</li>
                    <li>Branches: {metrics.covered_branches}/{metrics.total_branches} ({metrics.branch_coverage_percent:.2f}%)</li>
                    <li>Functions: {metrics.covered_functions}/{metrics.total_functions} ({metrics.function_coverage_percent:.2f}%)</li>
                    <li>Files: {metrics.covered_files}/{metrics.total_files} ({metrics.file_coverage_percent:.2f}%)</li>
                </ul>
            </div>
            
            <div class="file-list">
                <h2>File Coverage</h2>
        """

        for file_path, file_metrics in metrics.file_coverage.items():
            coverage_class = "coverage-high"
            if file_metrics["line_coverage_percent"] < 50:
                coverage_class = "coverage-low"
            elif file_metrics["line_coverage_percent"] < 80:
                coverage_class = "coverage-medium"

            html += f"""
                <div class="file-item {coverage_class}">
                    <strong>{file_path}</strong>: {file_metrics['line_coverage_percent']:.2f}%
                    ({file_metrics['covered_lines']}/{file_metrics['total_lines']} lines)
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def generate_json_report(self, metrics: CoverageMetrics) -> str:
        """Generate JSON coverage report."""
        import json

        report_data = {
            "overall_coverage_percent": metrics.overall_coverage_percent,
            "summary": {
                "lines": {
                    "covered": metrics.covered_lines,
                    "total": metrics.total_lines,
                    "percent": metrics.line_coverage_percent,
                },
                "branches": {
                    "covered": metrics.covered_branches,
                    "total": metrics.total_branches,
                    "percent": metrics.branch_coverage_percent,
                },
                "functions": {
                    "covered": metrics.covered_functions,
                    "total": metrics.total_functions,
                    "percent": metrics.function_coverage_percent,
                },
                "files": {
                    "covered": metrics.covered_files,
                    "total": metrics.total_files,
                    "percent": metrics.file_coverage_percent,
                },
            },
            "file_coverage": metrics.file_coverage,
            "uncovered_lines": metrics.uncovered_lines,
            "uncovered_branches": metrics.uncovered_branches,
            "uncovered_functions": metrics.uncovered_functions,
        }

        return json.dumps(report_data, indent=2)

    def save_report(
        self, report: str, file_path: str, format_type: str = "text"
    ) -> bool:
        """Save coverage report to file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save coverage report: {e}")
            return False
