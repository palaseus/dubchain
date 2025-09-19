"""Test cases for testing/coverage.py module."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from dubchain.testing.coverage import (
    CoverageAnalyzer,
    CoverageCollector,
    CoverageMetrics,
    CoverageReporter,
)


class TestCoverageMetrics:
    """Test CoverageMetrics dataclass."""

    def test_coverage_metrics_creation(self):
        """Test CoverageMetrics creation with default values."""
        metrics = CoverageMetrics()

        assert metrics.total_lines == 0
        assert metrics.covered_lines == 0
        assert metrics.line_coverage_percent == 0.0
        assert metrics.total_branches == 0
        assert metrics.covered_branches == 0
        assert metrics.branch_coverage_percent == 0.0
        assert metrics.total_functions == 0
        assert metrics.covered_functions == 0
        assert metrics.function_coverage_percent == 0.0
        assert metrics.total_files == 0
        assert metrics.covered_files == 0
        assert metrics.file_coverage_percent == 0.0
        assert metrics.overall_coverage_percent == 0.0
        assert metrics.file_coverage == {}
        assert metrics.uncovered_lines == {}
        assert metrics.uncovered_branches == {}
        assert metrics.uncovered_functions == {}

    def test_coverage_metrics_custom_values(self):
        """Test CoverageMetrics creation with custom values."""
        metrics = CoverageMetrics(
            total_lines=100,
            covered_lines=80,
            line_coverage_percent=80.0,
            total_branches=20,
            covered_branches=16,
            branch_coverage_percent=80.0,
            total_functions=10,
            covered_functions=8,
            function_coverage_percent=80.0,
            total_files=5,
            covered_files=4,
            file_coverage_percent=80.0,
            overall_coverage_percent=80.0,
        )

        assert metrics.total_lines == 100
        assert metrics.covered_lines == 80
        assert metrics.line_coverage_percent == 80.0
        assert metrics.total_branches == 20
        assert metrics.covered_branches == 16
        assert metrics.branch_coverage_percent == 80.0
        assert metrics.total_functions == 10
        assert metrics.covered_functions == 8
        assert metrics.function_coverage_percent == 80.0
        assert metrics.total_files == 5
        assert metrics.covered_files == 4
        assert metrics.file_coverage_percent == 80.0
        assert metrics.overall_coverage_percent == 80.0


class TestCoverageCollector:
    """Test CoverageCollector class."""

    def test_coverage_collector_creation(self):
        """Test CoverageCollector creation."""
        collector = CoverageCollector()

        assert collector.coverage_data == {}
        assert collector.logger is not None

    def test_coverage_collector_start_collection(self):
        """Test start_collection method."""
        collector = CoverageCollector()
        collector.coverage_data = {"existing": "data"}

        collector.start_collection()

        assert collector.coverage_data == {}

    def test_coverage_collector_stop_collection(self):
        """Test stop_collection method."""
        collector = CoverageCollector()

        # Should not raise any exceptions
        collector.stop_collection()

    def test_coverage_collector_add_coverage_data(self):
        """Test add_coverage_data method."""
        collector = CoverageCollector()

        collector.add_coverage_data("test.py", 10)

        assert "test.py" in collector.coverage_data
        assert 10 in collector.coverage_data["test.py"]["lines"]
        assert collector.coverage_data["test.py"]["branches"] == set()
        assert collector.coverage_data["test.py"]["functions"] == set()

    def test_coverage_collector_add_coverage_data_with_branch(self):
        """Test add_coverage_data method with branch."""
        collector = CoverageCollector()

        collector.add_coverage_data("test.py", 10, "branch_1")

        assert "test.py" in collector.coverage_data
        assert 10 in collector.coverage_data["test.py"]["lines"]
        assert "branch_1" in collector.coverage_data["test.py"]["branches"]

    def test_coverage_collector_add_function_coverage(self):
        """Test add_function_coverage method."""
        collector = CoverageCollector()

        collector.add_function_coverage("test.py", "test_function")

        assert "test.py" in collector.coverage_data
        assert "test_function" in collector.coverage_data["test.py"]["functions"]
        assert collector.coverage_data["test.py"]["lines"] == set()
        assert collector.coverage_data["test.py"]["branches"] == set()

    def test_coverage_collector_get_coverage_data(self):
        """Test get_coverage_data method."""
        collector = CoverageCollector()
        collector.add_coverage_data("test.py", 10)

        data = collector.get_coverage_data()

        assert data == collector.coverage_data
        assert data is not collector.coverage_data  # Should be a copy


class TestCoverageAnalyzer:
    """Test CoverageAnalyzer class."""

    def test_coverage_analyzer_creation(self):
        """Test CoverageAnalyzer creation."""
        analyzer = CoverageAnalyzer()

        assert analyzer.metrics is not None
        assert analyzer.logger is not None

    def test_coverage_analyzer_analyze_coverage_empty(self):
        """Test analyze_coverage with empty data."""
        analyzer = CoverageAnalyzer()

        metrics = analyzer.analyze_coverage({})

        assert metrics.total_lines == 0
        assert metrics.covered_lines == 0
        assert metrics.line_coverage_percent == 0.0
        assert metrics.total_branches == 0
        assert metrics.covered_branches == 0
        assert metrics.branch_coverage_percent == 0.0
        assert metrics.total_functions == 0
        assert metrics.covered_functions == 0
        assert metrics.function_coverage_percent == 0.0
        assert metrics.total_files == 0
        assert metrics.covered_files == 0
        assert metrics.file_coverage_percent == 0.0
        assert metrics.overall_coverage_percent == 0.0

    def test_coverage_analyzer_analyze_coverage_with_data(self):
        """Test analyze_coverage with coverage data."""
        analyzer = CoverageAnalyzer()

        coverage_data = {
            "test.py": {
                "lines": {1, 2, 3},
                "branches": {"branch_1", "branch_2"},
                "functions": {"func1", "func2"},
            }
        }

        metrics = analyzer.analyze_coverage(coverage_data)

        assert metrics.total_lines == 3  # max(covered_lines)
        assert metrics.covered_lines == 3
        assert metrics.line_coverage_percent == 100.0
        assert metrics.total_branches == 2
        assert metrics.covered_branches == 2
        assert metrics.branch_coverage_percent == 100.0
        assert metrics.total_functions == 2
        assert metrics.covered_functions == 2
        assert metrics.function_coverage_percent == 100.0
        assert metrics.total_files == 1
        assert metrics.covered_files == 1
        assert metrics.file_coverage_percent == 100.0
        assert metrics.overall_coverage_percent == 100.0

    def test_coverage_analyzer_analyze_coverage_with_source_files(self):
        """Test analyze_coverage with source files."""
        analyzer = CoverageAnalyzer()

        coverage_data = {
            "test.py": {
                "lines": {1, 3},  # Line 2 is uncovered
                "branches": set(),
                "functions": set(),
            }
        }

        source_files = {"test.py": ["def func():", "    pass", "    return"]}

        metrics = analyzer.analyze_coverage(coverage_data, source_files)

        assert metrics.total_lines == 3
        assert metrics.covered_lines == 2
        assert abs(metrics.line_coverage_percent - 66.67) < 0.01
        assert metrics.uncovered_lines["test.py"] == [2]

    def test_coverage_analyzer_get_coverage_report(self):
        """Test get_coverage_report method."""
        analyzer = CoverageAnalyzer()

        coverage_data = {
            "test.py": {
                "lines": {1, 2, 3},
                "branches": {"branch_1"},
                "functions": {"func1"},
            }
        }

        analyzer.analyze_coverage(coverage_data)
        report = analyzer.get_coverage_report()

        assert "overall_metrics" in report
        assert "file_coverage" in report
        assert "uncovered_lines" in report
        assert "uncovered_branches" in report
        assert "uncovered_functions" in report

        overall = report["overall_metrics"]
        assert overall["total_lines"] == 3
        assert overall["covered_lines"] == 3
        assert overall["line_coverage_percent"] == 100.0

    def test_coverage_analyzer_check_coverage_threshold(self):
        """Test check_coverage_threshold method."""
        analyzer = CoverageAnalyzer()

        coverage_data = {
            "test.py": {
                "lines": {1, 2, 3},
                "branches": {"branch_1"},
                "functions": {"func1"},
            }
        }

        analyzer.analyze_coverage(coverage_data)

        assert analyzer.check_coverage_threshold(80.0) is True
        assert analyzer.check_coverage_threshold(100.0) is True
        assert analyzer.check_coverage_threshold(101.0) is False

    def test_coverage_analyzer_get_files_below_threshold(self):
        """Test get_files_below_threshold method."""
        analyzer = CoverageAnalyzer()

        coverage_data = {
            "test1.py": {"lines": {1, 2, 3}, "branches": set(), "functions": set()},
            "test2.py": {
                "lines": {1},  # Only 1 line covered
                "branches": set(),
                "functions": set(),
            },
        }

        source_files = {
            "test1.py": ["line1", "line2", "line3"],
            "test2.py": ["line1", "line2", "line3", "line4", "line5"],  # 5 lines total
        }

        analyzer.analyze_coverage(coverage_data, source_files)

        below_threshold = analyzer.get_files_below_threshold(80.0)

        assert "test2.py" in below_threshold
        assert "test1.py" not in below_threshold


class TestCoverageReporter:
    """Test CoverageReporter class."""

    def test_coverage_reporter_creation(self):
        """Test CoverageReporter creation."""
        reporter = CoverageReporter()

        assert reporter.logger is not None

    def test_coverage_reporter_generate_text_report(self):
        """Test generate_text_report method."""
        reporter = CoverageReporter()

        metrics = CoverageMetrics(
            total_lines=100,
            covered_lines=80,
            line_coverage_percent=80.0,
            total_branches=20,
            covered_branches=16,
            branch_coverage_percent=80.0,
            total_functions=10,
            covered_functions=8,
            function_coverage_percent=80.0,
            total_files=5,
            covered_files=4,
            file_coverage_percent=80.0,
            overall_coverage_percent=80.0,
            file_coverage={
                "test.py": {
                    "total_lines": 50,
                    "covered_lines": 40,
                    "line_coverage_percent": 80.0,
                }
            },
        )

        report = reporter.generate_text_report(metrics)

        assert "Coverage Report" in report
        assert "Overall Coverage: 80.00%" in report
        assert "Lines: 80/100 (80.00%)" in report
        assert "Branches: 16/20 (80.00%)" in report
        assert "Functions: 8/10 (80.00%)" in report
        assert "Files: 4/5 (80.00%)" in report
        assert "test.py: 80.00% (40/50 lines)" in report

    def test_coverage_reporter_generate_html_report(self):
        """Test generate_html_report method."""
        reporter = CoverageReporter()

        metrics = CoverageMetrics(
            total_lines=100,
            covered_lines=80,
            line_coverage_percent=80.0,
            total_branches=20,
            covered_branches=16,
            branch_coverage_percent=80.0,
            total_functions=10,
            covered_functions=8,
            function_coverage_percent=80.0,
            total_files=5,
            covered_files=4,
            file_coverage_percent=80.0,
            overall_coverage_percent=80.0,
            file_coverage={
                "test.py": {
                    "total_lines": 50,
                    "covered_lines": 40,
                    "line_coverage_percent": 80.0,
                }
            },
        )

        report = reporter.generate_html_report(metrics)

        assert "<!DOCTYPE html>" in report
        assert "<title>Coverage Report</title>" in report
        assert "Overall Coverage: <strong>80.00%</strong>" in report
        assert "Lines: 80/100 (80.00%)" in report
        assert "Branches: 16/20 (80.00%)" in report
        assert "Functions: 8/10 (80.00%)" in report
        assert "Files: 4/5 (80.00%)" in report
        assert "<strong>test.py</strong>: 80.00%" in report

    def test_coverage_reporter_generate_json_report(self):
        """Test generate_json_report method."""
        reporter = CoverageReporter()

        metrics = CoverageMetrics(
            total_lines=100,
            covered_lines=80,
            line_coverage_percent=80.0,
            total_branches=20,
            covered_branches=16,
            branch_coverage_percent=80.0,
            total_functions=10,
            covered_functions=8,
            function_coverage_percent=80.0,
            total_files=5,
            covered_files=4,
            file_coverage_percent=80.0,
            overall_coverage_percent=80.0,
            file_coverage={
                "test.py": {
                    "total_lines": 50,
                    "covered_lines": 40,
                    "line_coverage_percent": 80.0,
                }
            },
        )

        report = reporter.generate_json_report(metrics)

        # Parse JSON to verify it's valid
        data = json.loads(report)

        assert data["overall_coverage_percent"] == 80.0
        assert data["summary"]["lines"]["covered"] == 80
        assert data["summary"]["lines"]["total"] == 100
        assert data["summary"]["lines"]["percent"] == 80.0
        assert data["summary"]["branches"]["covered"] == 16
        assert data["summary"]["branches"]["total"] == 20
        assert data["summary"]["branches"]["percent"] == 80.0
        assert data["summary"]["functions"]["covered"] == 8
        assert data["summary"]["functions"]["total"] == 10
        assert data["summary"]["functions"]["percent"] == 80.0
        assert data["summary"]["files"]["covered"] == 4
        assert data["summary"]["files"]["total"] == 5
        assert data["summary"]["files"]["percent"] == 80.0
        assert "test.py" in data["file_coverage"]

    def test_coverage_reporter_save_report_success(self):
        """Test save_report method with success."""
        reporter = CoverageReporter()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = f.name

        try:
            result = reporter.save_report("Test report content", temp_path, "text")

            assert result is True

            with open(temp_path, "r") as f:
                content = f.read()
                assert content == "Test report content"
        finally:
            os.unlink(temp_path)

    def test_coverage_reporter_save_report_failure(self):
        """Test save_report method with failure."""
        reporter = CoverageReporter()

        # Try to save to a directory (should fail)
        result = reporter.save_report("Test report content", "/tmp", "text")

        assert result is False
