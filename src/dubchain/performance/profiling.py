"""
Automated profiling harness for DubChain performance optimization.

This module provides comprehensive profiling capabilities including:
- CPU time profiling with call stack analysis
- Memory allocation and usage profiling
- Hotspot detection and ranking
- Performance regression detection
- Profiling artifact generation (flamegraphs, callgrind)
"""

import asyncio
import cProfile
import io
import json
import os
import pstats
import sys
import time
import tracemalloc
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import psutil
import linecache
from collections import defaultdict, Counter

try:
    import pyspy
    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


@dataclass
class ProfilingConfig:
    """Configuration for profiling operations."""
    
    # CPU profiling
    enable_cpu_profiling: bool = True
    cpu_sample_rate: float = 0.001  # 1ms sampling
    cpu_stack_depth: int = 100
    
    # Memory profiling
    enable_memory_profiling: bool = True
    memory_trace_limit: int = 25  # MB
    memory_snapshot_interval: float = 1.0  # seconds
    
    # Hotspot detection
    enable_hotspot_detection: bool = True
    hotspot_threshold: float = 0.05  # 5% of total time
    min_calls_for_hotspot: int = 10
    
    # Output configuration
    output_directory: str = "profiling_artifacts"
    generate_flamegraph: bool = True
    generate_callgrind: bool = True
    generate_json_report: bool = True
    
    # Performance budgets
    max_cpu_time_percent: float = 80.0
    max_memory_usage_mb: float = 1000.0
    max_allocation_rate_mb_per_sec: float = 100.0


@dataclass
class FunctionProfile:
    """Profile data for a single function."""
    
    function_name: str
    file_path: str
    line_number: int
    total_time: float
    cumulative_time: float
    call_count: int
    time_per_call: float
    memory_allocations: int = 0
    memory_peak: int = 0
    is_hotspot: bool = False


@dataclass
class ProfilingResult:
    """Results from a profiling session."""
    
    session_id: str
    start_time: float
    end_time: float
    duration: float
    
    # CPU profiling results
    cpu_functions: List[FunctionProfile] = field(default_factory=list)
    cpu_hotspots: List[FunctionProfile] = field(default_factory=list)
    total_cpu_time: float = 0.0
    
    # Memory profiling results
    memory_peak: int = 0
    memory_allocations: int = 0
    memory_fragmentation: float = 0.0
    memory_hotspots: List[FunctionProfile] = field(default_factory=list)
    
    # System metrics
    system_cpu_percent: float = 0.0
    system_memory_percent: float = 0.0
    
    # Artifacts
    flamegraph_path: Optional[str] = None
    callgrind_path: Optional[str] = None
    json_report_path: Optional[str] = None
    
    # Performance budget compliance
    budget_violations: List[str] = field(default_factory=list)


class CPUTimeProfiler:
    """CPU time profiler using cProfile."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.profiler = cProfile.Profile()
        self.stats: Optional[pstats.Stats] = None
        
    def start(self) -> None:
        """Start CPU profiling."""
        self.profiler.enable()
        
    def stop(self) -> ProfilingResult:
        """Stop CPU profiling and return results."""
        self.profiler.disable()
        self.stats = pstats.Stats(self.profiler)
        return self._analyze_stats()
        
    def _analyze_stats(self) -> ProfilingResult:
        """Analyze profiling statistics."""
        if not self.stats:
            raise RuntimeError("Profiling not started")
            
        # Get function statistics
        functions = []
        total_time = 0.0
        
        for func, (cc, nc, tt, ct, callers) in self.stats.stats.items():
            filename, line_number, function_name = func
            
            # Skip built-in functions
            if filename.startswith('<'):
                continue
                
            total_time += tt
            
            function_profile = FunctionProfile(
                function_name=function_name,
                file_path=filename,
                line_number=line_number,
                total_time=tt,
                cumulative_time=ct,
                call_count=cc,
                time_per_call=tt / cc if cc > 0 else 0,
            )
            functions.append(function_profile)
            
        # Sort by total time
        functions.sort(key=lambda x: x.total_time, reverse=True)
        
        # Identify hotspots
        hotspots = []
        for func in functions:
            if (func.total_time / total_time >= self.config.hotspot_threshold and 
                func.call_count >= self.config.min_calls_for_hotspot):
                func.is_hotspot = True
                hotspots.append(func)
                
        return ProfilingResult(
            session_id=f"cpu_{int(time.time())}",
            start_time=0,  # Will be set by caller
            end_time=0,    # Will be set by caller
            duration=total_time,
            cpu_functions=functions,
            cpu_hotspots=hotspots,
            total_cpu_time=total_time,
        )


class MemoryProfiler:
    """Memory allocation profiler."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.tracemalloc_started = False
        self.snapshots: List[tracemalloc.Snapshot] = []
        self.allocation_tracker: Dict[str, int] = defaultdict(int)
        
    def start(self) -> None:
        """Start memory profiling."""
        if not self.tracemalloc_started:
            # Start tracemalloc without limit parameter for compatibility
            tracemalloc.start()
            self.tracemalloc_started = True
            
    def stop(self) -> ProfilingResult:
        """Stop memory profiling and return results."""
        if not self.tracemalloc_started:
            raise RuntimeError("Memory profiling not started")
            
        # Take final snapshot
        final_snapshot = tracemalloc.take_snapshot()
        self.snapshots.append(final_snapshot)
        
        # Analyze memory usage
        return self._analyze_memory_usage()
        
    def take_snapshot(self) -> None:
        """Take a memory snapshot."""
        if self.tracemalloc_started:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append(snapshot)
            
    def _analyze_memory_usage(self) -> ProfilingResult:
        """Analyze memory usage from snapshots."""
        if not self.snapshots:
            return ProfilingResult(
                session_id=f"memory_{int(time.time())}",
                start_time=0,
                end_time=0,
                duration=0,
            )
            
        # Get top memory allocations
        top_stats = self.snapshots[-1].statistics('lineno')
        
        functions = []
        total_allocations = 0
        peak_memory = 0
        
        for stat in top_stats:
            try:
                traceback_line = stat.traceback.format()[-1]
                if ':' in traceback_line:
                    filename, line_number = traceback_line.split(':')[:2]
                    function_name = self._extract_function_name(filename, int(line_number))
                else:
                    filename = traceback_line
                    line_number = 0
                    function_name = "unknown_function"
            except (ValueError, IndexError):
                filename = "unknown_file"
                line_number = 0
                function_name = "unknown_function"
            
            total_allocations += stat.count
            
            function_profile = FunctionProfile(
                function_name=function_name,
                file_path=filename,
                line_number=int(line_number),
                total_time=0,  # Not applicable for memory
                cumulative_time=0,
                call_count=stat.count,
                time_per_call=0,
                memory_allocations=stat.count,
                memory_peak=stat.size,
            )
            functions.append(function_profile)
            
        # Calculate peak memory
        if self.snapshots:
            peak_memory = max(snapshot.traceback_limit for snapshot in self.snapshots)
            
        # Sort by memory usage
        functions.sort(key=lambda x: x.memory_peak, reverse=True)
        
        # Identify memory hotspots
        hotspots = []
        for func in functions:
            if (func.memory_peak >= self.config.hotspot_threshold * peak_memory and
                func.memory_allocations >= self.config.min_calls_for_hotspot):
                func.is_hotspot = True
                hotspots.append(func)
                
        return ProfilingResult(
            session_id=f"memory_{int(time.time())}",
            start_time=0,
            end_time=0,
            duration=0,
            memory_peak=peak_memory,
            memory_allocations=total_allocations,
            memory_hotspots=hotspots,
        )
        
    def _extract_function_name(self, filename: str, line_number: int) -> str:
        """Extract function name from filename and line number."""
        try:
            line = linecache.getline(filename, line_number)
            if 'def ' in line:
                return line.split('def ')[1].split('(')[0].strip()
            elif 'class ' in line:
                return line.split('class ')[1].split('(')[0].split(':')[0].strip()
            else:
                return f"line_{line_number}"
        except:
            return f"line_{line_number}"


class HotspotDetector:
    """Detects performance hotspots in code."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.hotspot_history: List[FunctionProfile] = []
        
    def detect_hotspots(self, profiling_result: ProfilingResult) -> List[FunctionProfile]:
        """Detect hotspots from profiling results."""
        hotspots = []
        
        # CPU hotspots
        for func in profiling_result.cpu_hotspots:
            if self._is_significant_hotspot(func, profiling_result.total_cpu_time):
                hotspots.append(func)
                
        # Memory hotspots
        for func in profiling_result.memory_hotspots:
            if self._is_significant_memory_hotspot(func, profiling_result.memory_peak):
                hotspots.append(func)
                
        # Store in history
        self.hotspot_history.extend(hotspots)
        
        return hotspots
        
    def _is_significant_hotspot(self, func: FunctionProfile, total_time: float) -> bool:
        """Check if function is a significant CPU hotspot."""
        return (func.total_time / total_time >= self.config.hotspot_threshold and
                func.call_count >= self.config.min_calls_for_hotspot)
                
    def _is_significant_memory_hotspot(self, func: FunctionProfile, peak_memory: int) -> bool:
        """Check if function is a significant memory hotspot."""
        return (func.memory_peak >= self.config.hotspot_threshold * peak_memory and
                func.memory_allocations >= self.config.min_calls_for_hotspot)
                
    def get_hotspot_trends(self) -> Dict[str, List[float]]:
        """Get trends for hotspot functions over time."""
        trends = defaultdict(list)
        
        for func in self.hotspot_history:
            trends[func.function_name].append(func.total_time)
            
        return dict(trends)


class PerformanceProfiler:
    """Main performance profiler that coordinates CPU and memory profiling."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.cpu_profiler = CPUTimeProfiler(config)
        self.memory_profiler = MemoryProfiler(config)
        self.hotspot_detector = HotspotDetector(config)
        self.results: List[ProfilingResult] = []
        
    def profile_function(self, func: Callable, *args, **kwargs) -> ProfilingResult:
        """Profile a single function execution."""
        start_time = time.time()
        
        # Start profiling
        if self.config.enable_cpu_profiling:
            self.cpu_profiler.start()
        if self.config.enable_memory_profiling:
            self.memory_profiler.start()
            
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Stop profiling
            cpu_result = None
            memory_result = None
            
            if self.config.enable_cpu_profiling:
                cpu_result = self.cpu_profiler.stop()
            if self.config.enable_memory_profiling:
                memory_result = self.memory_profiler.stop()
                
            # Combine results
            combined_result = self._combine_results(cpu_result, memory_result, start_time, time.time())
            
            # Detect hotspots
            hotspots = self.hotspot_detector.detect_hotspots(combined_result)
            combined_result.cpu_hotspots = [h for h in hotspots if h.total_time > 0]
            combined_result.memory_hotspots = [h for h in hotspots if h.memory_peak > 0]
            
            # Check performance budgets
            self._check_performance_budgets(combined_result)
            
            # Store results
            self.results.append(combined_result)
            
            return combined_result
            
        except Exception as e:
            # Stop profiling even on exception
            if self.config.enable_cpu_profiling:
                self.cpu_profiler.stop()
            if self.config.enable_memory_profiling:
                self.memory_profiler.stop()
            raise
            
    @contextmanager
    def profile_context(self, session_name: str):
        """Context manager for profiling code blocks."""
        start_time = time.time()
        
        # Start profiling
        if self.config.enable_cpu_profiling:
            self.cpu_profiler.start()
        if self.config.enable_memory_profiling:
            self.memory_profiler.start()
            
        try:
            yield
        finally:
            # Stop profiling
            cpu_result = None
            memory_result = None
            
            if self.config.enable_cpu_profiling:
                cpu_result = self.cpu_profiler.stop()
            if self.config.enable_memory_profiling:
                memory_result = self.memory_profiler.stop()
                
            # Combine results
            combined_result = self._combine_results(cpu_result, memory_result, start_time, time.time())
            combined_result.session_id = session_name
            
            # Detect hotspots
            hotspots = self.hotspot_detector.detect_hotspots(combined_result)
            combined_result.cpu_hotspots = [h for h in hotspots if h.total_time > 0]
            combined_result.memory_hotspots = [h for h in hotspots if h.memory_peak > 0]
            
            # Check performance budgets
            self._check_performance_budgets(combined_result)
            
            # Store results
            self.results.append(combined_result)
            
    def _combine_results(self, cpu_result: Optional[ProfilingResult], 
                        memory_result: Optional[ProfilingResult],
                        start_time: float, end_time: float) -> ProfilingResult:
        """Combine CPU and memory profiling results."""
        result = ProfilingResult(
            session_id=f"combined_{int(time.time())}",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
        )
        
        if cpu_result:
            result.cpu_functions = cpu_result.cpu_functions
            result.cpu_hotspots = cpu_result.cpu_hotspots
            result.total_cpu_time = cpu_result.total_cpu_time
            
        if memory_result:
            result.memory_peak = memory_result.memory_peak
            result.memory_allocations = memory_result.memory_allocations
            result.memory_hotspots = memory_result.memory_hotspots
            
        # Get system metrics
        process = psutil.Process()
        result.system_cpu_percent = process.cpu_percent()
        result.system_memory_percent = process.memory_percent()
        
        return result
        
    def _check_performance_budgets(self, result: ProfilingResult) -> None:
        """Check if results violate performance budgets."""
        violations = []
        
        # CPU budget check
        if result.system_cpu_percent > self.config.max_cpu_time_percent:
            violations.append(f"CPU usage {result.system_cpu_percent:.1f}% exceeds budget {self.config.max_cpu_time_percent}%")
            
        # Memory budget check
        memory_mb = result.memory_peak / (1024 * 1024)
        if memory_mb > self.config.max_memory_usage_mb:
            violations.append(f"Memory usage {memory_mb:.1f}MB exceeds budget {self.config.max_memory_usage_mb}MB")
            
        result.budget_violations = violations
        
    def generate_artifacts(self, result: ProfilingResult) -> None:
        """Generate profiling artifacts (flamegraph, callgrind, JSON)."""
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        if self.config.generate_json_report:
            self._generate_json_report(result)
            
        if self.config.generate_flamegraph and PY_SPY_AVAILABLE:
            self._generate_flamegraph(result)
            
        if self.config.generate_callgrind:
            self._generate_callgrind(result)
            
    def _generate_json_report(self, result: ProfilingResult) -> None:
        """Generate JSON report."""
        report_data = {
            "session_id": result.session_id,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "duration": result.duration,
            "cpu_functions": [
                {
                    "function_name": func.function_name,
                    "file_path": func.file_path,
                    "line_number": func.line_number,
                    "total_time": func.total_time,
                    "cumulative_time": func.cumulative_time,
                    "call_count": func.call_count,
                    "time_per_call": func.time_per_call,
                    "is_hotspot": func.is_hotspot,
                }
                for func in result.cpu_functions
            ],
            "memory_hotspots": [
                {
                    "function_name": func.function_name,
                    "file_path": func.file_path,
                    "line_number": func.line_number,
                    "memory_allocations": func.memory_allocations,
                    "memory_peak": func.memory_peak,
                    "is_hotspot": func.is_hotspot,
                }
                for func in result.memory_hotspots
            ],
            "system_metrics": {
                "cpu_percent": result.system_cpu_percent,
                "memory_percent": result.system_memory_percent,
                "memory_peak": result.memory_peak,
            },
            "budget_violations": result.budget_violations,
        }
        
        json_path = os.path.join(self.config.output_directory, f"{result.session_id}.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        result.json_report_path = json_path
        
    def _generate_flamegraph(self, result: ProfilingResult) -> None:
        """Generate flamegraph using py-spy."""
        if not PY_SPY_AVAILABLE:
            return
            
        flamegraph_path = os.path.join(self.config.output_directory, f"{result.session_id}_flamegraph.svg")
        
        try:
            # This would require py-spy integration
            # For now, we'll create a placeholder
            with open(flamegraph_path, 'w') as f:
                f.write("<!-- Flamegraph placeholder - requires py-spy integration -->")
            result.flamegraph_path = flamegraph_path
        except Exception as e:
            print(f"Failed to generate flamegraph: {e}")
            
    def _generate_callgrind(self, result: ProfilingResult) -> None:
        """Generate callgrind format file."""
        if not result.cpu_functions:
            return
            
        callgrind_path = os.path.join(self.config.output_directory, f"{result.session_id}.callgrind")
        
        with open(callgrind_path, 'w') as f:
            f.write("events: Ticks\n")
            f.write("positions: line\n")
            f.write("summary: Ticks\n")
            
            for func in result.cpu_functions:
                f.write(f"fl={func.file_path}\n")
                f.write(f"fn={func.function_name}\n")
                f.write(f"{func.line_number} {int(func.total_time * 1000000)}\n")
                
        result.callgrind_path = callgrind_path


class ProfilingHarness:
    """Main profiling harness that orchestrates all profiling operations."""
    
    def __init__(self, config: Optional[ProfilingConfig] = None):
        self.config = config or ProfilingConfig()
        self.profiler = PerformanceProfiler(self.config)
        self.baseline_results: Dict[str, ProfilingResult] = {}
        
    def run_baseline_profiling(self, workloads: Dict[str, Callable]) -> Dict[str, ProfilingResult]:
        """Run baseline profiling on standard workloads."""
        print("Running baseline profiling...")
        
        for name, workload in workloads.items():
            print(f"Profiling workload: {name}")
            
            try:
                result = self.profiler.profile_function(workload)
                self.baseline_results[name] = result
                
                # Generate artifacts
                self.profiler.generate_artifacts(result)
                
                print(f"  Duration: {result.duration:.3f}s")
                print(f"  CPU hotspots: {len(result.cpu_hotspots)}")
                print(f"  Memory hotspots: {len(result.memory_hotspots)}")
                print(f"  Budget violations: {len(result.budget_violations)}")
                
            except Exception as e:
                print(f"  Error profiling {name}: {e}")
                
        return self.baseline_results
        
    def compare_with_baseline(self, result: ProfilingResult, baseline_name: str) -> Dict[str, Any]:
        """Compare profiling result with baseline."""
        if baseline_name not in self.baseline_results:
            raise ValueError(f"Baseline {baseline_name} not found")
            
        baseline = self.baseline_results[baseline_name]
        
        comparison = {
            "duration_change": result.duration - baseline.duration,
            "duration_change_percent": ((result.duration - baseline.duration) / baseline.duration) * 100,
            "cpu_time_change": result.total_cpu_time - baseline.total_cpu_time,
            "memory_peak_change": result.memory_peak - baseline.memory_peak,
            "new_hotspots": [],
            "resolved_hotspots": [],
            "regressions": [],
        }
        
        # Compare hotspots
        baseline_hotspots = {f.function_name for f in baseline.cpu_hotspots + baseline.memory_hotspots}
        current_hotspots = {f.function_name for f in result.cpu_hotspots + result.memory_hotspots}
        
        comparison["new_hotspots"] = list(current_hotspots - baseline_hotspots)
        comparison["resolved_hotspots"] = list(baseline_hotspots - current_hotspots)
        
        # Check for regressions
        if comparison["duration_change_percent"] > 10:  # 10% regression threshold
            comparison["regressions"].append(f"Duration increased by {comparison['duration_change_percent']:.1f}%")
            
        if result.memory_peak > baseline.memory_peak * 1.1:  # 10% memory increase
            comparison["regressions"].append(f"Memory usage increased by {((result.memory_peak - baseline.memory_peak) / baseline.memory_peak) * 100:.1f}%")
            
        return comparison
        
    def generate_hotspot_report(self) -> str:
        """Generate a comprehensive hotspot report."""
        report = ["# DubChain Performance Hotspot Report\n"]
        
        # Overall statistics
        total_sessions = len(self.profiler.results)
        total_hotspots = sum(len(r.cpu_hotspots) + len(r.memory_hotspots) for r in self.profiler.results)
        
        report.append(f"## Summary")
        report.append(f"- Total profiling sessions: {total_sessions}")
        report.append(f"- Total hotspots identified: {total_hotspots}")
        report.append(f"- Baseline workloads: {len(self.baseline_results)}\n")
        
        # Top hotspots across all sessions
        all_hotspots = []
        for result in self.profiler.results:
            all_hotspots.extend(result.cpu_hotspots)
            all_hotspots.extend(result.memory_hotspots)
            
        # Group by function name
        hotspot_groups = defaultdict(list)
        for hotspot in all_hotspots:
            hotspot_groups[hotspot.function_name].append(hotspot)
            
        # Sort by average impact
        sorted_hotspots = sorted(
            hotspot_groups.items(),
            key=lambda x: sum(h.total_time + h.memory_peak for h in x[1]) / len(x[1]),
            reverse=True
        )
        
        report.append("## Top Performance Hotspots\n")
        for i, (function_name, hotspots) in enumerate(sorted_hotspots[:10], 1):
            avg_time = sum(h.total_time for h in hotspots) / len(hotspots)
            avg_memory = sum(h.memory_peak for h in hotspots) / len(hotspots)
            occurrences = len(hotspots)
            
            report.append(f"{i}. **{function_name}**")
            report.append(f"   - Occurrences: {occurrences}")
            report.append(f"   - Average CPU time: {avg_time:.3f}s")
            report.append(f"   - Average memory: {avg_memory / 1024 / 1024:.1f}MB")
            report.append(f"   - File: {hotspots[0].file_path}:{hotspots[0].line_number}")
            report.append("")
            
        return "\n".join(report)
