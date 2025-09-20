#!/usr/bin/env python3
"""
Generate comprehensive optimization plan for DubChain.

This script analyzes profiling results and generates a prioritized optimization plan
with impact/risk matrix and implementation roadmap.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dubchain.performance.optimizations import OptimizationType, OptimizationConfig


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with impact and risk analysis."""
    
    name: str
    optimization_type: OptimizationType
    description: str
    estimated_impact_percent: float
    risk_level: str  # low, medium, high
    implementation_effort: str  # low, medium, high
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    expected_improvement: str = ""
    implementation_notes: str = ""
    testing_requirements: List[str] = field(default_factory=list)


@dataclass
class OptimizationPlan:
    """Comprehensive optimization plan."""
    
    plan_id: str
    generated_at: float
    baseline_analysis: Dict[str, Any]
    recommendations: List[OptimizationRecommendation]
    implementation_phases: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    success_metrics: Dict[str, Any]
    timeline: Dict[str, Any]


class OptimizationPlanGenerator:
    """Generates comprehensive optimization plans."""
    
    def __init__(self):
        self.optimization_templates = self._load_optimization_templates()
        
    def _load_optimization_templates(self) -> Dict[str, OptimizationRecommendation]:
        """Load optimization templates with predefined recommendations."""
        return {
            "consensus_batching": OptimizationRecommendation(
                name="consensus_batching",
                optimization_type=OptimizationType.CONSENSUS_BATCHING,
                description="Batch block validation and consensus operations for improved throughput",
                estimated_impact_percent=25.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="20-30% improvement in consensus throughput",
                implementation_notes="Requires careful validation of batch operations to maintain consensus safety",
                testing_requirements=["consensus_correctness_tests", "batch_validation_tests", "performance_benchmarks"]
            ),
            
            "consensus_lock_reduction": OptimizationRecommendation(
                name="consensus_lock_reduction",
                optimization_type=OptimizationType.CONSENSUS_LOCK_REDUCTION,
                description="Reduce lock contention in consensus mechanisms using lock striping and read-write locks",
                estimated_impact_percent=15.0,
                risk_level="medium",
                implementation_effort="high",
                expected_improvement="15-20% reduction in consensus latency",
                implementation_notes="Requires careful analysis of lock ordering to prevent deadlocks",
                testing_requirements=["concurrency_tests", "deadlock_detection", "performance_under_load"]
            ),
            
            "consensus_o1_structures": OptimizationRecommendation(
                name="consensus_o1_structures",
                optimization_type=OptimizationType.CONSENSUS_O1_STRUCTURES,
                description="Implement O(1) data structures for validator selection and vote aggregation",
                estimated_impact_percent=20.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="O(n) to O(1) improvement in validator operations",
                implementation_notes="Use consistent hashing and optimized data structures",
                testing_requirements=["data_structure_tests", "validator_selection_tests", "scalability_tests"]
            ),
            
            "network_async_io": OptimizationRecommendation(
                name="network_async_io",
                optimization_type=OptimizationType.NETWORK_ASYNC_IO,
                description="Implement async I/O for non-blocking network operations",
                estimated_impact_percent=40.0,
                risk_level="medium",
                implementation_effort="high",
                expected_improvement="40-50% improvement in network throughput",
                implementation_notes="Requires careful handling of async operations and error handling",
                testing_requirements=["async_io_tests", "network_stress_tests", "error_handling_tests"]
            ),
            
            "network_batching": OptimizationRecommendation(
                name="network_batching",
                optimization_type=OptimizationType.NETWORK_BATCHING,
                description="Batch network messages for improved efficiency",
                estimated_impact_percent=30.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="30-40% reduction in network overhead",
                implementation_notes="Implement message coalescing and batch processing",
                testing_requirements=["message_batching_tests", "latency_tests", "throughput_tests"]
            ),
            
            "network_zero_copy": OptimizationRecommendation(
                name="network_zero_copy",
                optimization_type=OptimizationType.NETWORK_ZERO_COPY,
                description="Implement zero-copy serialization for network messages",
                estimated_impact_percent=20.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="20-25% reduction in memory allocations",
                implementation_notes="Use binary protocols and buffer pools",
                testing_requirements=["serialization_tests", "memory_profiling", "performance_benchmarks"]
            ),
            
            "vm_jit_caching": OptimizationRecommendation(
                name="vm_jit_caching",
                optimization_type=OptimizationType.VM_JIT_CACHING,
                description="Implement JIT compilation and bytecode caching for frequently executed contracts",
                estimated_impact_percent=50.0,
                risk_level="high",
                implementation_effort="high",
                expected_improvement="50-70% improvement in contract execution speed",
                implementation_notes="Requires careful cache invalidation and security considerations",
                testing_requirements=["vm_correctness_tests", "cache_invalidation_tests", "security_tests"]
            ),
            
            "vm_gas_optimization": OptimizationRecommendation(
                name="vm_gas_optimization",
                optimization_type=OptimizationType.VM_GAS_OPTIMIZATION,
                description="Optimize gas metering and instruction execution",
                estimated_impact_percent=15.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="15-20% improvement in gas efficiency",
                implementation_notes="Optimize gas calculation algorithms and instruction costs",
                testing_requirements=["gas_metering_tests", "instruction_tests", "gas_accuracy_tests"]
            ),
            
            "vm_state_caching": OptimizationRecommendation(
                name="vm_state_caching",
                optimization_type=OptimizationType.VM_STATE_CACHING,
                description="Implement intelligent state caching with LRU and version checking",
                estimated_impact_percent=25.0,
                risk_level="medium",
                implementation_effort="medium",
                expected_improvement="25-35% reduction in state access latency",
                implementation_notes="Requires careful cache invalidation and consistency checks",
                testing_requirements=["state_consistency_tests", "cache_performance_tests", "invalidation_tests"]
            ),
            
            "storage_binary_formats": OptimizationRecommendation(
                name="storage_binary_formats",
                optimization_type=OptimizationType.STORAGE_BINARY_FORMATS,
                description="Replace JSON with binary formats (msgpack, protobuf) for storage",
                estimated_impact_percent=35.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="35-45% improvement in serialization/deserialization speed",
                implementation_notes="Implement fallback to JSON for compatibility",
                testing_requirements=["serialization_tests", "compatibility_tests", "performance_benchmarks"]
            ),
            
            "storage_write_batching": OptimizationRecommendation(
                name="storage_write_batching",
                optimization_type=OptimizationType.STORAGE_WRITE_BATCHING,
                description="Batch storage write operations for improved I/O efficiency",
                estimated_impact_percent=30.0,
                risk_level="medium",
                implementation_effort="medium",
                expected_improvement="30-40% improvement in storage throughput",
                implementation_notes="Requires careful handling of batch failures and consistency",
                testing_requirements=["batch_write_tests", "consistency_tests", "failure_recovery_tests"]
            ),
            
            "storage_multi_tier_cache": OptimizationRecommendation(
                name="storage_multi_tier_cache",
                optimization_type=OptimizationType.STORAGE_MULTI_TIER_CACHE,
                description="Implement multi-tier caching (memory, disk, remote)",
                estimated_impact_percent=40.0,
                risk_level="medium",
                implementation_effort="high",
                expected_improvement="40-60% improvement in data access speed",
                implementation_notes="Requires sophisticated cache management and eviction policies",
                testing_requirements=["cache_hierarchy_tests", "eviction_tests", "performance_under_load"]
            ),
            
            "crypto_parallel_verification": OptimizationRecommendation(
                name="crypto_parallel_verification",
                optimization_type=OptimizationType.CRYPTO_PARALLEL_VERIFICATION,
                description="Parallelize signature verification operations",
                estimated_impact_percent=60.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="60-80% improvement in signature verification throughput",
                implementation_notes="Use thread pools and vectorized operations where available",
                testing_requirements=["parallel_verification_tests", "correctness_tests", "scalability_tests"]
            ),
            
            "crypto_hardware_acceleration": OptimizationRecommendation(
                name="crypto_hardware_acceleration",
                optimization_type=OptimizationType.CRYPTO_HARDWARE_ACCELERATION,
                description="Use hardware acceleration for cryptographic operations",
                estimated_impact_percent=80.0,
                risk_level="medium",
                implementation_effort="high",
                expected_improvement="80-90% improvement in crypto operations with hardware support",
                implementation_notes="Requires hardware detection and fallback mechanisms",
                testing_requirements=["hardware_detection_tests", "fallback_tests", "performance_benchmarks"]
            ),
            
            "memory_allocation_reduction": OptimizationRecommendation(
                name="memory_allocation_reduction",
                optimization_type=OptimizationType.MEMORY_ALLOCATION_REDUCTION,
                description="Reduce temporary allocations and object churn on hot paths",
                estimated_impact_percent=20.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="20-30% reduction in memory allocations",
                implementation_notes="Use object pools and buffer reuse",
                testing_requirements=["memory_profiling", "allocation_tests", "gc_pressure_tests"]
            ),
            
            "memory_gc_tuning": OptimizationRecommendation(
                name="memory_gc_tuning",
                optimization_type=OptimizationType.MEMORY_GC_TUNING,
                description="Optimize garbage collection settings and timing",
                estimated_impact_percent=15.0,
                risk_level="medium",
                implementation_effort="low",
                expected_improvement="15-25% reduction in GC overhead",
                implementation_notes="Requires careful tuning and monitoring",
                testing_requirements=["gc_profiling", "memory_pressure_tests", "latency_tests"]
            ),
            
            "batching_state_writes": OptimizationRecommendation(
                name="batching_state_writes",
                optimization_type=OptimizationType.BATCHING_STATE_WRITES,
                description="Batch state write operations for improved efficiency",
                estimated_impact_percent=25.0,
                risk_level="medium",
                implementation_effort="medium",
                expected_improvement="25-35% improvement in state write throughput",
                implementation_notes="Requires careful handling of batch consistency",
                testing_requirements=["state_consistency_tests", "batch_write_tests", "failure_recovery_tests"]
            ),
            
            "batching_tx_validation": OptimizationRecommendation(
                name="batching_tx_validation",
                optimization_type=OptimizationType.BATCHING_TX_VALIDATION,
                description="Batch transaction validation operations",
                estimated_impact_percent=30.0,
                risk_level="low",
                implementation_effort="medium",
                expected_improvement="30-40% improvement in transaction validation throughput",
                implementation_notes="Group transactions by common validation requirements",
                testing_requirements=["validation_correctness_tests", "batch_validation_tests", "performance_benchmarks"]
            ),
        }
        
    def generate_optimization_plan(self, profiling_results: Dict[str, Any]) -> OptimizationPlan:
        """Generate comprehensive optimization plan from profiling results."""
        
        # Analyze profiling results
        baseline_analysis = self._analyze_baseline_results(profiling_results)
        
        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations(baseline_analysis)
        
        # Create implementation phases
        implementation_phases = self._create_implementation_phases(recommendations)
        
        # Assess risks
        risk_assessment = self._assess_risks(recommendations)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(recommendations)
        
        # Create timeline
        timeline = self._create_timeline(implementation_phases)
        
        return OptimizationPlan(
            plan_id=f"optimization_plan_{int(time.time())}",
            generated_at=time.time(),
            baseline_analysis=baseline_analysis,
            recommendations=recommendations,
            implementation_phases=implementation_phases,
            risk_assessment=risk_assessment,
            success_metrics=success_metrics,
            timeline=timeline
        )
        
    def _analyze_baseline_results(self, profiling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze baseline profiling results to identify optimization opportunities."""
        analysis = {
            "total_profiling_sessions": len(profiling_results.get("profiling_results", {})),
            "total_benchmarks": profiling_results.get("benchmark_results", {}).get("total_benchmarks", 0),
            "hotspots_identified": 0,
            "performance_bottlenecks": [],
            "optimization_opportunities": [],
            "subsystem_analysis": {}
        }
        
        # Analyze profiling results
        for subsystem, results in profiling_results.get("profiling_results", {}).items():
            subsystem_analysis = {
                "cpu_hotspots": len(results.get("cpu_hotspots", [])),
                "memory_hotspots": len(results.get("memory_hotspots", [])),
                "total_cpu_time": results.get("total_cpu_time", 0),
                "memory_peak": results.get("memory_peak", 0),
                "performance_issues": []
            }
            
            # Identify performance issues
            if results.get("total_cpu_time", 0) > 1.0:  # More than 1 second
                subsystem_analysis["performance_issues"].append("high_cpu_usage")
                
            if results.get("memory_peak", 0) > 100 * 1024 * 1024:  # More than 100MB
                subsystem_analysis["performance_issues"].append("high_memory_usage")
                
            analysis["subsystem_analysis"][subsystem] = subsystem_analysis
            analysis["hotspots_identified"] += subsystem_analysis["cpu_hotspots"] + subsystem_analysis["memory_hotspots"]
            
        # Analyze benchmark results
        benchmarks = profiling_results.get("benchmark_results", {}).get("benchmarks", [])
        for benchmark in benchmarks:
            if benchmark.get("mean_time", 0) > 0.1:  # More than 100ms
                analysis["performance_bottlenecks"].append({
                    "name": benchmark["name"],
                    "mean_time": benchmark["mean_time"],
                    "throughput": benchmark.get("throughput", 0)
                })
                
        return analysis
        
    def _generate_recommendations(self, baseline_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on baseline analysis."""
        recommendations = []
        
        # Analyze each subsystem
        for subsystem, analysis in baseline_analysis.get("subsystem_analysis", {}).items():
            if subsystem == "core_blockchain":
                if "high_cpu_usage" in analysis["performance_issues"]:
                    recommendations.append(self.optimization_templates["consensus_batching"])
                    recommendations.append(self.optimization_templates["consensus_o1_structures"])
                    
                if "high_memory_usage" in analysis["performance_issues"]:
                    recommendations.append(self.optimization_templates["memory_allocation_reduction"])
                    
            elif subsystem == "consensus":
                if analysis["cpu_hotspots"] > 5:
                    recommendations.append(self.optimization_templates["consensus_batching"])
                    recommendations.append(self.optimization_templates["consensus_lock_reduction"])
                    
            elif subsystem == "virtual_machine":
                if analysis["cpu_hotspots"] > 3:
                    recommendations.append(self.optimization_templates["vm_jit_caching"])
                    recommendations.append(self.optimization_templates["vm_gas_optimization"])
                    
            elif subsystem == "network":
                if analysis["cpu_hotspots"] > 2:
                    recommendations.append(self.optimization_templates["network_async_io"])
                    recommendations.append(self.optimization_templates["network_batching"])
                    
            elif subsystem == "storage":
                if analysis["cpu_hotspots"] > 2:
                    recommendations.append(self.optimization_templates["storage_binary_formats"])
                    recommendations.append(self.optimization_templates["storage_write_batching"])
                    
            elif subsystem == "crypto":
                if analysis["cpu_hotspots"] > 1:
                    recommendations.append(self.optimization_templates["crypto_parallel_verification"])
                    recommendations.append(self.optimization_templates["crypto_hardware_acceleration"])
                    
        # Add general optimizations based on bottlenecks
        if baseline_analysis["performance_bottlenecks"]:
            recommendations.append(self.optimization_templates["memory_gc_tuning"])
            recommendations.append(self.optimization_templates["batching_tx_validation"])
            
        # Remove duplicates and sort by impact
        unique_recommendations = {}
        for rec in recommendations:
            if rec.name not in unique_recommendations:
                unique_recommendations[rec.name] = rec
                
        return sorted(unique_recommendations.values(), key=lambda x: x.estimated_impact_percent, reverse=True)
        
    def _create_implementation_phases(self, recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Create implementation phases based on risk and dependencies."""
        phases = []
        
        # Phase 1: Low risk, high impact optimizations
        phase1 = {
            "name": "Quick Wins",
            "duration_weeks": 2,
            "optimizations": [],
            "description": "Low-risk optimizations with immediate impact"
        }
        
        # Phase 2: Medium risk, medium impact optimizations
        phase2 = {
            "name": "Core Optimizations",
            "duration_weeks": 4,
            "optimizations": [],
            "description": "Medium-risk optimizations requiring careful implementation"
        }
        
        # Phase 3: High risk, high impact optimizations
        phase3 = {
            "name": "Advanced Optimizations",
            "duration_weeks": 6,
            "optimizations": [],
            "description": "High-risk optimizations with significant potential impact"
        }
        
        # Categorize optimizations
        for rec in recommendations:
            if rec.risk_level == "low" and rec.estimated_impact_percent >= 20:
                phase1["optimizations"].append(rec.name)
            elif rec.risk_level == "medium" or (rec.risk_level == "low" and rec.estimated_impact_percent < 20):
                phase2["optimizations"].append(rec.name)
            elif rec.risk_level == "high":
                phase3["optimizations"].append(rec.name)
                
        # Add phases that have optimizations
        for phase in [phase1, phase2, phase3]:
            if phase["optimizations"]:
                phases.append(phase)
                
        return phases
        
    def _assess_risks(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Assess overall risks of the optimization plan."""
        risk_levels = [rec.risk_level for rec in recommendations]
        
        risk_counts = {
            "low": risk_levels.count("low"),
            "medium": risk_levels.count("medium"),
            "high": risk_levels.count("high")
        }
        
        # Calculate overall risk level
        if risk_counts["high"] > 2:
            overall_risk = "high"
        elif risk_counts["high"] > 0 or risk_counts["medium"] > 3:
            overall_risk = "medium"
        else:
            overall_risk = "low"
            
        return {
            "overall_risk_level": overall_risk,
            "risk_breakdown": risk_counts,
            "high_risk_optimizations": [rec.name for rec in recommendations if rec.risk_level == "high"],
            "mitigation_strategies": [
                "Implement feature gates for all optimizations",
                "Use comprehensive testing and validation",
                "Implement fallback mechanisms",
                "Monitor performance continuously",
                "Use gradual rollout strategies"
            ]
        }
        
    def _define_success_metrics(self, recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Define success metrics for the optimization plan."""
        total_expected_improvement = sum(rec.estimated_impact_percent for rec in recommendations)
        
        return {
            "overall_expected_improvement": total_expected_improvement,
            "key_performance_indicators": {
                "block_creation_latency": "Target: < 100ms (median)",
                "transaction_throughput": "Target: > 1000 TPS",
                "memory_usage": "Target: < 1GB per node",
                "cpu_usage": "Target: < 80% under normal load",
                "network_latency": "Target: < 500ms (p95)"
            },
            "optimization_specific_metrics": {
                rec.name: {
                    "expected_improvement": rec.estimated_impact_percent,
                    "success_criteria": rec.expected_improvement
                }
                for rec in recommendations
            },
            "measurement_methods": [
                "Automated performance benchmarks",
                "Continuous profiling in CI/CD",
                "Production performance monitoring",
                "Load testing and stress testing",
                "Memory and CPU profiling"
            ]
        }
        
    def _create_timeline(self, implementation_phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create implementation timeline."""
        total_duration = sum(phase["duration_weeks"] for phase in implementation_phases)
        
        return {
            "total_duration_weeks": total_duration,
            "phases": implementation_phases,
            "milestones": [
                {
                    "week": 1,
                    "milestone": "Phase 1: Quick Wins - Start",
                    "deliverables": ["Low-risk optimizations implementation"]
                },
                {
                    "week": 3,
                    "milestone": "Phase 1: Quick Wins - Complete",
                    "deliverables": ["Performance testing and validation"]
                },
                {
                    "week": 4,
                    "milestone": "Phase 2: Core Optimizations - Start",
                    "deliverables": ["Medium-risk optimizations implementation"]
                },
                {
                    "week": 8,
                    "milestone": "Phase 2: Core Optimizations - Complete",
                    "deliverables": ["Comprehensive testing and integration"]
                },
                {
                    "week": 9,
                    "milestone": "Phase 3: Advanced Optimizations - Start",
                    "deliverables": ["High-risk optimizations implementation"]
                },
                {
                    "week": 15,
                    "milestone": "Phase 3: Advanced Optimizations - Complete",
                    "deliverables": ["Final testing and production deployment"]
                }
            ],
            "critical_path": [
                "consensus_batching",
                "network_async_io",
                "storage_binary_formats",
                "crypto_parallel_verification"
            ]
        }
        
    def save_optimization_plan(self, plan: OptimizationPlan, output_file: str) -> None:
        """Save optimization plan to file."""
        plan_data = {
            "plan_id": plan.plan_id,
            "generated_at": plan.generated_at,
            "baseline_analysis": plan.baseline_analysis,
            "recommendations": [
                {
                    "name": rec.name,
                    "optimization_type": rec.optimization_type.value,
                    "description": rec.description,
                    "estimated_impact_percent": rec.estimated_impact_percent,
                    "risk_level": rec.risk_level,
                    "implementation_effort": rec.implementation_effort,
                    "dependencies": rec.dependencies,
                    "conflicts": rec.conflicts,
                    "prerequisites": rec.prerequisites,
                    "expected_improvement": rec.expected_improvement,
                    "implementation_notes": rec.implementation_notes,
                    "testing_requirements": rec.testing_requirements
                }
                for rec in plan.recommendations
            ],
            "implementation_phases": plan.implementation_phases,
            "risk_assessment": plan.risk_assessment,
            "success_metrics": plan.success_metrics,
            "timeline": plan.timeline
        }
        
        with open(output_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
            
    def generate_markdown_report(self, plan: OptimizationPlan) -> str:
        """Generate markdown report for optimization plan."""
        report_lines = [
            "# DubChain Performance Optimization Plan",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(plan.generated_at))}",
            f"Plan ID: {plan.plan_id}",
            "",
            "## Executive Summary",
            "",
            f"This optimization plan includes {len(plan.recommendations)} optimizations across all DubChain subsystems.",
            f"Expected overall improvement: {plan.success_metrics['overall_expected_improvement']:.1f}%",
            f"Implementation timeline: {plan.timeline['total_duration_weeks']} weeks",
            f"Overall risk level: {plan.risk_assessment['overall_risk_level']}",
            "",
            "## Baseline Analysis",
            "",
            f"- Total profiling sessions: {plan.baseline_analysis['total_profiling_sessions']}",
            f"- Total benchmarks: {plan.baseline_analysis['total_benchmarks']}",
            f"- Hotspots identified: {plan.baseline_analysis['hotspots_identified']}",
            f"- Performance bottlenecks: {len(plan.baseline_analysis['performance_bottlenecks'])}",
            "",
            "### Subsystem Analysis",
            "",
        ]
        
        for subsystem, analysis in plan.baseline_analysis.get("subsystem_analysis", {}).items():
            report_lines.extend([
                f"#### {subsystem.replace('_', ' ').title()}",
                f"- CPU hotspots: {analysis['cpu_hotspots']}",
                f"- Memory hotspots: {analysis['memory_hotspots']}",
                f"- Total CPU time: {analysis['total_cpu_time']:.3f}s",
                f"- Memory peak: {analysis['memory_peak'] / 1024 / 1024:.1f}MB",
                f"- Performance issues: {', '.join(analysis['performance_issues']) if analysis['performance_issues'] else 'None'}",
                "",
            ])
            
        report_lines.extend([
            "## Optimization Recommendations",
            "",
        ])
        
        # Group recommendations by impact
        high_impact = [r for r in plan.recommendations if r.estimated_impact_percent >= 30]
        medium_impact = [r for r in plan.recommendations if 20 <= r.estimated_impact_percent < 30]
        low_impact = [r for r in plan.recommendations if r.estimated_impact_percent < 20]
        
        for impact_level, recommendations in [("High Impact", high_impact), ("Medium Impact", medium_impact), ("Low Impact", low_impact)]:
            if recommendations:
                report_lines.extend([
                    f"### {impact_level}",
                    "",
                ])
                
                for rec in recommendations:
                    report_lines.extend([
                        f"#### {rec.name.replace('_', ' ').title()}",
                        f"- **Description**: {rec.description}",
                        f"- **Expected Improvement**: {rec.expected_improvement}",
                        f"- **Risk Level**: {rec.risk_level}",
                        f"- **Implementation Effort**: {rec.implementation_effort}",
                        f"- **Implementation Notes**: {rec.implementation_notes}",
                        "",
                    ])
                    
        report_lines.extend([
            "## Implementation Plan",
            "",
            "### Phases",
            "",
        ])
        
        for i, phase in enumerate(plan.implementation_phases, 1):
            report_lines.extend([
                f"#### Phase {i}: {phase['name']}",
                f"- **Duration**: {phase['duration_weeks']} weeks",
                f"- **Description**: {phase['description']}",
                f"- **Optimizations**: {', '.join(phase['optimizations'])}",
                "",
            ])
            
        report_lines.extend([
            "## Risk Assessment",
            "",
            f"**Overall Risk Level**: {plan.risk_assessment['overall_risk_level']}",
            "",
            "### Risk Breakdown",
            "",
        ])
        
        for risk_level, count in plan.risk_assessment['risk_breakdown'].items():
            report_lines.append(f"- {risk_level.title()}: {count} optimizations")
            
        if plan.risk_assessment['high_risk_optimizations']:
            report_lines.extend([
                "",
                "### High-Risk Optimizations",
                "",
            ])
            for opt in plan.risk_assessment['high_risk_optimizations']:
                report_lines.append(f"- {opt}")
                
        report_lines.extend([
            "",
            "### Mitigation Strategies",
            "",
        ])
        
        for strategy in plan.risk_assessment['mitigation_strategies']:
            report_lines.append(f"- {strategy}")
            
        report_lines.extend([
            "",
            "## Success Metrics",
            "",
            f"**Overall Expected Improvement**: {plan.success_metrics['overall_expected_improvement']:.1f}%",
            "",
            "### Key Performance Indicators",
            "",
        ])
        
        for kpi, target in plan.success_metrics['key_performance_indicators'].items():
            report_lines.append(f"- **{kpi.replace('_', ' ').title()}**: {target}")
            
        report_lines.extend([
            "",
            "## Timeline",
            "",
            f"**Total Duration**: {plan.timeline['total_duration_weeks']} weeks",
            "",
            "### Milestones",
            "",
        ])
        
        for milestone in plan.timeline['milestones']:
            report_lines.extend([
                f"**Week {milestone['week']}**: {milestone['milestone']}",
                f"- {milestone['deliverables']}",
                "",
            ])
            
        report_lines.extend([
            "### Critical Path",
            "",
        ])
        
        for opt in plan.timeline['critical_path']:
            report_lines.append(f"- {opt}")
            
        return "\n".join(report_lines)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate DubChain optimization plan")
    parser.add_argument(
        "--profiling-results",
        default="baseline_profiling_results/baseline_results.json",
        help="Path to profiling results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="optimization_plan_results",
        help="Output directory for optimization plan"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load profiling results
    if not os.path.exists(args.profiling_results):
        print(f"Error: Profiling results file not found: {args.profiling_results}")
        print("Please run baseline profiling first:")
        print("python scripts/run_baseline_profiling.py")
        sys.exit(1)
        
    with open(args.profiling_results, 'r') as f:
        profiling_results = json.load(f)
        
    # Generate optimization plan
    generator = OptimizationPlanGenerator()
    plan = generator.generate_optimization_plan(profiling_results)
    
    # Save plan
    plan_file = output_dir / "optimization_plan.json"
    generator.save_optimization_plan(plan, str(plan_file))
    
    # Generate markdown report
    markdown_report = generator.generate_markdown_report(plan)
    report_file = output_dir / "optimization_plan.md"
    with open(report_file, 'w') as f:
        f.write(markdown_report)
        
    print(f"✅ Optimization plan generated successfully!")
    print(f"📁 Plan saved to: {plan_file}")
    print(f"📋 Report saved to: {report_file}")
    print(f"")
    print(f"📊 Summary:")
    print(f"  - Optimizations: {len(plan.recommendations)}")
    print(f"  - Expected improvement: {plan.success_metrics['overall_expected_improvement']:.1f}%")
    print(f"  - Timeline: {plan.timeline['total_duration_weeks']} weeks")
    print(f"  - Risk level: {plan.risk_assessment['overall_risk_level']}")


if __name__ == "__main__":
    main()
