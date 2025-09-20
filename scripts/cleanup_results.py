#!/usr/bin/env python3
"""
Cleanup script for DubChain test results and artifacts.

This script helps organize and clean up test results, moving them to the
appropriate directories in the results/ folder.
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Dict


def find_result_files(root_dir: str) -> Dict[str, List[str]]:
    """Find all result files in the root directory."""
    result_files = {
        'json_results': [],
        'db_files': [],
        'coverage_files': [],
        'benchmark_dirs': [],
        'profiling_dirs': [],
        'other_artifacts': []
    }
    
    # JSON result files
    result_files['json_results'] = glob.glob(os.path.join(root_dir, '*_results.json'))
    result_files['json_results'].extend(glob.glob(os.path.join(root_dir, '*_benchmark_results.json')))
    
    # Database files
    result_files['db_files'] = glob.glob(os.path.join(root_dir, 'test*.db*'))
    result_files['db_files'].extend(glob.glob(os.path.join(root_dir, '*.sqlite*')))
    
    # Coverage files
    result_files['coverage_files'] = glob.glob(os.path.join(root_dir, 'htmlcov'))
    result_files['coverage_files'].extend(glob.glob(os.path.join(root_dir, '.coverage*')))
    
    # Benchmark directories
    result_files['benchmark_dirs'] = glob.glob(os.path.join(root_dir, '*_benchmark_results'))
    
    # Profiling directories
    result_files['profiling_dirs'] = glob.glob(os.path.join(root_dir, '*_profiling_artifacts'))
    
    # Other artifacts
    result_files['other_artifacts'] = glob.glob(os.path.join(root_dir, '*.prof'))
    result_files['other_artifacts'].extend(glob.glob(os.path.join(root_dir, '*.profile')))
    
    return result_files


def move_files_to_results(files_dict: Dict[str, List[str]], results_dir: str) -> None:
    """Move files to appropriate subdirectories in results/."""
    # Ensure results directory structure exists
    subdirs = ['test_results', 'benchmarks', 'profiling', 'artifacts']
    for subdir in subdirs:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
    
    # Move JSON results
    for file_path in files_dict['json_results']:
        if os.path.exists(file_path):
            dest = os.path.join(results_dir, 'test_results', os.path.basename(file_path))
            shutil.move(file_path, dest)
            print(f"Moved {file_path} -> {dest}")
    
    # Move benchmark directories
    for dir_path in files_dict['benchmark_dirs']:
        if os.path.exists(dir_path):
            dest = os.path.join(results_dir, 'benchmarks', os.path.basename(dir_path))
            shutil.move(dir_path, dest)
            print(f"Moved {dir_path} -> {dest}")
    
    # Move profiling directories
    for dir_path in files_dict['profiling_dirs']:
        if os.path.exists(dir_path):
            dest = os.path.join(results_dir, 'profiling', os.path.basename(dir_path))
            shutil.move(dir_path, dest)
            print(f"Moved {dir_path} -> {dest}")
    
    # Move database and coverage files
    for file_path in files_dict['db_files'] + files_dict['coverage_files']:
        if os.path.exists(file_path):
            dest = os.path.join(results_dir, 'artifacts', os.path.basename(file_path))
            shutil.move(file_path, dest)
            print(f"Moved {file_path} -> {dest}")
    
    # Move other artifacts
    for file_path in files_dict['other_artifacts']:
        if os.path.exists(file_path):
            dest = os.path.join(results_dir, 'artifacts', os.path.basename(file_path))
            shutil.move(file_path, dest)
            print(f"Moved {file_path} -> {dest}")


def cleanup_results_directory(results_dir: str) -> None:
    """Clean up old results to save disk space."""
    if not os.path.exists(results_dir):
        return
    
    # Remove files older than 30 days
    import time
    current_time = time.time()
    thirty_days_ago = current_time - (30 * 24 * 60 * 60)
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) < thirty_days_ago:
                os.remove(file_path)
                print(f"Removed old file: {file_path}")


def main():
    """Main cleanup function."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root_dir, 'results')
    
    print("🧹 DubChain Results Cleanup")
    print("=" * 40)
    
    # Find all result files
    print("🔍 Scanning for result files...")
    files_dict = find_result_files(root_dir)
    
    total_files = sum(len(files) for files in files_dict.values())
    if total_files == 0:
        print("✅ No result files found to organize.")
        return
    
    print(f"📁 Found {total_files} files/directories to organize:")
    for category, files in files_dict.items():
        if files:
            print(f"  - {category}: {len(files)} items")
    
    # Move files to results directory
    print("\n📦 Organizing files...")
    move_files_to_results(files_dict, results_dir)
    
    # Clean up old results
    print("\n🗑️  Cleaning up old results...")
    cleanup_results_directory(results_dir)
    
    print("\n✅ Cleanup complete!")
    print(f"📂 Results organized in: {results_dir}")


if __name__ == "__main__":
    main()
