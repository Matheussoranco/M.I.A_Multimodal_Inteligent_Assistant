#!/usr/bin/env python3
"""
Development and Quality Assurance Script for M.I.A
Run comprehensive tests, quality checks, and performance analysis
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.mia.code_quality_manager import CodeQualityManager
from src.mia.documentation_generator import DocumentationGenerator
from src.mia.performance_monitor import PerformanceMonitor
from src.mia.cache_manager import CacheManager

def run_tests():
    """Run comprehensive test suite."""
    print("="*60)
    print("🧪 RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    # Run unit tests
    print("\n1. Running unit tests...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/', '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✅ Unit tests passed")
        else:
            print("❌ Unit tests failed")
            print(result.stdout)
            print(result.stderr)
    except FileNotFoundError:
        print("⚠️  pytest not found, running basic tests...")
        try:
            subprocess.run([
                sys.executable, 'tests/test_priority_4.py'
            ], cwd=Path(__file__).parent.parent)
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
    
    # Run integration tests
    print("\n2. Running integration tests...")
    try:
        from tests.test_priority_4 import TestIntegration
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ Integration tests passed")
        else:
            print("❌ Integration tests failed")
            
    except Exception as e:
        print(f"⚠️  Integration tests could not run: {e}")

def run_quality_checks():
    """Run code quality analysis."""
    print("\n="*60)
    print("📊 RUNNING CODE QUALITY ANALYSIS")
    print("="*60)
    
    try:
        quality_manager = CodeQualityManager()
        report = quality_manager.run_quality_checks()
        
        print(f"\n📈 Quality Score: {report.quality_score:.1f}/100")
        print(f"📁 Files analyzed: {report.total_files}")
        print(f"⚠️  Issues found: {report.issues_count}")
        
        # Show severity breakdown
        severity_counts = {}
        for issue in report.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            
        for severity, count in severity_counts.items():
            emoji = "🔴" if severity == "error" else "🟡" if severity == "warning" else "🔵"
            print(f"   {emoji} {severity.title()}: {count}")
        
        # Show top suggestions
        if report.suggestions:
            print("\n💡 Top Improvement Suggestions:")
            for i, suggestion in enumerate(report.suggestions[:5], 1):
                print(f"   {i}. {suggestion}")
        
        # Generate HTML report
        quality_manager.generate_quality_report_html(report)
        print(f"\n📋 Detailed report saved to: reports/quality_report.html")
        
        return report.quality_score
        
    except Exception as e:
        print(f"❌ Quality check failed: {e}")
        return 0

def run_performance_analysis():
    """Run performance analysis."""
    print("\n="*60)
    print("⚡ RUNNING PERFORMANCE ANALYSIS")
    print("="*60)
    
    try:
        # Initialize performance monitor
        perf_monitor = PerformanceMonitor()
        cache_manager = CacheManager()
        
        print("\n1. Starting performance monitoring...")
        perf_monitor.start_monitoring()
        
        # Simulate some load
        print("2. Running performance tests...")
        start_time = time.time()
        
        # Test cache performance
        for i in range(1000):
            cache_manager.put(f"test_key_{i}", f"test_value_{i}")
        
        for i in range(1000):
            cache_manager.get(f"test_key_{i}")
        
        end_time = time.time()
        
        # Get metrics
        time.sleep(1)  # Allow metrics to be collected
        metrics = perf_monitor.get_current_metrics()
        summary = perf_monitor.get_performance_summary()
        cache_stats = cache_manager.get_stats()
        
        print(f"\n📊 Performance Results:")
        print(f"   Cache operations: {end_time - start_time:.2f} seconds")
        
        if metrics:
            print(f"   CPU usage: {metrics.cpu_percent:.1f}%")
            print(f"   Memory usage: {metrics.memory_percent:.1f}%")
            print(f"   Active threads: {metrics.active_threads}")
            
        if summary:
            print(f"   Average CPU: {summary.get('average_cpu_percent', 0):.1f}%")
            print(f"   Average memory: {summary.get('average_memory_percent', 0):.1f}%")
            
        print(f"   Cache hit rate: {cache_stats['memory_cache']['hit_rate']:.1%}")
        print(f"   Cache entries: {cache_stats['memory_cache']['size']}")
        
        # Memory consumers
        top_consumers = perf_monitor.get_memory_top_consumers(5)
        if top_consumers:
            print("\n🔝 Top Memory Consumers:")
            for consumer in top_consumers:
                print(f"   {consumer['size_mb']:.1f}MB - {consumer['filename']}")
        
        # Cleanup
        perf_monitor.stop_monitoring()
        perf_monitor.cleanup()
        cache_manager.clear_all()
        
        print("\n✅ Performance analysis completed")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")

def generate_documentation():
    """Generate comprehensive documentation."""
    print("\n="*60)
    print("📚 GENERATING DOCUMENTATION")
    print("="*60)
    
    try:
        doc_generator = DocumentationGenerator()
        doc_generator.generate_all_docs()
        print("✅ Documentation generated successfully")
        print("📁 Documentation available in: docs/generated/")
        
    except Exception as e:
        print(f"❌ Documentation generation failed: {e}")

def run_automated_fixes():
    """Run automated code fixes."""
    print("\n="*60)
    print("🔧 RUNNING AUTOMATED FIXES")
    print("="*60)
    
    try:
        quality_manager = CodeQualityManager()
        quality_manager.run_automated_fixes()
        print("✅ Automated fixes applied")
        
    except Exception as e:
        print(f"❌ Automated fixes failed: {e}")

def run_full_qa_pipeline():
    """Run complete QA pipeline."""
    print("🚀 STARTING FULL QA PIPELINE")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Run tests
    run_tests()
    
    # 2. Run quality checks
    quality_score = run_quality_checks()
    
    # 3. Run performance analysis
    run_performance_analysis()
    
    # 4. Generate documentation
    generate_documentation()
    
    # 5. Run automated fixes if quality is low
    if quality_score < 80:
        print(f"\n⚠️  Quality score ({quality_score:.1f}) is below threshold (80)")
        print("Running automated fixes...")
        run_automated_fixes()
        
        # Re-run quality check
        print("\nRe-running quality check after fixes...")
        new_quality_score = run_quality_checks()
        
        improvement = new_quality_score - quality_score
        if improvement > 0:
            print(f"✅ Quality improved by {improvement:.1f} points")
        else:
            print("⚠️  No significant improvement after automated fixes")
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print("🎉 QA PIPELINE COMPLETED")
    print("="*60)
    print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
    print(f"📊 Final quality score: {quality_score:.1f}/100")
    
    # Summary
    print("\n📋 Generated Reports:")
    print("   - reports/quality_report.html")
    print("   - reports/quality_report.json")
    print("   - docs/generated/")
    
    return quality_score

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/dev_qa.py [command]")
        print("")
        print("Commands:")
        print("  test         - Run test suite")
        print("  quality      - Run quality checks")
        print("  performance  - Run performance analysis")
        print("  docs         - Generate documentation")
        print("  fix          - Run automated fixes")
        print("  full         - Run full QA pipeline")
        return
    
    command = sys.argv[1].lower()
    
    if command == "test":
        run_tests()
    elif command == "quality":
        run_quality_checks()
    elif command == "performance":
        run_performance_analysis()
    elif command == "docs":
        generate_documentation()
    elif command == "fix":
        run_automated_fixes()
    elif command == "full":
        run_full_qa_pipeline()
    else:
        print(f"Unknown command: {command}")
        print("Use --help for available commands")

if __name__ == "__main__":
    main()
