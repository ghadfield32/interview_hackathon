# path: api/src/airflow_project/utils/performance_benchmark.py
"""
NBA Pipeline Performance Benchmarking Tool
Comprehensive testing and performance measurement utilities
"""

import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Callable, Any
import pandas as pd
import json
from contextlib import contextmanager
import tracemalloc
import duckdb

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking for NBA pipeline"""

    def __init__(self):
        self.results = {}
        self.baseline_metrics = {}
        self.optimized_metrics = {}

    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information for benchmarking context"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
                "platform": psutil.sys.platform
            }
        except Exception as e:
            logger.warning(f"Could not gather system info: {e}")
            return {"error": str(e)}

    @contextmanager
    def measure_performance(self, operation_name: str):
        """Context manager for measuring operation performance"""
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().used

        # Start memory tracking
        tracemalloc.start()

        try:
            yield
        finally:
            # Stop memory tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory().used

            # Calculate metrics
            execution_time = end_time - start_time
            cpu_usage = (start_cpu + end_cpu) / 2  # Average CPU usage
            memory_delta = end_memory - start_memory
            memory_peak_mb = peak / (1024 * 1024)

            # Store results
            self.results[operation_name] = {
                "execution_time_seconds": execution_time,
                "cpu_usage_percent": cpu_usage,
                "memory_delta_bytes": memory_delta,
                "memory_peak_mb": memory_peak_mb,
                "timestamp": time.time()
            }

            logger.info(f"📊 {operation_name}: {execution_time:.2f}s, CPU: {cpu_usage:.1f}%, Memory: {memory_peak_mb:.1f}MB")

    def benchmark_function(self, func: Callable, func_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark a single function execution"""
        with self.measure_performance(func_name):
            result = func(*args, **kwargs)

        return result

    def benchmark_pipeline_components(self, pipeline_instance) -> Dict[str, Any]:
        """Benchmark individual pipeline components"""
        component_results = {}

        # Benchmark data loading
        if hasattr(pipeline_instance, 'load_and_validate_data_consolidated'):
            with self.measure_performance("data_loading"):
                validation_results = pipeline_instance.load_and_validate_data_consolidated()
            component_results["data_loading"] = validation_results

        # Benchmark lineup building
        if hasattr(pipeline_instance, 'build_simplified_lineups'):
            with self.measure_performance("lineup_building"):
                pipeline_instance.build_simplified_lineups()

        # Benchmark metrics computation
        if hasattr(pipeline_instance, 'compute_metrics_batch'):
            with self.measure_performance("metrics_computation"):
                lineups_df, players_df = pipeline_instance.compute_metrics_batch()
            component_results["metrics_computation"] = {"lineups": len(lineups_df), "players": len(players_df)}

        # Benchmark output formatting
        if hasattr(pipeline_instance, 'format_final_output'):
            with self.measure_performance("output_formatting"):
                lineups_final, players_final = pipeline_instance.format_final_output(lineups_df, players_df)
            component_results["output_formatting"] = {"lineups": len(lineups_df), "players": len(players_df)}

        return component_results

    def compare_performance(self, baseline_metrics: Dict[str, Any], 
                           optimized_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline vs optimized performance"""
        comparison = {}

        for metric in baseline_metrics:
            if metric in optimized_metrics:
                baseline_val = baseline_metrics[metric]
                optimized_val = optimized_metrics[metric]

                if isinstance(baseline_val, (int, float)) and isinstance(optimized_val, (int, float)):
                    if baseline_val > 0:
                        improvement_pct = ((baseline_val - optimized_val) / baseline_val) * 100
                        speedup_factor = baseline_val / optimized_val if optimized_val > 0 else 0

                        comparison[metric] = {
                            "baseline": baseline_val,
                            "optimized": optimized_val,
                            "improvement_percent": round(improvement_pct, 1),
                            "speedup_factor": round(speedup_factor, 1),
                            "improvement_direction": "faster" if metric == "execution_time_seconds" else "better"
                        }

        return comparison

    def memory_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        memory_analysis = {}

        for operation, metrics in self.results.items():
            if "memory_peak_mb" in metrics:
                memory_analysis[operation] = {
                    "peak_memory_mb": metrics["memory_peak_mb"],
                    "memory_efficiency": "LOW" if metrics["memory_peak_mb"] > 1000 else "MEDIUM" if metrics["memory_peak_mb"] > 500 else "HIGH"
                }

        return memory_analysis

    def generate_performance_report(self, output_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        system_info = self._get_system_info()

        # Calculate overall performance metrics
        total_time = sum(result.get("execution_time_seconds", 0) for result in self.results.values())
        avg_cpu = sum(result.get("cpu_usage_percent", 0) for result in self.results.values()) / len(self.results) if self.results else 0
        max_memory = max(result.get("memory_peak_mb", 0) for result in self.results.values()) if self.results else 0

        # Performance grading
        if total_time < 30:
            performance_grade = "A+"
        elif total_time < 60:
            performance_grade = "A"
        elif total_time < 120:
            performance_grade = "B"
        elif total_time < 300:
            performance_grade = "C"
        else:
            performance_grade = "D"

        report = {
            "benchmark_timestamp": time.time(),
            "system_info": system_info,
            "overall_performance": {
                "total_execution_time_seconds": total_time,
                "average_cpu_usage_percent": round(avg_cpu, 1),
                "peak_memory_usage_mb": round(max_memory, 1),
                "performance_grade": performance_grade,
                "operations_benchmarked": len(self.results)
            },
            "component_benchmarks": self.results,
            "memory_analysis": self.memory_analysis(),
            "performance_insights": self._generate_optimization_summary()
        }

        # Save report if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to: {output_file}")

        return report

    def _generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate optimization insights and recommendations"""
        insights = {
            "bottlenecks": [],
            "optimization_opportunities": [],
            "performance_highlights": []
        }

        # Identify bottlenecks (operations taking >50% of total time)
        total_time = sum(result.get("execution_time_seconds", 0) for result in self.results.values())

        for operation, metrics in self.results.items():
            execution_time = metrics.get("execution_time_seconds", 0)
            time_percentage = (execution_time / total_time * 100) if total_time > 0 else 0

            if time_percentage > 50:
                insights["bottlenecks"].append({
                    "operation": operation,
                    "time_percentage": round(time_percentage, 1),
                    "execution_time": execution_time
                })

            # Performance highlights (fast operations)
            if execution_time < 1.0:
                insights["performance_highlights"].append({
                    "operation": operation,
                    "execution_time": execution_time,
                    "status": "EXCELLENT"
                })

        # Optimization recommendations
        if insights["bottlenecks"]:
            insights["optimization_opportunities"].append("Focus optimization efforts on bottleneck operations")

        if any(result.get("memory_peak_mb", 0) > 1000 for result in self.results.values()):
            insights["optimization_opportunities"].append("Consider memory optimization for high-memory operations")

        return insights


class DatabasePerformanceTester:
    """Specialized database performance testing"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.con = None

    def __enter__(self):
        self.con = duckdb.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con:
            self.con.close()

    def test_query_performance(self, query: str, query_name: str, iterations: int = 3) -> Dict[str, Any]:
        """Test performance of a specific SQL query"""
        times = []

        for i in range(iterations):
            start_time = time.time()
            result = self.con.execute(query)
            execution_time = time.time() - start_time
            times.append(execution_time)

        return {
            "query_name": query_name,
            "iterations": iterations,
            "execution_times": times,
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_deviation": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }

    def test_data_loading_performance(self, csv_path: str, table_name: str) -> Dict[str, Any]:
        """Test CSV loading performance"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used

        # Load CSV
        self.con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{csv_path}', header=true)
        """)

        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        # Get row count
        row_count = self.con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        return {
            "operation": f"load_csv_{table_name}",
            "execution_time_seconds": end_time - start_time,
            "memory_delta_bytes": end_memory - start_memory,
            "rows_loaded": row_count,
            "rows_per_second": row_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
        }


def run_full_benchmark(pipeline_class, pipeline_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run comprehensive benchmark suite"""
    benchmark = PerformanceBenchmark()

    logger.info("🚀 Starting comprehensive performance benchmark...")

    try:
        # Benchmark pipeline instantiation
        with benchmark.measure_performance("pipeline_instantiation"):
            pipeline = pipeline_class(**(pipeline_args or {}))

        # Benchmark individual components
        component_results = benchmark.benchmark_pipeline_components(pipeline)

        # Benchmark full pipeline execution
        if hasattr(pipeline, 'run_optimized_pipeline'):
            with benchmark.measure_performance("full_pipeline_execution"):
                result = pipeline.run_optimized_pipeline()

        # Generate comprehensive report
        report = benchmark.generate_performance_report()

        logger.info("✅ Performance benchmark completed successfully")
        return report

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise


def print_benchmark_summary(report: Dict[str, Any]) -> None:
    """Print formatted benchmark summary"""
    print("=" * 80)
    print("NBA PIPELINE PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)

    # Overall performance
    overall = report.get("overall_performance", {})
    print(f"🎯 Overall Performance Grade: {overall.get('performance_grade', 'N/A')}")
    print(f"⏱️  Total Execution Time: {overall.get('total_execution_time_seconds', 0):.2f} seconds")
    print(f"💻 Average CPU Usage: {overall.get('average_cpu_usage_percent', 0):.1f}%")
    print(f"🧠 Peak Memory Usage: {overall.get('peak_memory_usage_mb', 0):.1f} MB")
    print(f"🔧 Operations Benchmarked: {overall.get('operations_benchmarked', 0)}")

    # Component breakdown
    print(f"\n📊 Component Performance Breakdown:")
    for operation, metrics in report.get("component_benchmarks", {}).items():
        time_sec = metrics.get("execution_time_seconds", 0)
        memory_mb = metrics.get("memory_peak_mb", 0)
        print(f"   {operation}: {time_sec:.2f}s, {memory_mb:.1f}MB")

    # Performance insights
    insights = report.get("performance_insights", {})
    if insights.get("bottlenecks"):
        print(f"\n⚠️  Performance Bottlenecks:")
        for bottleneck in insights["bottlenecks"]:
            print(f"   {bottleneck['operation']}: {bottleneck['time_percentage']:.1f}% of total time")

    if insights.get("performance_highlights"):
        print(f"\n✅ Performance Highlights:")
        for highlight in insights["performance_highlights"]:
            print(f"   {highlight['operation']}: {highlight['execution_time']:.2f}s ({highlight['status']})")

    if insights.get("optimization_opportunities"):
        print(f"\n💡 Optimization Opportunities:")
        for opportunity in insights["optimization_opportunities"]:
            print(f"   {opportunity}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    logger.info("Performance benchmarking tool ready for use")

    # Example: Test database performance
    # with DatabasePerformanceTester("test.duckdb") as db_tester:
    #     result = db_tester.test_query_performance("SELECT COUNT(*) FROM test_table", "count_test")
    #     print(f"Query performance: {result}")
