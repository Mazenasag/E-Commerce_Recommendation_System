#!/usr/bin/env python
"""Analyze Locust stress test results"""
import pandas as pd
import json
from pathlib import Path

def analyze_stress_test(test_name, csv_prefix):
    """Analyze a stress test and generate report"""
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {test_name}")
    print(f"{'='*70}\n")
    
    # Load statistics
    stats_file = f"reports/{csv_prefix}_stats.csv"
    if not Path(stats_file).exists():
        print(f"❌ File not found: {stats_file}")
        return None
    
    df = pd.read_csv(stats_file)
    
    # Overall metrics
    total_row = df[df['Name'] == 'Aggregated'].iloc[0]
    
    print("📊 OVERALL METRICS:")
    print(f"   Total Requests: {total_row['Request Count']:,.0f}")
    print(f"   Total Failures: {total_row['Failure Count']:,.0f} ({total_row['Failure Count']/total_row['Request Count']*100:.2f}%)")
    print(f"   ⭐ Average Latency: {total_row['Average Response Time']:.2f}ms")
    print(f"   ⭐ P95 Latency: {total_row['95%']:.2f}ms")
    print(f"   Median Latency: {total_row['Median Response Time']:.2f}ms")
    print(f"   P99 Latency: {total_row['99%']:.2f}ms")
    print(f"   Requests/Second: {total_row['Requests/s']:.2f}")
    print()
    
    # Per-endpoint metrics
    print("📈 PER-ENDPOINT METRICS:")
    print(f"{'Endpoint':<40} {'Avg (ms)':<12} {'P95 (ms)':<12} {'RPS':<10} {'Failures':<10}")
    print("-" * 90)
    
    for _, row in df.iterrows():
        if row['Name'] != 'Aggregated':
            name = row['Name'][:38]
            avg = row['Average Response Time']
            p95 = row['95%']
            rps = row['Requests/s']
            failures = f"{row['Failure Count']:.0f} ({row['Failure Count']/row['Request Count']*100:.1f}%)" if row['Request Count'] > 0 else "0"
            print(f"{name:<40} {avg:<12.2f} {p95:<12.2f} {rps:<10.2f} {failures:<10}")
    
    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    avg_latency = total_row['Average Response Time']
    p95_latency = total_row['95%']
    failure_rate = (total_row['Failure Count'] / total_row['Request Count'] * 100) if total_row['Request Count'] > 0 else 0
    
    # Determine status
    if "500" in test_name:
        avg_target = 500
        p95_target = 1000
    else:
        avg_target = 800
        p95_target = 1500
    
    avg_status = "✅" if avg_latency < avg_target else "⚠️" if avg_latency < avg_target * 1.5 else "❌"
    p95_status = "✅" if p95_latency < p95_target else "⚠️" if p95_latency < p95_target * 1.5 else "❌"
    failure_status = "✅" if failure_rate < 1 else "⚠️" if failure_rate < 2 else "❌"
    
    print(f"   Average Latency: {avg_status} {avg_latency:.2f}ms (target: <{avg_target}ms)")
    print(f"   P95 Latency: {p95_status} {p95_latency:.2f}ms (target: <{p95_target}ms)")
    print(f"   Failure Rate: {failure_status} {failure_rate:.2f}% (target: <1%)")
    
    return {
        'test_name': test_name,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'failure_rate': failure_rate,
        'rps': total_row['Requests/s']
    }

if __name__ == "__main__":
    results = []
    
    # Analyze 500 user test
    if Path("reports/stress_test_500_stats.csv").exists():
        results.append(analyze_stress_test("500 Concurrent Users", "stress_test_500"))
    
    # Analyze 1000 user test
    if Path("reports/stress_test_1000_stats.csv").exists():
        results.append(analyze_stress_test("1000 Concurrent Users", "stress_test_1000"))
    
    # Comparison
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("COMPARISON: 500 vs 1000 Users")
        print(f"{'='*70}\n")
        print(f"{'Metric':<25} {'500 Users':<20} {'1000 Users':<20} {'Change':<15}")
        print("-" * 80)
        
        r500 = results[0]
        r1000 = results[1]
        
        avg_change = ((r1000['avg_latency'] - r500['avg_latency']) / r500['avg_latency']) * 100
        p95_change = ((r1000['p95_latency'] - r500['p95_latency']) / r500['p95_latency']) * 100
        
        print(f"{'Average Latency':<25} {r500['avg_latency']:<20.2f} {r1000['avg_latency']:<20.2f} {avg_change:+.1f}%")
        print(f"{'P95 Latency':<25} {r500['p95_latency']:<20.2f} {r1000['p95_latency']:<20.2f} {p95_change:+.1f}%")
        print(f"{'Failure Rate':<25} {r500['failure_rate']:<20.2f}% {r1000['failure_rate']:<20.2f}% {r1000['failure_rate']-r500['failure_rate']:+.2f}%")
        print(f"{'RPS':<25} {r500['rps']:<20.2f} {r1000['rps']:<20.2f} {((r1000['rps']-r500['rps'])/r500['rps']*100):+.1f}%")
    
    print(f"\n{'='*70}")
    print("✅ Analysis complete! Check HTML reports for detailed charts.")
    print(f"{'='*70}\n")

