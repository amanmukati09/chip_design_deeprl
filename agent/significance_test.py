# agent/significance_test.py
import json
from scipy import stats
import numpy as np

with open("agent/benchmark_results.json") as f:
    data = json.load(f)["results"]

sa      = np.array([r['sa_mean'] for r in data])
hybrid  = np.array([r['hybrid_mean'] for r in data])
portfolio = np.maximum(sa, hybrid)

print(f"SA mean:        {sa.mean():.3f}")
print(f"Hybrid mean:    {hybrid.mean():.3f}")
print(f"Portfolio mean: {portfolio.mean():.3f}  (never worse than SA, by construction)")

stat, p = stats.wilcoxon(sa, hybrid)
print(f"\nWilcoxon signed-rank (SA vs Hybrid): stat={stat:.3f}, p={p:.4f}")

t, p_t = stats.ttest_rel(sa, hybrid)
print(f"Paired t-test:                       t={t:.3f}, p={p_t:.4f}")

# Excluding the two known basin-of-attraction outliers
mask = np.array([r['circuit'] not in ('c1908', 'c2670') for r in data])
print(f"\nExcluding c1908/c2670:")
print(f"  SA mean:     {sa[mask].mean():.3f}")
print(f"  Hybrid mean: {hybrid[mask].mean():.3f}")