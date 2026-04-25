"""
optimizer/benchmark.py
─────────────────────────────────────────────────────
Runs all optimizers on one or more benchmark circuits
and prints a clean comparison table.

Usage:
    python optimizer/benchmark.py
    python optimizer/benchmark.py s1196
    python optimizer/benchmark.py s1196 c17 s27

Results table columns:
    Circuit | Gates | Original | SA | GA | GA+SA | GNN-SA | Best
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_v1                    import load_circuit
from optimizer.simulated_annealing    import simulated_annealing
from optimizer.genetic_algorithm      import genetic_algorithm
from optimizer.hybrid_optimizer       import hybrid_optimize
from optimizer.gnn_optimizer          import gnn_simulated_annealing


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BENCHMARK_DIR = "data/benchmarks"

# Default circuits to test (can override via command line)
DEFAULT_CIRCUITS = ["s1196"]

# Optimizer settings — same for all runs (fair comparison)
SA_CONFIG = dict(
    initial_temp        = 50.0,
    cooling_rate        = 0.95,
    min_temp            = 0.1,
    iterations_per_temp = 10,
    verbose             = False,
)

GA_CONFIG = dict(
    population_size = 20,
    generations     = 50,
    survival_rate   = 0.3,
    mutation_rate   = 0.7,
    verbose         = False,
)

HYBRID_CONFIG = dict(
    ga_population   = 20,
    ga_generations  = 50,
    ga_survival     = 0.3,
    ga_mutation     = 0.7,
    sa_initial_temp = 50.0,
    sa_cooling      = 0.95,
    sa_min_temp     = 0.1,
    sa_iterations   = 10,
    verbose         = False,
)

GNN_SA_CONFIG = dict(
    initial_temp        = 50.0,
    cooling_rate        = 0.95,
    min_temp            = 0.1,
    iterations_per_temp = 10,
    verify_every        = 20,
    verbose             = False,
)


# ─────────────────────────────────────────────────────────────
# RUNNER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def run_sa(circuit):
    t0 = time.perf_counter()
    _, cost, _ = simulated_annealing(circuit, **SA_CONFIG)
    return cost, time.perf_counter() - t0


def run_ga(circuit):
    t0 = time.perf_counter()
    _, cost, _ = genetic_algorithm(circuit, **GA_CONFIG)
    return cost, time.perf_counter() - t0


def run_hybrid(circuit):
    t0 = time.perf_counter()
    _, cost, _ = hybrid_optimize(circuit, **HYBRID_CONFIG)
    return cost, time.perf_counter() - t0


def run_gnn_sa(circuit, predictor):
    t0 = time.perf_counter()
    _, cost, _ = gnn_simulated_annealing(circuit, predictor, **GNN_SA_CONFIG)
    return cost, time.perf_counter() - t0


def improvement(original, optimized):
    return round((original - optimized) / original * 100, 2)


# ─────────────────────────────────────────────────────────────
# BENCHMARK ONE CIRCUIT
# ─────────────────────────────────────────────────────────────

def benchmark_circuit(circuit_name, predictor):
    filepath = os.path.join(BENCHMARK_DIR, f"{circuit_name}.bench")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {circuit_name}.bench not found")
        return None

    circuit, _ = load_circuit(filepath)
    original   = circuit.cost

    print(f"\n  Benchmarking: {circuit_name}  "
          f"({circuit.gate_count} gates, original cost: {original})")

    results = {"name": circuit_name, "gates": circuit.gate_count,
               "original": original}

    # SA
    print(f"    Running SA...", end="  ", flush=True)
    cost, t = run_sa(circuit)
    results["sa"]      = cost
    results["sa_time"] = round(t, 2)
    print(f"done  ({improvement(original, cost)}%  {round(t,2)}s)")

    # GA
    print(f"    Running GA...", end="  ", flush=True)
    cost, t = run_ga(circuit)
    results["ga"]      = cost
    results["ga_time"] = round(t, 2)
    print(f"done  ({improvement(original, cost)}%  {round(t,2)}s)")

    # GA+SA Hybrid
    print(f"    Running GA+SA...", end="  ", flush=True)
    cost, t = run_hybrid(circuit)
    results["hybrid"]      = cost
    results["hybrid_time"] = round(t, 2)
    print(f"done  ({improvement(original, cost)}%  {round(t,2)}s)")

    # GNN-SA
    if predictor:
        print(f"    Running GNN-SA...", end="  ", flush=True)
        cost, t = run_gnn_sa(circuit, predictor)
        results["gnn_sa"]      = cost
        results["gnn_sa_time"] = round(t, 2)
        print(f"done  ({improvement(original, cost)}%  {round(t,2)}s)")
    else:
        results["gnn_sa"]      = None
        results["gnn_sa_time"] = None

    return results


# ─────────────────────────────────────────────────────────────
# PRINT RESULTS TABLE
# ─────────────────────────────────────────────────────────────

def print_table(all_results):
    print()
    print("=" * 95)
    print("  BENCHMARK RESULTS")
    print("=" * 95)

    # Header
    print(f"  {'Circuit':<10} {'Gates':>6} {'Original':>10} "
          f"{'SA':>10} {'GA':>10} {'GA+SA':>10} {'GNN-SA':>10} "
          f"{'Best Imp%':>10}")
    print("-" * 95)

    for r in all_results:
        orig = r['original']

        sa_str     = f"{round(r['sa'],1)} ({improvement(orig,r['sa'])}%)"
        ga_str     = f"{round(r['ga'],1)} ({improvement(orig,r['ga'])}%)"
        hybrid_str = f"{round(r['hybrid'],1)} ({improvement(orig,r['hybrid'])}%)"

        if r['gnn_sa']:
            gnn_str = f"{round(r['gnn_sa'],1)} ({improvement(orig,r['gnn_sa'])}%)"
            candidates = [r['sa'], r['ga'], r['hybrid'], r['gnn_sa']]
        else:
            gnn_str    = "N/A"
            candidates = [r['sa'], r['ga'], r['hybrid']]

        best_imp = improvement(orig, min(candidates))

        print(f"  {r['name']:<10} {r['gates']:>6} {round(orig,1):>10} "
              f"{sa_str:>16} {ga_str:>14} {hybrid_str:>16} "
              f"{gnn_str:>16} {best_imp:>9}%")

    print("=" * 95)

    # Timing table
    print()
    print("  TIMING (seconds)")
    print("-" * 60)
    print(f"  {'Circuit':<10} {'SA':>8} {'GA':>8} {'GA+SA':>8} {'GNN-SA':>8}")
    print("-" * 60)
    for r in all_results:
        gnn_t = r['gnn_sa_time'] if r['gnn_sa_time'] else "N/A"
        print(f"  {r['name']:<10} {r['sa_time']:>8} {r['ga_time']:>8} "
              f"{r['hybrid_time']:>8} {str(gnn_t):>8}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Which circuits to benchmark
    circuits = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CIRCUITS

    print("=" * 60)
    print("  CIRCUIT OPTIMIZATION BENCHMARK")
    print("=" * 60)

    # Load GNN predictor
    try:
        from ml.predictor import GNNPredictor
        predictor = GNNPredictor()
        print("  GNN predictor loaded.")
    except Exception as e:
        print(f"  GNN predictor unavailable ({e}). Running without GNN-SA.")
        predictor = None

    # Run benchmarks
    all_results = []
    for name in circuits:
        result = benchmark_circuit(name, predictor)
        if result:
            all_results.append(result)

    # Print final table
    if all_results:
        print_table(all_results)