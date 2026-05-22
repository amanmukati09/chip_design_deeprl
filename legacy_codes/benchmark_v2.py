"""
optimizer/benchmark.py
─────────────────────────────────────────────────────
Runs all optimizers on benchmark circuits with
iteration scaling proportional to circuit size.

Usage:
    python optimizer/benchmark.py
    python optimizer/benchmark.py s1196 c2670
    python optimizer/benchmark.py --all
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
from heuristics.manager               import scale_iterations


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BENCHMARK_DIR     = "data/benchmarks"
DEFAULT_CIRCUITS  = ["s1196"]

ALL_CIRCUITS = [
    "c17", "c880", "s820", "s832", "s953",
    "s1196", "s1238", "s1488", "s1494",
    "c1355", "c1908", "c2670", "c3540",
    "s5378", "s13207", "c7552", "s35932", "s38584",
]

GA_CONFIG = dict(
    population_size = 20,
    generations     = 50,
    survival_rate   = 0.3,
    mutation_rate   = 0.7,
    verbose         = False,
)


def build_sa_config(gate_count):
    return dict(
        initial_temp        = 50.0,
        cooling_rate        = 0.95,
        min_temp            = 0.1,
        iterations_per_temp = scale_iterations(gate_count),
        verbose             = False,
    )

def build_hybrid_config(gate_count):
    return dict(
        ga_population   = 20,
        ga_generations  = 50,
        ga_survival     = 0.3,
        ga_mutation     = 0.7,
        sa_initial_temp = 50.0,
        sa_cooling      = 0.95,
        sa_min_temp     = 0.1,
        sa_iterations   = scale_iterations(gate_count),
        verbose         = False,
    )

def build_gnn_config(gate_count):
    return dict(
        initial_temp        = 50.0,
        cooling_rate        = 0.95,
        min_temp            = 0.1,
        iterations_per_temp = scale_iterations(gate_count),
        verify_every        = 20,
        verbose             = False,
    )


# ─────────────────────────────────────────────────────────────
# RUNNERS
# ─────────────────────────────────────────────────────────────

def run_sa(circuit):
    t0 = time.perf_counter()
    _, cost, _ = simulated_annealing(circuit, **build_sa_config(circuit.gate_count))
    return cost, round(time.perf_counter() - t0, 2)

def run_ga(circuit):
    t0 = time.perf_counter()
    _, cost, _ = genetic_algorithm(circuit, **GA_CONFIG)
    return cost, round(time.perf_counter() - t0, 2)

def run_hybrid(circuit):
    t0 = time.perf_counter()
    _, cost, _ = hybrid_optimize(circuit, **build_hybrid_config(circuit.gate_count))
    return cost, round(time.perf_counter() - t0, 2)

def run_gnn_sa(circuit, predictor):
    t0 = time.perf_counter()
    _, cost, _ = gnn_simulated_annealing(
        circuit, predictor, **build_gnn_config(circuit.gate_count)
    )
    return cost, round(time.perf_counter() - t0, 2)

def imp(original, optimized):
    return round((original - optimized) / original * 100, 2)


# ─────────────────────────────────────────────────────────────
# BENCHMARK ONE CIRCUIT
# ─────────────────────────────────────────────────────────────

def benchmark_circuit(name, predictor):
    filepath = os.path.join(BENCHMARK_DIR, f"{name}.bench")
    if not os.path.exists(filepath):
        print(f"  [SKIP] {name}.bench not found")
        return None

    circuit, _ = load_circuit(filepath)
    original   = circuit.cost
    iters      = scale_iterations(circuit.gate_count)

    print(f"\n  {name}  ({circuit.gate_count} gates  "
          f"orig={round(original,1)}  iters={iters})")

    r = {"name": name, "gates": circuit.gate_count,
         "original": original, "iters": iters}

    print(f"    SA      ...", end="  ", flush=True)
    r["sa"], r["sa_t"] = run_sa(circuit)
    print(f"{imp(original, r['sa'])}%  {r['sa_t']}s")

    print(f"    GA      ...", end="  ", flush=True)
    r["ga"], r["ga_t"] = run_ga(circuit)
    print(f"{imp(original, r['ga'])}%  {r['ga_t']}s")

    print(f"    GA+SA   ...", end="  ", flush=True)
    r["hybrid"], r["hybrid_t"] = run_hybrid(circuit)
    print(f"{imp(original, r['hybrid'])}%  {r['hybrid_t']}s")

    if predictor:
        print(f"    GNN-SA  ...", end="  ", flush=True)
        r["gnn_sa"], r["gnn_t"] = run_gnn_sa(circuit, predictor)
        print(f"{imp(original, r['gnn_sa'])}%  {r['gnn_t']}s")
    else:
        r["gnn_sa"] = r["gnn_t"] = None

    return r


# ─────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────

def print_table(results):
    print()
    print("=" * 100)
    print("  BENCHMARK RESULTS  (with iteration scaling)")
    print("=" * 100)
    print(f"  {'Circuit':<10} {'Gates':>6} {'Iters':>6} {'Original':>10} "
          f"{'SA':>12} {'GA':>10} {'GA+SA':>12} {'GNN-SA':>12} {'Best':>8}")
    print("-" * 100)

    for r in results:
        orig = r['original']
        sa_s     = f"{imp(orig,r['sa'])}%"
        ga_s     = f"{imp(orig,r['ga'])}%"
        hy_s     = f"{imp(orig,r['hybrid'])}%"
        gn_s     = f"{imp(orig,r['gnn_sa'])}%" if r['gnn_sa'] else "N/A"
        cands    = [r['sa'], r['ga'], r['hybrid']]
        if r['gnn_sa']:
            cands.append(r['gnn_sa'])
        best_s   = f"{imp(orig, min(cands))}%"

        print(f"  {r['name']:<10} {r['gates']:>6} {r['iters']:>6} "
              f"{round(orig,1):>10} "
              f"{sa_s:>12} {ga_s:>10} {hy_s:>12} {gn_s:>12} {best_s:>8}")

    print("=" * 100)

    print()
    print("  TIMING (seconds)")
    print("-" * 65)
    print(f"  {'Circuit':<10} {'Iters':>6} {'SA':>8} {'GA':>8} "
          f"{'GA+SA':>8} {'GNN-SA':>8}")
    print("-" * 65)
    for r in results:
        gn_t = r['gnn_t'] if r['gnn_t'] else "N/A"
        print(f"  {r['name']:<10} {r['iters']:>6} {r['sa_t']:>8} "
              f"{r['ga_t']:>8} {r['hybrid_t']:>8} {str(gn_t):>8}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args     = sys.argv[1:]
    run_all  = "--all" in args
    circuits = [a for a in args if not a.startswith("--")]

    if run_all:
        circuits = ALL_CIRCUITS
    elif not circuits:
        circuits = DEFAULT_CIRCUITS

    print("=" * 65)
    print("  CIRCUIT OPTIMIZATION BENCHMARK  (scaled iterations)")
    print("=" * 65)

    try:
        from ml.predictor import GNNPredictor
        predictor = GNNPredictor()
        print("  GNN predictor loaded.")
    except Exception as e:
        print(f"  GNN unavailable ({e}). Skipping GNN-SA.")
        predictor = None

    results = []
    for name in circuits:
        r = benchmark_circuit(name, predictor)
        if r:
            results.append(r)

    if results:
        print_table(results)