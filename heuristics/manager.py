"""
heuristics/manager.py
─────────────────────────────────────────────────────
Complexity Router — the novel contribution of this system.

Automatically selects the best optimizer based on circuit size,
based on empirical benchmark results across 15 MCNC circuits:

  < 500 gates  → GA+SA hybrid
                 GNN generalizes poorly to small circuits
                 GA+SA wins consistently in this range

  ≥ 500 gates  → GNN-accelerated SA
                 GNN-SA wins 11/15 circuits tested
                 Best result: 21.75% on c2670 (1193 gates)

Iteration scaling:
  iterations_per_temp = max(10, gate_count // 50)
  Ensures SA explores the space proportionally to circuit size.
  Without this, large circuits are under-explored.

Usage:
    from heuristics.manager import optimize
    result = optimize(circuit)
    print(result)

    # or force a specific optimizer:
    result = optimize(circuit, optimizer="sa")
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline_v1                 import load_circuit
from optimizer.simulated_annealing import simulated_annealing
from optimizer.genetic_algorithm   import genetic_algorithm
from optimizer.hybrid_optimizer    import hybrid_optimize
from optimizer.gnn_optimizer       import gnn_simulated_annealing


# ─────────────────────────────────────────────────────────────
# THRESHOLDS  (empirically determined from benchmark results)
# ─────────────────────────────────────────────────────────────

SMALL_CIRCUIT_THRESHOLD = 500   # gates
LARGE_CIRCUIT_THRESHOLD = 10000 # beyond this, scale iterations only

# GNN predictor — loaded once, reused across all calls
_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    try:
        from ml.predictor import GNNPredictor
        _predictor = GNNPredictor()
        return _predictor
    except Exception as e:
        print(f"[Manager] GNN unavailable: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# ITERATION SCALING
# ─────────────────────────────────────────────────────────────

def scale_iterations(gate_count: int) -> int:
    """
    Scale SA iterations proportionally to circuit size.

    Without scaling, SA runs the same 1220 iterations
    whether the circuit has 100 gates or 20000 gates.
    A 20000-gate circuit needs ~37x more iterations
    to achieve equivalent exploration coverage.

    Capped at 200 to keep runtime reasonable on CPU.
    """
    return min(200, max(10, gate_count // 50))


# ─────────────────────────────────────────────────────────────
# OPTIMIZER CONFIGS  (built per circuit, not hardcoded)
# ─────────────────────────────────────────────────────────────

def build_configs(gate_count: int) -> dict:
    iters = scale_iterations(gate_count)
    return {
        "sa": dict(
            initial_temp        = 50.0,
            cooling_rate        = 0.95,
            min_temp            = 0.1,
            iterations_per_temp = iters,
            verbose             = False,
        ),
        "ga": dict(
            population_size = 20,
            generations     = 50,
            survival_rate   = 0.3,
            mutation_rate   = 0.7,
            verbose         = False,
        ),
        "hybrid": dict(
            ga_population   = 20,
            ga_generations  = 50,
            ga_survival     = 0.3,
            ga_mutation     = 0.7,
            sa_initial_temp = 50.0,
            sa_cooling      = 0.95,
            sa_min_temp     = 0.1,
            sa_iterations   = iters,
            verbose         = False,
        ),
        "gnn_sa": dict(
            initial_temp        = 50.0,
            cooling_rate        = 0.95,
            min_temp            = 0.1,
            iterations_per_temp = iters,
            verify_every        = 20,
            verbose             = False,
        ),
    }


# ─────────────────────────────────────────────────────────────
# ROUTER LOGIC
# ─────────────────────────────────────────────────────────────

def select_optimizer(gate_count: int, force: str = None) -> str:
    """
    Selects best optimizer based on circuit size.

    Args:
        gate_count : number of gates in the circuit
        force      : override selection ('sa','ga','hybrid','gnn_sa')

    Returns:
        optimizer name as string
    """
    if force and force in ("sa", "ga", "hybrid", "gnn_sa"):
        return force

    predictor = _get_predictor()

    if gate_count < SMALL_CIRCUIT_THRESHOLD:
        return "hybrid"        # GA+SA wins on small circuits
    elif predictor is not None:
        return "gnn_sa"        # GNN-SA wins on medium/large
    else:
        return "hybrid"        # fallback if GNN unavailable


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def optimize(circuit, force_optimizer: str = None, verbose: bool = True):
    """
    Optimizes a circuit using the automatically selected optimizer.

    Args:
        circuit          : Circuit object (from load_circuit)
        force_optimizer  : optional override ('sa','ga','hybrid','gnn_sa')
        verbose          : print progress

    Returns:
        dict with full optimization report
    """
    gate_count = circuit.gate_count
    optimizer  = select_optimizer(gate_count, force_optimizer)
    configs    = build_configs(gate_count)
    iters      = scale_iterations(gate_count)

    if verbose:
        print("=" * 55)
        print("  COMPLEXITY ROUTER")
        print("=" * 55)
        print(f"  Circuit     : {circuit.name}")
        print(f"  Gates       : {gate_count}")
        print(f"  Original    : {round(circuit.cost, 4)}")
        print(f"  Selected    : {optimizer.upper()}")
        print(f"  Iterations  : {iters} per temp step")
        print("-" * 55)

    t0 = time.perf_counter()

    if optimizer == "sa":
        _, cost, report = simulated_annealing(circuit, **configs["sa"])

    elif optimizer == "ga":
        _, cost, report = genetic_algorithm(circuit, **configs["ga"])

    elif optimizer == "hybrid":
        _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])

    elif optimizer == "gnn_sa":
        predictor = _get_predictor()
        if predictor is None:
            # Fallback to hybrid if GNN fails at runtime
            if verbose:
                print("  [Fallback] GNN unavailable, using hybrid.")
            optimizer = "hybrid"
            _, cost, report = hybrid_optimize(circuit, **configs["hybrid"])
        else:
            _, cost, report = gnn_simulated_annealing(
                circuit, predictor, **configs["gnn_sa"]
            )

    elapsed     = round(time.perf_counter() - t0, 3)
    improvement = round((circuit.cost - cost) / circuit.cost * 100, 4)

    result = {
        "circuit_name"   : circuit.name,
        "gate_count"     : gate_count,
        "original_cost"  : round(circuit.cost, 4),
        "optimized_cost" : round(cost, 4),
        "improvement_pct": improvement,
        "optimizer_used" : optimizer,
        "iterations_used": iters,
        "time_seconds"   : elapsed,
        "detail"         : report,
    }

    if verbose:
        print(f"  Optimized   : {round(cost, 4)}")
        print(f"  Improvement : {improvement}%")
        print(f"  Time        : {elapsed}s")
        print("=" * 55)

    return result


def optimize_file(filepath: str, force_optimizer: str = None, verbose: bool = True):
    """Convenience wrapper — takes a file path instead of Circuit object."""
    circuit, _ = load_circuit(filepath)
    return optimize(circuit, force_optimizer, verbose)


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\nTest 1 — Small circuit (c17, 6 gates) → expect hybrid")
    result = optimize_file("data/benchmarks/c17.bench")
    print(f"  Used: {result['optimizer_used']}  "
          f"Improvement: {result['improvement_pct']}%\n")

    print("Test 2 — Medium circuit (s1196, 547 gates) → expect gnn_sa")
    result = optimize_file("data/benchmarks/s1196.bench")
    print(f"  Used: {result['optimizer_used']}  "
          f"Improvement: {result['improvement_pct']}%\n")

    print("Test 3 — Large circuit (c2670, 1193 gates) → expect gnn_sa")
    result = optimize_file("data/benchmarks/c2670.bench")
    print(f"  Used: {result['optimizer_used']}  "
          f"Improvement: {result['improvement_pct']}%\n")

    print("Test 4 — Force SA on s1196")
    result = optimize_file("data/benchmarks/s1196.bench",
                           force_optimizer="sa")
    print(f"  Used: {result['optimizer_used']}  "
          f"Improvement: {result['improvement_pct']}%\n")