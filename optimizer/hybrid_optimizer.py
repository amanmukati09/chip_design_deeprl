# optimizer/hybrid_optimizer.py
# Hybrid GA + SA optimizer
# GA explores globally, SA refines locally
# This is the core of your original paper's approach

import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.genetic_algorithm   import genetic_algorithm
from optimizer.simulated_annealing import simulated_annealing
from optimizer.cost_function       import compute_pac_cost
from core.circuit                  import Circuit
from core.graph_builder            import build_graph
from core.feature_extractor        import extract_features


def hybrid_optimize(circuit,
                    # GA settings
                    ga_population   = 20,
                    ga_generations  = 50,
                    ga_survival     = 0.3,
                    ga_mutation     = 0.3,
                    # SA settings
                    sa_initial_temp = 50.0,
                    sa_cooling      = 0.95,
                    sa_min_temp     = 0.1,
                    sa_iterations   = 10,
                    verbose         = True):
    """
    Hybrid GA + SA optimizer.

    Pipeline:
        1. GA explores the design space globally
           Finds a good solution in N generations
        2. SA refines that solution locally
           Fine-tunes from where GA left off
        3. Return best result from combined run

    This matches your paper's DeepRL + Multiple HA approach.
    GA plays the role of global explorer.
    SA plays the role of local refiner.
    Later RL will replace GA as the global explorer.

    Args:
        circuit       : Circuit object from pipeline
        ga_*          : Genetic Algorithm settings
        sa_*          : Simulated Annealing settings
        verbose       : print progress

    Returns:
        best_gates    : optimized gates dict
        best_cost     : lowest cost found
        report        : full optimization report dict
    """

    original_cost = circuit.cost

    if verbose:
        print()
        print("=" * 55)
        print("  HYBRID OPTIMIZER  (GA → SA)")
        print("=" * 55)
        print(f"  Circuit      : {circuit.name}")
        print(f"  Gates        : {circuit.gate_count}")
        print(f"  Original cost: {original_cost}")
        print("=" * 55)

    # ── Phase 1: Genetic Algorithm ───────────────────
    if verbose:
        print("\n[Phase 1] Running Genetic Algorithm...")

    ga_gates, ga_cost, ga_history = genetic_algorithm(
        circuit,
        population_size = ga_population,
        generations     = ga_generations,
        survival_rate   = ga_survival,
        mutation_rate   = ga_mutation,
        verbose         = verbose
    )

    ga_improvement = ((original_cost - ga_cost)
                      / original_cost * 100)

    if verbose:
        print(f"\n[Phase 1] GA complete.")
        print(f"  GA best cost : {ga_cost}")
        print(f"  GA improvement: {round(ga_improvement, 2)}%")

    # ── Phase 2: Simulated Annealing ─────────────────
    # Build a new Circuit object from GA's best result
    # so SA can refine it
    if verbose:
        print(f"\n[Phase 2] Running Simulated Annealing "
              f"on GA's best result...")

    # Create temporary circuit from GA output
    ga_circuit = Circuit(
        name    = circuit.name + "_ga",
        inputs  = circuit.inputs,
        outputs = circuit.outputs,
        gates   = ga_gates,
        graph   = build_graph(circuit.inputs,
                              circuit.outputs,
                              ga_gates)
    )
    ga_circuit.cost = ga_cost
    ga_circuit      = extract_features(ga_circuit)

    sa_gates, sa_cost, sa_history = simulated_annealing(
        ga_circuit,
        initial_temp        = sa_initial_temp,
        cooling_rate        = sa_cooling,
        min_temp            = sa_min_temp,
        iterations_per_temp = sa_iterations,
        verbose             = verbose
    )

    sa_improvement = ((original_cost - sa_cost)
                      / original_cost * 100)

    # ── Final Result ──────────────────────────────────
    best_gates = sa_gates
    best_cost  = sa_cost

    total_improvement = ((original_cost - best_cost)
                         / original_cost * 100)

    report = {
        'circuit_name'      : circuit.name,
        'gate_count'        : circuit.gate_count,
        'original_cost'     : original_cost,
        'ga_cost'           : round(ga_cost, 4),
        'sa_cost'           : round(sa_cost, 4),
        'best_cost'         : round(best_cost, 4),
        'ga_improvement'    : round(ga_improvement, 2),
        'sa_improvement'    : round(sa_improvement, 2),
        'total_improvement' : round(total_improvement, 2),
        'ga_generations'    : ga_generations,
        'ga_population'     : ga_population,
        'sa_initial_temp'   : sa_initial_temp,
    }

    if verbose:
        print()
        print("=" * 55)
        print("  HYBRID OPTIMIZATION COMPLETE")
        print("=" * 55)
        print(f"  Original cost     : {original_cost}")
        print(f"  After GA          : {round(ga_cost, 4)}"
              f"  ({round(ga_improvement, 2)}% better)")
        print(f"  After SA refinement: {round(sa_cost, 4)}"
              f"  ({round(sa_improvement, 2)}% better)")
        print(f"  Total improvement : {round(total_improvement, 2)}%")
        print("=" * 55)

    return best_gates, best_cost, report


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from pipeline_v1 import load_circuit

    # Test on s1196 — same circuit GA tested on
    circuit, _ = load_circuit("data/benchmarks/s1196.bench")

    best_gates, best_cost, report = hybrid_optimize(
        circuit,
        ga_population  = 20,
        ga_generations = 50,
        sa_initial_temp= 50.0,
        sa_cooling     = 0.95,
        verbose        = True
    )

    print("\nFINAL REPORT:")
    for k, v in report.items():
        print(f"  {k:25s} : {v}")