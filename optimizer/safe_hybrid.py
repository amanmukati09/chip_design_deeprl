# optimizer/safe_hybrid.py
import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizer.safe_genetic_algorithm import safe_genetic_algorithm
from optimizer.safe_simulated_annealing import safe_simulated_annealing

def safe_hybrid_optimizer(circuit, verbose=True):
    """
    The Memetic Hybrid: GA finds the global region, SA finds the local peak.
    """
    if verbose:
        print("\n" + "="*60)
        print("  STARTING HYBRID OPTIMIZATION: GA + SA")
        print("="*60)

    # 1. PHASE 1: GLOBAL EXPLORATION (GA)
    # We run GA for a moderate number of generations to find a strong topology.
    print("\n[Phase 1] Running Genetic Algorithm for Global Exploration...")
    ga_gates, ga_cost, _ = safe_genetic_algorithm(
        circuit, 
        pop_size=30, 
        generations=40, 
        verbose=False
    )
    
    ga_improv = ((circuit.cost - ga_cost) / circuit.cost) * 100
    if verbose:
        print(f"GA Phase Completed. Cost: {ga_cost:.4f} ({ga_improv:.2f}% improvement)")

    # 2. HANDOFF
    # Create a temporary circuit object to pass to the SA
    temp_circuit = copy.deepcopy(circuit)
    temp_circuit.gates = ga_gates
    temp_circuit.cost = ga_cost

    # 3. PHASE 2: LOCAL POLISHING (SA)
    # We use a lower initial temperature because the circuit is already quite optimized.
    print("\n[Phase 2] Running Simulated Annealing for Local Polishing...")
    final_gates, final_cost, _ = safe_simulated_annealing(
        temp_circuit,
        initial_temp=30.0,  # Lower starting temp
        cooling_rate=0.95,
        iterations_per_temp=30,
        verbose=False
    )

    final_improv = ((circuit.cost - final_cost) / circuit.cost) * 100
    if verbose:
        print(f"SA Polishing Completed. Final Cost: {final_cost:.4f}")
        print(f"Total Hybrid Improvement: {final_improv:.2f}%")
        print("="*60)

    return final_gates, final_cost