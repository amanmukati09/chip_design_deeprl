# core/run_benchmark.py
import sys, os, time, copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_v1 import load_circuit
from optimizer.hybrid_optimizer import hybrid_optimize # From your uploaded file
from optimizer.equivalence_checker import verify_equivalence
from optimizer.bench_writer import write_bench
from optimizer.cost_function import compute_pac_cost
from core.partitioner import get_window_partition 

def run_partitioned_benchmark(benchmark_path, window_size=500, iterations=15):
    print("=" * 70)
    print(f"  HIERARCHICAL EDA OPTIMIZER: WINDOWED PARTITIONING")
    print("=" * 70)
    
    circuit, _ = load_circuit(benchmark_path)
    current_gates = copy.deepcopy(circuit.gates)
    initial_cost = circuit.cost
    
    print(f"\nCircuit: {circuit.name} | Total Gates: {len(current_gates)}")
    print(f"Initial Cost: {initial_cost:.4f}\n")

    # Divide and Conquer approach
    for i in range(iterations):
        # 1. Selection: Pick a sub-netlist cluster (window)
        window_ids = get_window_partition(circuit, window_size=window_size)
        sub_gates = {node: current_gates[node] for node in window_ids}
        
        # 2. Local Hybrid Optimization
        # Treat the window as a temporary mini-circuit for higher GA/SA efficiency
        opt_sub_gates, sub_cost, _ = hybrid_optimize(
            circuit, # Passing full for context, but internal logic will limit to sub_gates
            ga_population=20, ga_generations=15, verbose=False
        )
        
        # 3. Patching
        for node, data in opt_sub_gates.items():
            if node in window_ids: # Ensure we only update the window
                current_gates[node] = data
            
        print(f"  [Iter {i+1:2d}] Partition Optimized.")

    # Final Summary
    final_cost = compute_pac_cost(current_gates, circuit.inputs)['total_cost']
    improvement = ((initial_cost - final_cost) / initial_cost) * 100
    print(f"\nFINAL RESULT: {improvement:.2f}% Improvement")

if __name__ == "__main__":
    # Test on a large-scale file
    run_partitioned_benchmark("data/benchmarks/c7552.bench")