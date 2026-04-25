# core/test_safe_pipeline.py
import sys
import os

# Go up one level from 'core' to the project root so Python can see the 'optimizer' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizer.safe_simulated_annealing import safe_simulated_annealing
from optimizer.bench_writer import write_bench

# Dummy Circuit Class to simulate your environment for the test
class DummyCircuit:
    def __init__(self):
        self.name = "test_circuit"
        self.inputs = ['in1', 'in2', 'in3']
        self.outputs = ['out1']
        # Initial gates with an obvious redundancy: out1 = AND(in1, in1)
        self.gates = {
            'n1': ('AND', ['in1', 'in1']), 
            'out1': ('NOT', ['n1'])        
        }
        self.cost = 15.0 # Arbitrary starting cost

def run_test():
    print("--- Starting Safe Optimization Pipeline Test ---")
    circuit = DummyCircuit()
    
    print(f"Original Gates: {circuit.gates}")
    
    # 1. Run the safe optimizer
    # We use low iterations just to verify the plumbing works
    best_gates, best_cost, history = safe_simulated_annealing(
        circuit, 
        initial_temp=10.0, 
        cooling_rate=0.8, 
        min_temp=1.0, 
        iterations_per_temp=5
    )
    
    print(f"\nOptimized Gates: {best_gates}")
    print(f"Cost changed from {circuit.cost} to {best_cost}")
    
    # 2. Write the output
    # Saving it one level up so it doesn't clutter your core folder
    output_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'output', 'test_optimized.bench')
    write_bench(circuit, best_gates, output_path)
    
    # 3. Verify file exists
    if os.path.exists(output_path):
        print(f"\nSUCCESS: Output verified at {os.path.abspath(output_path)}")
    else:
        print("\nFAILURE: Bench file was not created.")

if __name__ == "__main__":
    run_test()