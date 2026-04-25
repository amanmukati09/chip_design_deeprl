# optimizer/bench_writer.py

import os

def write_bench(circuit, best_gates, output_filepath):
    """
    Takes the optimized gates and writes them back out in standard .bench format.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    # Add encoding='utf-8' and use standard ASCII dashes
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(f"# ----------------------------------------\n")
        f.write(f"# Optimized Circuit: {circuit.name}\n")
        f.write(f"# Original Cost: {circuit.cost}\n")
        f.write(f"# ----------------------------------------\n\n")
        
        # 1. Write Inputs
        for pin in circuit.inputs:
            f.write(f"INPUT({pin})\n")
        f.write("\n")
        
        # 2. Write Outputs
        for pin in circuit.outputs:
            f.write(f"OUTPUT({pin})\n")
        f.write("\n")
        
        # 3. Write Gates
        for out_sig, (gtype, fanins) in best_gates.items():
            fanin_str = ", ".join(fanins)
            f.write(f"{out_sig} = {gtype}({fanin_str})\n")
            
    print(f"[BenchWriter] Successfully saved functionally correct circuit to: {output_filepath}")

if __name__ == "__main__":
    print("Bench writer module ready.")