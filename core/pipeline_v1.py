# core/pipeline.py
# Connects everything into one clean pipeline
# Parser → Graph → Circuit → Features → Cost

import sys
import os

# Make sure all modules are found
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.netlist_parser    import parse_bench
from core.graph_builder     import build_graph
from core.circuit           import Circuit
from core.feature_extractor import extract_features
from optimizer.cost_function import compute_pac_cost, print_cost_report


def load_circuit(filepath, name=None):
    """
    Full pipeline: .bench file → ready Circuit object with cost.

    This is the single entry point for the entire system.
    Every other module (optimizer, GNN, API) calls this.

    Args:
        filepath : path to .bench file
        name     : circuit name (optional, inferred from filename)

    Returns:
        circuit  : fully populated Circuit object
    """

    # Infer name from filename if not given
    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]

    print(f"\n[Pipeline] Loading circuit: {name}")
    print(f"[Pipeline] File: {filepath}")

    # ── Step 1: Parse ────────────────────────────
    print("[Pipeline] Step 1/4 — Parsing netlist...")
    inputs, outputs, gates = parse_bench(filepath)

    # ── Step 2: Build Graph ───────────────────────
    print("[Pipeline] Step 2/4 — Building graph...")
    graph = build_graph(inputs, outputs, gates)

    # ── Step 3: Create Circuit Object ────────────
    print("[Pipeline] Step 3/4 — Creating circuit object...")
    circuit         = Circuit(name, inputs, outputs, gates, graph)

    # ── Step 4: Extract Features ──────────────────
    print("[Pipeline] Step 4/4 — Extracting features...")
    circuit         = extract_features(circuit)

    # ── Step 5: Compute PAC Cost ──────────────────
    cost_dict       = compute_pac_cost(gates, inputs)
    circuit.cost    = cost_dict['total_cost']
    circuit.features.update({
        'power'      : cost_dict['power'],
        'area'       : cost_dict['area'],
        'wirelength' : cost_dict['wirelength'],
    })

    print(f"[Pipeline] Done. Cost: {circuit.cost}")
    return circuit, cost_dict


def run_pipeline(filepath, verbose=True):
    """
    Runs full pipeline and prints results.
    Use this for quick testing.
    """
    circuit, cost_dict = load_circuit(filepath)

    if verbose:
        circuit.summary()
        print_cost_report(cost_dict, circuit.name)

    return circuit, cost_dict


# ── Quick test ───────────────────────────────────
if __name__ == "__main__":

    # Test on c17
    print("=" * 50)
    print("  FULL PIPELINE TEST")
    print("=" * 50)

    circuit, cost = run_pipeline("data/benchmarks/c17.bench")

    print("\nKey facts about this circuit:")
    print(f"  Name       : {circuit.name}")
    print(f"  Gates      : {circuit.gate_count}")
    print(f"  Depth      : {circuit.depth}")
    print(f"  Max fan-in : {circuit.features['max_fan_in']}")
    print(f"  Gate types : {circuit.features['gate_types']}")
    print(f"  PAC Cost   : {circuit.cost}")
    print()
    print("This circuit is ready for optimization.")