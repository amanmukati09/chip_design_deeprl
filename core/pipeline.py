"""
core/pipeline.py  (Version 2 — multi-format support)
─────────────────────────────────────────────────────
Same interface as before. Only change:
  parse_bench_file() → parse_circuit() from parser_factory

Supports .bench, .isc, and any future formats
without changing any optimizer or API code.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.parsers.parser_factory import parse_circuit
from core.graph_builder          import build_graph
from core.circuit                import Circuit
from core.feature_extractor      import extract_features
from optimizer.cost_function     import compute_pac_cost


def load_circuit(filepath: str, verbose: bool = True):
    """
    Full pipeline: file → Circuit object with cost computed.

    Accepts any supported format: .bench, .isc, .v (when added)

    Args:
        filepath : path to circuit file
        verbose  : print progress steps

    Returns:
        circuit  : Circuit object, fully populated
        report   : dict with basic circuit stats
    """
    name = os.path.splitext(os.path.basename(filepath))[0]

    if verbose:
        print(f"\n[Pipeline] Loading circuit: {name}")
        print(f"[Pipeline] File: {filepath}")

    # Step 1 — Parse (format auto-detected)
    if verbose:
        print(f"[Pipeline] Step 1/4 — Parsing netlist...")
    inputs, outputs, gates, name = parse_circuit(filepath)

    # Step 2 — Build graph
    if verbose:
        print(f"[Pipeline] Step 2/4 — Building graph...")
    graph = build_graph(inputs, outputs, gates)

    # Step 3 — Create circuit object
    if verbose:
        print(f"[Pipeline] Step 3/4 — Creating circuit object...")
    circuit = Circuit(
        name    = name,
        inputs  = inputs,
        outputs = outputs,
        gates   = gates,
        graph   = graph,
    )

    # Step 4 — Extract features + compute cost
    if verbose:
        print(f"[Pipeline] Step 4/4 — Extracting features...")
    circuit = extract_features(circuit)
    cost_dict = compute_pac_cost(gates, inputs)
    circuit.cost = cost_dict['total_cost']

    if verbose:
        print(f"[Pipeline] Done. Cost: {round(circuit.cost, 4)}")

    report = {
        'name'      : name,
        'inputs'    : len(inputs),
        'outputs'   : len(outputs),
        'gates'     : len(gates),
        'cost'      : circuit.cost,
        'format'    : os.path.splitext(filepath)[1].lower(),
    }

    return circuit, report


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_files = [
        "data/benchmarks/c1355.bench",
        "data/benchmarks/c1355.isc",
    ]

    for f in test_files:
        if not os.path.exists(f):
            print(f"[SKIP] {f}")
            continue
        circuit, report = load_circuit(f)
        print(f"  Gates: {report['gates']}  "
              f"Cost: {round(report['cost'],4)}  "
              f"Format: {report['format']}\n")