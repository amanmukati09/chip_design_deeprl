# core/feature_extractor.py
# Computes graph-level features from a Circuit object

import networkx as nx

def extract_features(circuit):
    """
    Computes and stores features directly into the Circuit object.
    Features computed:
        - gate_count
        - depth (longest path)
        - avg_fan_in
        - avg_fan_out
        - max_fan_in
        - max_fan_out
        - input_count
        - output_count
        - edge_count
        - gate_types
    """
    G = circuit.graph

    # ── Fan-in & Fan-out ────────────────────────────
    fan_in  = {}
    fan_out = {}

    for node in G.nodes():
        fan_in[node]  = G.in_degree(node)
        fan_out[node] = G.out_degree(node)

    circuit.fan_in  = fan_in
    circuit.fan_out = fan_out

    # ── Depth (longest path input → output) ─────────
    try:
        circuit.depth = nx.dag_longest_path_length(G)
    except nx.NetworkXUnfeasible:
        circuit.depth = -1

    # ── Gate-level stats ────────────────────────────
    gate_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get('node_type') == 'gate'
    ]

    fan_in_vals  = [fan_in[n]  for n in gate_nodes]
    fan_out_vals = [fan_out[n] for n in gate_nodes]

    avg_fan_in  = sum(fan_in_vals)  / len(fan_in_vals)  if fan_in_vals  else 0
    avg_fan_out = sum(fan_out_vals) / len(fan_out_vals) if fan_out_vals else 0
    max_fan_in  = max(fan_in_vals)  if fan_in_vals  else 0
    max_fan_out = max(fan_out_vals) if fan_out_vals else 0

    # ── Gate type distribution ───────────────────────
    gate_types = {}
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'gate':
            gtype = data.get('gate_type', 'UNKNOWN')
            gate_types[gtype] = gate_types.get(gtype, 0) + 1

    # ── Store all features ───────────────────────────
    circuit.features = {
        'gate_count'   : circuit.gate_count,
        'depth'        : circuit.depth,
        'input_count'  : len(circuit.inputs),
        'output_count' : len(circuit.outputs),
        'avg_fan_in'   : round(avg_fan_in,  2),
        'avg_fan_out'  : round(avg_fan_out, 2),
        'max_fan_in'   : max_fan_in,
        'max_fan_out'  : max_fan_out,
        'gate_types'   : gate_types,
        'edge_count'   : G.number_of_edges(),
    }

    return circuit