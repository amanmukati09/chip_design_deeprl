"""
optimizer/gnn_optimizer.py
─────────────────────────────────────────────────────
GNN-accelerated optimizer.

Wraps Simulated Annealing but replaces the expensive
compute_pac_cost() call with GNNPredictor.predict()
during the inner loop.

Every N steps, it verifies with the real cost function
to stay accurate. This gives SA the speed of GNN
with the accuracy of the real cost function.

Why this approach:
  - Does NOT modify any existing files
  - SA logic stays identical — only the cost call changes
  - GNN at 2.5ms vs real cost at ~50ms = ~20x speedup
  - Verification every 20 steps keeps error bounded

Usage:
    python optimizer/gnn_optimizer.py
"""

import random
import math
import copy
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.cost_function       import compute_pac_cost
from optimizer.simulated_annealing import mutate_swap_gate, mutate_remove_buffer
from core.graph_builder            import build_graph
from core.feature_extractor        import extract_features
from core.circuit import Circuit
from ml.predictor                  import GNNPredictor
from ml.gnn_model                  import sample_to_tensors
import torch


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION FOR GNN
# Converts a Circuit object → node_features + edge_index
# Same format as the training data
# ─────────────────────────────────────────────────────────────

GATE_TYPE_MAP = {
    'NOT': 0, 'BUFF': 1, 'AND': 2, 'OR': 3,
    'NAND': 4, 'NOR': 5, 'XOR': 6, 'XNOR': 7, 'DFF': 8
}

# def graph_to_gnn_input(graph):
#     """
#     Converts a NetworkX graph directly to GNN tensors.
#     Faster than going through Circuit + extract_features.
#     """
#     nodes     = list(graph.nodes())
#     node_idx  = {n: i for i, n in enumerate(nodes)}

#     node_features = []
#     for node in nodes:
#         gate_type = graph.nodes[node].get('gate_type', 'BUFF')
#         type_id   = GATE_TYPE_MAP.get(gate_type, 1)
#         fan_in    = graph.in_degree(node)
#         fan_out   = graph.out_degree(node)
#         node_features.append([type_id, fan_in, fan_out])

#     edge_index = [
#         [node_idx[u], node_idx[v]]
#         for u, v in graph.edges()
#         if u in node_idx and v in node_idx
#     ]

#     return torch.tensor(node_features, dtype=torch.float32), edge_index


def gates_to_gnn_input(gates, inputs):
    """
    Converts gates dict directly to GNN tensors.
    No NetworkX graph needed — 10x faster than build_graph.
    
    gates  : {out_signal: (gate_type, [input_signals])}
    inputs : list of primary input signal names
    """
    # All nodes = primary inputs + gate outputs
    all_nodes = list(inputs) + list(gates.keys())
    node_idx  = {n: i for i, n in enumerate(all_nodes)}

    # Compute fan_out by counting how many gates use each signal
    fan_out = {n: 0 for n in all_nodes}
    for out_signal, (gate_type, gate_inputs) in gates.items():
        for inp in gate_inputs:
            if inp in fan_out:
                fan_out[inp] += 1

    # Build node features and edge index
    node_features = []
    edge_index    = []

    for node in all_nodes:
        if node in gates:
            gate_type, gate_inputs = gates[node]
            type_id = GATE_TYPE_MAP.get(gate_type, 1)
            fan_in  = len(gate_inputs)
            # Add edges
            for inp in gate_inputs:
                if inp in node_idx:
                    edge_index.append([node_idx[inp], node_idx[node]])
        else:
            # Primary input
            type_id = 0
            fan_in  = 0

        node_features.append([type_id, fan_in, fan_out.get(node, 0)])

    return torch.tensor(node_features, dtype=torch.float32), edge_index

# def circuit_to_gnn_input(circuit):
#     """
#     Converts Circuit object to (node_features tensor, edge_index list).
#     Matches the format used in data_collector.py.
#     """
#     graph = circuit.graph
#     nodes = list(graph.nodes())
#     node_idx = {n: i for i, n in enumerate(nodes)}

#     # Node features: [gate_type_id, fan_in, fan_out]
#     node_features = []
#     for node in nodes:
#         gate_type = graph.nodes[node].get('gate_type', 'BUFF')
#         type_id   = GATE_TYPE_MAP.get(gate_type, 1)
#         fan_in    = graph.in_degree(node)
#         fan_out   = graph.out_degree(node)
#         node_features.append([type_id, fan_in, fan_out])

#     # Edge index: list of [src_idx, dst_idx]
#     edge_index = [
#         [node_idx[u], node_idx[v]]
#         for u, v in graph.edges()
#         if u in node_idx and v in node_idx
#     ]

#     node_feat_tensor = torch.tensor(node_features, dtype=torch.float32)
#     return node_feat_tensor, edge_index


# ─────────────────────────────────────────────────────────────
# GNN-ACCELERATED SIMULATED ANNEALING
# ─────────────────────────────────────────────────────────────

def gnn_simulated_annealing(
    circuit,
    predictor,
    initial_temp        = 50.0,
    cooling_rate        = 0.95,
    min_temp            = 0.1,
    iterations_per_temp = 10,
    verify_every        = 20,    # use real cost every N accepted moves
    verbose             = True
):
    """
    Simulated Annealing with GNN as fast cost estimator.

    Args:
        circuit          : Circuit object (with features extracted)
        predictor        : GNNPredictor instance
        initial_temp     : starting temperature
        cooling_rate     : temperature multiplier per step
        min_temp         : stop when temp falls below this
        iterations_per_temp : SA steps per temperature level
        verify_every     : call real cost function every N accepted moves
        verbose          : print progress

    Returns:
        best_gates, best_cost, report_dict
    """
    MUTATIONS = [mutate_swap_gate, mutate_remove_buffer]

    # ── Initial state ─────────────────────────────────────────
    current_gates = copy.deepcopy(circuit.gates)

    # Get initial cost from real function (ground truth baseline)
    current_cost  = circuit.cost if circuit.cost else compute_pac_cost(circuit)

    best_gates    = copy.deepcopy(current_gates)
    best_cost     = current_cost

    temp          = initial_temp
    total_iters   = 0
    accepted      = 0
    gnn_calls     = 0
    real_calls    = 1   # counted the initial one above
    verify_counter = 0

    if verbose:
        print("=" * 55)
        print("  GNN-ACCELERATED SIMULATED ANNEALING")
        print("=" * 55)
        print(f"  Circuit     : {circuit.name}")
        print(f"  Gates       : {circuit.gate_count}")
        print(f"  Start cost  : {round(current_cost, 4)}")
        print(f"  Start temp  : {initial_temp}")
        print(f"  Verify every: {verify_every} accepted moves")
        print("-" * 55)

    # ── SA loop ───────────────────────────────────────────────
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            total_iters += 1

            # Mutate
            mutation_fn  = random.choice(MUTATIONS)
            new_gates    = mutation_fn(current_gates)



            # Build temporary circuit for GNN input
# Fast GNN input — extract features directly from gates
# without building a full Circuit object

            node_feat, edge_index  = gates_to_gnn_input(new_gates, circuit.inputs)
            new_cost               = predictor.predict(node_feat, edge_index)



            gnn_calls += 1

            # ── Accept / reject (SA rule) ─────────────────────
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_gates  = new_gates
                current_cost   = new_cost
                accepted      += 1
                verify_counter += 1

                # ── Periodic verification with real cost ───────
                # if verify_counter >= verify_every:
                #     verify_counter = 0
                #     real_circuit = Circuit(
                #         name    = "verify",
                #         inputs  = circuit.inputs,
                #         outputs = circuit.outputs,
                #         gates   = current_gates,
                #         graph   = build_graph(circuit.inputs,
                #                               circuit.outputs,
                #                               current_gates)
                #     )
                #     real_circuit   = extract_features(real_circuit)
                #     verified_cost  = compute_pac_cost(current_gates, circuit.inputs)['total_cost']                    
                #     real_calls    += 1
                #     # Correct the GNN drift
                #     current_cost   = verified_cost


                if verify_counter >= verify_every:
                    verify_counter = 0
                    verified_cost  = compute_pac_cost(current_gates, circuit.inputs)['total_cost']
                    real_calls    += 1
                    current_cost   = verified_cost

                    # Track best (always update from current_cost)
                    if current_cost < best_cost:
                        best_cost  = current_cost
                        best_gates = copy.deepcopy(current_gates)

        temp *= cooling_rate

    # ── Final verification with real cost function ────────────
    final_circuit = Circuit(
        name    = circuit.name + "_gnn_opt",
        inputs  = circuit.inputs,
        outputs = circuit.outputs,
        gates   = best_gates,
        graph   = build_graph(circuit.inputs, circuit.outputs, best_gates)
    )
    final_circuit = extract_features(final_circuit)
    final_cost    = compute_pac_cost(best_gates, circuit.inputs)['total_cost']    
    real_calls   += 1

    improvement = ((circuit.cost - final_cost) / circuit.cost * 100)

    report = {
        'original_cost' : round(circuit.cost,  4),
        'best_cost'     : round(final_cost,    4),
        'improvement'   : round(improvement,   2),
        'total_iters'   : total_iters,
        'accepted'      : accepted,
        'gnn_calls'     : gnn_calls,
        'real_calls'    : real_calls,
        'gnn_ratio'     : round(gnn_calls / (gnn_calls + real_calls) * 100, 1),
    }

    if verbose:
        print(f"  Iterations  : {total_iters}")
        print(f"  Accepted    : {accepted}")
        print(f"  GNN calls   : {gnn_calls}  ({report['gnn_ratio']}% of evaluations)")
        print(f"  Real calls  : {real_calls}")
        print(f"  Final cost  : {round(final_cost, 4)}")
        print(f"  Improvement : {round(improvement, 2)}%")
        print("=" * 55)

    return best_gates, final_cost, report


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.pipeline import load_circuit

    print("Loading predictor...")
    predictor = GNNPredictor()

    print("Loading circuit...")
    circuit, _ = load_circuit("data/benchmarks/s1196.bench")

    print("\nRunning standard SA (baseline)...")
    from optimizer.simulated_annealing import simulated_annealing
    t0 = time.perf_counter()
    _, sa_cost, _ = simulated_annealing(
        circuit,
        initial_temp        = 50.0,
        cooling_rate        = 0.95,
        min_temp            = 0.1,
        iterations_per_temp = 10,
        verbose             = False
    )
    sa_time = time.perf_counter() - t0
    sa_improvement = (circuit.cost - sa_cost) / circuit.cost * 100

    print(f"  Standard SA  : cost={round(sa_cost,4)}  "
          f"improvement={round(sa_improvement,2)}%  "
          f"time={round(sa_time,2)}s")

    print("\nRunning GNN-accelerated SA...")
    t0 = time.perf_counter()
    _, gnn_cost, report = gnn_simulated_annealing(
        circuit,
        predictor,
        initial_temp        = 50.0,
        cooling_rate        = 0.95,
        min_temp            = 0.1,
        iterations_per_temp = 10,
        verify_every        = 20,
        verbose             = True
    )
    gnn_time = time.perf_counter() - t0
    gnn_improvement = (circuit.cost - gnn_cost) / circuit.cost * 100

    print(f"  GNN SA       : cost={round(gnn_cost,4)}  "
          f"improvement={round(gnn_improvement,2)}%  "
          f"time={round(gnn_time,2)}s")

    print()
    print("=" * 55)
    print("  COMPARISON")
    print("=" * 55)
    print(f"  Original cost    : {circuit.cost}")
    print(f"  Standard SA      : {round(sa_cost,4)}  ({round(sa_improvement,2)}%)")
    print(f"  GNN-accel SA     : {round(gnn_cost,4)}  ({round(gnn_improvement,2)}%)")
    print(f"  Standard SA time : {round(sa_time,2)}s")
    print(f"  GNN SA time      : {round(gnn_time,2)}s")
    print(f"  GNN evaluations  : {report['gnn_ratio']}% of all cost calls")
    print("=" * 55)