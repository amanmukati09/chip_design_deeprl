# ml/data_collector.py
# Collects circuit + cost pairs during optimization runs
# This becomes our GNN training data
#
# Every time optimizer evaluates a circuit variation,
# we save:
#   - graph structure (nodes + edges)
#   - node features (gate type, fan-in, fan-out)
#   - PAC cost (label for training)

import os
import json
import copy
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.cost_function  import compute_pac_cost
from core.graph_builder       import build_graph
from core.feature_extractor   import extract_features
from core.circuit             import Circuit

# ─────────────────────────────────────────────────────────────
# GATE TYPE ENCODING
# Convert gate type string to number for ML
# ─────────────────────────────────────────────────────────────
GATE_TYPE_MAP = {
    'INPUT' : 0,
    'OUTPUT': 1,
    'NOT'   : 2,
    'BUFF'  : 3,
    'AND'   : 4,
    'OR'    : 5,
    'NAND'  : 6,
    'NOR'   : 7,
    'XOR'   : 8,
    'XNOR'  : 9,
    'DFF'   : 10,
    'UNKNOWN': 11,
}


def gates_to_graph_data(gates, inputs, outputs):
    """
    Converts gates dict into a format suitable for GNN training.

    Returns:
        node_features : list of [gate_type_id, fan_in, fan_out]
        edge_index    : list of [src, dst] pairs
        node_names    : list of signal names (for reference)
    """
    # Build graph to get structural info
    G = build_graph(inputs, outputs, gates)

    # Create node list (consistent ordering)
    node_names = list(G.nodes())
    node_idx   = {name: i for i, name in enumerate(node_names)}

    # Build node features
    node_features = []
    for node in node_names:
        data    = G.nodes[node]
        ntype   = data.get('node_type', 'gate')
        gtype   = data.get('gate_type', 'UNKNOWN')

        if ntype == 'input':
            type_id = GATE_TYPE_MAP['INPUT']
        elif ntype == 'output':
            type_id = GATE_TYPE_MAP.get(gtype, GATE_TYPE_MAP['OUTPUT'])
        else:
            type_id = GATE_TYPE_MAP.get(gtype, GATE_TYPE_MAP['UNKNOWN'])

        fan_in  = G.in_degree(node)
        fan_out = G.out_degree(node)

        node_features.append([type_id, fan_in, fan_out])

    # Build edge index
    edge_index = []
    for src, dst in G.edges():
        src_idx = node_idx[src]
        dst_idx = node_idx[dst]
        edge_index.append([src_idx, dst_idx])

    return node_features, edge_index, node_names


def collect_sample(gates, inputs, outputs):
    """
    Creates one training sample from a circuit variation.

    Returns:
        sample : dict with graph data + cost label
    """
    # Compute cost (ground truth label)
    cost_dict = compute_pac_cost(gates, inputs)

    # Convert to graph data
    node_features, edge_index, node_names = gates_to_graph_data(
        gates, inputs, outputs
    )

    sample = {
        'node_features' : node_features,
        'edge_index'    : edge_index,
        'node_count'    : len(node_names),
        'edge_count'    : len(edge_index),
        'cost'          : cost_dict['total_cost'],
        'power'         : cost_dict['power'],
        'area'          : cost_dict['area'],
        'wirelength'    : cost_dict['wirelength'],
    }

    return sample


class DataCollector:
    """
    Wraps optimization runs and collects training data.

    Usage:
        collector = DataCollector()
        collector.record(gates, inputs, outputs)
        collector.save("ml/data/s1196_samples.json")
    """

    def __init__(self, max_samples=5000):
        self.samples     = []
        self.max_samples = max_samples

    def record(self, gates, inputs, outputs):
        """Records one circuit variation as a training sample."""
        if len(self.samples) >= self.max_samples:
            return  # don't exceed memory limit

        sample = collect_sample(gates, inputs, outputs)
        self.samples.append(sample)

    def save(self, filepath):
        """Saves all collected samples to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.samples, f)

        print(f"[DataCollector] Saved {len(self.samples)} "
              f"samples to {filepath}")

    def load(self, filepath):
        """Loads samples from JSON file."""
        with open(filepath, 'r') as f:
            self.samples = json.load(f)

        print(f"[DataCollector] Loaded {len(self.samples)} "
              f"samples from {filepath}")

    @property
    def size(self):
        return len(self.samples)


def generate_training_data(circuit,
                            n_variations=500,
                            save_path=None,
                            verbose=True):
    """
    Generates training data by running hybrid optimizer
    and collecting all circuit variations evaluated.

    This is how we solve the 'no training data' problem.
    Every mutation the optimizer tries becomes a data point.

    Args:
        circuit     : Circuit object
        n_variations: how many variations to generate
        save_path   : where to save (optional)
        verbose     : print progress

    Returns:
        collector   : DataCollector with all samples
    """
    from optimizer.simulated_annealing import apply_random_mutation

    collector = DataCollector(max_samples=n_variations)

    if verbose:
        print(f"\n[DataCollector] Generating {n_variations} "
              f"training samples from {circuit.name}...")

    # Always record original circuit first
    collector.record(circuit.gates, circuit.inputs, circuit.outputs)

    current_gates = copy.deepcopy(circuit.gates)

    for i in range(n_variations - 1):
        # Apply random mutation
        new_gates = apply_random_mutation(
            current_gates, inputs=circuit.inputs
        )

        # Record this variation
        collector.record(new_gates, circuit.inputs, circuit.outputs)

        # Occasionally reset to original to explore diverse space
        if i % 50 == 0:
            current_gates = copy.deepcopy(circuit.gates)
        else:
            current_gates = new_gates

        if verbose and i % 100 == 0:
            print(f"  Generated {i+1}/{n_variations} samples...")

    if verbose:
        print(f"  Done. Total samples: {collector.size}")

    if save_path:
        collector.save(save_path)

    return collector


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from core.pipeline import load_circuit

    # Generate training data from s1196
    circuit, _ = load_circuit("data/benchmarks/s1196.bench")

    collector = generate_training_data(
        circuit,
        n_variations = 500,
        save_path    = "ml/data/s1196_samples.json",
        verbose      = True
    )

    print(f"\nSample 0 (original circuit):")
    s = collector.samples[0]
    print(f"  Nodes     : {s['node_count']}")
    print(f"  Edges     : {s['edge_count']}")
    print(f"  Cost      : {s['cost']}")
    print(f"  Power     : {s['power']}")
    print(f"  Area      : {s['area']}")

    print(f"\nSample 1 (first mutation):")
    s = collector.samples[1]
    print(f"  Nodes     : {s['node_count']}")
    print(f"  Edges     : {s['edge_count']}")
    print(f"  Cost      : {s['cost']}")

    print(f"\nData range:")
    costs = [s['cost'] for s in collector.samples]
    print(f"  Min cost  : {round(min(costs), 4)}")
    print(f"  Max cost  : {round(max(costs), 4)}")
    print(f"  Avg cost  : {round(sum(costs)/len(costs), 4)}")