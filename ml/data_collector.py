# ml/data_collector.py
# Collects circuit + cost pairs during optimization runs
# This becomes our GNN training data

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
    """Converts gates dict into GNN training format."""
    G = build_graph(inputs, outputs, gates)

    node_names = list(G.nodes())
    node_idx   = {name: i for i, name in enumerate(node_names)}

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

    edge_index = [
        [node_idx[src], node_idx[dst]]
        for src, dst in G.edges()
    ]

    return node_features, edge_index, node_names


def collect_sample(gates, inputs, outputs):
    """Creates one training sample from a circuit variation."""
    cost_dict = compute_pac_cost(gates, inputs)

    node_features, edge_index, node_names = gates_to_graph_data(
        gates, inputs, outputs
    )

    return {
        'node_features' : node_features,
        'edge_index'    : edge_index,
        'node_count'    : len(node_names),
        'edge_count'    : len(edge_index),
        'cost'          : cost_dict['total_cost'],
        'power'         : cost_dict['power'],
        'area'          : cost_dict['area'],
        'wirelength'    : cost_dict['wirelength'],
    }


class DataCollector:
    """Wraps optimization runs and collects training data."""

    def __init__(self, max_samples=5000):
        self.samples     = []
        self.max_samples = max_samples

    def record(self, gates, inputs, outputs):
        if len(self.samples) >= self.max_samples:
            return
        sample = collect_sample(gates, inputs, outputs)
        self.samples.append(sample)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.samples, f)
        print(f"[DataCollector] Saved {len(self.samples)} samples to {filepath}")

    def load(self, filepath):
        with open(filepath, 'r') as f:
            self.samples = json.load(f)
        print(f"[DataCollector] Loaded {len(self.samples)} samples from {filepath}")

    @property
    def size(self):
        return len(self.samples)


def generate_training_data(circuit,
                            n_variations=500,
                            save_path=None,
                            verbose=True):
    """
    Generates training data covering the FULL cost range.

    Strategy:
      Tier 1 (20%) — random mutations from original
                     covers cost range ~840-856
      Tier 2 (50%) — samples collected DURING SA runs
                     covers cost range ~700-840 (the range GNN needs)
      Tier 3 (30%) — mutations from SA-optimized circuits
                     covers cost range ~690-750

    This ensures the GNN learns to predict cost at every
    level of optimization, not just near the starting point.
    """
    from optimizer.simulated_annealing import apply_random_mutation, simulated_annealing

    collector = DataCollector(max_samples=n_variations)

    n_tier1 = int(n_variations * 0.20)   # ~100 samples near original
    n_sa    = 3                           # number of SA runs for tier 2+3
    n_tier3 = int(n_variations * 0.30)   # ~150 mutations from SA result

    # ── Tier 1: mutations near original ──────────────────────
    if verbose:
        print(f"\n[DataCollector] Tier 1: {n_tier1} samples near original...")

    collector.record(circuit.gates, circuit.inputs, circuit.outputs)
    current_gates = copy.deepcopy(circuit.gates)

    for i in range(n_tier1 - 1):
        new_gates = apply_random_mutation(current_gates, inputs=circuit.inputs)
        collector.record(new_gates, circuit.inputs, circuit.outputs)
        # Walk away from original — don't reset
        current_gates = new_gates
        if verbose and i % 25 == 0:
            print(f"  Tier 1: {i+1}/{n_tier1}")

    # ── Tier 2+3: run SA, collect samples along the way ──────
    if verbose:
        print(f"\n[DataCollector] Tier 2: collecting samples during SA runs...")

    # Monkey-patch: wrap SA to intercept every accepted move
    import math, random as rnd

    def sa_with_collection(start_gates, collector, circuit,
                           initial_temp=50.0, cooling=0.95,
                           min_temp=0.1, iters=10):
        """Runs SA and records every accepted circuit."""
        GATE_SWAP_MAP = {
            'XOR': ['XNOR', 'NAND'], 'XNOR': ['XOR', 'NAND'],
            'AND': ['NAND', 'OR'],   'OR': ['NOR', 'AND'],
            'NAND': ['AND', 'NOR'],  'NOR': ['OR', 'NAND'],
            'NOT': ['BUFF'],         'BUFF': ['NOT'],
            'DFF': ['DFF'],
        }

        def mutate(gates):
            g = copy.deepcopy(gates)
            target = rnd.choice(list(g.keys()))
            gtype, ginputs = g[target]
            opts = GATE_SWAP_MAP.get(gtype, [])
            if opts:
                g[target] = (rnd.choice(opts), ginputs)
            return g

        current = copy.deepcopy(start_gates)
        current_cost = compute_pac_cost(current, circuit.inputs)['total_cost']
        best = copy.deepcopy(current)
        best_cost = current_cost
        temp = initial_temp

        while temp > min_temp:
            for _ in range(iters):
                new = mutate(current)
                new_cost = compute_pac_cost(new, circuit.inputs)['total_cost']
                delta = new_cost - current_cost
                if delta < 0 or rnd.random() < math.exp(-delta / temp):
                    current = new
                    current_cost = new_cost
                    # Record every accepted move — this is our tier 2 data
                    collector.record(current, circuit.inputs, circuit.outputs)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best = copy.deepcopy(current)
            temp *= cooling

        return best, best_cost

    for run in range(n_sa):
        if verbose:
            print(f"  SA run {run+1}/{n_sa}  "
                  f"(collected so far: {collector.size})")
        best_gates, best_cost = sa_with_collection(
            circuit.gates, collector, circuit
        )

    # ── Tier 3: mutations from SA-optimized circuits ──────────
    if verbose:
        print(f"\n[DataCollector] Tier 3: {n_tier3} mutations from SA result...")

    current_gates = copy.deepcopy(best_gates)
    remaining     = n_variations - collector.size

    for i in range(remaining):
        new_gates = apply_random_mutation(current_gates, inputs=circuit.inputs)
        collector.record(new_gates, circuit.inputs, circuit.outputs)
        current_gates = new_gates
        if verbose and i % 50 == 0:
            print(f"  Tier 3: {i+1}/{remaining}")

    if verbose:
        costs = [s['cost'] for s in collector.samples]
        print(f"\n  Done. Total samples : {collector.size}")
        print(f"  Cost range          : {round(min(costs),2)} – {round(max(costs),2)}")
        print(f"  Mean cost           : {round(sum(costs)/len(costs),2)}")

    if save_path:
        collector.save(save_path)

    return collector


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    from pipeline_v1 import load_circuit

    circuit, _ = load_circuit("data/benchmarks/s1196.bench")

    collector = generate_training_data(
        circuit,
        n_variations = 2000,
        save_path    = "ml/data/s1196_samples_v2.json",
        verbose      = True
    )

    print(f"\nSample 0 (original):")
    s = collector.samples[0]
    print(f"  Cost : {s['cost']}")

    costs = [s['cost'] for s in collector.samples]
    print(f"\nFinal cost range:")
    print(f"  Min : {round(min(costs), 2)}")
    print(f"  Max : {round(max(costs), 2)}")
    print(f"  Mean: {round(sum(costs)/len(costs), 2)}")