# ml/data_collector.py
# Multi-circuit GNN training data collector.
# Uses correct pattern-based mutations (not old random gate swaps).
# Trains across 6 circuits spanning 292-1193 gates.
# Node features normalized by sqrt(gate_count) for cross-circuit generalization.

import os
import json
import copy
import sys
import math
import random
import networkx as nx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.cost_function import compute_pac_cost
from core.graph_builder      import build_graph
from concurrent.futures import ProcessPoolExecutor, as_completed

GATE_TYPE_MAP = {
    'INPUT': 0, 'OUTPUT': 1, 'NOT': 2, 'BUFF': 3,
    'AND': 4, 'OR': 5, 'NAND': 6, 'NOR': 7,
    'XOR': 8, 'XNOR': 9, 'DFF': 10, 'UNKNOWN': 11,
}

SAMPLES_PER_CIRCUIT = 500

TRAINING_CIRCUITS = [
    # use your full list — skip files that don't exist
    # data_collector already does os.path.exists() check
    "data/benchmarks/c432.v",
    "data/benchmarks/c499.v",
    "data/benchmarks/c6288.bench",
    "data/benchmarks/c7552.bench",
    "data/benchmarks/s349.bench",
    "data/benchmarks/s382.bench",
    "data/benchmarks/s400.bench",
    "data/benchmarks/s344.bench",
    "data/benchmarks/s832.bench",
    "data/benchmarks/s386.bench",
    "data/benchmarks/s444.bench",
    "data/benchmarks/s510.bench",
    "data/benchmarks/s641.bench",
    "data/benchmarks/s713.bench",
    "data/benchmarks/s820.bench",
    "data/benchmarks/c880.bench",
    "data/benchmarks/s953.bench",
    "data/benchmarks/s1196.bench",
    "data/benchmarks/s1238.bench",
    "data/benchmarks/c1355.bench",
    "data/benchmarks/s1488.bench",
    "data/benchmarks/s1494.bench",
    "data/benchmarks/c1908.bench",
    "data/benchmarks/c2670.bench",
    "data/benchmarks/c3540.bench",
    "data/benchmarks/c5315.bench",
    "data/benchmarks/s5378.bench",
    "data/benchmarks/s9234.bench", # extra 
    "data/benchmarks/s13207.bench",
    "data/benchmarks/s15850.bench", #extra after 64 DIM
    "data/benchmarks/s35932.bench", # extra 
    "data/benchmarks/s38417.bench", # extra


]


def gates_to_graph_data(gates, inputs, outputs, gate_count: int):
    """
    Converts gates to GNN training format.
    fan_in and fan_out normalized by sqrt(gate_count)
    so features are comparable across circuit sizes.
    """
    G      = build_graph(inputs, outputs, gates)
    nodes  = list(G.nodes())
    node_idx = {name: i for i, name in enumerate(nodes)}
    norm   = max(1.0, math.sqrt(gate_count))

    node_features = []
    depth_map = {}

    try:
        topo_order = list(nx.topological_sort(G))

        for node in topo_order:

            preds = list(G.predecessors(node))

            if len(preds) == 0:
                depth_map[node] = 0
            else:
                depth_map[node] = 1 + max(
                    depth_map[p] for p in preds
                )

    except Exception:
        # fallback if graph somehow malformed
        for node in G.nodes():
            depth_map[node] = 0
    
    for node in nodes:
        data  = G.nodes[node]
        ntype = data.get('node_type', 'gate')
        gtype = data.get('gate_type', 'UNKNOWN')

        depth = depth_map[node] / max(1.0, math.log2(gate_count))

        if ntype == 'input':
            type_id = GATE_TYPE_MAP['INPUT']
        elif ntype == 'output':
            type_id = GATE_TYPE_MAP.get(gtype, GATE_TYPE_MAP['OUTPUT'])
        else:
            type_id = GATE_TYPE_MAP.get(gtype, GATE_TYPE_MAP['UNKNOWN'])

        fan_in  = round(G.in_degree(node)  / norm, 4)
        fan_out = round(G.out_degree(node) / norm, 4)
        # node_features.append([type_id, fan_in, fan_out])
        # node_features.append([type_id, fan_in, fan_out, round(depth, 4)])
        relative_size = round(gate_count / 5000.0, 4)

        node_features.append([
            type_id,
            fan_in,
            fan_out,
            round(depth, 4),
            relative_size
        ])

    edge_index = [
        [node_idx[src], node_idx[dst]]
        for src, dst in G.edges()
    ]
    return node_features, edge_index




def collect_sample(
        gates,
        inputs,
        outputs,
        gate_count: int,
        mutation_depth: int = 0):
    cost_dict = compute_pac_cost(gates, inputs)
    node_features, edge_index = gates_to_graph_data(
        gates, inputs, outputs, gate_count
    )
    return {
        'node_features': node_features,
        'edge_index'   : edge_index,
        'node_count'   : len(node_features),
        'edge_count'   : len(edge_index),
        'gate_count'   : gate_count,
        'cost'         : cost_dict['total_cost'],
        'power'        : cost_dict['power'],
        'area'         : cost_dict['area'],
        'wirelength'   : cost_dict['wirelength'],
        'mutation_depth': mutation_depth,
    }


def _sa_walk(start_gates, record_fn, circuit,
              initial_temp=100.0, cooling=0.95,
              min_temp=0.1, iters=10):
    """Runs SA with correct mutations, records every accepted move."""
    from optimizer.mutations import apply_safe_mutation

    current      = copy.deepcopy(start_gates)
    current_cost = compute_pac_cost(current, circuit.inputs)['total_cost']
    best         = copy.deepcopy(current)
    best_cost    = current_cost
    temp         = initial_temp

    while temp > min_temp:
        for _ in range(iters):
            new_gates = apply_safe_mutation(
                circuit.inputs, circuit.outputs, current,
                max_attempts=15, validate=False
            )
            if new_gates is None:
                continue
            new_cost = compute_pac_cost(new_gates, circuit.inputs)['total_cost']
            delta    = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current      = new_gates
                current_cost = new_cost
                record_fn(current)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best      = copy.deepcopy(current)
        temp *= cooling
    return best, best_cost


def generate_samples_for_circuit(circuit, n_samples: int,
                                   verbose: bool = True) -> list:
    """
    Three-tier sampling:
      Tier 1 (30%) — mutations near original
      Tier 2 (50%) — SA walk (covers full cost range)
      Tier 3 (20%) — mutations from SA result
    """
    from optimizer.mutations import apply_safe_mutation

    samples = []
    gc      = circuit.gate_count
    n1      = int(n_samples * 0.30)
    n2      = int(n_samples * 0.50)
    n3      = n_samples - n1 - n2

    if verbose:
        print(f"    {circuit.name} ({gc} gates): "
              f"tier1={n1} tier2={n2} tier3={n3}")

    # Original circuit always first
    samples.append(collect_sample(
        circuit.gates, circuit.inputs, circuit.outputs, gc
    ))

    # Tier 1
    current = copy.deepcopy(circuit.gates)
    depth = 0
    for _ in range(n1 - 1):
        m = apply_safe_mutation(circuit.inputs, circuit.outputs,
                                 current, max_attempts=15, validate=False)
        depth += 1

        if m is not None:
            samples.append(collect_sample(m, circuit.inputs,
                                           circuit.outputs, gc,         mutation_depth=depth))
            current = m

    # Tier 2 — SA walk
    sa_collected = []

    def record(gates):
        if len(sa_collected) < n2:
            sa_collected.append(collect_sample(
                gates, circuit.inputs, circuit.outputs, gc))

    best_gates, _ = _sa_walk(
        circuit.gates, record, circuit,
        initial_temp=100.0, cooling=0.95, min_temp=0.1,
        iters=max(5, gc // 100)
    )
    if len(sa_collected) < n2:
        _sa_walk(circuit.gates, record, circuit,
                  initial_temp=50.0, cooling=0.97, min_temp=0.1,
                  iters=max(5, gc // 100))

    samples.extend(sa_collected)

    # Tier 3 — from SA result
    current = copy.deepcopy(best_gates)
    depth = 0

    for _ in range(n3):
        m = apply_safe_mutation(circuit.inputs, circuit.outputs,
                                 current, max_attempts=15, validate=False)
        depth+=1
        if m is not None:
            samples.append(collect_sample(m, circuit.inputs,
                                           circuit.outputs, gc,         mutation_depth=depth))
            current = m

    if verbose:
        costs = [s['cost'] for s in samples]
        print(f"      collected={len(samples)}  "
              f"range=[{round(min(costs),1)}, {round(max(costs),1)}]")

    return samples

def worker_task(filepath, base_samples, verbose=True):
    """
    Runs one circuit collection in a separate process.
    """

    from core.pipeline import load_circuit

    if not os.path.exists(filepath):
        return {
            "filepath": filepath,
            "samples": [],
            "skipped": True
        }

    circuit, _ = load_circuit(filepath)
    gc = circuit.gate_count

    # Adaptive sampling
    if gc < 300:
        adaptive_samples = int(base_samples * 0.6)

    elif gc < 700:
        adaptive_samples = int(base_samples * 0.8)

    elif gc < 1500:
        adaptive_samples = int(base_samples * 1)

    elif gc < 3000:
        adaptive_samples = int(base_samples * 1.2)

    else:
        adaptive_samples = int(base_samples * 1.3)

    samples = generate_samples_for_circuit(
        circuit,
        adaptive_samples,
        verbose=verbose
    )

    return {
        "filepath": filepath,
        "samples": samples,
        "gate_count": gc,
        "sample_count": len(samples),
        "skipped": False
    }

def generate_multi_circuit_data(
        circuits=None,
        samples_per_circuit: int = SAMPLES_PER_CIRCUIT,
        save_path: str = "ml/data/multi_circuit_samples.json",
        verbose: bool = True) -> list:

    """
    Generates training data across multiple circuits
    using multiprocessing for major speedup.
    """

    if circuits is None:
        circuits = TRAINING_CIRCUITS

    all_samples = []

    if verbose:
        print("=" * 60)
        print("  MULTI-CIRCUIT GNN DATA COLLECTION")
        print("=" * 60)
        print(f"  Circuits    : {len(circuits)}")
        print(f"  Base samples: {samples_per_circuit}")
        print(f"  CPU workers : {os.cpu_count()}")
        print("-" * 60)

    with ProcessPoolExecutor(
            max_workers=max(1, os.cpu_count() - 1)
    ) as executor:

        futures = []

        for filepath in circuits:
            futures.append(
                executor.submit(
                    worker_task,
                    filepath,
                    samples_per_circuit,
                    verbose
                )
            )

        for future in as_completed(futures):

            result = future.result()

            if result["skipped"]:
                print(f"  [SKIP] {result['filepath']}")
                continue

            all_samples.extend(result["samples"])

            if verbose:
                print(
                    f"  [DONE] "
                    f"{os.path.basename(result['filepath'])} | "
                    f"gates={result['gate_count']} | "
                    f"samples={result['sample_count']}"
                )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(all_samples, f)

    if verbose and len(all_samples) > 0:
        costs = [s['cost'] for s in all_samples]
        gcs   = [s['gate_count'] for s in all_samples]

        print("-" * 60)
        print(f"  Total samples : {len(all_samples)}")
        print(f"  Cost range    : [{round(min(costs),1)}, {round(max(costs),1)}]")
        print(f"  Gate range    : [{min(gcs)}, {max(gcs)}]")
        print(f"  Saved         : {save_path}")
        print("=" * 60)

    return all_samples

# Legacy single-circuit collector (backward compat for old code)
class DataCollector:
    def __init__(self, max_samples=5000):
        self.samples     = []
        self.max_samples = max_samples

    def record(self, gates, inputs, outputs):
        if len(self.samples) >= self.max_samples:
            return
        gc = len(gates)
        self.samples.append(collect_sample(gates, inputs, outputs, gc))

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.samples, f)
        print(f"[DataCollector] Saved {len(self.samples)} to {filepath}")

    @property
    def size(self):
        return len(self.samples)


if __name__ == "__main__":
    samples = generate_multi_circuit_data(verbose=True)

    from collections import Counter
    dist = Counter(s['gate_count'] for s in samples)
    print("\nDistribution by gate count:")
    for gc, cnt in sorted(dist.items()):
        print(f"  {gc:5d} gates : {cnt} samples")