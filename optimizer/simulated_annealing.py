# optimizer/simulated_annealing.py
# Simulated Annealing optimizer for circuit PAC cost reduction
#
# How it works:
#   Start with a circuit
#   Make a small random change (mutation)
#   If new circuit is better → accept it
#   If new circuit is worse  → accept it with small probability
#   (this prevents getting stuck in local optima)
#   Cool down over time → become more selective
#   Return best circuit found

import random
import math
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimizer.cost_function  import compute_pac_cost
from core.graph_builder       import build_graph

# ─────────────────────────────────────────────────────────────
# GATE MUTATION RULES
# Which gates can be swapped for which
# Based on logical equivalence and optimization goals
# ─────────────────────────────────────────────────────────────
GATE_SWAP_MAP = {
    # Swap to lower-cost alternatives
    'XOR'  : ['XNOR', 'NAND', 'NOR'],
    'XNOR' : ['XOR',  'NAND', 'NOR'],
    'AND'  : ['NAND'],   # NAND cheaper than AND
    'OR'   : ['NOR'],    # NOR cheaper than OR
    'NAND' : ['NOR'],    # lateral swap
    'NOR'  : ['NAND'],   # lateral swap
    'NOT'  : ['BUFF'],
    'BUFF' : ['NOT'],
    'DFF'  : ['DFF'],
}


# ─────────────────────────────────────────────────────────────
# MUTATION FUNCTIONS
# Each function takes gates dict and returns modified copy
# ─────────────────────────────────────────────────────────────

def mutate_swap_gate(gates):
    """
    Randomly picks one gate and swaps it with
    a cheaper/equivalent gate type.
    Most common mutation — low risk, small change.
    """
    if not gates:
        return gates

    new_gates = copy.deepcopy(gates)
    # Pick a random gate to mutate
    target = random.choice(list(new_gates.keys()))
    gate_type, gate_inputs = new_gates[target]

    # Get possible swaps for this gate type
    options = GATE_SWAP_MAP.get(gate_type, [])
    if options:
        new_type = random.choice(options)
        new_gates[target] = (new_type, gate_inputs)

    return new_gates


def mutate_remove_buffer(gates):
    """
    Finds BUFF gates (which add no logic, just delay)
    and removes them by connecting inputs directly to outputs.
    Reduces area and power.
    """
    new_gates = copy.deepcopy(gates)
    buffers = [
        k for k, (gt, _) in new_gates.items()
        if gt == 'BUFF'
    ]

    if not buffers:
        return new_gates

    # Remove one random buffer
    target = random.choice(buffers)
    _, buff_inputs = new_gates[target]

    if buff_inputs:
        # Reconnect any gate that used target as input
        source = buff_inputs[0]
        for key in new_gates:
            gt, gi = new_gates[key]
            new_gi = [source if x == target else x for x in gi]
            new_gates[key] = (gt, new_gi)

        del new_gates[target]

    return new_gates


def mutate_swap_inputs(gates):
    """
    Picks a random gate and shuffles its input order.
    For commutative gates (AND, OR, NAND, NOR, XOR)
    this doesn't change logic but can affect routing.
    Simulates wire reordering for area optimization.
    """
    commutative = {'AND', 'OR', 'NAND', 'NOR', 'XOR', 'XNOR'}
    new_gates   = copy.deepcopy(gates)

    candidates = [
        k for k, (gt, gi) in new_gates.items()
        if gt in commutative and len(gi) > 1
    ]

    if not candidates:
        return new_gates

    target = random.choice(candidates)
    gt, gi = new_gates[target]
    random.shuffle(gi)
    new_gates[target] = (gt, gi)

    return new_gates


def mutate_add_not(gates, inputs):
    """
    Inserts a NOT gate on a random internal signal.
    Structural mutation — changes circuit topology.
    """
    new_gates     = copy.deepcopy(gates)
    all_signals   = list(new_gates.keys()) + list(inputs)

    if not all_signals:
        return new_gates

    target_signal = random.choice(all_signals)
    new_signal    = f"inv_{target_signal}"

    new_gates[new_signal] = ('NOT', [target_signal])

    candidates = [
        k for k, (gt, gi) in new_gates.items()
        if target_signal in gi and k != new_signal
    ]

    if candidates:
        rewire_target    = random.choice(candidates)
        gt, gi           = new_gates[rewire_target]
        gi               = [new_signal if x == target_signal
                             else x for x in gi]
        new_gates[rewire_target] = (gt, gi)

    return new_gates


def apply_random_mutation(gates, inputs=None):
    """
    Picks one of the mutation strategies at random.
    Called by the SA and GA optimizers each iteration.
    """
    if inputs is None:
        inputs = []

    mutations = [
        lambda g: mutate_swap_gate(g),
        lambda g: mutate_swap_gate(g),
        lambda g: mutate_remove_buffer(g),
        lambda g: mutate_swap_inputs(g),
        lambda g: mutate_add_not(g, inputs),
    ]
    mutation_fn = random.choice(mutations)
    return mutation_fn(gates)


# ─────────────────────────────────────────────────────────────
# SIMULATED ANNEALING CORE
# ─────────────────────────────────────────────────────────────

def simulated_annealing(circuit,
                        initial_temp   = 100.0,
                        cooling_rate   = 0.95,
                        min_temp       = 0.1,
                        iterations_per_temp = 10,
                        verbose        = True):
    """
    Runs Simulated Annealing on a Circuit object.

    Args:
        circuit             : Circuit object from pipeline
        initial_temp        : starting temperature
        cooling_rate        : how fast we cool (0.95 = 5% per step)
        min_temp            : stop when temp drops below this
        iterations_per_temp : how many mutations per temperature
        verbose             : print progress

    Returns:
        best_gates  : optimized gates dict
        best_cost   : lowest cost found
        history     : list of (temp, cost) for plotting
    """

    # ── Setup ────────────────────────────────────────
    current_gates = copy.deepcopy(circuit.gates)
    current_cost  = compute_pac_cost(
        current_gates, circuit.inputs
    )['total_cost']

    best_gates    = copy.deepcopy(current_gates)
    best_cost     = current_cost

    temp          = initial_temp
    history       = []
    iteration     = 0

    if verbose:
        print()
        print("=" * 50)
        print("  SIMULATED ANNEALING OPTIMIZER")
        print("=" * 50)
        print(f"  Circuit     : {circuit.name}")
        print(f"  Gates       : {circuit.gate_count}")
        print(f"  Start cost  : {current_cost}")
        print(f"  Start temp  : {initial_temp}")
        print(f"  Cooling     : {cooling_rate}")
        print("-" * 50)

    # ── Main SA Loop ─────────────────────────────────
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            iteration += 1

            # Mutate current solution
            new_gates = apply_random_mutation(current_gates)
            new_cost  = compute_pac_cost(
                new_gates, circuit.inputs
            )['total_cost']

            # Calculate cost difference
            delta = new_cost - current_cost

            # Accept decision
            if delta < 0:
                # Better solution — always accept
                current_gates = new_gates
                current_cost  = new_cost

                # Update best if improved
                if current_cost < best_cost:
                    best_gates = copy.deepcopy(current_gates)
                    best_cost  = current_cost

            else:
                # Worse solution — accept with probability
                # Higher temp = more likely to accept bad moves
                probability = math.exp(-delta / temp)
                if random.random() < probability:
                    current_gates = new_gates
                    current_cost  = new_cost

        # Record history
        history.append((round(temp, 4), round(current_cost, 4)))

        # Cool down
        temp *= cooling_rate

    if verbose:
        print(f"  Iterations  : {iteration}")
        print(f"  Final temp  : {round(temp, 4)}")
        print(f"  Best cost   : {best_cost}")
        improvement = ((circuit.cost - best_cost) / circuit.cost) * 100
        print(f"  Improvement : {round(improvement, 2)}%")
        print("=" * 50)

    return best_gates, best_cost, history


# ── Quick test ───────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from core.pipeline import load_circuit

    # Test on c17
    print("Loading circuit...")
    circuit, _ = load_circuit("data/benchmarks/c17.bench")

    print(f"Original cost: {circuit.cost}")

    # Run SA
    best_gates, best_cost, history = simulated_annealing(
        circuit,
        initial_temp        = 100.0,
        cooling_rate        = 0.95,
        min_temp            = 0.1,
        iterations_per_temp = 10,
        verbose             = True
    )

    print(f"\nOriginal cost : {circuit.cost}")
    print(f"Optimized cost: {best_cost}")
    improvement = ((circuit.cost - best_cost) / circuit.cost) * 100
    print(f"Improvement   : {round(improvement, 2)}%")

    # Show cost over time
    print(f"\nCost history (temp → cost):")
    for i, (temp, cost) in enumerate(history):
        if i % 5 == 0:  # print every 5th entry
            print(f"  Temp: {temp:8.4f} → Cost: {cost}")